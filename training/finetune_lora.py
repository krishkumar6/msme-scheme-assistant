"""QLoRA supervised fine-tuning of an open-weights model on the SFT set.

Needs a CUDA GPU. No GPU of your own? The free T4 on Google Colab or Kaggle is
enough for 1.5B models in 4-bit - see training/finetune_colab.ipynb, which runs
this script and exports a GGUF file that Ollama serves on a normal CPU.
Extras: pip install -e ".[finetune]"

    # sanity-check the dataset anywhere (stdlib only, no GPU):
    python training/finetune_lora.py --validate-only

    # train, then merge the adapter into full weights for serving:
    python training/finetune_lora.py --base Qwen/Qwen2.5-1.5B-Instruct --epochs 2 --merge

    # serve with any OpenAI-compatible server and point the app at it:
    vllm serve out/nabard-qa-lora/merged --served-model-name nabard-qa-lora --port 8001
    NABARD_QA_LLM_PROVIDER=local nabard-qa serve

Loss is computed on assistant tokens only, so the model learns to answer from
the provided sources (and to refuse when they don't contain the answer) rather
than to reproduce the long source context.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "training"


def validate(paths: list[Path]) -> None:
    """Structural checks + rough length stats, no third-party imports."""
    for path in paths:
        if not path.exists():
            sys.exit(f"missing {path} - run training/build_sft_dataset.py first")
        # "\n" only: splitlines() also splits on U+2028/U+0085 inside passage text
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
        problems, lengths, refusals = 0, [], 0
        for i, r in enumerate(rows):
            roles = [m["role"] for m in r.get("messages", [])]
            if roles != ["system", "user", "assistant"] or not all(m["content"].strip() for m in r["messages"]):
                problems += 1
                print(f"  {path.name}:{i + 1} bad message structure {roles}")
            lengths.append(sum(len(m["content"]) for m in r["messages"]) // 4)  # ~4 chars/token
            refusals += bool(r.get("meta", {}).get("refusal"))
        if lengths:
            print(f"{path.name}: {len(rows)} examples, {problems} problems, {refusals} refusal targets, "
                  f"~tokens median {statistics.median(lengths):.0f} / max {max(lengths)}")
        else:
            print(f"{path.name}: empty")


def _dtype_kwarg(dtype) -> dict:  # noqa: ANN001
    """transformers >= 4.56 renamed from_pretrained(torch_dtype=) to dtype=."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def _make_config(config_cls, **kw):  # noqa: ANN001, ANN003
    """Build SFTConfig across TRL / transformers versions.

    Renamed options are mapped (warmup_ratio -> warmup_steps as a fraction in
    transformers 5, max_length <-> max_seq_length, eval_strategy <->
    evaluation_strategy); options this version doesn't know are dropped with a
    warning instead of crashing."""
    import inspect

    params = inspect.signature(config_cls.__init__).parameters
    renames = {"warmup_ratio": "warmup_steps", "max_length": "max_seq_length",
               "eval_strategy": "evaluation_strategy"}
    for new, old in renames.items():
        if new in kw and new not in params and old in params:
            kw[old] = kw.pop(new)  # transformers 5: warmup_steps < 1 is read as a ratio
    for key in [k for k in kw if k not in params]:
        print(f"note: this TRL/transformers version has no '{key}' option; skipping it")
        kw.pop(key)
    return config_cls(**kw)


def train(args: argparse.Namespace) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not found. Use a free Colab/Kaggle T4 (training/finetune_colab.ipynb), "
                 "or --validate-only to check the data here.")
    # Native bf16 needs compute capability >= 8.0 (A100, L4, RTX 30xx+). On a T4 (7.5,
    # free Colab/Kaggle) recent PyTorch reports bf16 as "supported" via slow emulation,
    # so check the hardware directly and use fp16 there.
    major, minor = torch.cuda.get_device_capability(0)
    bf16 = major >= 8
    dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{major}{minor}) | precision: {'bf16' if bf16 else 'fp16'}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        **_dtype_kwarg(dtype),
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
    )

    data = load_dataset("json", data_files={"train": str(args.train), "validation": str(args.val)})
    # Conversational prompt/completion format: TRL computes the loss on the
    # completion (assistant answer) only, for any chat template.
    data = data.map(
        lambda r: {"prompt": r["messages"][:-1], "completion": r["messages"][-1:]},
        remove_columns=data["train"].column_names,
    )

    peft_config = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    config = _make_config(
        SFTConfig,
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        logging_steps=10,
        # ~350 examples / (batch 1 x accum 16) = ~22 steps per epoch, so check often
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_length=args.max_len,
        completion_only_loss=True,  # learn the answers, not the copied sources
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model, args=config, train_dataset=data["train"], eval_dataset=data["validation"],
        processing_class=tokenizer, peft_config=peft_config,
    )
    # fp16 mixed precision keeps fp32 master weights: torch's GradScaler cannot unscale
    # fp16/bf16 gradients. Recent stacks create the LoRA weights in the checkpoint's dtype
    # (bf16 for Qwen2.5), so upcast the trainable tensors before the optimizer is built.
    trainable = [p for p in trainer.model.parameters() if p.requires_grad]
    dtypes = sorted({str(p.dtype) for p in trainable})
    if not bf16:
        for p in trainable:
            if p.dtype != torch.float32:
                p.data = p.data.float()
    n_params = sum(p.numel() for p in trainable)
    print(f"trainable: {len(trainable)} tensors, {n_params / 1e6:.1f}M params "
          f"(were {', '.join(dtypes)}; now {'bf16' if bf16 else 'fp32 master weights, fp16 compute'})")
    trainer.train()
    trainer.save_model(str(args.out / "adapter"))
    tokenizer.save_pretrained(str(args.out / "adapter"))
    print("eval:", trainer.evaluate())

    if args.merge:
        import gc

        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        merge(args.out / "adapter", args.out / "merged")


def merge(adapter: Path, out: Path) -> None:
    """Fold the LoRA weights into a plain fp16 copy of the base model (CPU, no quantization).

    Base and adapter are loaded separately: AutoPeftModelForCausalLM forwards
    from_pretrained kwargs into PEFT's adapter injection, which breaks on newer stacks."""
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:  # Colab ships torchao 0.10; recent PEFT raises on any old torchao while injecting
        import peft.import_utils
        import peft.tuners.lora.torchao as tao

        peft.import_utils.is_torchao_available = tao.is_torchao_available = lambda: False
    except ImportError:
        pass
    if not (adapter / "adapter_config.json").exists():
        sys.exit(f"no trained adapter in {adapter} - run training first")
    base_id = PeftConfig.from_pretrained(str(adapter)).base_model_name_or_path
    base = AutoModelForCausalLM.from_pretrained(base_id, **_dtype_kwarg(torch.float16), low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(str(out))
    print("merged model written to", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", type=Path, default=DATA / "sft_train.jsonl")
    ap.add_argument("--val", type=Path, default=DATA / "sft_val.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "nabard-qa-lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch-size", type=int, default=1)  # fits a 16 GB T4 at 4k tokens
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--merge-only", action="store_true", help="skip training; merge an already saved adapter")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    if args.merge_only:
        merge(args.out / "adapter", args.out / "merged")
        return
    validate([args.train, args.val])
    if not args.validate_only:
        train(args)


if __name__ == "__main__":
    main()
