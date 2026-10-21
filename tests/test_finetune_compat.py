"""finetune_lora._make_config must survive TRL / transformers API renames
(the script itself needs a GPU, so only the compatibility layer is tested)."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "finetune_lora", Path(__file__).resolve().parents[1] / "training" / "finetune_lora.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)


@dataclass
class NewConfig:  # transformers 5 / recent TRL: no warmup_ratio, warmup_steps takes a fraction
    output_dir: str
    warmup_steps: float = 0
    max_length: int = 1024
    eval_strategy: str = "no"
    completion_only_loss: bool | None = None


@dataclass
class OldConfig:  # older stack
    output_dir: str
    warmup_ratio: float = 0.0
    max_seq_length: int = 1024
    evaluation_strategy: str = "no"


def test_new_stack_maps_warmup_ratio():
    c = ft._make_config(NewConfig, output_dir="o", warmup_ratio=0.03, max_length=4096,
                        eval_strategy="steps", completion_only_loss=True)
    assert c.warmup_steps == 0.03 and c.max_length == 4096 and c.completion_only_loss


def test_old_stack_maps_new_names_back(capsys):
    c = ft._make_config(OldConfig, output_dir="o", warmup_ratio=0.03, max_length=4096,
                        eval_strategy="steps", completion_only_loss=True)
    assert c.warmup_ratio == 0.03 and c.max_seq_length == 4096 and c.evaluation_strategy == "steps"
    assert "completion_only_loss" in capsys.readouterr().out  # unknown option skipped, not a crash


def test_validate_only_runs_without_ml_stack(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, '
                 '{"role": "assistant", "content": "a"}]}', encoding="utf-8")
    ft.validate([p])
    with pytest.raises(SystemExit):
        ft.validate([tmp_path / "missing.jsonl"])
