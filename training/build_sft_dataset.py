"""Build the fine-tuning dataset from the indexed NABARD corpus.

Steps
1. Sample content-rich passages from the vector store (stratified by report
   year and text/table).
2. Ask an LLM (Groq gpt-oss by default, Gemini as the alternative) for grounded
   question/answer pairs per passage, validated with Pydantic. Resumable:
   passages already processed are skipped; failed calls are retried next run.
3. Assemble RAFT-style SFT examples in chat format: the "oracle" passage is
   mixed with retrieved distractor passages under shuffled [S#] ids and the
   target answer cites the oracle. ~15% of examples contain only distractors and
   the target is a refusal, which teaches the model to say "not in the reports".
4. Thumbs-up answers from the feedback store are added as extra examples.
5. Each synthetic question is also stored as a retrieval training pair
   (origin="synthetic") for the query adapter.

Outputs (data/training/): synthetic_qa.jsonl, sft_train.jsonl, sft_val.jsonl

    python training/build_sft_dataset.py --limit 150                     # Groq (default)
    python training/build_sft_dataset.py --provider gemini --limit 150
    python training/build_sft_dataset.py --assemble-only                 # rebuild SFT files, no API calls
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from nabard_qa.config import get_settings
from nabard_qa.generation.prompts import SYSTEM_PROMPT, build_user_message
from nabard_qa.schemas import Chunk, ScoredChunk
from nabard_qa.service import QAService

log = logging.getLogger("build_sft")

GEN_SYSTEM = """\
You create evaluation and training data for a question-answering assistant over \
MSME scheme documents (scheme guidelines, FAQs, state MSME policies, Ministry of \
MSME annual reports) and NABARD annual reports. Given one passage, write questions \
that a farmer, an artisan, an MSME / agri-entrepreneur, an FPO or SHG leader, a \
bank officer or a policy researcher might genuinely ask, which this passage fully \
answers.

Rules
- If the passage sets scheme eligibility, subsidy, loan limits or terms, make one \
question first-person, the way an applicant asks about their own case (e.g. "I am \
a woman starting a 20 lakh unit in a village. How much PMEGP subsidy can I get?"), \
and answer it by applying the passage's rules to that case.
- Each question must be answerable from the passage alone, and must make sense \
without seeing it: name the scheme, fund, institution and year explicitly; never \
say "the passage", "this table" or "the above".
- Vary the type: exact figures, eligibility / terms, how something works, and \
comparisons within the passage.
- Answers: 1-4 sentences, copy figures exactly with units and dates, and \
mark every claim with [S1]. Do not add facts that are not in the passage.
- If the passage is boilerplate (addresses, signatures, a list of names) or too \
fragmentary to support a real question, return an empty list."""


class QAPair(BaseModel):
    question: str = Field(description="Standalone user question")
    answer: str = Field(description="Grounded answer citing [S1]")
    question_type: Literal["figure", "eligibility", "explanation", "comparison", "other"]


class QAPairs(BaseModel):
    pairs: list[QAPair]


REFUSAL = ("The documents I have do not contain this information. "
           "Please check the scheme's official guidelines or portal, or ask the implementing agency.")


# Passages worth training on describe what an applicant can get or must meet. Annual
# reports also hold admin chapters (Hindi Pakhwada, vigilance, staff training, receipt
# formats) that only produce trivia questions.
_SCHEME_CONTENT = re.compile(
    r"\b(subsid\w*|eligib\w*|loan|credit|guarantee|margin money|interest|collateral|refinance|"
    r"incentive|beneficiar\w*|entrepreneur\w*|grant|assistance|stipend|scheme)\b", re.I)
_ADMIN = re.compile(r"\b(Hindi Pakhwada|Rajbhasha|Official Language|vigilance|RTI|pre-receipt|"
                    r"Received a sum of|workshop was organi[sz]ed|Swachhta)\b", re.I)


def is_scheme_content(c: Chunk) -> bool:
    return len(_SCHEME_CONTENT.findall(c.text)) >= 3 and not _ADMIN.search(c.text)


def sample_chunks(chunks: list[Chunk], limit: int, seed: int, match: str | None = None) -> list[Chunk]:
    """Round-robin over (document, kind) strata, so small scheme documents get as many
    passages as 300-page annual reports. ``match``: regex on the document title; when
    set, only passages about scheme benefits and conditions are kept."""
    rich = [c for c in chunks if len(c.text.split()) >= 60 and (not match or re.search(match, c.report))
            and (not match or is_scheme_content(c))]
    strata: dict[tuple, list[Chunk]] = defaultdict(list)
    for c in rich:
        strata[(c.report, c.kind)].append(c)
    rng = random.Random(seed)
    for v in strata.values():
        rng.shuffle(v)
    out: list[Chunk] = []
    while len(out) < limit and any(strata.values()):
        for key in list(strata):
            if strata[key] and len(out) < limit:
                out.append(strata[key].pop())
    return out


JSON_INSTRUCTIONS = """

Respond with a single JSON object and nothing else, in exactly this shape:
{"pairs": [{"question": "...", "answer": "... [S1]", "question_type": "figure|eligibility|explanation|comparison|other"}]}
Use {"pairs": []} when the passage cannot support a real question."""


class PairGenerator:
    """Synthetic Q&A writer: ``groq`` or ``gemini`` through their OpenAI-compatible
    JSON mode, validated with Pydantic."""

    # after this many back-to-back rate-limit waits, move to the fallback provider
    SWITCH_AFTER_429S = 3

    def __init__(self, provider: str, settings, fallback: str | None = None) -> None:  # noqa: ANN001
        self.s = settings
        self.fallback = fallback if fallback and fallback != provider else None
        self._configure(provider)

    def _configure(self, provider: str) -> None:
        self.provider = provider
        cfg = {
            "groq": (self.s.groq_api_key, self.s.groq_base_url, self.s.groq_model, self.s.groq_reasoning_effort),
            "gemini": (self.s.gemini_api_key, self.s.gemini_base_url, self.s.gemini_model,
                       self.s.gemini_reasoning_effort),
        }[provider]
        self.key, self.base_url, self.model, self.effort = cfg
        if not self.key:
            raise SystemExit(f"{provider.upper()}_API_KEY is not set in .env")

    def _switch_to_fallback(self, why: str) -> bool:
        if not self.fallback:
            return False
        log.warning("%s: %s - switching to %s for the remaining passages", self.provider, why, self.fallback)
        nxt, self.fallback = self.fallback, None
        self._configure(nxt)
        return True

    def __call__(self, user: str) -> list[QAPair] | None:
        for attempt in range(2):  # one retry if the JSON doesn't validate
            text = self._post_json(user)
            if text is None:
                return None
            try:
                return QAPairs.model_validate_json(_strip_fences(text)).pairs
            except ValueError as e:
                log.warning("invalid JSON from %s (attempt %d): %s", self.provider, attempt + 1, str(e)[:200])
        return None

    def _request(self, user: str) -> urllib.request.Request:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": GEN_SYSTEM + JSON_INSTRUCTIONS},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "temperature": 0.4,
            "max_tokens": 3000,
        }
        if self.effort:
            body["reasoning_effort"] = self.effort
        return urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}",
                     "User-Agent": "nabard-qa/0.1"},
        )

    def _post_json(self, user: str) -> str | None:
        req = self._request(user)
        rate_limited = 0
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.load(resp)["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    detail = e.read().decode("utf-8", "replace") if e.code == 429 else ""
                    rate_limited += e.code == 429
                    # a daily cap won't lift by waiting; a per-minute limit that keeps
                    # recurring is cheaper to escape than to sit out
                    if (re.search(r"per day|\bTPD\b|\bRPD\b|daily", detail, re.I)
                            or rate_limited >= self.SWITCH_AFTER_429S):
                        why = "daily limit reached" if re.search(r"per day|TPD|RPD|daily", detail, re.I) \
                            else f"{rate_limited} rate limits in a row"
                        if self._switch_to_fallback(why):
                            req, rate_limited = self._request(user), 0
                            continue
                    # free tiers: Groq ~8k tokens/min, Gemini a few requests/min -> wait as told
                    try:
                        wait = float(e.headers.get("retry-after", ""))
                    except ValueError:
                        wait = 5.0 * 2**attempt
                    wait = min(max(wait, 1.0), 90.0)
                    log.info("%s HTTP %s, waiting %.0fs", self.provider, e.code, wait)
                    time.sleep(wait)
                    continue
                log.error("%s HTTP %s: %s", self.provider, e.code, e.read().decode("utf-8", "replace")[:300])
                return None
            except OSError as e:
                log.warning("%s network error: %s", self.provider, e)
                time.sleep(5)
        log.error("%s: giving up after repeated rate limits", self.provider)
        return None


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text


def generate_pairs(gen: PairGenerator, chunks: list[Chunk], out_path: Path, per_chunk: int) -> None:
    done = set()
    if out_path.exists():
        done = {json.loads(line)["chunk_id"] for line in out_path.read_text(encoding="utf-8").split("\n") if line}
    todo = [c for c in chunks if c.id not in done]
    log.info("%d passages to process with %s:%s (%d already done)", len(todo), gen.provider, gen.model, len(done))
    ok = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for i, c in enumerate(todo, start=1):
            passage = f"Report: {c.report}\nSection: {c.section or '-'}\nPages: {c.page_label}\n\n{c.text}"
            pairs = gen(f"Write up to {per_chunk} question/answer pairs.\n\n<passage>\n{passage}\n</passage>")
            if pairs is None:
                continue  # failed call: not recorded, so a re-run retries this passage
            pairs = [p for p in pairs[:per_chunk] if "[S1]" in p.answer and len(p.question) > 15]
            fh.write(json.dumps({"chunk_id": c.id, "generator": f"{gen.provider}:{gen.model}",
                                 "pairs": [p.model_dump() for p in pairs]}, ensure_ascii=False) + "\n")
            fh.flush()
            ok += 1
            if i % 10 == 0 or i == len(todo):
                log.info("processed %d/%d passages (%d succeeded)", i, len(todo), ok)


def assemble(svc: QAService, qa_path: Path, out_dir: Path, distractors: int, refusal_rate: float,
             val_fraction: float, seed: int) -> None:
    rng = random.Random(seed)
    chunks, _ = svc.store.all()
    by_id = {c.id: c for c in chunks}
    examples, retrieval_pairs = [], []

    for line in qa_path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        oracle = by_id.get(row["chunk_id"])
        if oracle is None:
            continue
        family = (oracle.section or "").split(" > ")[0]
        for pair in row["pairs"]:
            q = pair["question"]
            retrieved = svc.retriever.retrieve(q, k=distractors + 4).chunks
            negatives = [sc.chunk for sc in retrieved
                         if sc.chunk.id != oracle.id and (sc.chunk.section or "").split(" > ")[0] != family][:distractors]
            retrieval_pairs.append((q, oracle.id, [n.id for n in negatives]))
            refusal = rng.random() < refusal_rate and len(negatives) >= 2
            ctx = list(negatives) if refusal else [oracle, *negatives]
            rng.shuffle(ctx)
            if refusal:
                answer = REFUSAL
            else:
                ref = f"S{ctx.index(oracle) + 1}"
                answer = pair["answer"].replace("[S1]", f"[{ref}]")
            scored = [ScoredChunk(chunk=c, score=0.0) for c in ctx]
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(q, scored)},
                    {"role": "assistant", "content": answer},
                ],
                "meta": {"origin": "synthetic", "chunk_id": oracle.id, "refusal": refusal, "type": pair["question_type"]},
            })

    # thumbs-up answers from real users
    for it in svc.feedback.export_interactions():
        if it.get("rating") != 1:
            continue
        ctx = [by_id[r["chunk_id"]] for r in it["retrieved"] if r["chunk_id"] in by_id]
        if not ctx:
            continue
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(it["question"], [ScoredChunk(chunk=c, score=0.0) for c in ctx])},
                {"role": "assistant", "content": it["answer"]},
            ],
            "meta": {"origin": "feedback", "answer_id": it["answer_id"]},
        })

    # split by source passage so validation questions never share a passage with training
    keys = sorted({e["meta"].get("chunk_id") or e["meta"].get("answer_id") for e in examples})
    rng.shuffle(keys)
    val_keys = set(keys[: max(1, int(len(keys) * val_fraction))]) if keys else set()
    def key(e: dict) -> str:
        return e["meta"].get("chunk_id") or e["meta"].get("answer_id")

    train = [e for e in examples if key(e) not in val_keys]
    val = [e for e in examples if key(e) in val_keys]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("sft_train.jsonl", train), ("sft_val.jsonl", val)):
        (out_dir / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    added = svc.feedback.add_training_pairs("synthetic", retrieval_pairs)
    log.info("SFT: %d train / %d val examples; %d new retrieval pairs for the adapter", len(train), len(val), added)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["groq", "gemini"], default="groq",
                    help="who writes the Q&A. groq (gpt-oss, Apache-2.0 open weights) is the cleanest licence "
                         "for training data; check Google's terms before training on Gemini output")
    ap.add_argument("--limit", type=int, default=200, help="passages to generate questions for")
    ap.add_argument("--fallback-provider", choices=["groq", "gemini"],
                    help="switch to this provider when the main one hits its daily limit "
                         "(check its terms on training models with its output)")
    ap.add_argument("--match", help='only documents whose title matches this regex, e.g. "^(?!NABARD)"')
    ap.add_argument("--per-chunk", type=int, default=3)
    ap.add_argument("--distractors", type=int, default=3)
    ap.add_argument("--refusal-rate", type=float, default=0.15)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--assemble-only", action="store_true", help="skip generation, rebuild SFT files")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    s = get_settings()
    out_dir = s.data_dir / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_path = out_dir / "synthetic_qa.jsonl"
    svc = QAService.from_settings(s)

    if not args.assemble_only:
        gen = PairGenerator(args.provider, s, fallback=args.fallback_provider)
        chunks, _ = svc.store.all()
        picked = sample_chunks(chunks, args.limit, args.seed, args.match)
        log.info("sampled %d passages; roughly %d API calls", len(picked), len(picked))
        generate_pairs(gen, picked, qa_path, args.per_chunk)
    if not qa_path.exists():
        raise SystemExit(f"{qa_path} not found - run without --assemble-only first")
    assemble(svc, qa_path, out_dir, args.distractors, args.refusal_rate, args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
