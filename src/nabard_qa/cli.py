"""Command-line entry point: ``nabard-qa <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import PROJECT_ROOT, get_settings


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpx2", "chromadb", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_ingest(args: argparse.Namespace) -> None:
    from .ingestion.pipeline import ingest
    from .retrieval.embeddings import Embedder
    from .retrieval.vector_store import ChromaStore

    s = get_settings()
    embedder = Embedder(s.embedding_model, s.model_cache_dir, s.query_instruction)
    store = ChromaStore(s.index_dir, s.collection_name)
    paths = [Path(p) for p in args.paths] if args.paths else None
    print(json.dumps(ingest(s, embedder, store, paths).__dict__, indent=2))


def _service():  # type: ignore[no-untyped-def]
    from .service import QAService

    return QAService.from_settings(get_settings())


def cmd_ask(args: argparse.Namespace) -> None:
    from .schemas import AskRequest

    svc = _service()
    req = AskRequest(question=args.question, year=args.year, profile=args.profile,
                     mode="deep" if args.deep else "fast")
    for event in svc.stream(req):
        if event["type"] == "delta":
            print(event["text"], end="", flush=True)
        elif event["type"] == "done":
            res = event["result"]
            print(f"\n\n[{res['provider']} | grounded={res['grounded']} | {res['latency_ms']} ms | answer_id={res['answer_id']}]")
            for c in res["citations"]:
                mark = "*" if c["cited"] else " "
                print(f" {mark}[{c['ref']}] {c['report']} {c['pages']} - {c['section'] or ''} (score {c['score']})")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("nabard_qa.api.app:create_app", factory=True, host=args.host, port=args.port, log_level="info")


def cmd_eval(args: argparse.Namespace) -> None:
    from .evaluation import evaluate_answers, evaluate_retrieval, load_golden

    svc = _service()
    items = load_golden(Path(args.golden))
    out = {"retrieval": evaluate_retrieval(svc, items, k=args.k)}
    if not args.retrieval_only:
        out["answers"] = evaluate_answers(svc, items)
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_bootstrap_pairs(args: argparse.Namespace) -> None:
    from .feedback.learner import heading_pairs
    from .feedback.store import FeedbackStore
    from .retrieval.vector_store import ChromaStore

    s = get_settings()
    chunks, _ = ChromaStore(s.index_dir, s.collection_name).all()
    added = FeedbackStore(s.db_path).add_training_pairs("heading", heading_pairs(chunks))
    print(f"added {added} heading pseudo-query pairs")


def cmd_train_adapter(args: argparse.Namespace) -> None:
    from .evaluation import evaluate_retrieval, load_golden
    from .feedback.learner import train_adapter

    svc = _service()
    golden = Path(args.golden)
    items = load_golden(golden) if golden.exists() else []
    if items:
        svc.retriever.adapter = type(svc.retriever.adapter)(None)
        before = evaluate_retrieval(svc, items)
    report = train_adapter(svc.s, svc.retriever.embedder, svc.store, svc.feedback, tuple(args.origins))
    print(json.dumps(report.__dict__ if report else {"trained": False}, indent=2))
    if items and report and report.accepted:
        svc.reload_adapter()
        after = evaluate_retrieval(svc, items)
        print(json.dumps({"golden_before": before, "golden_after": after}, indent=2))


def cmd_stats(args: argparse.Namespace) -> None:
    from .feedback.store import FeedbackStore
    from .retrieval.vector_store import ChromaStore

    s = get_settings()
    print(json.dumps({
        "chunks_indexed": ChromaStore(s.index_dir, s.collection_name).count(),
        "adapter": s.adapter_path.exists(),
        **FeedbackStore(s.db_path).stats(),
    }, indent=2))


def cmd_export(args: argparse.Namespace) -> None:
    from .feedback.store import FeedbackStore

    rows = FeedbackStore(get_settings().db_path).export_interactions()
    Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} interactions to {args.out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nabard-qa", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="extract, chunk, embed and index PDFs (default: data/raw/*.pdf)")
    p.add_argument("paths", nargs="*")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask one question from the terminal")
    p.add_argument("question")
    p.add_argument("--year", help="restrict to one report year, e.g. 2017-18")
    p.add_argument("--profile", help="applicant profile for eligibility answers")
    p.add_argument("--deep", action="store_true", help="many more passages, long-context model (comparisons, summaries)")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("serve", help="run the API + web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    default_golden = str(PROJECT_ROOT / "eval" / "golden_set.jsonl")
    p = sub.add_parser("eval", help="evaluate retrieval (and answers) on the golden set")
    p.add_argument("--golden", default=default_golden)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--retrieval-only", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("bootstrap-pairs", help="seed adapter training pairs from section headings (no LLM)")
    p.set_defaults(func=cmd_bootstrap_pairs)

    p = sub.add_parser("train-adapter", help="train the feedback query adapter")
    p.add_argument("--origins", nargs="+", default=["feedback", "synthetic", "heading"])
    p.add_argument("--golden", default=default_golden)
    p.set_defaults(func=cmd_train_adapter)

    p = sub.add_parser("stats", help="index and feedback statistics")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("export-feedback", help="dump interactions + feedback as JSONL")
    p.add_argument("out")
    p.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    _setup_logging(get_settings().log_level)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
