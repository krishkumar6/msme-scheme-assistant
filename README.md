# MSME & NABARD Scheme Assistant: a RAG-based GenAI QnA chatbot

Ask about MSME schemes (eligibility, subsidy, loan limits, guarantees), state MSME policy,
and NABARD's schemes, refinance, development funds and financial statements, in plain
English, Hindi or Hinglish. Every answer cites the document and page it came from. If the
indexed documents don't cover a question, the assistant says so instead of guessing.

Corpus: 18 official documents, 2,057 printed pages, 4,913 indexed passages. They're listed with
titles in `data/raw/manifest.json`.

| Source | Documents |
|---|---|
| Central MSME schemes | MSME Schemes Booklet 2025-26 (DC-MSME); PMEGP revised guidelines; CGTMSE CGS-I scheme document (1 Apr 2025); PM Vishwakarma guidelines; PMMY/MUDRA FAQ; Stand-Up India factsheet; MSE-CDP guidelines (2022) and FAQs |
| Ministry of MSME | Annual Reports 2024-25 and 2025-26 |
| State policy | Uttar Pradesh MSME Promotion Policy 2022; Tamil Nadu MSME Policy 2021; Aatmanirbhar Gujarat Scheme for Assistance to MSMEs (GR 05-10-2022) |
| NABARD | Annual Reports 2017-18, 2018-19, 2019-20, 2024-25, 2025-26 |

To add more PDFs, drop them in `data/raw/`, give each a title and year in `manifest.json`, and run
`nabard-qa ingest`. Without a manifest entry, a document is labelled "NABARD Annual Report <year>".

Scanned PDFs have no text to extract (Gujarat publishes its GRs this way). Give them a text layer
first: `python scripts/make_searchable_pdf.py scanned.pdf data/raw/name.pdf`. It OCRs only the
image-only pages, using the OCR engine built into Windows (`scripts/ocr_pages.ps1`; no Tesseract or
model download), and writes each recognised line as invisible text at its position, so ingestion
treats the file like any other. The 28-page Gujarat GR takes about 35 s. On Linux, `ocrmypdf` does
the same job.

Tables printed sideways on an upright page (Tamil Nadu's incentive annexure) are detected from the
text direction, turned upright, and read cell by cell from their ruling lines, giving one
`scheme | location | incentive | limit | …` line per row.

```
                ┌──────────── ingestion (offline, idempotent) ────────────┐
 PDFs ─► PyMuPDF layout + pdfminer glyph repair ─► clean ─► structure-aware chunks
        (rotated spreads split into printed pages,      (sections, captions, table rows,
         running headers removed by frequency)            TOC filtered, content-hash ids)
                                                    │
                                     bge-small embeddings ─► ChromaDB  +  BM25 index
                └─────────────────────────────────────────────────────────┘

 question ─► follow-up rewrite ─► dense (query adapter) ⊕ BM25 ─► RRF ─► cross-encoder rerank
          ─► + feedback boost from similar past questions ─► relevance gate
          ─► LLM chain (grounded prompt, [S#] citations, streaming) ─► answer + sources
                                                    │
                         👍/👎 + per-source relevant/irrelevant ─► SQLite
                           ├─ immediate: boosts/demotes passages for similar questions
                           ├─ periodic: trains the query-embedding adapter
                           └─ dataset: thumbs-up answers become fine-tuning examples
```

## What's notable

**The PDF text layer is broken, and ingestion recovers it.** Most narrative pages of the 2017-18
report use Georgia subsets that name glyphs by ID (`/g68`) and ship a ToUnicode map that covers
only some codes. Standard extractors (pypdf, plain PyMuPDF) return a substitution cipher on those
pages: `rìê aüèîÜü êé†ü†û∞` instead of "The Indian economy". `ingestion/pdf_extract.py` decodes the
glyph IDs (TrueType standard order: `gid + 29` for ASCII, plus a verified table of ligatures, ₹,
dashes, quotes and small caps), then puts the corrected characters into PyMuPDF's layout by matching
each character's position on the page. Rotated landscape spreads are split into their two printed
pages, and the printed page numbers are captured so citations match the page the reader sees.
`tests/test_pdf_integration.py` locks this behaviour in.

**Tables stay usable.** Cells on the same baseline are joined as `Kerala | 1,11,351 | 8,668 | …`.
Table chunks are split by row, and each piece repeats the caption and the header row, so a number is
never separated from what it measures.

**Retrieval is hybrid and gated.** BM25 catches the exact tokens these reports are full of (ST-SAO,
LTIF, `1,59,822`), and dense search catches paraphrases. A cross-encoder reranks the fused list. The
reranker score also acts as a calibrated "is anything relevant?" gate, which is how the assistant
declines out-of-scope questions.

**The feedback loop changes retrieval.** Users rate an answer and can mark individual sources as
relevant or irrelevant. Each rating:
1. takes effect immediately: passages rated on *similar* past questions (cosine ≥ 0.82) get a bounded
   boost or demotion. Passages rated *relevant* are also **recalled** into the candidate set when a
   new phrasing of the question doesn't retrieve them. The cross-encoder still scores them, so
   feedback can't push an irrelevant passage past the relevance gate;
2. becomes a contrastive training pair for the **query adapter**, a linear map on query embeddings
   trained with InfoNCE and kept close to the identity. Documents never need re-embedding, and the
   adapter is only activated if it beats the base embeddings on held-out pairs;
3. if it's a thumbs-up, becomes a fine-tuning example.

## Results (measured on this machine, `eval/golden_set.jsonl`)

53 hand-checked questions: 50 answerable and 3 out of scope. Of the answerable ones, 37 are NABARD
figures and terms; they name their report year, because with five report years an undated question
has a different correct answer. 9 are central MSME scheme rules: PMEGP, CGTMSE, PM Vishwakarma,
MUDRA, Stand-Up India and MSE-CDP. 4 are state policy: Tamil Nadu targets and capital subsidy,
Gujarat interest and capital subsidy. Every expected fact was checked against the indexed text.

| Metric | Value |
|---|---|
| Retrieval hit@6 (all expected facts in the top-6 context) | **0.96** (48/50; the 4 state questions all hit) |
| Retrieval hit@1 / MRR | 0.60 / 0.735 |
| MSME scheme answers (Groq `gpt-oss-120b`): fact recall / cited | **1.0 / 1.0** (9/9) |
| Out-of-scope questions declined (SBI home loan rate, IPL, Tamil Nadu MSME policy) | **3/3** (measured before Tamil Nadu was added; that item is now Kerala and not yet re-run) |
| Fact recall of answers, *extractive mode, no LLM* (2-report corpus) | 0.784 |

The two retrieval misses are table rows whose multi-line row label was split from the numbers
during extraction (SHG savings) and an auditor name that only appears in a signature block.

**Personal eligibility questions.** The cross-encoder was trained on short web queries. It scores a
long first-person question ("I am a 24 year old woman in a rural area… can I get PMEGP subsidy?")
at −2.7 even when the right passage is retrieved, which fails the gate. When nothing clears the
gate, the service condenses the question into a search query that keeps the eligibility attributes
("PMEGP subsidy eligibility female 24 rural food processing 30 lakh", which scores +2.3) and retries
once. That question now gets the correct, cited answer: Special Category, 35% of ₹30 lakh =
₹10.5 lakh. If still nothing is relevant, the reply is a fixed decline with no LLM call. Before
this, the model was given an empty context and guessed a wrong subsidy rate.

**Feedback loop, measured.** The demo ran on a copy of the feedback DB.
- A user asked "What guarantee cover does CGTMSE give to women entrepreneurs?", rated source #3
  relevant and #1 irrelevant.
- Then a different phrasing was asked: "How much CGTMSE guarantee coverage do women-led small
  businesses get?". The rated-irrelevant passage was demoted (−0.37). The rated-relevant passage,
  which this phrasing hadn't retrieved at all, was recalled to rank 7 (+0.37). One rating is
  bounded on purpose; agreement from more users raises it further.
- An unrelated question (PM Vishwakarma toolkit) kept exactly the same ranking.
- `tests/test_feedback.py` locks in the reordering, the similarity scoping and the recall.

**Query adapter (the embedding side).** It was last retrained after the Tamil Nadu and Gujarat
documents were added (87 new heading pseudo-queries, on top of the earlier heading, synthetic and
LLM-written MSME pairs). On held-out pairs, recall@5 went from 0.606 to 0.628 and MRR from 0.434
to 0.463, so it was accepted. Golden-set retrieval barely moves (MRR 0.702 without it, 0.707 with
it; hit@6 0.96 both ways), because the cross-encoder decides the final order. No real user feedback exists yet; ratings collected in the
UI become `feedback` pairs for the next `nabard-qa train-adapter`.

**Groq (`openai/gpt-oss-120b`), partial run:** 25 of 40 questions were rate-limited (HTTP 429)
on the free tier before retry-with-backoff was added. On the 15 questions it answered: fact recall
0.833 on 12 answerable, 11/12 cited, 3/3 out-of-scope declined.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate on Linux/macOS
pip install -e ".[dev]"
cp .env.example .env                                  # add GROQ_API_KEY and/or GEMINI_API_KEY

# put the reports in data/raw/ (download from nabard.org → Publications → Annual Reports)
nabard-qa ingest           # ~2 min extraction + ~7 min CPU embedding the first time; re-runs are incremental
nabard-qa serve            # http://127.0.0.1:8000
```

The first run downloads two small ONNX models (bge-small-en-v1.5 and ms-marco-MiniLM-L-6) into
`data/models/`. No GPU is needed. Without an API key the app still runs: it returns quoted extracts
with citations and says that it is doing so.

Docker:
```bash
docker compose run --rm ingest && docker compose up -d api
```

### Deployment

The image ships with the search index built in: `scripts/fetch_index.py` downloads a 30 MB bundle
(Chroma index + query adapter) from this repo's GitHub release and checks its SHA-256, and both ONNX
models are downloaded at build time. A cold start needs no PDFs and no ingest.

It runs on **Render's free plan** (`render.yaml`: Dashboard → New → Blueprint → this repo, then
enter `GROQ_API_KEY` and `GEMINI_API_KEY`). Render rebuilds on every push to `main`; GitHub Actions
runs the tests.

Fitting 512 MB took one change. Peak memory was ~750 MB, and almost all of the excess came from the
cross-encoder scoring passages 16 at a time: every pair was padded to the longest passage, and the
ONNX runtime kept that buffer. Scoring one pair at a time gives identical scores, is faster on CPU
(2.8 s vs 6.6 s for 40 passages), and brings the peak to ~465 MB: Chroma ~95 MB, embedding model
~150 MB, reranker ~100 MB, a query ~45 MB.

The free plan also has only 0.1 CPU, and the first deploy took ~34 s per question, almost all of
it in the cross-encoder. Two changes cut reranking 2.4x on the golden set: the int8-quantized ONNX
file of the same model, and scoring only the first 128 words of each passage. hit@6 stayed at 0.96
and hit@1 rose from 0.54 to 0.60 (the opening of a passage is its most specific part). 96 words was
too few (hit@6 0.86), and reranking 10 candidates instead of 15 lost two questions.

| Reranker (15 candidates) | hit@6 | hit@1 | MRR | time/question (local) |
|---|---|---|---|---|
| fp32, full passage | 0.96 | 0.54 | 0.707 | 0.85 s |
| int8, full passage | 0.96 | 0.54 | 0.707 | 0.47 s |
| **int8, first 128 words** (default) | **0.96** | **0.60** | **0.735** | **0.35 s** |
| int8, first 96 words | 0.86 | 0.48 | 0.606 | 0.29 s |

The free plan sleeps after 15 minutes idle (about a minute to wake) and has no persistent disk, so
feedback collected there resets on each deploy or restart.

## CLI

| Command | Purpose |
|---|---|
| `nabard-qa ingest [paths…]` | extract → chunk → embed → index. Unchanged chunks are skipped and removed ones are deleted |
| `nabard-qa ask "…" [--year 2017-18] [--profile "woman entrepreneur, SC, Assam"]` | answer in the terminal |
| `nabard-qa serve` | API + web UI |
| `nabard-qa eval [--retrieval-only]` | golden-set metrics |
| `nabard-qa bootstrap-pairs` | seed adapter training pairs from section headings |
| `nabard-qa train-adapter` | train the feedback query adapter; prints golden retrieval before and after |
| `nabard-qa stats` / `export-feedback out.jsonl` | monitoring and data export |

## API

| Endpoint | |
|---|---|
| `POST /api/ask` | `{question, history?, year?, profile?}` → answer, citations (report, pages, section, snippet, cited?), grounded flag, `answer_id` |
| `POST /api/ask/stream` | same input; server-sent events: `sources` → `delta`… → `done` |
| `POST /api/feedback` | `{answer_id, rating: 1\|-1, comment?, relevant_chunk_ids?, irrelevant_chunk_ids?}` |
| `GET /health`, `GET /api/stats` | status, provider, index size, adapter state |
| `POST /api/admin/train-adapter` | retrain and hot-reload the adapter (`Authorization: Bearer $NABARD_QA_ADMIN_TOKEN`) |

Requests are rate-limited per IP and capped by question length. Each response carries a request ID
that also appears in the logs. If a model call fails, the service falls back to extracts instead of
returning an error.

## Generation and prompts

* Models: Groq `openai/gpt-oss-120b` answers first, Gemini `gemini-3.6-flash` is the backup and the
  long-context Deep-mode model, and the fine-tuned local model can be added as the last fallback. All
  three are called through the OpenAI-compatible chat API, with streaming. Models and order are set in `.env`.
* The system prompt (`generation/prompts.py`) is fixed text. It requires a citation for every claim,
  an explicit "not in the reports" instead of guesses, figures copied exactly with units and dates,
  separate figures for each report year, and a structure for scheme questions: what it is,
  eligibility, terms, how to access. When an applicant profile is supplied, it checks the profile
  against each eligibility condition. Answers come back in the user's language. Passages are wrapped
  as XML data and the prompt says to ignore any instructions found inside them.
* **Personalisation.** The UI has an optional profile: state, gender, social category, rural or
  urban, sector, new or existing unit, age, amount needed. It is kept in the browser's localStorage
  and sent with each question as `profile`. It does two things:
  - the prompt checks each eligibility condition against it ("SC applicant -> 35% rate applies: met");
  - for a personal question that names no central scheme or state ("which capital subsidy can I
    get?"), the applicant's state is added to the search query, so a Tamil Nadu user gets the Tamil
    Nadu policy and a Gujarat user gets the Gujarat GR (`service.personalised_query`).
* Follow-up questions are rewritten into standalone search queries at `low` reasoning effort.
* **Provider chain.** Keys come from `GROQ_API_KEY` and `GEMINI_API_KEY`. With
  `NABARD_QA_LLM_PROVIDER=auto`, every provider that has a key forms a chain in
  `NABARD_QA_LLM_CHAIN` order (default `groq,gemini`). If a provider fails before writing any
  text (rate limit, quota, outage, retired model), the next one answers. Quoted extracts are the last
  resort. The provider that actually answered is shown on each answer.
* **Deep mode** (UI "Mode: Deep", CLI `--deep`, API `"mode": "deep"`) is for comparisons, summaries and
  "list all …" questions. It reranks 80 candidates and sends about 40 passages (~15–20k tokens) to the
  long-context provider (`NABARD_QA_DEEP_PROVIDER=gemini`), with the normal chain as its backup. Normal
  questions only need about 6 passages, so they go to the fast provider.
  If the long-context call fails, the question is re-asked through the normal chain with only the top
  passages. The 40-passage prompt (~10k tokens) is larger than Groq's free-tier limit of 8k tokens per
  minute, so it is not passed down the chain as-is.
* `python scripts/list_models.py groq|gemini` lists the models your key can see. Listed does not mean
  usable: `gemini-2.5-*` is listed but returns 404 for new accounts. Speed also varies a lot between
  models, so check with `scripts/probe_provider.py gemini`. The default is `gemini-3.6-flash`
  (`3.8-flash` was overloaded during testing).
* Measured with `scripts/stream_timing.py`: Fast (Groq `gpt-oss-120b`) shows first text at ~3.3 s and
  completes at ~4 s; Deep (Gemini `3.6-flash`, 40 passages) shows first text at ~10.4 s and completes at
  ~11–14 s. Answers stream over SSE: Groq in hundreds of small pieces, Gemini in ~20 larger chunks.

## Fine-tuning

Only free tiers are needed, and no local GPU:

| Step | Where | Command |
|---|---|---|
| 1. Write training Q&A | your PC + Groq or Gemini key | `python training/build_sft_dataset.py --limit 150` (Groq by default, `--provider gemini`; `--fallback-provider gemini` switches over when Groq's daily cap is hit) |
| 2. Check the dataset | your PC | `python training/finetune_lora.py --validate-only` |
| 3. QLoRA training + GGUF export | **free Colab/Kaggle T4** | open `training/finetune_colab.ipynb`, upload `finetune_lora.py`, `sft_train.jsonl`, `sft_val.jsonl` |
| 4. Run the model on CPU | your PC + llama.cpp | download `adapter.zip` from Colab into `out/nabard-qa-lora/`, convert it with `llama.cpp/convert_lora_to_gguf.py` (config files only), then `scripts/serve_local_model.ps1` (port 8001, matches the `local` provider defaults) |
| 4b. Or with Ollama | your PC + [Ollama](https://ollama.com) | `ollama create nabard-qa -f training/ollama/Modelfile.adapter` (base `qwen2.5:1.5b-instruct` + adapter), then set `NABARD_QA_LOCAL_BASE_URL=http://localhost:11434/v1` and `NABARD_QA_LOCAL_MODEL=nabard-qa` |

Notes:
* Step 1 resumes where it stopped, waits out rate limits, and validates every JSON reply.
  Groq's free tier (8k tokens/minute) allows roughly 3 passages a minute. If the daily cap is hit,
  re-run the next day and only the remaining passages are processed.
* `--fallback-provider gemini` switches to Gemini as soon as Groq reports its daily cap ("tokens per
  day"), or after 3 rate-limit waits in a row, instead of retrying until tomorrow. Each pair records
  which model wrote it (`generator` in `synthetic_qa.jsonl`). `--match "^(?!NABARD)"` limits
  sampling to the MSME documents. It samples evenly across documents and skips admin passages
  (Hindi Pakhwada, vigilance, receipt formats) that only produce trivia.
* Licence: `gpt-oss-120b` (the Groq default) has Apache-2.0 open weights. Google's Gemini API terms
  restrict using outputs to develop competing models, so check them before training on Gemini output.
  Gemini-written pairs can be filtered out by their `generator` field.
* Step 3 picks fp16 automatically on a T4 (which has no bf16) and defaults to batch size 1 with
  16-step gradient accumulation to fit 16 GB.
* The Modelfile sets `num_ctx 8192`. Ollama's smaller default would silently cut off the source
  passages.
* Step 4 skips merging. The 74 MB adapter is applied at load time to a 4-bit (Q4_K_M) base, so you
  download 1 GB instead of the 3 GB fp16 base and its merge. That matches training, which used a
  4-bit (QLoRA) base too. Colab's preinstalled torchao 0.10 breaks PEFT's merge step; the notebook
  uninstalls it.
* All four steps have been run end to end: Qwen2.5-1.5B-Instruct, 354 training examples, 2 epochs on
  a free Colab T4, served by llama.cpp on a laptop CPU.

The dataset is RAFT-style. Each question is paired with its source passage mixed among retrieved
distractor passages, under shuffled `[S#]` labels. About 15% of examples contain only distractors and
the target is a refusal. Thumbs-up answers from real users are added. The train/validation split is
by source passage, so no passage appears in both. The loss covers the answer tokens only. The same
synthetic questions also feed the query adapter (`origin="synthetic"`).

Measured effect of the adapter on 8 answerable and 1 out-of-scope golden questions. Retrieval and
prompts were identical; the adapter was switched on and off with llama-server's `/lora-adapters`.

| | Fine-tuned | Base Qwen2.5-1.5B |
|---|---|---|
| Fact recall | 0.75 | 0.75 |
| Answers citing `[S#]` | **8/8** | 6/8 |
| Out-of-scope question | **declined** | invented an answer ("SBI home loan rate is 8.5% [S1]") |
| Wrongly refused answerable questions | 0 | 0 |

So fine-tuning taught the model the citation format and to refuse, not new facts. Facts come from
retrieval. Both models missed st-01. That was partly an extraction bug, now fixed: table cells set
in another font start a fraction of a point higher, so values sorted ahead of their row label. Even
with the corrected table, the 1.5B model still takes a value from the wrong column, while
`gpt-oss-120b` answers it correctly.

A second adapter (v2) was trained on Kaggle with 263 more pairs from the MSME scheme documents
(`training/ollama/nabard-qa-lora-v2-f16.gguf`). On 14 answerable and 3 out-of-scope golden
questions (9 MSME, 5 NABARD), one run each:

| | v1 (NABARD only, default) | v2 (+ MSME) |
|---|---|---|
| Fact recall | **0.762** | 0.738 |
| Answers citing `[S#]` | **14/14** | 13/14 |
| Wrongly refused answerable questions | **0** | 1 (vish-01) |
| Out-of-scope declined | 2/3 | 2/3 |

v2 won cgtmse-01 but lost pmegp-01 and vish-01. With 17 questions that is within noise, so v2 did
not earn the switch and v1 stays the default. Both models answered na-04 (Tamil Nadu MSME policy)
from the UP policy instead of declining.

A 1.5B model on CPU takes 45–65 s per answer here, against about 3 s on Groq. Its value is free,
offline or on-premise serving, so add it as the last fallback:
`NABARD_QA_LLM_CHAIN=groq,gemini,local`. If its server isn't running, the chain skips it
immediately.

## Project layout

```
src/nabard_qa/
  ingestion/   glyphs.py · pdf_extract.py · cleaning.py · chunking.py · pipeline.py
  retrieval/   embeddings.py · vector_store.py · lexical.py · reranker.py · adapter.py · hybrid.py
  generation/  prompts.py · llm.py   (Groq / Gemini / local, all OpenAI-compatible; extractive)
  feedback/    store.py (SQLite) · learner.py
  service.py   evaluation.py   cli.py   config.py   schemas.py
  api/         app.py · static/index.html
training/      build_sft_dataset.py · finetune_lora.py
eval/          golden_set.jsonl
scripts/       debug_retrieval.py · calibrate_gate.py · make_searchable_pdf.py + ocr_pages.ps1 (OCR)
tests/         70 tests: unit tests with fakes, a service and API end-to-end test, and a real-PDF integration test
```

## Known limitations

* Scheme coverage is what's in `data/raw`. Not included:
  - the Udyam registration notification (no official PDF was found);
  - state policies other than Uttar Pradesh, Tamil Nadu and Gujarat. Karnataka's 2025-30 policy
    had no official PDF that could be found;
  - the 2022 PMEGP guidelines (the KVIC site was unreachable; the revised guidelines are included).
  The UP policy's capital-subsidy table didn't survive extraction, so those rates can't be answered.
* The Gujarat GR is OCR text: its rates read correctly, but some table labels are garbled
  ("Cate o of taluka"). For a vague question such as "which capital subsidy can I get?" from a
  Gujarat profile, its capital-subsidy table ranks 9th, just outside the 6 passages the model sees.
  Naming the subsidy ("capital investment subsidy") brings it to 3rd.
* The MSME-aware adapter (v2) covers only 90 MSME passages because the Groq and Gemini daily quotas
  ran out, and it did not beat v1. Generating the remaining ~34 passages and retraining may change that.
* Some table rows lose their multi-line row labels during extraction, so the numbers survive but
  their row name doesn't.
* The rate limiter and feedback boost cache run in-process. For multiple replicas, put a gateway in
  front and use a shared database instead of SQLite.
