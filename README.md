# MSME & NABARD Scheme Assistant

**Ask a question about a government scheme, get an answer you can check.**

**Live demo:** https://msme-scheme-assistant.onrender.com
(free server, so the first question after a quiet spell takes a minute to wake it up, and answers take about 20–35 seconds)

---

## Why I built this

If you run a small business in India, there is a lot of money meant for you: subsidies for
setting up a unit, collateral-free loans, credit guarantees, toolkits for artisans. Finding out
what *you* qualify for is the hard part. The rules are spread across long PDFs from different
ministries and states, written in dense official language, and they change every few years.

This assistant reads those documents so you don't have to. You ask in plain English, Hindi or
Hinglish. For example:

- *"I'm a woman starting a ₹30 lakh food processing unit in a village. How much PMEGP subsidy can I get?"*
- *"What does CGTMSE guarantee cover, and what is the maximum loan?"*
- *"What toolkit incentive do carpenters get under PM Vishwakarma?"*
- *"What refinance limits does NABARD give cooperative banks in the North East?"*

You get a short answer, and **every claim is tagged with its source**, like **[S2]**. Click the
tag and you see the exact passage, document and page it came from. If the documents don't
cover your question, it says so instead of making something up. I cared about that more than
anything else: a confident wrong answer about a subsidy rate can cost someone real money.

There's also an optional profile (your state, category, sector, and so on). With it, answers
check each eligibility rule against your situation, and a question like *"which capital
subsidy can I get?"* searches your own state's policy.

## What it knows

18 official documents: 2,057 pages, split into 4,913 searchable passages.

| | Documents |
|---|---|
| **Central MSME schemes** | MSME Schemes Booklet 2025-26, PMEGP guidelines, CGTMSE credit guarantee scheme (Apr 2025), PM Vishwakarma guidelines, MUDRA FAQ, Stand-Up India factsheet, MSE-CDP cluster development guidelines and FAQs |
| **Ministry of MSME** | Annual Reports 2024-25 and 2025-26 |
| **State policies** | Uttar Pradesh MSME Policy 2022, Tamil Nadu MSME Policy 2021, Gujarat's Aatmanirbhar MSME scheme (2022) |
| **NABARD** | Annual Reports 2017-18, 2018-19, 2019-20, 2024-25, 2025-26 |

## How it works, in one picture

```
 Official PDFs ─► extract text (and repair it) ─► split into passages ─► search index
                                                                            │
 Your question ─► find likely passages (meaning + keywords) ─► re-rank them carefully
              ─► nothing relevant? say so honestly
              ─► otherwise an LLM writes a short answer, citing each passage it used
                                                                            │
 Your 👍 / 👎 and "this source was / wasn't useful" ─► search gets better for similar questions
```

This pattern is called RAG (retrieval-augmented generation). The language model never answers
from memory. It only gets to rephrase what the retrieved passages say.

## The interesting problems

Most of the work wasn't the chatbot. It was getting clean, trustworthy text out of government
PDFs and making search good enough that the right passage is actually there.

**Some PDFs lie about their own text.** Copy text from the 2017-18 NABARD report and you get
`rìê aüèîÜü êé†ü†û∞` instead of "The Indian economy". The fonts are broken: they name characters by
internal IDs and only half-map them to real letters. I decode those IDs directly, then put the
corrected letters back into the page layout by matching each character's position. Two-page
landscape spreads are split back into their printed pages, so a citation says "p. 47" and you
find it on page 47.

**Tables are where the numbers live, and they break easily.** A subsidy rate is useless if it
gets separated from the row that says what it's for. Table rows are kept together as
`scheme | location | incentive | limit`. Each piece of a long table repeats its caption and
header row. Two tables needed special care:
- Tamil Nadu prints its incentive table **sideways** on an upright page. The extractor notices
  the text runs bottom-to-top, turns the page upright, and reads it cell by cell from the
  grid lines.
- In one NABARD table, the numbers sat a fraction of a point higher than their labels, so every
  value slid up a row. Grouping cells that start within 2 points of each other fixed it.

**Gujarat only publishes scans.** Its scheme document is photographs of paper, with no text at
all. `scripts/make_searchable_pdf.py` runs the OCR engine that comes with Windows (no extra
download) and lays the recognised text invisibly over each page. After that, the rest of the
pipeline treats it like any other PDF. The 28 pages took 35 seconds.

**Search uses two methods because each misses things.** Keyword search (BM25) catches exact terms
like `ST-SAO` or `₹1,59,822`. Meaning-based search catches paraphrases ("loan guarantee" vs
"credit guarantee"). A third, more careful model (a cross-encoder) then re-reads the top 15
candidates against the question and reorders them. Its score also decides when nothing is
relevant enough to answer from.

**Long, personal questions confused the search.** *"I am a 24 year old woman in a rural area…
can I get PMEGP subsidy?"* scored badly even when the right passage was found, because the
re-ranker was trained on short web searches. When that happens, the question is condensed
into a short search ("PMEGP subsidy eligibility female rural food processing 30 lakh") and
tried once more. That question now gets the right answer: special category, 35% of ₹30 lakh
= ₹10.5 lakh. Before the fix, the model was handed no sources and guessed a wrong rate. Now,
when nothing is found, it doesn't call the model at all.

**Feedback actually changes the search.** When you rate an answer, or mark a source as useful
or not:
- **Right away**, passages you rated move up or down for *similar* questions from anyone. A
  useful passage can even be pulled in when a new phrasing of the question would have missed
  it. Unrelated questions aren't affected.
- **Over time**, ratings train a small adjustment to how questions are matched to passages (a
  "query adapter"). It's only switched on if it scores better on questions it hasn't seen.
- **Good answers** become training examples for the fine-tuned model.

I checked this end to end. After one user rated sources for *"What guarantee cover does CGTMSE
give to women entrepreneurs?"*, a differently worded question from someone else saw the
unhelpful passage demoted and the helpful one, which that wording hadn't found at all, pulled
into the results. A question about PM Vishwakarma kept exactly the same ranking.

**Making it run on a free server.** The free plan gives 512 MB of memory and a tenth of a CPU.
The app first needed about 750 MB. Nearly all of the excess came from the re-ranker scoring 16
passages at once, each padded to the longest one. Scoring one at a time gave identical scores,
cut peak memory to about 465 MB, and was faster too. For speed, I switched to the 8-bit version
of the same re-ranker and let it read the first 128 words of each passage. That made re-ranking
2.4 times faster, and accuracy went *up*, because the start of a passage is usually its most
specific part:

| Re-ranker setting | Right passage in top 6 | Right passage first | Time per question (laptop) |
|---|---|---|---|
| original (full precision, whole passage) | 96% | 54% | 0.85 s |
| 8-bit, whole passage | 96% | 54% | 0.47 s |
| **8-bit, first 128 words** (used now) | **96%** | **60%** | **0.35 s** |
| 8-bit, first 96 words | 86% | 48% | 0.29 s |

Even so, a tenth of a CPU is slow: about 20–28 seconds to find passages, plus a few seconds to
write the answer. A $7/month instance would bring that to around 5 seconds.

## How well it works

I wrote 53 test questions by hand and checked every expected answer against the documents:
37 about NABARD figures and policies, 9 about central MSME schemes, 4 about state policies,
and 3 that it *shouldn't* answer (like "Who won the IPL in 2018?").

| | Result |
|---|---|
| The right passage is among the 6 the model sees | **96%** (48 of 50) |
| ...and it's the very first one | 60% |
| MSME scheme answers containing the correct facts, with citations | **9 of 9** |
| Out-of-scope questions it declined | **3 of 3** |

The two misses: a table whose row label got split from its numbers, and an auditor's name that
only appears in a signature block.

The out-of-scope check was run when the third question was about Tamil Nadu's policy, before
that document was added. That question now asks about Kerala and hasn't been re-run yet.

## The fine-tuned model

I also fine-tuned a small open model (Qwen2.5-1.5B) so the assistant can run fully offline, on
a laptop, with no API at all. Everything ran on free tiers:
- the training questions were written by an LLM from the documents (`training/build_sft_dataset.py`);
- training ran on a free Colab / Kaggle GPU (`training/finetune_colab.ipynb`);
- the result runs on an ordinary CPU through llama.cpp (`scripts/serve_local_model.ps1`).

The training set mixes each question's true source passage with similar-looking distractors.
About 15% of examples have *no* correct passage, where the right answer is "the documents don't
say". The goal was a model that cites its sources and knows when to refuse, not one that
memorises facts.

That's exactly what happened:

| | Fine-tuned | Original model |
|---|---|---|
| Answers with the correct facts | 75% | 75% |
| Answers that cite their sources | **8 of 8** | 6 of 8 |
| Out-of-scope question | **declined** | made up an answer ("SBI home loan rate is 8.5% [S1]") |

A second version trained with extra MSME questions didn't beat the first (76% vs 74% facts, one
wrongly refused question), so the first one stays the default. On a laptop CPU it takes 45–65
seconds per answer, so it's the last fallback, not the main model.

## Run it yourself

You need Python 3.11+ and, optionally, a free API key from [Groq](https://console.groq.com)
and/or [Google AI Studio](https://aistudio.google.com) (Gemini). No GPU needed.

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate on Linux/macOS
pip install -e ".[dev]"
cp .env.example .env                                  # add GROQ_API_KEY and/or GEMINI_API_KEY

# put the PDFs in data/raw/ (titles and years go in data/raw/manifest.json)
nabard-qa ingest           # builds the search index; re-runs only process what changed
nabard-qa serve            # open http://127.0.0.1:8000
```

The first run downloads two small models (about 150 MB). With no API key at all, it still
works: it shows the most relevant quoted passages instead of a written answer, and says so.

**Docker:** `docker compose run --rm ingest && docker compose up -d api`

**Deploying:** the Docker image downloads a prebuilt 30 MB search index from this repo's
GitHub release (checksum verified), so a server needs no PDFs and no indexing. `render.yaml`
sets it up on Render: Dashboard → New → Blueprint → this repo, then paste the two API keys.
The free plan sleeps after 15 idle minutes and forgets collected feedback on each redeploy.

**Adding documents:** drop PDFs in `data/raw/`, give each a title and year in `manifest.json`,
and run `nabard-qa ingest`. For scanned PDFs, run `scripts/make_searchable_pdf.py` on them first.

## Reference

### Command line

| Command | What it does |
|---|---|
| `nabard-qa ingest [paths…]` | extract, split, embed and index PDFs (unchanged passages are skipped) |
| `nabard-qa ask "…" [--year 2017-18] [--profile "woman, SC, Assam"]` | answer one question in the terminal |
| `nabard-qa serve` | web app + API |
| `nabard-qa eval [--retrieval-only]` | run the 53 test questions |
| `nabard-qa bootstrap-pairs` / `train-adapter` | seed and retrain the feedback query adapter |
| `nabard-qa stats` / `export-feedback out.jsonl` | usage stats and feedback export |

### API

| Endpoint | |
|---|---|
| `POST /api/ask` | `{question, history?, year?, profile?, mode?}` → answer, citations, `grounded`, `answer_id` |
| `POST /api/ask/stream` | same, streamed as server-sent events: `sources` → `delta`… → `done` |
| `POST /api/feedback` | `{answer_id, rating: 1 or -1, comment?, relevant_chunk_ids?, irrelevant_chunk_ids?}` |
| `GET /health`, `GET /api/stats` | status, models, index size |
| `POST /api/admin/train-adapter` | retrain the query adapter (needs `NABARD_QA_ADMIN_TOKEN`) |

Requests are rate-limited per IP. If every language model is unavailable, you get quoted
passages instead of an error.

### Models and settings

- **Answers:** Groq `openai/gpt-oss-120b` first, then Gemini `gemini-3.6-flash`, optionally the
  local fine-tuned model last (`NABARD_QA_LLM_CHAIN=groq,gemini,local`). If one is down or
  rate-limited, the next answers straight away. The page shows which one answered.
- **Deep mode** (in the UI's Mode menu) sends about 40 passages to Gemini, for comparisons and
  "list all…" questions.
- **Search:** `BAAI/bge-small-en-v1.5` embeddings in ChromaDB, BM25, and an
  `ms-marco-MiniLM-L-6-v2` re-ranker, all running locally on CPU.
- **Prompt** (`generation/prompts.py`): cite every claim; say when the documents don't cover
  something; copy figures exactly with units and dates; give each year's figure separately; for
  schemes, cover what it is, who's eligible, the terms and how to apply; answer in the user's
  language. Passages are passed as data, and any instructions inside them are ignored.
- All settings live in `.env` (see `.env.example`).

### Project layout

```
src/nabard_qa/
  ingestion/   PDF extraction and repair, cleaning, splitting into passages
  retrieval/   embeddings, ChromaDB, BM25, re-ranker, query adapter, hybrid search
  generation/  prompts and the LLM provider chain
  feedback/    ratings store (SQLite) and adapter training
  api/         FastAPI app and the web page
  service.py   ties it all together
training/      dataset builder, LoRA fine-tuning script and notebook
eval/          the 53 hand-checked test questions
scripts/       OCR, timing and debugging tools, index download for deploys
tests/         71 tests
```

## What's not great yet

- **Speed on the free server**: 20–35 seconds per answer, almost all of it the re-ranker on a
  tenth of a CPU. A paid instance fixes it without code changes.
- **Coverage**: only three states (UP, Tamil Nadu, Gujarat). I couldn't find official PDFs for
  Karnataka's new policy, the Udyam registration rules, or the 2022 PMEGP guidelines. UP's
  capital-subsidy table didn't extract cleanly, so those rates can't be answered.
- **The Gujarat document is OCR text**: the rates read correctly, but some table labels are
  garbled ("Cate o of taluka"). A vague question from a Gujarat profile sometimes misses its
  subsidy table. Naming the subsidy ("capital investment subsidy") finds it.
- **Some multi-line table row labels** still get separated from their numbers.
- **No real user feedback yet**: the feedback loop is tested with demo ratings, and the live
  server's free disk forgets ratings when it redeploys.
- **Built for one server**: rate limits and feedback caches live in memory. Scaling out would
  need a shared database instead of SQLite.
