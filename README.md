# LendLens

**Loan-file intelligence for used-car lending.** One messy application bundle in — evidence-linked underwriting signals, transparent policy flags, cross-file fraud linkage, grounded Q&A, and an offline eval harness out.

Built with Streamlit + SQLite.

> **This is the `rag-groq` branch** — Ask the File runs as live retrieval-augmented
> generation through a Groq-hosted LLM and expects a `GROQ_API_KEY` (free tier at
> console.groq.com). Everything else (pipeline, rules, Ring Watch, evals) still runs
> offline. For the zero-key demo, use `main`.

## Branches

| Branch | Q&A path | Keys |
|---|---|---|
| `main` | Offline demo: cached answers + extractive evidence | none |
| `rag-groq` | Live RAG: hybrid retrieval → LLM synthesis (Groq) with inline `[seg_id]` citations | `GROQ_API_KEY` |

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
streamlit run app.py
```

That's it. The app seeds itself with a bundled demo portfolio of four fictional applicants (a clean file, a FOIR-breach refer, and a two-file fraud ring) plus an 11-section lending policy corpus.

Sanity-check the whole chain before a demo:

```bash
python run_checks.py   # 15 assertions: pipeline → rules → rings → retrieval → guardrail → evals
```

## What it does

A used-car loan file is unstructured by nature: an application narrative, an employer letter, a dealer quote, and pages of bank statement lines. The lending decision, though, needs *structure* — income, obligations, conduct, collateral — and needs every number to be traceable back to the document that produced it.

**Pipeline** — `core/pipeline.py`
Segments the bundle (statement lines stay grouped so ledger context survives chunking), classifies each segment into 8 document categories, and extracts amounts, organizations, phone numbers, account references, and fully parsed transactions.

**Signals** — `core/signals.py`
Builds the applicant profile: bank-derived income vs stated income, existing EMIs from NACH debits (not from what the applicant chose to declare), bounce counts, FOIR, LTV, tenure. Then runs transparent policy rules — every flag carries severity, the policy section breached, and the exact evidence segment IDs.

**Ring Watch** — cross-file linkage
Rules score files one at a time; fraud rings operate across files. Shared phones, employers, or accounts between unrelated applications route every linked file to the fraud desk — including files that look individually approvable. In the demo portfolio, one linked file passes rules with only a minor LTV exception; the linkage is the only thing that catches it.

**Ask the File** — `core/chat.py`
Retrieval-augmented generation over the applicant's documents plus the policy corpus. Retrieval is hybrid: from-scratch Okapi BM25 fused with dense embeddings via Reciprocal Rank Fusion (when an embedding key exists). The top-8 segments — and only those — are handed to the LLM (`llama-3.3-70b-versatile` via Groq, configurable), which answers with inline `[seg_id]` citations and admits gaps rather than inventing figures. Account/phone digits are masked in displayed snippets.

**Fairness guardrail** — `core/guardrail.py`
A hard filter, not a prompt suggestion: questions that tie religion, caste, gender, marital status, or community to a credit judgment are intercepted before retrieval, with an explanation of what the model *will* assess.

**Evals** — `core/evals.py`
Three offline layers, because a pipeline you can't measure is a pipeline you can't ship:

| Layer | Question it answers | Demo portfolio (offline mode) |
|---|---|---|
| Extraction | Do derived fields match hand-labelled gold? | 100% (28/28 fields) |
| Retrieval | Does the right evidence come back? (hit@3 / hit@5 / MRR) | 75% / 94% / 0.61 |
| Decision | Does the rule engine reproduce expected verdicts? | 4/4 |

The retrieval misses are instructive: questions like "any bounced payments?" miss statement lines that say `RTN CHG` — a vocabulary gap that lexical search can't close and exactly what the dense half of hybrid retrieval fixes when enabled.

## Modes

| | No key (fallback) | Live (this branch's default) |
|---|---|---|
| Extraction | Regex + lexicon + txn parser | Same, LLM-assisted classification |
| Retrieval | BM25 + RRF | BM25 + dense embeddings + RRF |
| Q&A | Extractive evidence only | LLM RAG (Groq), grounded in retrieved segments with inline citations |
| Keys needed | none | `GROQ_API_KEY`, optional `OPENAI_API_KEY` |

Copy `.env.example` to `.env` and add your key. Everything degrades gracefully if the API is unreachable.

## Project layout

```
app.py               Streamlit entry + navigation
core/
  config.py          modes, policy thresholds
  store.py           SQLite persistence (zero-setup, swap-for-Postgres design)
  pipeline.py        segment → classify → extract → embed
  signals.py         profile build, policy rules, ring detection
  retrieval.py       BM25 from scratch + RRF hybrid
  chat.py            grounded Q&A with citations
  guardrail.py       fairness hard-filter
  evals.py           extraction / retrieval / decision harness
  seed.py            demo portfolio ingest (gold refs resolved by substring)
ui/                  one module per page
data/
  portfolio.json     4 fictional applicant files + gold labels (original)
  policy.md          11-section lending policy (original)
run_checks.py        end-to-end assertions
```

## Design notes

- **Evidence-linked everything.** A flag without a pointer to the source segment is an opinion. Every profile number, flag, and answer carries segment IDs.
- **IR before ML.** Most of the value is retrieval and deterministic parsing; the LLM is reserved for the last mile (synthesis, classification edge cases). This keeps the offline path fully functional and the live path auditable.
- **Rules are a feature, not a fallback.** Underwriting rules are transparent, versionable, and cite policy sections — reason codes come free (POL-9.1).
- **Gold labels resolve by substring, not by ID**, so re-chunking never silently corrupts the eval set.

All applicant data is fictional and generated for this demo.
