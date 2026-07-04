"""Ingestion pipeline: raw loan-file text → categorized, indexed segments.

A loan file arrives as one messy text bundle (application form narrative,
bank statement lines, employer letter, dealer quote). Stages:

  1. segment  — block-aware chunking; bank-statement lines stay grouped
  2. classify — category tagging via lexicon (LLM refinement in live mode)
  3. extract  — amounts, dates, orgs, phones, account refs, txn parsing
  4. embed    — dense vectors when a key exists, else lexical-only
"""

import re

from core.config import dense_available

_MIN_WORDS, _MAX_WORDS = 25, 110

# ------------------------------------------------------------- segmentation

_TXN_LINE = re.compile(r"^\s*\d{2}-[A-Za-z]{3}-\d{4}\s*\|")


def segment_text(raw: str) -> list[dict]:
    """Chunk a document bundle for retrieval.

    Prose paragraphs are merged/split toward 25–110 words. Consecutive
    transaction lines are grouped into statement blocks (max 8 lines) so
    ledger context survives chunking.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
    chunks: list[str] = []
    buf = ""

    for block in blocks:
        lines = block.splitlines()
        if any(_TXN_LINE.match(ln) for ln in lines):
            if buf:
                chunks.append(buf)
                buf = ""
            header = [ln for ln in lines if not _TXN_LINE.match(ln)]
            txns = [ln for ln in lines if _TXN_LINE.match(ln)]
            for i in range(0, len(txns), 8):
                piece = "\n".join(header[:1] + txns[i : i + 8])
                chunks.append(piece)
            continue

        candidate = f"{buf} {block}".strip() if buf else block
        words = len(candidate.split())
        if words < _MIN_WORDS:
            buf = candidate
        elif words <= _MAX_WORDS:
            chunks.append(candidate)
            buf = ""
        else:
            if buf:
                chunks.append(buf)
            chunks.extend(_split_long(block))
            buf = ""
    if buf:
        chunks.append(buf)
    return [{"seq": i, "text": c} for i, c in enumerate(chunks)]


def _split_long(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, cur = [], ""
    for s in sentences:
        candidate = f"{cur} {s}".strip()
        if len(candidate.split()) > _MAX_WORDS and cur:
            out.append(cur)
            cur = s
        else:
            cur = candidate
    if cur:
        out.append(cur)
    return out


# ------------------------------------------------------------ classification

_CATEGORY_LEXICON = {
    "identity": ["pan", "aadhaar", "date of birth", "residing", "address", "kyc", "passport", "voter"],
    "income": ["salary", "income", "take-home", "ctc", "in-hand", "earn", "wages", "stipend", "net pay"],
    "employment": ["employed", "employer", "designation", "company", "joined", "tenure", "hr department", "offer letter", "self-employed", "proprietor"],
    "banking": ["account", "statement", "balance", "neft", "imps", "upi", "credited", "debited", "cheque"],
    "obligations": ["emi", "loan", "credit card", "outstanding", "borrowed", "repay", "dues", "installment"],
    "vehicle": ["car", "vehicle", "model", "variant", "registration", "odometer", "dealer", "quotation", "ex-showroom", "swift", "baleno", "i20", "wagonr", "creta"],
    "declarations": ["declare", "confirm", "undertake", "consent", "true and correct", "signature"],
}


def classify_segment(seg: dict) -> dict:
    text = seg["text"].lower()
    if _TXN_LINE.match(seg["text"].strip().splitlines()[-1] if seg["text"].strip() else ""):
        seg["category"] = "banking"
        return seg
    scores = {c: sum(text.count(w) for w in words) for c, words in _CATEGORY_LEXICON.items()}
    best = max(scores, key=scores.get)
    seg["category"] = best if scores[best] > 0 else "declarations"
    return seg


# ---------------------------------------------------------------- extraction

_AMOUNT_RE = re.compile(r"(?:₹|\bRs\.?\s?|\bINR\s?)(\d[\d,]*(?:\.\d{1,2})?)", re.I)
_PHONE_RE = re.compile(r"\b([6-9]\d{9})\b")
_ACCT_RE = re.compile(r"\b(?:a/c|account)\s*(?:no\.?|number)?\s*[:#]?\s*[xX*]*(\d{4,})\b", re.I)
_DATE_RE = re.compile(r"\b(\d{2}-[A-Za-z]{3}-\d{4})\b")
_ORG_RE = re.compile(r"\b([A-Z][A-Za-z&.]+(?:\s[A-Z][A-Za-z&.]+){0,3}\s(?:Pvt\.?\s?Ltd\.?|Ltd\.?|LLP|Industries|Motors|Enterprises|Solutions|Technologies))")

TXN = re.compile(
    r"^\s*(?P<date>\d{2}-[A-Za-z]{3}-\d{4})\s*\|\s*(?P<mode>[A-Z/ ]+?)\s*\|\s*(?P<desc>.+?)\s*\|\s*(?P<sign>[+-])\s*(?P<amount>[\d,]+)\s*\|\s*bal\s*(?P<balance>[\d,]+)\s*$",
    re.M,
)


def _num(s: str) -> float:
    cleaned = s.replace(",", "")
    return float(cleaned) if cleaned else 0.0


def parse_transactions(text: str) -> list[dict]:
    """Parse pipe-delimited statement lines into structured transactions."""
    out = []
    for m in TXN.finditer(text):
        out.append(
            {
                "date": m.group("date"),
                "mode": m.group("mode").strip(),
                "desc": m.group("desc").strip(),
                "amount": _num(m.group("amount")) * (1 if m.group("sign") == "+" else -1),
                "balance": _num(m.group("balance")),
            }
        )
    return out


def extract_entities(seg: dict) -> dict:
    text = seg["text"]
    amounts = [_num(a) for a in _AMOUNT_RE.findall(text)]
    seg["entities"] = {
        "amounts": amounts[:10],
        "phones": list(dict.fromkeys(_PHONE_RE.findall(text)))[:4],
        "accounts": list(dict.fromkeys(_ACCT_RE.findall(text)))[:4],
        "dates": _DATE_RE.findall(text)[:8],
        "orgs": list(dict.fromkeys(_ORG_RE.findall(text)))[:4],
        "txn_count": len(TXN.findall(text)),
    }
    return seg


# ---------------------------------------------------------------- embeddings

def embed_texts(texts: list[str]) -> list[list[float] | None]:
    """Dense embeddings via OpenAI when available; None otherwise.

    The hybrid retriever treats missing vectors as 'lexical only', so
    ingestion never blocks on a missing key.
    """
    if not dense_available():
        return [None] * len(texts)
    from openai import OpenAI

    client = OpenAI()
    out: list[list[float] | None] = []
    for i in range(0, len(texts), 96):
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts[i : i + 96])
        out.extend(d.embedding for d in resp.data)
    return out


# ------------------------------------------------------------------- ingest

def run_pipeline(raw_text: str, id_prefix: str = "seg", progress=None) -> list[dict]:
    """Full ingest of one document bundle. Returns enriched segments."""

    def tick(label, frac):
        if progress:
            progress(label, frac)

    tick("Segmenting documents…", 0.15)
    segments = segment_text(raw_text)

    tick("Classifying sections…", 0.4)
    segments = [classify_segment(s) for s in segments]

    tick("Extracting amounts, orgs, phones, transactions…", 0.65)
    segments = [extract_entities(s) for s in segments]

    tick("Indexing for retrieval…", 0.85)
    vectors = embed_texts([s["text"] for s in segments])
    for i, (s, v) in enumerate(zip(segments, vectors)):
        s["embedding"] = v
        s["id"] = f"{id_prefix}_{i:03d}"

    tick("Done", 1.0)
    return segments
