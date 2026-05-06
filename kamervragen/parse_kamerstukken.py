#!/usr/bin/env python3
"""
parse_kamerstukken.py — Extract structured Q&A pairs from raw Rekenkamer kamerstukken.

Reads raw document text from kamervragen-raw.csv and the document index from
rekenkamer.xlsx, submits each document to Claude (Haiku via the Batches API),
and saves extracted question-answer pairs to kamervragen-structured.csv.

Usage
-----
    python parse_kamerstukken.py submit [--limit N]
    python parse_kamerstukken.py status  [BATCH_ID]
    python parse_kamerstukken.py retrieve [BATCH_ID]
    python parse_kamerstukken.py run     [--limit N]
"""

import json
import os
import re
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# ── Configuration ─────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL      = "claude-haiku-4-5-20251001"
MAX_TOKENS = 8192

from paths import RAW_CSV, STRUCTURED_CSV as OUTPUT_CSV, DATA_DIR

# Raw Excel input: configurable via env var INPUT_XLSX, defaults to data/rekenkamer.xlsx
INPUT_XLSX = Path(os.environ.get("INPUT_XLSX", str(DATA_DIR / "rekenkamer.xlsx")))
STATE_FILE = Path(__file__).parent / "batch_state.json"
META_FILE  = Path(__file__).parent / "batch_meta.json"

SRC_COLS = ["_id", "title", "doc_url", "published_at", "source"]

# ── JSON Schema ───────────────────────────────────────────────────────────────

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "dossier_nummer": {"type": "string"},
                "rapport_titel":  {"type": "string"},
                "datum":          {"type": "string"},
                "vergaderjaar":   {"type": "string"},
                "actor":   {"type": "string"}
            },
            "required": ["dossier_nummer", "rapport_titel"]
        },
        "extracted_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_number": {"type": "integer"},
                    "question_text":   {"type": "string"},
                    "answer_text":     {"type": "string"},
                    "answer_refers_to": {
                        "type": "integer",
                        "description": "If the answer is 'Zie antwoord vraag N', set this to N, else null."
                    },
                    "references": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "dossier":    {"type": "string"},
                                "onderwerp":  {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["question_number", "question_text", "answer_text"]
            }
        }
    },
    "required": ["metadata", "extracted_pairs"]
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def light_clean(text):
    if not isinstance(text, str):
        return ""
    # Strip boilerplate headers and footers
    text = re.sub(r'Postbus 20015.*?www\.rekenkamer\.nl', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Normalise whitespace
    text = re.sub(r'\.\.,', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def sanitize_df(df):
    """Replace newlines and tabs in all string columns so they don't break TSV."""
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(
        lambda col: col.str.replace(r'[\r\n]+', ' ', regex=True).str.replace('\t', ' ')
    )
    return df

def load_df():
    df = pd.read_excel(INPUT_XLSX)
    df = df[df._id.str.contains("beantwo")].reset_index(drop=True)
    return df

SYSTEM_PROMPT = """\
Je bent een nauwkeurige data-extractor voor parlementaire documenten van de Algemene Rekenkamer.

TAAK: Extraheer ALLE vraag-antwoord paren uit het document. Sla geen enkele vraag over.

REGELS:
- Neem elke genummerde vraag op, ook als het antwoord kort is of verwijst naar een ander antwoord.
- Kopieer de vraagstekst en antwoordtekst letterlijk — parafraseer niet.
- Als een antwoord luidt "Zie antwoord vraag N", noteer dan de volledige antwoordtekst én zet question_number N in answer_refers_to.
- Verwijzingen naar Kamerstukken (bijv. "33 123, nr. 4") horen in het references-veld.
- Haal de metadata (dossier_nummer, rapport_titel, datum, vergaderjaar, actor (vaak een Kamercommissie, zoals Vaste Commissie voor de Rijksuitgaven of Commissie Veiligheid en Jusitie. Als er meerdere actoren zijn, scheidt ze met ;)) uit de aanhef of het briefhoofd.
"""

# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_export_raw():
    """Write the filtered kamervragen rows to a raw CSV (no API needed)."""
    df = load_df()
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    sanitize_df(df).to_csv(RAW_CSV, index=False, sep="\t")
    print(f"Exported {len(df)} kamervragen rows to {RAW_CSV}")


def cmd_submit(limit=None):
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    df = load_df()
    if limit:
        df = df.head(limit)

    # Save filtered raw data
    cmd_export_raw()

    requests = []
    for idx, row in df.iterrows():
        content = light_clean(str(row.get("doc_content", "")))
        if not content:
            print(f"  [SKIP idx={idx}] empty content after cleaning")
            continue
        requests.append(Request(
            custom_id=str(idx),
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
                tools=[{
                    "name": "extract_document",
                    "description": "Extraheer alle Q&A paren en metadata uit het document.",
                    "input_schema": OUTPUT_SCHEMA
                }],
                tool_choice={"type": "tool", "name": "extract_document"},
            ),
        ))

    if not requests:
        print("No valid documents to submit.")
        return

    # Chunk into batches of 50 to avoid 502s on large payloads
    CHUNK = 50
    batch_ids = []
    for i in range(0, len(requests), CHUNK):
        chunk = requests[i:i + CHUNK]
        for attempt in range(4):
            try:
                batch = client.messages.batches.create(requests=chunk)
                batch_ids.append(batch.id)
                print(f"  Submitted chunk {i//CHUNK + 1}: {batch.id} ({len(chunk)} docs)")
                break
            except anthropic.InternalServerError as e:
                if attempt == 3:
                    raise
                wait = 2 ** attempt
                print(f"  Server error on attempt {attempt+1}, retrying in {wait}s: {e}")
                time.sleep(wait)

    STATE_FILE.write_text(json.dumps({"batch_ids": batch_ids}, indent=2))

    src_meta = {str(idx): {c: str(row.get(c, "")) for c in SRC_COLS} for idx, row in df.iterrows()}
    META_FILE.write_text(json.dumps(src_meta, indent=2))

    print(f"Submitted {len(requests)} documents across {len(batch_ids)} batch(es).")


def _load_batch_ids(bid_arg):
    """Return list of batch IDs from arg or state file."""
    if bid_arg:
        return [bid_arg]
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        # Support both old single-id format and new multi-id format
        return state.get("batch_ids") or [state["batch_id"]]
    return []


def cmd_status(batch_ids):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    for bid in batch_ids:
        batch = client.messages.batches.retrieve(bid)
        c = batch.request_counts
        print(f"\nBatch ID: {bid}")
        print(f"Status:   {batch.processing_status.upper()}")
        print(f"Success:  {c.succeeded}  Failed: {c.errored}  "
              f"Canceled: {getattr(c, 'canceled', '?')}  "
              f"Expired: {getattr(c, 'expired', '?')}  "
              f"Pending: {c.processing}")


def cmd_retrieve(batch_ids):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    src_meta = json.loads(META_FILE.read_text()) if META_FILE.exists() else {}

    rows = []
    for bid in batch_ids:
        batch = client.messages.batches.retrieve(bid)
        if batch.processing_status != "ended":
            print(f"Batch {bid} not ready yet ('{batch.processing_status}') — skipping.")
            continue

        print(f"Retrieving {bid}...")
        for result in client.messages.batches.results(bid):
            if result.result.type != "succeeded":
                print(f"  [FAILED id={result.custom_id}] {getattr(result.result, 'error', result.result)}")
                continue

            block = result.result.message.content[0]
            data  = block.input if block.type == "tool_use" else json.loads(block.text)
            meta  = data.get("metadata", {})
            pairs = data.get("extracted_pairs", [])
            src   = src_meta.get(result.custom_id, {})

            if not pairs:
                print(f"  [EMPTY id={result.custom_id}] {src.get('_id', '?')}")

            for pair in pairs:
                refs = [
                    f"{r.get('dossier', '')}: {r.get('onderwerp', '')}".strip(": ")
                    for r in pair.get("references", [])
                ]
                rows.append({
                    "batch_idx":        result.custom_id,
                    "src_id":           src.get("_id"),
                    "src_title":        src.get("title"),
                    "src_url":          src.get("doc_url"),
                    "src_published_at": src.get("published_at"),
                    "dossier_nummer":   meta.get("dossier_nummer"),
                    "rapport_titel":    meta.get("rapport_titel"),
                    "datum":            meta.get("datum"),
                    "vergaderjaar":     meta.get("vergaderjaar"),
                    "vraag_nr":         pair.get("question_number"),
                    "vraag":            pair.get("question_text"),
                    "antwoord":         pair.get("answer_text"),
                    "antwoord_zie_nr":  pair.get("answer_refers_to"),
                    "verwijzingen":     " | ".join(refs),
                })

    final_df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sanitize_df(final_df).to_csv(OUTPUT_CSV, index=False, sep="\t")
    print(f"Saved {len(final_df)} extracted questions to {OUTPUT_CSV}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("export-raw")
    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--limit", type=int, default=None)
    sub.add_parser("retrieve").add_argument("batch_id", nargs="?")
    sub.add_parser("status").add_argument("batch_id", nargs="?")

    args = parser.parse_args()

    if args.command == "export-raw":
        cmd_export_raw()
    elif args.command == "submit":
        cmd_submit(args.limit)
    elif args.command in ("status", "retrieve"):
        bids = _load_batch_ids(getattr(args, "batch_id", None))
        if not bids:
            print("No Batch ID found. Run 'submit' first.")
            return
        if args.command == "status":
            cmd_status(bids)
        else:
            cmd_retrieve(bids)


if __name__ == "__main__":
    main()
