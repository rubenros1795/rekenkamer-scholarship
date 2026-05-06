#!/usr/bin/env python3
"""
llm_annotate_questions.py — LLM batch annotation of questions (schema A v0.7).

Annotates all questions in kamervragen-structured.csv with FEIT/CAU/OOR/ADV/?
labels using Claude via the Batches API (async, 50 % cost vs. real-time).

Usage
-----
    # Submit a batch (prints the batch ID):
    python llm_annotate_questions.py submit

    # Test with 20 rows first:
    python llm_annotate_questions.py submit --limit 20

    # Check processing status:
    python llm_annotate_questions.py status [BATCH_ID]

    # Download results and save annotated CSV:
    python llm_annotate_questions.py retrieve [BATCH_ID]

    # One-shot: submit + poll + save:
    python llm_annotate_questions.py run [--limit N]

If BATCH_ID is omitted the script reads it from batch_state.json, which is
written automatically by 'submit'.

Dependencies
------------
    pip install anthropic pandas
    # (anthropic is not yet in pyproject.toml — add it or install manually)
"""

import json
import sys
import time
import argparse
from pathlib import Path

import os

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from paths import STRUCTURED_CSV as INPUT_CSV, ANNOTATED_Q_V07 as OUTPUT_CSV

# ── Configuration ─────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

MODEL       = "claude-opus-4-6"
MAX_TOKENS  = 512               # 384 caused truncation; bumped to 512
POLL_INTERVAL = 60              # seconds between status checks

STATE_FILE       = Path(__file__).parent / "batch_state.json"
STATE_FILE_RETRY = Path(__file__).parent / "batch_state_q_retry.json"

# ── Annotation taxonomy (schema v0.7) ─────────────────────────────────────────

LABEL_ENUM = ["FEIT", "CAU", "OOR", "ADV", "?"]

MODIFIER_ENUM: list[str] = []  # geen modificatoren

# JSON Schema for structured output — enforced by the API, not just the prompt.
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": LABEL_ENUM,
        },
        "confidence": {
            "type": "string",
            "enum": ["H", "M", "L"],
        },
        "reasoning": {
            "type": "string",
        },
    },
    "required": ["label", "confidence", "reasoning"],
    "additionalProperties": False,
}

# ── Codebook (system prompt) ──────────────────────────────────────────────────
# Schema v0.7: 4 labels (AGN vervallen), geen modificatoren.
# Prompt-cached on every request to reduce cost.

SYSTEM_PROMPT = """\
Je bent een expert annotator voor vragen die de Tweede Kamer stelt aan de \
Algemene Rekenkamer (AR) naar aanleiding van gepubliceerde rapporten.

Jouw taak: label elke vraag met één primair label, een zekerheidscore en \
een korte redenering. Er zijn geen modificatoren.

De labels zijn geordend van passief naar actief — hoeveel eigen inbreng \
wordt er van de AR gevraagd?

  FEIT → CAU → OOR → ADV

────────────────────────────────────────
PRIMAIRE LABELS
────────────────────────────────────────

FEIT — Feitenvraag
  Het antwoord is direct raadpleegbaar in het rapport. De AR hoeft niets \
te beredeneren.
  Signalen: hoeveel · welke · wanneer · wat zijn · wat staat er in het \
rapport · bedragen/percentages · overzicht/uitsplitsing · verwijzing naar \
tabel of bijlage · waaruit bestaat · heeft de AR zicht op · kunt u \
specificeren · heeft de AR X onderzocht (scopevraag).
  Let op: "waaruit bestaat X" → FEIT (inhoud beschrijven). \
"Waarom bestaat X" → CAU (oorzaak verklaren). \
"Heeft de AR X onderzocht? Zo nee, waarom niet?" → FEIT (de scopevraag \
domineert). "Kunt u specificeren/aangeven/welke [zaken uit rapport]" → \
FEIT ook bij vergelijkende bijzin, zolang het antwoord al gedocumenteerd is.

CAU — Causaliteitsvraag
  De AR legt een causaal mechanisme uit: een oorzaak, reden, belemmering of \
gevolg tussen twee concrete variabelen. De AR construeert een verklaring — \
het antwoord staat niet kant-en-klaar in het rapport.
  Signalen: waarom · wat is de oorzaak van · wat veroorzaakt · wat zijn de \
belemmeringen voor · hoe heeft dit kunnen gebeuren · wat is de reden van · \
hoe komt het dat · welke gevolgen heeft X op Y.
  Let op: "welke gevolgen heeft X op Y" → CAU (mechanisme). \
"Wat zijn de gevolgen als X niet gebeurt" → OOR (hypothetisch).

OOR — Oordeelsvraag
  De AR neemt een evaluatief, interpretatief, hypothetisch of prognostisch \
standpunt in. Vier varianten (allemaal OOR):
  • Evaluatief: "hoe beoordeelt u", "in hoeverre acht u voldoende", \
"is er voldoende", "acht u het terecht", "ziet u verbetering"
  • Interpretatief: "wat betekent", "hoe verhoudt X zich tot Y" (met \
waardeoordeel), "wat impliceert", "kunt u nader toelichten wat wordt \
bedoeld met [eigen AR-formulering]", "wat is de consequentie van"
  • Hypothetisch: "wat zijn de gevolgen als", "zou X leiden tot"
  • Prognostisch: "ligt het in de lijn der verwachting", "wanneer verwacht \
u", "is het realistisch dat", "wordt X teruggevorderd", "gaat u een vervolg \
geven", "op welke manier zet u uw onderzoek voort"

ADV — Advisvraag
  De AR geeft een prescriptief advies, aanbeveling of wordt gevraagd actie \
te ondernemen — inclusief verzoeken om nieuw of vervolgonderzoek.
  Signalen: welke aanbevelingen doet u · welke lessen trekt u · wat moet er \
worden gedaan · op welke wijze kan X worden verbeterd · hoe kan X worden \
opgelost · wat adviseert u · wat zou een betere X zijn · in hoeverre kan \
voorkomen worden dat · zou de AR X kunnen onderzoeken · bent u bereid \
onderzoek te doen naar · kan dit worden uitgebreid naar.
  Let op: "in hoeverre kan voorkomen worden dat X" → ADV (preventieve \
maatregel), niet OOR. "Zou de AR X kunnen onderzoeken?" → ADV (verzoek om \
nieuwe actie). "Gaat u een vervolg geven?" → OOR (vraag naar plannen, \
niet een prescriptief verzoek). "Zou het een voorkeur verdienen?" → OOR \
(evaluatief, ondanks "voorkeur").

? — Oncodeerbaar
  Uitsluitend als de vraag door extractie-artefacten onleesbaar is.

────────────────────────────────────────
BESLISREGELS GRENSGEVALLEN
────────────────────────────────────────

FEIT vs. CAU
• "Heeft de AR zicht op X?" / "Heeft de AR X onderzocht?" → FEIT (scopevraag).
• "Waaruit bestaat X?" → FEIT. "Waarom is X zo?" → CAU.
• Staat de verklaring letterlijk in het rapport? → FEIT. \
Moet de AR haar zelf construeren? → CAU.

FEIT vs. OOR
• "Kunt u specificeren/aangeven/welke [zaken al in rapport]" → FEIT, \
ook bij vergelijkende bijzin als beide waarden gedocumenteerd zijn.
• "Hoe verhoudt X zich tot Y?" → OOR als waardeoordeel vereist; FEIT als \
de vergelijking rekenkundig is en beide waarden gedocumenteerd zijn.
• Bevestigingsvraag: citeerbaar uit rapport → FEIT; AR moet interpreteren \
of oordelen → OOR.

CAU vs. OOR
• "Welke gevolgen heeft X op Y?" → CAU. \
"Wat zijn de gevolgen als X niet gebeurt?" → OOR.
• "Wat is de reden van X?" → CAU. "In hoeverre is X voldoende?" → OOR.
• "Kunt u nader toelichten wat wordt bedoeld met [AR's eigen tekst]?" → OOR.
• "Wat is de consequentie hiervan?" → OOR.
• Twijfelgeval: normatief woord (voldoende, terecht) of betekenisvraag \
(wat betekent) aanwezig? → OOR. Anders → CAU.

OOR vs. ADV
• Vraag evalueert/interpreteert/prognosticeert wat is/was/zou zijn \
of de AR's plannen → OOR.
• Vraag vraagt de AR actie te ondernemen (beleid, aanbeveling, \
nieuw onderzoek starten) → ADV.
• "Gaat u een vervolg geven?" → OOR (plannen toelichten). \
"Bent u bereid X te onderzoeken?" → ADV (prescriptief verzoek).
• "In hoeverre kan voorkomen worden dat X?" → ADV (preventieve maatregel).
• Bij gelijke gewichten → ADV.

Meerdere deelvragen
• Label de meest substantiële deelvraag (doorgaans de laatste).
• Bij gelijke gewichten: FEIT < CAU < OOR < ADV.

────────────────────────────────────────
PRIORITEITSREGEL (bij gemengde vragen)
────────────────────────────────────────

ADV > OOR > CAU > FEIT

• Bevat de vraag zowel een prescriptief als een evaluatief element → ADV.
• Bevat de vraag zowel een oordeelselement als een causaal element → OOR.
• Bevat de vraag zowel een causaal als een feitelijk element → CAU.
• "Wat is de oorzaak van X, en wat vindt u hiervan?" → OOR (oordeel wint).
• "Hoeveel zijn er, en wat zijn uw aanbevelingen?" → ADV (ADV wint).

────────────────────────────────────────
ZEKERHEIDSSCHAAL
────────────────────────────────────────

H — Één label past duidelijk; geen serieus alternatief.
M — Één plausibel alternatief; keuze verdedigbaar maar niet dwingend.
L — Twee of meer plausibele alternatieven.

────────────────────────────────────────
OUTPUTFORMAAT
────────────────────────────────────────

Geef uitsluitend een JSON-object met drie velden:
  label       – één van: FEIT, CAU, OOR, ADV, ?
  confidence  – H, M of L
  reasoning   – max. 2 zinnen in het Nederlands; benoem de beslisregel \
die de doorslag gaf.
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _parse_json(raw: str) -> dict:
    """Parse JSON from model output, stripping markdown code fences if present."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]           # drop opening ```json line
        raw = raw.rsplit("```", 1)[0].strip()  # drop closing ```
    return json.loads(raw)


def _load_data(limit: int | None = None) -> pd.DataFrame:
    dat = pd.read_csv(INPUT_CSV, sep="\t").reset_index(drop=True)
    if limit:
        dat = dat.head(limit)
    print(f"Loaded {len(dat)} questions from {INPUT_CSV.name}")
    return dat


def _build_requests(dat: pd.DataFrame) -> list[Request]:
    """One Batch request per question row, keyed by DataFrame index."""
    requests = []
    for idx, row in dat.iterrows():
        rapport      = str(row.get("rapport_titel", "") or "onbekend").strip()
        vergaderjaar = str(row.get("vergaderjaar", "") or "").strip()
        question     = str(row["vraag"]).strip()

        user_content = (
            f"Rapport: {rapport}\n"
            f"Vergaderjaar: {vergaderjaar}\n\n"
            f"Vraag: {question}"
        )

        requests.append(Request(
            custom_id=str(idx),
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # Prompt caching: the codebook is shared across all ~1 200
                    # requests so it will be cached after the first hit,
                    # reducing cost and latency significantly.
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_content}],
            ),
        ))
    return requests


def _resolve_batch_id(provided: str | None, parser: argparse.ArgumentParser) -> str:
    if provided:
        return provided
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text())
        batch_id = state["batch_id"]
        print(f"Using batch ID from {STATE_FILE.name}: {batch_id}")
        return batch_id
    parser.error(
        "No BATCH_ID given and no batch_state.json found. Run 'submit' first."
    )


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_submit(limit: int | None = None) -> str:
    client   = _client()
    dat      = _load_data(limit)
    requests = _build_requests(dat)

    print(f"Submitting {len(requests)} requests to the Batches API …")
    batch    = client.messages.batches.create(requests=requests)
    batch_id = batch.id

    STATE_FILE.write_text(json.dumps(
        {"batch_id": batch_id, "n_requests": len(requests), "limit": limit},
        indent=2,
    ))

    print(f"\nBatch submitted successfully.")
    print(f"  Batch ID : {batch_id}")
    print(f"  State    : {STATE_FILE}")
    print(f"\nCheck status:  python llm_annotate_questions.py status {batch_id}")
    print(f"Get results:   python llm_annotate_questions.py retrieve {batch_id}")
    return batch_id


def cmd_status(batch_id: str) -> None:
    batch  = _client().messages.batches.retrieve(batch_id)
    counts = batch.request_counts
    print(f"Batch {batch_id}")
    print(f"  Status      : {batch.processing_status}")
    print(f"  Processing  : {counts.processing}")
    print(f"  Succeeded   : {counts.succeeded}")
    print(f"  Errored     : {counts.errored}")
    print(f"  Canceled    : {counts.canceled}")
    print(f"  Expired     : {counts.expired}")


def cmd_retrieve(batch_id: str) -> None:
    client = _client()

    # ── Poll until the batch has ended ────────────────────────────────────────
    print(f"Waiting for batch {batch_id} to finish …")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        c = batch.request_counts
        print(
            f"  [{batch.processing_status}]  "
            f"processing={c.processing}  succeeded={c.succeeded}  errored={c.errored}"
        )
        time.sleep(POLL_INTERVAL)

    c = batch.request_counts
    print(
        f"\nBatch complete — succeeded={c.succeeded}  errored={c.errored}  "
        f"canceled={c.canceled}  expired={c.expired}"
    )

    # ── Collect and parse results ──────────────────────────────────────────────
    annotations: dict[int, dict] = {}
    errors: list[str] = []

    for result in client.messages.batches.results(batch_id):
        idx = int(result.custom_id)

        if result.result.type == "succeeded":
            raw = result.result.message.content[0].text
            try:
                annotations[idx] = _parse_json(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"row {idx}: JSON parse error — {exc}")
                annotations[idx] = {
                    "label": "?", "modifiers": [], "confidence": "L",
                    "reasoning": f"parse error: {exc}",
                }
        elif result.result.type == "errored":
            err = result.result.error.type
            errors.append(f"row {idx}: API error — {err}")
            annotations[idx] = {
                "label": "?", "modifiers": [], "confidence": "L",
                "reasoning": f"API error: {err}",
            }
        else:
            # canceled or expired
            errors.append(f"row {idx}: {result.result.type}")
            annotations[idx] = {
                "label": "?", "modifiers": [], "confidence": "L",
                "reasoning": result.result.type,
            }

    if errors:
        print(f"\n{len(errors)} annotation errors (first 10):")
        for e in errors[:10]:
            print(f"  {e}")

    # ── Merge into the source DataFrame ───────────────────────────────────────
    dat = _load_data()  # full dataset (no limit)

    def _get(idx: int, field: str, default="") -> str:
        return str(annotations.get(idx, {}).get(field, default))

    dat["llm_label"]      = [_get(i, "label")      for i in dat.index]
    dat["llm_confidence"] = [_get(i, "confidence")  for i in dat.index]
    dat["llm_reasoning"]  = [_get(i, "reasoning")   for i in dat.index]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    dat.to_csv(OUTPUT_CSV, sep="\t", index=False)
    print(f"\nSaved {len(dat)} annotated rows → {OUTPUT_CSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nLabel distribution:")
    print(dat["llm_label"].value_counts().to_string())
    print("\nConfidence distribution:")
    print(dat["llm_confidence"].value_counts().to_string())


def cmd_submit_retry() -> str:
    """Resubmit rows where llm_label == '?' in the current annotated CSV."""
    if not OUTPUT_CSV.exists():
        sys.exit(f"No annotated CSV at {OUTPUT_CSV}. Run 'retrieve' first.")

    dat_ann = pd.read_csv(OUTPUT_CSV, sep="\t")
    error_idx = dat_ann.index[dat_ann["llm_label"] == "?"].tolist()

    if not error_idx:
        print("No '?' rows found — nothing to retry.")
        return ""

    print(f"Found {len(error_idx)} '?' rows to retry.")
    dat_orig = _load_data()  # full original data (index matches annotated CSV)
    retry_dat = dat_orig.loc[error_idx]

    client   = _client()
    requests = _build_requests(retry_dat)
    batch    = client.messages.batches.create(requests=requests)
    batch_id = batch.id

    STATE_FILE_RETRY.write_text(json.dumps(
        {"batch_id": batch_id, "n_requests": len(requests)}, indent=2,
    ))
    print(f"Retry batch submitted: {batch_id}")
    print(f"Retrieve with: python llm_annotate_questions.py retrieve-merge {batch_id}")
    return batch_id


def cmd_retrieve_merge(batch_id: str) -> None:
    """Retrieve retry batch and patch '?' rows in the annotated CSV."""
    client = _client()

    print(f"Waiting for retry batch {batch_id} …")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        c = batch.request_counts
        print(f"  [{batch.processing_status}] processing={c.processing} succeeded={c.succeeded}")
        time.sleep(POLL_INTERVAL)

    c = batch.request_counts
    print(f"\nRetry complete — succeeded={c.succeeded}  errored={c.errored}")

    retry_ann: dict[int, dict] = {}
    for result in client.messages.batches.results(batch_id):
        idx = int(result.custom_id)
        if result.result.type == "succeeded":
            raw = result.result.message.content[0].text
            try:
                retry_ann[idx] = _parse_json(raw)
            except json.JSONDecodeError as exc:
                retry_ann[idx] = {"label": "?", "confidence": "L", "reasoning": f"parse error: {exc}"}
        else:
            err = getattr(result.result, "error", result.result.type)
            retry_ann[idx] = {"label": "?", "confidence": "L", "reasoning": str(err)}

    dat = pd.read_csv(OUTPUT_CSV, sep="\t")
    patched = 0
    for idx, ann in retry_ann.items():
        if ann.get("label", "?") != "?":
            dat.at[idx, "llm_label"]      = ann["label"]
            dat.at[idx, "llm_confidence"] = ann["confidence"]
            dat.at[idx, "llm_reasoning"]  = ann["reasoning"]
            patched += 1

    dat.to_csv(OUTPUT_CSV, sep="\t", index=False)
    print(f"Patched {patched} rows → {OUTPUT_CSV.name}")
    print(f"Remaining '?' labels: {(dat['llm_label'] == '?').sum()}")
    print("\nLabel distribution:")
    print(dat["llm_label"].value_counts().to_string())


def cmd_run(limit: int | None = None) -> None:
    batch_id = cmd_submit(limit)
    cmd_retrieve(batch_id)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate kamervragen with Claude Batches API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Submit a new annotation batch")
    p_submit.add_argument(
        "--limit", type=int, default=None,
        metavar="N", help="Annotate only the first N rows (for testing)",
    )

    p_status = sub.add_parser("status", help="Check batch processing status")
    p_status.add_argument(
        "batch_id", nargs="?", default=None,
        help="Batch ID (defaults to last submitted, from batch_state.json)",
    )

    p_retrieve = sub.add_parser("retrieve", help="Download results and save CSV")
    p_retrieve.add_argument(
        "batch_id", nargs="?", default=None,
        help="Batch ID (defaults to last submitted, from batch_state.json)",
    )

    p_run = sub.add_parser("run", help="Submit + poll + save in one go")
    p_run.add_argument(
        "--limit", type=int, default=None,
        metavar="N", help="Annotate only the first N rows (for testing)",
    )

    sub.add_parser("submit-retry", help="Resubmit '?' rows from the annotated CSV")

    p_rm = sub.add_parser("retrieve-merge", help="Retrieve retry batch and patch annotated CSV")
    p_rm.add_argument(
        "batch_id", nargs="?", default=None,
        help="Retry batch ID (defaults to batch_state_q_retry.json)",
    )

    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(args.limit)

    elif args.command == "status":
        cmd_status(_resolve_batch_id(args.batch_id, parser))

    elif args.command == "retrieve":
        cmd_retrieve(_resolve_batch_id(args.batch_id, parser))

    elif args.command == "run":
        cmd_run(args.limit)

    elif args.command == "submit-retry":
        cmd_submit_retry()

    elif args.command == "retrieve-merge":
        bid = args.batch_id
        if not bid:
            if STATE_FILE_RETRY.exists():
                bid = json.loads(STATE_FILE_RETRY.read_text())["batch_id"]
                print(f"Using retry batch ID from {STATE_FILE_RETRY.name}: {bid}")
            else:
                parser.error("No batch_id given and no batch_state_q_retry.json found.")
        cmd_retrieve_merge(bid)


if __name__ == "__main__":
    main()
