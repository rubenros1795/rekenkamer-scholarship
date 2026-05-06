#!/usr/bin/env python3
"""
llm_annotate_answers.py — LLM batch annotation of AR answers (schema B v0.3).

Annotates AR answers in kamervragen-structured.csv with FEIT/CAU/OOR/ADV/DEFL
labels using Claude via the Batches API (async, 50 % cost vs. real-time).

Usage
-----
    python llm_annotate_answers.py submit [--limit N]
    python llm_annotate_answers.py status  [BATCH_ID]
    python llm_annotate_answers.py retrieve [BATCH_ID]
    python llm_annotate_answers.py run     [--limit N]
"""

import json
import os
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from paths import STRUCTURED_CSV as INPUT_CSV, ANNOTATED_A_V03 as OUTPUT_CSV

# ── Configuration ─────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

MODEL         = "claude-opus-4-6"
MAX_TOKENS    = 384
POLL_INTERVAL = 60

STATE_FILE       = Path(__file__).parent / "batch_state_answers.json"
STATE_FILE_RETRY = Path(__file__).parent / "batch_state_a_retry.json"

MIN_ANSWER_LEN = 30  # kortere antwoorden worden overgeslagen

# ── Annotation taxonomy (antwoordschema v0.3) ──────────────────────────────────

LABEL_ENUM = ["FEIT", "CAU", "OOR", "ADV", "DEFL"]

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "label":      {"type": "string", "enum": LABEL_ENUM},
        "confidence": {"type": "string", "enum": ["H", "M", "L"]},
        "reasoning":  {"type": "string"},
    },
    "required": ["label", "confidence", "reasoning"],
    "additionalProperties": False,
}

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
Je bent een expert annotator voor antwoorden die de Algemene Rekenkamer (AR) \
geeft op vragen van de Tweede Kamer.

Jouw taak: label elk antwoord met één primair label, een zekerheidscore en \
een korte redenering.

De labels beschrijven wat de AR primair levert. Ze zijn geordend van \
passief naar actief:

  FEIT → CAU → OOR → ADV → DEFL

────────────────────────────────────────
PRIMAIRE LABELS
────────────────────────────────────────

FEIT — AR rapporteert of beschrijft
  AR levert feitelijke informatie, beschrijft bevindingen, of rapporteert \
wat het onderzoek heeft opgeleverd. Hieronder vallen ook causale bevindingen \
die de AR rechtstreeks citeert — zolang de AR een gedocumenteerde uitkomst \
meldt zonder zelf een verklaring te construeren.
  Signalen: "wij hebben vastgesteld dat", "uit ons onderzoek blijkt", \
"de situatie is", beschrijvingen van processen of oorzaken die direct uit \
het rapport worden geciteerd, verwijzing naar rapportpagina.
  Cruciaal: causale woorden (oorzaak, veroorzaakt, omdat) alleen maken een \
antwoord NIET tot CAU. "Dit wordt veroorzaakt door X" = FEIT als het een \
geciteerde bevinding is. Alleen als de AR een verklaring construeert die \
verder gaat dan één gedocumenteerde bevinding → CAU.

CAU — AR construeert een causale verklaring
  AR legt uit waarom iets zo is door meerdere factoren samen te brengen, een \
mechanisme te reconstrueren, of te synthetiseren — verder dan het citeren \
van een enkelvoudige bevinding.
  Signalen: "de oorzaak ligt in de combinatie van..." · "dit verklaren wij \
door..." · "het mechanisme werkt als volgt:..." · uitgebreide toelichting \
op een causaal verband met meerdere schakels · "wij schrijven dit toe aan..." \
· "de wisselwerking tussen A en B leidt ertoe dat..."
  Onderscheid met FEIT: FEIT = AR meldt een gedocumenteerde (causale) \
uitkomst. CAU = AR elaboreert op het mechanisme of synthetiseert meerdere \
factoren tot een verklaring in eigen woorden.
  Onderscheid met OOR: CAU verklaart hoe/waarom iets zo is gekomen \
(mechanistisch). OOR beoordeelt of het goed/toereikend/betekenisvol is \
(normatief). Antwoord dat causaal begint maar eindigt met een oordeel → OOR.

OOR — AR neemt een standpunt in
  AR evalueert, interpreteert of vormt een oordeel. AR gaat verder dan \
rapporteren of verklaren: ze neemt positie in over de kwaliteit, \
toereikendheid of betekenis van iets.
  Expliciete signalen: "wij vinden dat", "wij oordelen dat", "naar ons \
oordeel", "wij constateren dat X onvoldoende is", "wij vinden het \
belangrijk dat".
  Impliciete OOR — let op deze patronen:
  • AR vergelijkt uitkomst met norm/standaard: "bleef onder de \
tolerantiegrens", "had nog niet het gewenste niveau", "voldeed niet aan \
de norm" → OOR (impliciet normatief oordeel, ook zonder "wij vinden").
  • AR maakt gekwalificeerde inschatting: "lijkt er op te wijzen dat", \
"kunnen we niet met zekerheid zeggen, maar...", "heeft bijgedragen aan" \
→ OOR (AR weegt bewijs en neemt epistemisch standpunt in).
  • AR verdedigt eigen methodologische keuzes of timing → OOR.
  • AR wijst verantwoordelijkheid toe: "het is primair de \
verantwoordelijkheid van X" → OOR (interpretatief).
  • AR evalueert beleidsimpact: "is niet goed van de grond gekomen", \
"heeft onvoldoende resultaat opgeleverd" → OOR.
  FEIT/OOR-grensregel: vergelijkt het antwoord een gemeten waarde met een \
normatief referentiepunt (tolerantiegrens, verwachting, gewenst niveau)? \
→ OOR. Meldt het alleen een waarde zonder norm-vergelijking? → FEIT.

ADV — AR geeft een aanbeveling
  AR schrijft voor wat er moet of zou moeten. Prescriptief advies of \
concrete aanbevelingen gericht op toekomstige actie.
  Signalen: "wij bevelen aan", "het is van belang dat", "de minister \
dient", "wij adviseren", concrete stappen die ondernomen moeten worden, \
"X moet zorgen dat...", "wij verwachten van Y dat...".

DEFL — AR beantwoordt de vraag niet substantieel
  AR verwijst naar de minister, geeft aan dat het buiten scope valt, heeft \
het niet onderzocht, of verwijst naar een ander antwoord of toekomstig \
onderzoek. Geen inhoudelijk antwoord op de gestelde vraag.
  Signalen: "dit hebben wij niet onderzocht", "voor een antwoord verwijzen \
wij u naar de minister", "dit valt buiten de reikwijdte", "wij kunnen \
hierover geen uitspraak doen", "het is aan de minister om", "zie het \
antwoord bij vraag X", "dit zullen wij in de komende periode verkennen", \
"onze rapportage is voorzien voor [toekomstige datum]".
  Let op: een uitleg van WAAROM de AR niet kan antwoorden is nog steeds \
DEFL — de inhoud van de uitleg telt niet als antwoord. \
"Zie het antwoord bij vraag X" zonder eigen inhoud → DEFL.

────────────────────────────────────────
PRIORITEITSREGEL (1 label kiezen)
────────────────────────────────────────

Bij gemengde antwoorden: ADV > OOR > CAU > FEIT > DEFL.
• Bevat het antwoord zowel oordeel als aanbeveling → ADV.
• Bevat het antwoord zowel een causale verklaring als een oordeel → OOR.
• Bevat het antwoord zowel een geciteerde bevinding als een geconstrueerde \
verklaring → CAU.
• Inhoud (FEIT/CAU/OOR) wint altijd van deflectie.
• Begint met "niet onderzocht" maar bevat vervolgens een aanbeveling → ADV.
• Alleen DEFL als er geen substantiële inhoud is.

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
  label       – één van: FEIT, CAU, OOR, ADV, DEFL
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
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _load_data(limit: int | None = None) -> pd.DataFrame:
    dat = pd.read_csv(INPUT_CSV, sep="\t").reset_index(drop=True)
    # Houd alleen rijen met een substantieel antwoord
    dat = dat[dat["antwoord"].notna() & (dat["antwoord"].str.len() >= MIN_ANSWER_LEN)]
    if limit:
        dat = dat.head(limit)
    print(f"Loaded {len(dat)} rows with answers from {INPUT_CSV.name}")
    return dat


def _build_requests(dat: pd.DataFrame) -> list[Request]:
    requests = []
    for idx, row in dat.iterrows():
        rapport  = str(row.get("rapport_titel", "") or "onbekend").strip()
        vraag    = str(row.get("vraag", "") or "").strip()
        antwoord = str(row["antwoord"]).strip()

        user_content = (
            f"Rapport: {rapport}\n\n"
            f"Vraag (context): {vraag}\n\n"
            f"Antwoord van de AR:\n{antwoord}"
        )

        requests.append(Request(
            custom_id=str(idx),
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
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
    parser.error("No BATCH_ID given and no batch_state_answers.json found. Run 'submit' first.")


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
    print(f"\nBatch submitted: {batch_id}")
    print(f"Check:    python llm_annotate_answers.py status {batch_id}")
    print(f"Retrieve: python llm_annotate_answers.py retrieve {batch_id}")
    return batch_id


def cmd_status(batch_id: str) -> None:
    batch  = _client().messages.batches.retrieve(batch_id)
    counts = batch.request_counts
    print(f"Batch {batch_id}  [{batch.processing_status}]")
    print(f"  processing={counts.processing}  succeeded={counts.succeeded}  "
          f"errored={counts.errored}  canceled={counts.canceled}  expired={counts.expired}")


def cmd_retrieve(batch_id: str) -> None:
    client = _client()

    print(f"Waiting for batch {batch_id} …")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        c = batch.request_counts
        print(f"  [{batch.processing_status}] processing={c.processing} succeeded={c.succeeded}")
        time.sleep(POLL_INTERVAL)

    c = batch.request_counts
    print(f"\nBatch complete — succeeded={c.succeeded}  errored={c.errored}")

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
                annotations[idx] = {"label": "?", "confidence": "L", "reasoning": f"parse error: {exc}"}
        else:
            err = getattr(result.result, "error", result.result.type)
            errors.append(f"row {idx}: {err}")
            annotations[idx] = {"label": "?", "confidence": "L", "reasoning": str(err)}

    if errors:
        print(f"\n{len(errors)} errors (first 10):")
        for e in errors[:10]:
            print(f"  {e}")

    dat = _load_data()

    def _get(idx: int, field: str) -> str:
        return str(annotations.get(idx, {}).get(field, ""))

    dat["llm_label"]      = [_get(i, "label")      for i in dat.index]
    dat["llm_confidence"] = [_get(i, "confidence")  for i in dat.index]
    dat["llm_reasoning"]  = [_get(i, "reasoning")   for i in dat.index]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    dat.to_csv(OUTPUT_CSV, sep="\t", index=False)
    print(f"\nSaved {len(dat)} annotated rows → {OUTPUT_CSV.name}")
    print("\nLabel distribution:")
    print(dat["llm_label"].value_counts().to_string())


def cmd_submit_retry() -> str:
    """Resubmit rows where llm_label == '?' in the current annotated CSV."""
    if not OUTPUT_CSV.exists():
        sys.exit(f"No annotated CSV at {OUTPUT_CSV}. Run 'retrieve' first.")

    # _load_data() gives the filtered dataset with its original (non-contiguous) index.
    # The annotated CSV was saved from that same filtered set (index=False),
    # so its positional order matches but index labels differ — realign here.
    orig = _load_data()
    dat_ann = pd.read_csv(OUTPUT_CSV, sep="\t")
    dat_ann.index = orig.index  # restore original index so it matches custom_ids

    error_idx = dat_ann.index[dat_ann["llm_label"] == "?"].tolist()

    if not error_idx:
        print("No '?' rows found — nothing to retry.")
        return ""

    print(f"Found {len(error_idx)} '?' rows to retry.")
    retry_dat = orig.loc[error_idx]

    client   = _client()
    requests = _build_requests(retry_dat)
    batch    = client.messages.batches.create(requests=requests)
    batch_id = batch.id

    STATE_FILE_RETRY.write_text(json.dumps(
        {"batch_id": batch_id, "n_requests": len(requests)}, indent=2,
    ))
    print(f"Retry batch submitted: {batch_id}")
    print(f"Retrieve with: python llm_annotate_answers.py retrieve-merge {batch_id}")
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

    # Reload annotated CSV and realign index to match original filtered index.
    orig = _load_data()
    dat = pd.read_csv(OUTPUT_CSV, sep="\t")
    dat.index = orig.index

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
    cmd_retrieve(cmd_submit(limit))


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate AR antwoorden with Claude Batches API")
    sub    = parser.add_subparsers(dest="command", required=True)

    for name in ("submit", "run"):
        p = sub.add_parser(name)
        p.add_argument("--limit", type=int, default=None, metavar="N")

    for name in ("status", "retrieve"):
        p = sub.add_parser(name)
        p.add_argument("batch_id", nargs="?", default=None)

    sub.add_parser("submit-retry", help="Resubmit '?' rows from the annotated CSV")

    p_rm = sub.add_parser("retrieve-merge", help="Retrieve retry batch and patch annotated CSV")
    p_rm.add_argument("batch_id", nargs="?", default=None)

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
                parser.error("No batch_id given and no batch_state_a_retry.json found.")
        cmd_retrieve_merge(bid)


if __name__ == "__main__":
    main()
