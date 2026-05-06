#!/usr/bin/env python3
"""
annotate_ui.py — Streamlit annotation tool for kamervragen.

Two modes selectable from the sidebar:

  Vragen      Annotate questions with schema A (v0.7).
              Source  : kamervragen-structured.csv
              State   : annotation_state_q_v07.json
              Output  : kamervragen_manual_q_v0.7.csv

  Antwoorden  Annotate AR answers with schema B (v0.3)
              Source  : kamervragen-structured.csv
              State   : annotation_state_answers_manual.json
              Output  : kamervragen_manual_a_v0.3.csv

Run with:
    uv run streamlit run "Analysis - Kamervragen/annotate_ui.py"
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import streamlit as st

from paths import (
    STRUCTURED_CSV,
    MANUAL_Q_V07,
    MANUAL_A_V03,
    RAW_CSV,
)

# ── Directories & state files ─────────────────────────────────────────────────

_HERE = Path(__file__).parent

STATE_Q = _HERE / "annotation_state_q_v07.json"
STATE_A = _HERE / "annotation_state_answers_manual.json"

MIN_ANTWOORD_LEN = 30
RAW_PREVIEW_CHARS = 2500

# ── Schema A — Questions (v0.7) ───────────────────────────────────────────────

LABELS_A = ["FEIT", "CAU", "OOR", "ADV", "?"]

LABEL_INFO_A: dict[str, dict] = {
    "FEIT": {
        "badge": "🔵",
        "short": "Feitenvraag",
        "expects": "Het antwoord is direct raadpleegbaar in het rapport. AR hoeft niets te beredeneren.",
        "signals": (
            "hoeveel · welke · wanneer · wat zijn · wat staat er in het rapport · "
            "bedragen/percentages · overzicht/uitsplitsing · verwijzing naar tabel/bijlage · "
            "waaruit bestaat · heeft de AR zicht op · heeft de AR X onderzocht"
        ),
        "tip": "Stelregel: kan het antwoord letterlijk geciteerd worden? Ja → FEIT.",
    },
    "CAU": {
        "badge": "🟤",
        "short": "Causaliteitsvraag",
        "expects": "AR construeert een causale verklaring: oorzaak, reden, belemmering of gevolg. Staat niet kant-en-klaar in het rapport.",
        "signals": (
            "waarom · wat is de oorzaak van · wat veroorzaakt · wat zijn de belemmeringen voor · "
            "hoe heeft dit kunnen gebeuren · wat is de reden van · hoe komt het dat · welke gevolgen heeft X op Y"
        ),
        "tip": "CAU zoekt een mechanisme ('hoe is het zo gekomen'). Verschil met OOR: CAU verklaart, OOR oordeelt.",
    },
    "OOR": {
        "badge": "🟠",
        "short": "Oordeelsvraag",
        "expects": "AR neemt evaluatief, interpretatief, hypothetisch of prognostisch standpunt in.",
        "signals": (
            "Evaluatief: hoe beoordeelt u · in hoeverre acht u voldoende · ziet u verbetering\n"
            "Interpretatief: wat betekent · hoe verhoudt X zich tot Y · wat impliceert\n"
            "Hypothetisch: wat zijn de gevolgen als · zou X leiden tot\n"
            "Prognostisch: ligt het in de lijn der verwachting · wanneer verwacht u · gaat u een vervolg geven"
        ),
        "tip": "OOR heeft vier gedaanten (evaluatief/interpretatief/hypothetisch/prognostisch). Gemeenschappelijk: de AR neemt een standpunt in.",
    },
    "ADV": {
        "badge": "🟡",
        "short": "Advisvraag",
        "expects": "AR geeft prescriptief advies of aanbeveling voor de toekomst. Bevat prescriptief werkwoord gericht op toekomstige actie.",
        "signals": (
            "welke aanbevelingen doet u · welke lessen trekt u · wat moet er worden gedaan · "
            "op welke wijze kan X worden verbeterd · wat adviseert u · bent u bereid X te onderzoeken · "
            "in hoeverre kan voorkomen worden dat"
        ),
        "tip": "OOR evalueert wat is/was/zou kunnen zijn. ADV schrijft voor wat moet. Prescriptief werkwoord → ADV.",
    },
    "?": {
        "badge": "⚫",
        "short": "Oncodeerbaar",
        "expects": "Gebruik uitsluitend als de vraag door PDF-extractie-artefacten onleesbaar is.",
        "signals": "technisch artefact · afgekapt · onleesbaar",
        "tip": "Noteer in het notitieveld wat de vermoedelijke inhoud is.",
    },
}

# ── Schema B — Answers (v0.3) ─────────────────────────────────────────────────

LABELS_B = ["FEIT", "CAU", "OOR", "ADV", "DEFL", "?"]

LABEL_INFO_B: dict[str, dict] = {
    "FEIT": {
        "badge": "🔵",
        "short": "AR rapporteert of beschrijft",
        "expects": (
            "AR levert feitelijke informatie of rapporteert gedocumenteerde bevindingen. "
            "Geen normatief standpunt, geen aanbeveling. Causale woorden zijn toegestaan "
            "als de AR een bevinding citeert (niet construeert)."
        ),
        "signals": (
            "\"wij hebben vastgesteld dat\" · \"uit ons onderzoek blijkt\" · "
            "\"de situatie is\" · verwijzing naar rapportpagina · "
            "\"dit wordt veroorzaakt door X\" (als citaat van bevinding)"
        ),
        "tip": "Stelregel: meldt het antwoord een gedocumenteerde uitkomst zonder eigen oordeel? → FEIT.",
    },
    "CAU": {
        "badge": "🟤",
        "short": "AR construeert causale verklaring",
        "expects": (
            "AR legt uit waarom iets zo is door meerdere factoren samen te brengen of een "
            "mechanisme te reconstrueren — verder dan het citeren van een enkelvoudige bevinding."
        ),
        "signals": (
            "\"de oorzaak ligt in de combinatie van...\" · \"dit verklaren wij door...\" · "
            "\"het mechanisme werkt als volgt\" · \"wij schrijven dit toe aan...\" · "
            "uitgebreide toelichting op causaal verband met meerdere schakels"
        ),
        "tip": "Verschil met FEIT: FEIT meldt een bevinding; CAU elaboreert op het mechanisme.",
    },
    "OOR": {
        "badge": "🟠",
        "short": "AR neemt een standpunt in",
        "expects": (
            "AR evalueert, interpreteert of vormt een oordeel. AR gaat verder dan rapporteren "
            "of verklaren: ze neemt positie in over kwaliteit, toereikendheid of betekenis."
        ),
        "signals": (
            "\"wij vinden dat\" · \"wij oordelen dat\" · \"naar ons oordeel\" · "
            "\"wij constateren dat X onvoldoende is\" · interpretatieve uitspraken · "
            "norm-vergelijking (\"bleef onder de tolerantiegrens\") · "
            "\"heeft bijgedragen aan\" · \"is niet goed van de grond gekomen\""
        ),
        "tip": "OOR vereist dat de AR een standpunt inneemt — niet louter beschrijven of aanbevelen.",
    },
    "ADV": {
        "badge": "🟡",
        "short": "AR geeft een aanbeveling",
        "expects": "AR schrijft voor wat er moet of zou moeten. Prescriptief advies gericht op toekomstige actie.",
        "signals": (
            "\"wij bevelen aan\" · \"het is van belang dat\" · \"de minister dient\" · "
            "\"wij adviseren\" · concrete stappen die ondernomen moeten worden"
        ),
        "tip": "ADV is prescriptief en toekomstgericht. OOR evalueert, ADV schrijft voor.",
    },
    "DEFL": {
        "badge": "🔴",
        "short": "AR beantwoordt de vraag niet substantieel",
        "expects": (
            "AR verwijst naar de minister, geeft aan dat het buiten scope valt, of heeft het "
            "niet onderzocht. Uitleg van een deflectie is nog steeds DEFL."
        ),
        "signals": (
            "\"dit hebben wij niet onderzocht\" · \"voor een antwoord verwijzen wij u naar de minister\" · "
            "\"dit valt buiten de reikwijdte\" · \"zie het antwoord bij vraag X\""
        ),
        "tip": "Gebruik DEFL alleen als er géén substantiële inhoud is. Gedeeltelijke inhoud → FEIT/OOR/ADV.",
    },
    "?": {
        "badge": "⚫",
        "short": "Oncodeerbaar",
        "expects": "Gebruik uitsluitend als het antwoord door extractie-artefacten onleesbaar is.",
        "signals": "technisch artefact · afgekapt · ontbreekt · onleesbaar",
        "tip": "Noteer in het notitieveld wat het probleem is.",
    },
}

CONFIDENCE: dict[str, str] = {
    "H": "Hoog — één label past duidelijk",
    "M": "Middel — één plausibel alternatief",
    "L": "Laag — twee of meer alternatieven",
}

# ── Mode config ───────────────────────────────────────────────────────────────

MODES = ["Vragen", "Antwoorden"]

MODE_META = {
    "Vragen": {
        "schema_label": "Schema A v0.7",
        "unit": "vraag",
        "units": "vragen",
        "labels": LABELS_A,
        "label_info": LABEL_INFO_A,
        "state_file": STATE_Q,
        "output_csv": MANUAL_Q_V07,
        "icon": "📋",
    },
    "Antwoorden": {
        "schema_label": "Schema B v0.3",
        "unit": "antwoord",
        "units": "antwoorden",
        "labels": LABELS_B,
        "label_info": LABEL_INFO_B,
        "state_file": STATE_A,
        "output_csv": MANUAL_A_V03,
        "icon": "💬",
    },
}

# ── State helpers ─────────────────────────────────────────────────────────────


def _load_state(state_file: Path) -> dict:
    if state_file.exists():
        state = json.loads(state_file.read_text())
        # Migrate old list format [{...}] → plain dict {...}
        anns = state.get("annotations", {})
        for k, v in list(anns.items()):
            if isinstance(v, list):
                anns[k] = v[0] if (v and isinstance(v[0], dict)) else {}
        return state
    return {"annotations": {}}


def _save_state(state: dict, state_file: Path) -> None:
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False))


# ── Progress helpers ──────────────────────────────────────────────────────────


def _q_is_annotated(val) -> bool:
    return isinstance(val, dict) and bool(val.get("label"))


def _a_is_annotated(val) -> bool:
    return isinstance(val, dict) and bool(val.get("label"))


# ── Data loaders ──────────────────────────────────────────────────────────────


@st.cache_data
def _load_questions() -> pd.DataFrame:
    return pd.read_csv(STRUCTURED_CSV, sep="\t").reset_index(drop=True)


@st.cache_data
def _load_answers() -> pd.DataFrame:
    df = pd.read_csv(STRUCTURED_CSV, sep="\t")
    mask = df["antwoord"].notna() & (df["antwoord"].str.len() >= MIN_ANTWOORD_LEN)
    return df[mask].reset_index(drop=True)


@st.cache_data
def _load_raw() -> dict[str, str]:
    if not RAW_CSV.exists():
        return {}
    raw = pd.read_csv(RAW_CSV, sep="\t", usecols=["_id", "doc_content"])
    return dict(zip(raw["_id"], raw["doc_content"].fillna("")))


# ── Export ────────────────────────────────────────────────────────────────────


def _export_q_csv(df: pd.DataFrame, annotations: dict, output_csv: Path) -> None:
    out = df.copy()
    for field in ("label", "confidence", "notes"):
        out[field] = [annotations.get(str(i), {}).get(field, "") for i in range(len(df))]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, sep="\t", index=False)


def _export_a_csv(df: pd.DataFrame, annotations: dict, output_csv: Path) -> None:
    out = df.copy()
    for field in ("label", "confidence", "notes"):
        out[field] = [annotations.get(str(i), {}).get(field, "") for i in range(len(df))]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, sep="\t", index=False)


# ── Sidebar ───────────────────────────────────────────────────────────────────


def _render_sidebar(mode: str) -> None:
    meta = MODE_META[mode]
    label_info = meta["label_info"]

    with st.sidebar:
        st.title(f"{meta['icon']} {meta['schema_label']}")

        st.subheader("Primaire labels")
        for label, info in label_info.items():
            if label == "?":
                continue
            with st.expander(f"{info['badge']} **{label}** — {info['short']}"):
                st.markdown(f"**Verwacht:** {info['expects']}")
                st.markdown(f"**Signaalwoorden:** {info['signals']}")
                st.info(f"💡 {info['tip']}")

        st.divider()
        st.subheader("Beslisregels")

        if mode == "Vragen":
            st.markdown("""
**FEIT vs. CAU**
Verklaring letterlijk in rapport? → **FEIT**
AR construeert verklaring zelf? → **CAU**
"Heeft de AR X onderzocht?" → **FEIT** (scopevraag)

**CAU vs. OOR**
Zoekt oorzaak/mechanisme? → **CAU**
Normatief oordeel of betekenisvraag? → **OOR**

**OOR vs. ADV**
Evalueert/prognosticeert? → **OOR**
Schrijft voor wat er moet / nieuw onderzoek? → **ADV**
"Gaat u een vervolg geven?" → **OOR**
"Bent u bereid X te onderzoeken?" → **ADV**

**Prioriteit bij gemengde vragen**
ADV > OOR > CAU > FEIT
""")
        else:
            st.markdown("""
**Prioriteit bij gemengde antwoorden**
ADV > OOR > CAU > FEIT > DEFL

**FEIT vs. CAU**
AR citeert bevinding → **FEIT**
AR elaboreert mechanisme → **CAU**

**FEIT/OOR grensregel**
Vergelijkt met norm/verwachting → **OOR**
Meldt alleen een waarde → **FEIT**

**DEFL**
Gedeeltelijke inhoud → niet DEFL.
""")


# ── Compact annotation form ───────────────────────────────────────────────────


def _render_annotation_form(
    labels: list[str],
    label_info: dict,
    slot_key: str,
    ann: dict,
) -> tuple[str | None, str, str]:
    """
    Compact annotation form: label buttons → hint caption → conf (horizontal) + notes.
    slot_key must be unique per annotation slot, e.g. "42_0", "42_1".
    Returns (selected_label, confidence, notes).
    """
    pending_key = f"pending_label_{slot_key}"
    if pending_key not in st.session_state:
        st.session_state[pending_key] = ann.get("label")

    # Label buttons
    label_cols = st.columns(len(labels))
    for i, lbl in enumerate(labels):
        info = label_info[lbl]
        is_active = st.session_state[pending_key] == lbl
        if label_cols[i].button(
            f"{info['badge']} {lbl}",
            key=f"lbl_{lbl}_{slot_key}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
            help=info["short"],
        ):
            st.session_state[pending_key] = lbl
            st.rerun()

    selected_label = st.session_state[pending_key]
    if selected_label and selected_label in label_info:
        info = label_info[selected_label]
        st.caption(f"{info['badge']} **{selected_label}** — {info['short']}  ·  💡 {info['tip']}")

    # Confidence (horizontal) + notes side by side
    col_conf, col_notes = st.columns([1, 2])
    with col_conf:
        current_conf = ann.get("confidence", "M")
        confidence = st.radio(
            "Zekerheid",
            options=list(CONFIDENCE),
            index=list(CONFIDENCE).index(current_conf) if current_conf in CONFIDENCE else 1,
            format_func=lambda x: x,
            horizontal=True,
            key=f"conf_{slot_key}",
            help="  \n".join(f"**{k}** — {v}" for k, v in CONFIDENCE.items()),
        )
    with col_notes:
        notes = st.text_area(
            "Notities",
            value=ann.get("notes", ""),
            height=68,
            key=f"notes_{slot_key}",
            placeholder="Toelichting bij twijfel of grensgevallen…",
        )

    return selected_label, confidence, notes


# ── Save & advance logic ──────────────────────────────────────────────────────


def _advance(annotations: dict, is_annotated, order: list, idx: int, n_total: int, idx_key: str) -> None:
    """Navigate to the next unannotated item and rerun."""
    next_pos = idx + 1
    for pos in range(idx + 1, n_total):
        if not is_annotated(annotations.get(str(order[pos]))):
            next_pos = pos
            break
    st.session_state[idx_key] = min(next_pos, n_total - 1)
    st.rerun()



def _save_a(
    row_idx: int,
    annotations: dict,
    state: dict,
    state_file: Path,
    order: list,
    idx: int,
    n_total: int,
    idx_key: str,
) -> None:
    lbl   = st.session_state.get(f"pending_label_{row_idx}")
    conf  = st.session_state.get(f"conf_{row_idx}", "M")
    raw_notes = st.session_state.get(f"notes_{row_idx}", "")
    notes = raw_notes.strip() if isinstance(raw_notes, str) else ""

    if not lbl:
        st.session_state["_save_error"] = "❌ Selecteer eerst een label."
        return

    annotations[str(row_idx)] = {"label": lbl, "confidence": conf, "notes": notes}
    _save_state(state, state_file)
    st.session_state.pop(f"pending_label_{row_idx}", None)
    _advance(annotations, _a_is_annotated, order, idx, n_total, idx_key)


# ── Source metadata expander ──────────────────────────────────────────────────


def _render_source_meta(row: pd.Series) -> None:
    with st.expander("📎 Brondocument", expanded=False):
        c1, c2 = st.columns(2)
        c1.markdown(f"**Rapport:** {row.get('rapport_titel', '')}")
        c1.markdown(f"**Vergaderjaar:** {row.get('vergaderjaar', '')}")
        c2.markdown(f"**Datum:** {row.get('datum', '')}")
        c2.markdown(f"**Dossier:** {row.get('dossier_nummer', '')}")
        src_url = str(row.get("src_url", "") or "").strip()
        if src_url:
            st.markdown(f"[📄 Bekijk brondocument]({src_url})")


# ── Main app ──────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Kamervragen Annotator",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        mode = st.selectbox(
            "Annotatiemodus",
            MODES,
            format_func=lambda m: f"{MODE_META[m]['icon']} {m}  ({MODE_META[m]['schema_label']})",
            key="mode",
        )
        st.divider()

    _render_sidebar(mode)

    meta       = MODE_META[mode]
    state_file = meta["state_file"]
    output_csv = meta["output_csv"]
    labels     = meta["labels"]
    label_info = meta["label_info"]
    unit       = meta["unit"]
    units      = meta["units"]

    # ── Load data ─────────────────────────────────────────────────────────────
    df          = _load_questions() if mode == "Vragen" else _load_answers()
    raw_content = _load_raw()
    n_total     = len(df)

    # ── Session state init (mode-scoped) ──────────────────────────────────────
    state_key    = f"state_{mode}"
    order_key    = f"order_{mode}"
    shuffled_key = f"shuffled_{mode}"
    idx_key      = f"current_idx_{mode}"

    if state_key not in st.session_state:
        st.session_state[state_key] = _load_state(state_file)

    state:       dict = st.session_state[state_key]
    annotations: dict = state["annotations"]

    is_annotated = _q_is_annotated if mode == "Vragen" else _a_is_annotated

    if order_key not in st.session_state:
        st.session_state[order_key]    = list(range(n_total))
        st.session_state[shuffled_key] = False

    if idx_key not in st.session_state:
        for pos, row_i in enumerate(st.session_state[order_key]):
            if not is_annotated(annotations.get(str(row_i))):
                st.session_state[idx_key] = pos
                break
        else:
            st.session_state[idx_key] = 0

    order:   list[int] = st.session_state[order_key]
    idx:     int       = st.session_state[idx_key]
    row_idx: int       = order[idx]
    row                = df.iloc[row_idx]

    # ── Progress + export ─────────────────────────────────────────────────────
    n_annotated = sum(1 for i in range(n_total) if is_annotated(annotations.get(str(i))))
    progress    = n_annotated / n_total if n_total else 0

    col_prog, col_export = st.columns([5, 1])
    with col_prog:
        st.progress(progress, text=f"**{n_annotated}/{n_total}** {units} geannoteerd ({progress:.1%})")
    with col_export:
        if st.button("💾 Exporteer CSV", use_container_width=True):
            if mode == "Vragen":
                _export_q_csv(df, annotations, output_csv)
            else:
                _export_a_csv(df, annotations, output_csv)
            _save_state(state, state_file)
            st.success(f"✅ {output_csv.name}")

    # ── Navigation row + Save & Next button ───────────────────────────────────
    c_prev, c_next, c_unan, c_num, c_go, c_shuf, c_save = st.columns([1, 1, 1, 2, 1, 1, 2])

    with c_prev:
        if st.button("⬅ Vorige", disabled=(idx == 0), use_container_width=True):
            st.session_state[idx_key] = idx - 1
            st.rerun()
    with c_next:
        if st.button("Volgende ➡", disabled=(idx >= n_total - 1), use_container_width=True):
            st.session_state[idx_key] = idx + 1
            st.rerun()
    with c_unan:
        if st.button("→ Onbeoordeeld", use_container_width=True):
            for pos in range(idx + 1, n_total):
                if not is_annotated(annotations.get(str(order[pos]))):
                    st.session_state[idx_key] = pos
                    st.rerun()
                    break
            else:
                st.info(f"Alle {units} zijn geannoteerd.")
    with c_num:
        jump = st.number_input("#", min_value=1, max_value=n_total, value=idx + 1,
                               label_visibility="collapsed")
    with c_go:
        if st.button("Ga naar", use_container_width=True):
            st.session_state[idx_key] = int(jump) - 1
            st.rerun()
    with c_shuf:
        shuffled = st.session_state.get(shuffled_key, False)
        if st.button("🔀 Herstel" if shuffled else "🔀 Shuffle", use_container_width=True):
            if shuffled:
                st.session_state[order_key]    = list(range(n_total))
                st.session_state[shuffled_key] = False
            else:
                new_order = list(range(n_total))
                random.shuffle(new_order)
                st.session_state[order_key]    = new_order
                st.session_state[shuffled_key] = True
            st.session_state[idx_key] = 0
            st.rerun()
    with c_save:
        # Primary save button — reads current pending values from session state,
        # so it works even though the form widgets render below.
        if st.button("✅ Sla op & volgende", type="primary", use_container_width=True, key="save_nav"):
            _save_a(row_idx, annotations, state, state_file, order, idx, n_total, idx_key)

    # Show any deferred save error (set by save helpers when validation fails)
    if err := st.session_state.pop("_save_error", None):
        st.error(err)

    st.divider()

    # ── Item header ───────────────────────────────────────────────────────────
    st.subheader(f"{unit.capitalize()} {idx + 1} / {n_total}")

    # ── Vragen mode ───────────────────────────────────────────────────────────
    if mode == "Vragen":
        vraag_text = str(row.get("vraag", "") or "").strip()
        ann = annotations.get(str(row_idx), {})

        st.info(vraag_text)

        antwoord = str(row.get("antwoord", "") or "").strip()
        if antwoord:
            with st.expander("💬 Antwoord (ter referentie)", expanded=False):
                st.markdown(antwoord)

        _render_annotation_form(labels, label_info, str(row_idx), ann)

        if st.button("✅ Sla op & ga naar volgende", type="primary",
                     use_container_width=True, key="save_bottom"):
            _save_a(row_idx, annotations, state, state_file, order, idx, n_total, idx_key)

        if err := st.session_state.pop("_save_error", None):
            st.error(err)

    # ── Antwoorden mode ───────────────────────────────────────────────────────
    else:
        ann = annotations.get(str(row_idx), {})

        with st.expander("❓ Vraag (context)", expanded=False):
            st.markdown(str(row.get("vraag", "") or ""))

        st.info(str(row.get("antwoord", "") or ""))

        _render_annotation_form(labels, label_info, str(row_idx), ann)

        if st.button("✅ Sla op & ga naar volgende", type="primary",
                     use_container_width=True, key="save_bottom"):
            _save_a(row_idx, annotations, state, state_file, order, idx, n_total, idx_key)

        if err := st.session_state.pop("_save_error", None):
            st.error(err)

    # ── Metadata footer (rarely needed — keep collapsed at bottom) ────────────
    st.divider()
    _render_source_meta(row)

    doc_text = raw_content.get(str(row.get("src_id", "") or ""), "")
    if doc_text:
        preview = doc_text[:RAW_PREVIEW_CHARS] + (" …" if len(doc_text) > RAW_PREVIEW_CHARS else "")
        with st.expander("📄 Brondocument (preview)", expanded=False):
            st.text(preview)


if __name__ == "__main__":
    main()
