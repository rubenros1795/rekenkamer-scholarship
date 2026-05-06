# Kamervragen — Analysis Pipeline

**Project:** Automated annotation of parliamentary questions (*kamervragen*) addressed to the Algemene Rekenkamer (AR).

**Goal:** Classify each question by the type of cognitive contribution the questioner expects from the AR (schema A: questions), and classify each AR answer by what the AR actually delivers (schema B: answers). Both manual and LLM annotations are produced and compared.

---

## Table of contents

1. [Data pipeline](#1-data-pipeline)
2. [Annotation schema A — Questions](#2-annotation-schema-a--questions)
3. [Annotation schema B — Answers](#3-annotation-schema-b--answers)
4. [Scripts](#4-scripts)
5. [Inter-annotator agreement](#5-inter-annotator-agreement)
6. [Embedding-based classifiers](#6-embedding-based-classifiers)
7. [File index](#7-file-index)

---

## 1. Data pipeline

### 1.1 Source data

Raw documents are sourced from the Rekenkamer website and stored in:

```
_DATA/rekenkamer.xlsx        # raw document index
results/kamervragen-raw.csv  # full document text
```

Each row in the raw CSV represents one source document (a formal *lijst van vragen en antwoorden* published by the AR).

### 1.2 Structured extraction — `parse_kamerstukken.py`

Extracts individual question–answer pairs from raw documents using Claude (Haiku model via the Batches API). Output is `results/kamervragen-structured.csv`.

**How it works:**
- Reads `_DATA/rekenkamer.xlsx` for the document list and `results/kamervragen-raw.csv` for full text
- Submits each document to Claude with a JSON schema requiring structured Q&A extraction
- Claude extracts: document metadata, question text, answer text, and referential links (`answer_refers_to`)
- Results are merged back and saved as a flat TSV

**CLI:**
```bash
uv run python "Analysis - Kamervragen/parse_kamerstukken.py" submit [--limit N]
uv run python "Analysis - Kamervragen/parse_kamerstukken.py" status
uv run python "Analysis - Kamervragen/parse_kamerstukken.py" retrieve
uv run python "Analysis - Kamervragen/parse_kamerstukken.py" run [--limit N]
```

**Output columns — `kamervragen-structured.csv`:**

| Column | Description |
|---|---|
| `batch_idx` | Source document index |
| `src_id` | Unique document identifier |
| `src_title` | Document title |
| `src_url` | URL to source document |
| `src_published_at` | Publication date |
| `dossier_nummer` | Parliamentary dossier number |
| `rapport_titel` | AR report title |
| `datum` | Date of the Q&A document |
| `vergaderjaar` | Parliamentary year (e.g. `2013-2014`) |
| `vraag_nr` | Question number within document |
| `vraag` | Full question text |
| `antwoord` | Full answer text (empty if referential) |
| `antwoord_zie_nr` | If referential, the question number it refers to |
| `verwijzingen` | External document references mentioned in the answer |

**Join key:** `src_id` + `vraag_nr` uniquely identifies each question–answer pair.

---

## 2. Annotation schema A — Questions

**File:** `annotation-guidelines.md` (current version: **v0.7**)

### 2.1 Theoretical basis

The schema describes which *cognitive contribution* the questioner expects from the AR. Labels are ordered along a single dimension from passive to active:

```
FEIT → CAU → OOR → ADV
```

The further right, the more the AR is expected to bring its own knowledge, reasoning, or judgment — and the less the answer can simply be cited from the report.

### 2.2 Labels

| Label | Name | Core question | Key signals |
|---|---|---|---|
| **FEIT** | Feitenvraag | What does the report say? | hoeveel · welke · wanneer · bedragen · kunt u uitsplitsen · verwijzing naar tabel |
| **CAU** | Causaliteitsvraag | Why / what is the cause? | waarom · wat is de oorzaak van · wat veroorzaakt · welke belemmeringen · hoe heeft dit kunnen gebeuren |
| **OOR** | Oordeelsvraag | How do you assess this? | hoe beoordeelt u · in hoeverre voldoende · wat betekent · wat zijn de gevolgen als · ligt het in de lijn der verwachting |
| **ADV** | Advisvraag | What should be done? | welke aanbevelingen · wat moet er worden gedaan · hoe kan X worden verbeterd · wat adviseert u |

**OOR has four variants** (all get the same label):
- *Evaluatief* — "hoe beoordeelt u / is er voldoende"
- *Interpretatief* — "wat betekent / hoe verhoudt X zich tot Y"
- *Hypothetisch* — "wat zijn de gevolgen als / zou X leiden tot"
- *Prognostisch* — "ligt het in de lijn der verwachting / wanneer verwacht u"

### 2.3 Key decision rules

**FEIT vs. CAU**
- Explanation literally in the report → **FEIT**
- AR must construct the explanation → **CAU**
- "Waaruit bestaat X?" → FEIT (describe content)
- "Waarom is X zo?" → CAU (explain cause)
- "Heeft de AR zicht op X?" → FEIT (scope question)

**CAU vs. OOR**
- Seeks a mechanism between two concrete variables → **CAU**
- Normative judgment, meaning interpretation, or estimate → **OOR**
- "Welke gevolgen heeft X op Y?" → CAU · "Wat zijn de gevolgen als X niet gebeurt?" → OOR
- Contains a normative word (voldoende, terecht) or meaning question (wat betekent) → **OOR**

**OOR vs. ADV**
- Evaluates or interprets what is/was/might be → **OOR**
- Seeks an intervention, improvement, or preventive measure → **ADV**
- "In hoeverre kan voorkomen worden dat X?" → **ADV** (despite "in hoeverre")

**Multiple sub-questions:** label the most substantive one (usually the last). Tie-break: FEIT < CAU < OOR < ADV.

**VEC note:** Questions directed implicitly at the minister (rather than the AR) are a *political* dimension orthogonal to the cognitive dimension. They are labeled by what they ask the AR to do (usually OOR), not by who is implicitly addressed.

### 2.4 Schema history

| Version | Key changes |
|---|---|
| v0.1 | INF / VEC / LEG / DEF / MET / POL / AGN + modifiers |
| v0.2 | Merged DEF/MET into modifiers; split INF → INF-CIJ / INF-OOR |
| v0.3 | Added CAU as primary label; LEG renamed; 6 modifiers (+CAU/+COMP/+GAP/+SCOPE/+POL/+FUT) |
| v0.4 | Eliminated all modifiers; 5 clean labels: FEIT/CAU/OOR/ADV/AGN; VEC outside schema |
| **v0.5** | Added decision rules from annotation practice: "waaruit" vs "waarom", "in hoeverre kan voorkomen worden" → ADV, OOR four variants documented |
| **v0.6** | FEIT/OOR boundary sharpened (comparative bijzin rule); ADV-rem added; implicit OOR patterns added to schema B (v0.1→v0.2) |
| **v0.7** | AGN removed — redirected to ADV (research requests) or OOR (agenda/planning queries); schema B (v0.3) adds CAU for constructed causal explanations; FEIT/OOR norm-comparison rule added |

---

## 3. Annotation schema B — Answers

**File:** `annotation-guidelines.md` (section "Annotatieschema B", version **v0.3**)

### 3.1 Labels

Labels describe what the AR *primarily delivers* in its answer. Ordered from passive to active:

```
FEIT → CAU → OOR → ADV → DEFL
```

| Label | Name | Core signal |
|---|---|---|
| **FEIT** | AR rapporteert of beschrijft | "wij hebben vastgesteld dat" · "uit ons onderzoek blijkt" · verwijzing naar rapportpagina |
| **CAU** | AR construeert causale verklaring | "de oorzaak ligt in de combinatie van..." · "dit verklaren wij door..." · mechanisme met meerdere schakels |
| **OOR** | AR neemt een standpunt in | "wij vinden dat" · "wij oordelen dat" · "naar ons oordeel" · interpretatieve uitspraken |
| **ADV** | AR geeft een aanbeveling | "wij bevelen aan" · "de minister dient" · concrete stappen die ondernomen moeten worden |
| **DEFL** | AR beantwoordt de vraag niet | "dit hebben wij niet onderzocht" · "voor een antwoord verwijzen wij u naar de minister" |

### 3.2 Priority rule (mixed answers)

**ADV > OOR > CAU > FEIT > DEFL**

- Answer contains both judgment and recommendation → **ADV**
- Answer contains both causal explanation and judgment → **OOR**
- Answer contains both cited finding and constructed explanation → **CAU**
- Partial content + referral → **FEIT/CAU/OOR** (content wins over deflection)
- No substantive content → **DEFL**

---

## 4. Scripts

### 4.1 LLM batch annotation — questions: `llm_annotate_questions.py`

Annotates all questions in `kamervragen-structured.csv` with schema A labels using Claude via the Batches API (50% cost reduction vs. real-time).

**Model:** `claude-opus-4-6`
**Output:** `results/kamervragen_annotated_q_v0.7.csv`
**State file:** `batch_state.json`

**CLI:**
```bash
uv run python "Analysis - Kamervragen/llm_annotate_questions.py" submit [--limit N]
uv run python "Analysis - Kamervragen/llm_annotate_questions.py" status  [BATCH_ID]
uv run python "Analysis - Kamervragen/llm_annotate_questions.py" retrieve [BATCH_ID]
uv run python "Analysis - Kamervragen/llm_annotate_questions.py" run     [--limit N]
```

**Output columns added to the CSV:**

| Column | Description |
|---|---|
| `llm_label` | Primary label: FEIT / CAU / OOR / ADV / ? |
| `llm_confidence` | Model certainty: H / M / L |
| `llm_reasoning` | 1–2 sentence justification in Dutch |

**Notes:**
- System prompt contains the full v0.7 schema with all decision rules
- Prompt caching (`cache_control: ephemeral`) is applied to the system prompt
- Structured output enforced via `output_config` / `json_schema`

---

### 4.2 LLM batch annotation — answers: `llm_annotate_answers.py`

Annotates all AR answers in `kamervragen-structured.csv` with schema B labels using Claude via the Batches API.

**Model:** `claude-opus-4-6`
**Output:** `results/kamervragen_annotated_a_v0.3.csv`
**State file:** `batch_state_answers.json`
**Min answer length:** 30 characters (shorter answers are skipped)

**CLI:**
```bash
uv run python "Analysis - Kamervragen/llm_annotate_answers.py" submit [--limit N]
uv run python "Analysis - Kamervragen/llm_annotate_answers.py" status  [BATCH_ID]
uv run python "Analysis - Kamervragen/llm_annotate_answers.py" retrieve [BATCH_ID]
uv run python "Analysis - Kamervragen/llm_annotate_answers.py" run     [--limit N]
```

**Output columns added to the CSV:**

| Column | Description |
|---|---|
| `llm_label` | Primary label: FEIT / CAU / OOR / ADV / DEFL |
| `llm_confidence` | Model certainty: H / M / L |
| `llm_reasoning` | 1–2 sentence justification in Dutch |

**User content format:** Each request includes rapport title, the original question (as context), and the AR answer.

---

### 4.3 Manual annotation — `annotate_ui.py`

Streamlit app for manual annotation of questions and answers.

**Launch:**
```bash
cd /home/rb/Documents/Code
uv run streamlit run "Rekenkamer/Analysis - Kamervragen/annotate_ui.py"
```

**Modes** (selected from the sidebar):

| Mode | Schema | Source data | State file | Output |
|---|---|---|---|---|
| Vragen | A v0.7 | `kamervragen-structured.csv` | `annotation_state_q_v07.json` | `kamervragen_manual_q_v0.7.csv` |
| Antwoorden | B v0.3 | `kamervragen-structured.csv` (antwoord ≥ 30 chars) | `annotation_state_answers_manual.json` | `kamervragen_manual_a_v0.3.csv` |

**UI features:**

| Feature | Details |
|---|---|
| Save & Next | Available in the nav bar (top) and below the form |
| Label buttons | One active at a time, highlighted; compact hint caption when selected |
| Confidence radio | H / M / L (horizontal) |
| Notes field | Free text for borderline cases |
| Navigation | ← Previous · Next → · Skip to unannotated · Jump by number |
| Progress bar | N/total annotated at all times |
| Shuffle | Randomise order (preserves annotations) |
| Auto-persist | State saved on every "Save" — survives restarts |
| Export | "Exporteer CSV" writes full data + annotation columns |
| Sidebar | Full schema reference with decision rules |

---

### 4.4 Inter-annotator agreement — `interrater_agreement.py`

Computes agreement metrics between manual and LLM annotations for questions or answers.

**Usage:**
```bash
# Questions (current schema)
uv run python "Analysis - Kamervragen/interrater_agreement.py" questions

# Questions with old→new label mapping (for old LLM annotations)
uv run python "Analysis - Kamervragen/interrater_agreement.py" questions --map-old

# Answers
uv run python "Analysis - Kamervragen/interrater_agreement.py" answers

# Custom files
uv run python "Analysis - Kamervragen/interrater_agreement.py" questions \
    --llm results/my_llm.csv --llm-col llm_label
```

**Join strategy:** items are matched on `src_id` + `vraag_nr`. Falls back to positional join if these columns are absent.

**Metrics reported:**

| Metric | Description |
|---|---|
| Percent agreement | Raw match rate |
| Cohen's κ | Chance-corrected nominal agreement |
| Weighted κ (linear) | Ordinal-aware; respects label order (FEIT < CAU < OOR < ADV) |

**Output also includes:** confusion matrix, per-label breakdown (N manual, N LLM, P(agree\|label)), and a sample of disagreements with join keys.

**Old→new label mapping** (`--map-old`):

| Old label | New label | Notes |
|---|---|---|
| INF-CIJ | FEIT | Direct mapping |
| INF-OOR | OOR | Ambiguous; could be FEIT/CAU/OOR |
| INF | FEIT | — |
| LEG | ADV | — |
| POL | OOR | Closest cognitive equivalent |
| VEC | OOR | Political dimension mapped to cognitive equivalent |
| AGN | AGN | Unchanged |

---

### 4.5 Other scripts

| Script | Purpose |
|---|---|
| `extract_metadata.py` | Computes document-level linguistic metadata (word counts, TTR, sentence length, nominalisations, etc.) → `results/kamervragen-metadata.csv` |
| `backfill_missing.py` | Fills gaps in structured CSV using a second extraction pass |
| `annotate_b_app.py` | Legacy annotation app using schema B (principal-agent framing: AR→FACT / AR→JUDGE / GOV→ACCOUNT / GOV→ACT / AR→EXPAND) — kept for reference |

---

## 5. Inter-annotator agreement

### 5.1 Current status

| Comparison | n | % agree | Cohen's κ | Weighted κ |
|---|---|---|---|---|
| Q: manual v0.6 vs LLM v0.6 | 117 | 70.1% | 0.598 | 0.601 |
| A: manual v0.2 vs LLM v0.2 | 79 | 65.8% | 0.525 | 0.615 |

Main disagreement patterns after systematic analysis (see `reports/analysis-logbook.md`):
- **Q:** LLM over-assigns OOR (pulling from FEIT and CAU); LLM over-triggers ADV on prescriptive sub-clauses
- **A:** LLM under-assigns OOR (misses implicit OOR via norm-comparison and hedged claims); LLM misclassifies explained deflections as FEIT

These patterns informed the v0.6→v0.7 and v0.2→v0.3 guideline revisions. Agreement figures for the new schema (v0.7/v0.3) will be available after the next annotation run.

### 5.2 Interpretation benchmarks

| κ range | Interpretation |
|---|---|
| < 0.20 | Slight agreement |
| 0.21–0.40 | Fair agreement |
| 0.41–0.60 | Moderate agreement |
| 0.61–0.80 | Substantial agreement |
| > 0.80 | Almost perfect agreement |

Target for production annotation: **κ ≥ 0.65**.

### 5.3 Known ambiguities

| Boundary | Issue |
|---|---|
| FEIT / CAU | When a causal explanation is quoted verbatim from the report, it is FEIT. Annotators sometimes code the causal framing of the *question* rather than the retrievability of the answer. |
| CAU / OOR | Questions asking for a mechanism ("welke gevolgen heeft X op Y") vs. a normative assessment ("in hoeverre is X voldoende"). The signal words overlap for some phrasings. |
| OOR / ADV | "In hoeverre kan voorkomen worden dat" looks like OOR but is ADV (seeks a preventive measure). Decision rule added in v0.5. |

---

## 6. Embedding-based classifiers

### 6.1 Overview

`train_classifier.py` trains a lightweight classifier (logistic regression or LinearSVC) on top of frozen **multilingual-e5-large** embeddings. Two classifiers can be trained independently: one for questions (schema A) and one for answers (schema B). The workflow is:

1. Load labelled data from the chosen gold-truth source
2. Encode texts with the local e5-large model (embeddings cached to `.embed_cache/`)
3. Evaluate with stratified 5-fold cross-validation
4. Save the final model (trained on all data) to `models/`

Because the encoder is frozen, this is fast: re-running with a different classifier or gold source only requires reloading the cached embeddings.

### 6.2 Gold-truth sources

| Option | Flag | Description | Approx. n (questions) |
|---|---|---|---|
| LLM high-confidence | `--gold llm-high` | LLM annotations with `llm_confidence = H` only *(default)* | ~1 100 |
| LLM all | `--gold llm-all` | All LLM annotations including M/L confidence | ~1 780 |
| Manual | `--gold manual` | Human annotations only | ~115 |

Manual annotations using old-schema labels (VEC → OOR, INF → FEIT) are remapped automatically.

### 6.3 Input text construction

| Schema | Text passed to encoder |
|---|---|
| Questions | `"query: {rapport_titel} | {vraag}"` |
| Answers | `"passage: {vraag} | {antwoord}"` |

The e5 prefix (`query:` / `passage:`) follows the multilingual-e5 convention. For answers, the original question is prepended as context because answer type is often ambiguous without it.

### 6.4 Usage

```bash
# Questions, high-confidence LLM gold (default)
uv run python "Analysis - Kamervragen/train_classifier.py" questions

# Answers, manual gold truth
uv run python "Analysis - Kamervragen/train_classifier.py" answers --gold manual

# Both schemas, SVM, all LLM annotations
uv run python "Analysis - Kamervragen/train_classifier.py" questions answers \
    --gold llm-all --classifier svm

# Force re-embed (clear cache)
uv run python "Analysis - Kamervragen/train_classifier.py" questions --no-cache

# Custom output dir, more CV folds
uv run python "Analysis - Kamervragen/train_classifier.py" questions \
    --output-dir results/models --cv-folds 10
```

### 6.5 Output files

For each trained model a triple of files is written to `models/` (or `--output-dir`):

| File | Contents |
|---|---|
| `{schema}_{gold}_{classifier}_clf.joblib` | Fitted sklearn classifier |
| `{schema}_{gold}_{classifier}_le.joblib` | LabelEncoder (int ↔ label name) |
| `{schema}_{gold}_{classifier}_metrics.json` | CV macro-F1, per-class F1, confusion matrix |

Example: `questions_llm-high_logreg_clf.joblib`

### 6.6 Using a saved model in code

```python
from train_classifier import predict

# Predict labels for new questions
labels = predict(
    texts=["Hoeveel instellingen voldeden niet aan de eisen?"],
    schema="questions",
    gold="llm-high",
    classifier="logreg",
)
# → ["FEIT"]

# Predict answer labels (pass raw answer text; question prefix handled internally)
labels = predict(
    texts=["Wij bevelen aan dat de minister vóór 2026 een actieplan opstelt."],
    schema="answers",
)
# → ["ADV"]
```

Or directly:

```python
import joblib, numpy as np
from train_classifier import embed

X = embed(["query: Titel | Vraag tekst"], use_cache=False)
clf = joblib.load("models/questions_llm-high_logreg_clf.joblib")
le  = joblib.load("models/questions_llm-high_logreg_le.joblib")
print(le.inverse_transform(clf.predict(X)))
```

### 6.7 Embedding cache

Embeddings are cached in `.embed_cache/` as `.npy` files keyed by a SHA-256 hash of the model path + all input texts. The cache is invalidated automatically whenever the texts or model change. Delete `.embed_cache/` or use `--no-cache` to force re-encoding.

---

## 7. File index

### Scripts (`Analysis - Kamervragen/`)

| File | Schema | Purpose |
|---|---|---|
| `parse_kamerstukken.py` | — | Extract Q&A pairs from raw documents |
| `llm_annotate_questions.py` | A v0.7 | LLM batch annotation of questions |
| `llm_annotate_answers.py` | B v0.3 | LLM batch annotation of answers |
| `annotate_ui.py` | A v0.7 / B v0.3 | Unified Streamlit manual annotation (Vragen / Antwoorden) |
| `interrater_agreement.py` | A / B | Inter-annotator agreement metrics |
| `train_classifier.py` | A / B | Train embedding-based classifiers |
| `extract_metadata.py` | — | Document-level linguistic metadata |
| `backfill_missing.py` | — | Fill gaps in structured CSV |
| `annotate_b_app.py` | B legacy | Legacy app (principal-agent schema) |

**Superseded** (preserved for reference):

| File | Replaced by |
|---|---|
| `annotation_app_questions.py` | `annotate_ui.py` (Vragen mode) |
| `annotation_app_answers.py` | `annotate_ui.py` (Antwoorden mode) |

### State files (`Analysis - Kamervragen/`)

| File | Created by | Purpose |
|---|---|---|
| `batch_state.json` | `parse_kamerstukken.py` | Batch ID for structured extraction |
| `batch_state_answers.json` | `llm_annotate_answers.py` | Batch ID for answer annotation |
| `annotation_state_q_v07.json` | `annotate_ui.py` (Vragen) | Manual annotation progress — questions |
| `annotation_state_answers_manual.json` | `annotate_ui.py` (Antwoorden) | Manual annotation progress — answers |
| `batch_meta.json` | `parse_kamerstukken.py` | Batch metadata |

### Results (`results/`)

| File | Created by | Contents |
|---|---|---|
| `kamervragen-structured.csv` | `parse_kamerstukken.py` | All extracted Q&A pairs (tab-separated) |
| `kamervragen-raw.csv` | upstream | Full document text |
| `kamervragen_annotated_q_v0.6.csv` | `llm_annotate_questions.py` (v0.6 run) | LLM question annotations — schema A v0.6 (FEIT/CAU/OOR/ADV/AGN) |
| `kamervragen_annotated_q_v0.7.csv` | `llm_annotate_questions.py` (v0.7 run) | LLM question annotations — schema A v0.7 (FEIT/CAU/OOR/ADV) |
| `kamervragen_manual_q_v0.6.csv` | `annotate_ui.py` (v0.6) | Manual question annotations — schema A v0.6 |
| `kamervragen_manual_q_v0.7.csv` | `annotate_ui.py` (v0.7) | Manual question annotations — schema A v0.7 |
| `kamervragen_annotated_a_v0.2.csv` | `llm_annotate_answers.py` (v0.2 run) | LLM answer annotations — schema B v0.2 (FEIT/OOR/ADV/DEFL) |
| `kamervragen_annotated_a_v0.3.csv` | `llm_annotate_answers.py` (v0.3 run) | LLM answer annotations — schema B v0.3 (FEIT/CAU/OOR/ADV/DEFL) |
| `kamervragen_manual_a_v0.2.csv` | `annotate_ui.py` (v0.2) | Manual answer annotations — schema B v0.2 |
| `kamervragen_manual_a_v0.3.csv` | `annotate_ui.py` (v0.3) | Manual answer annotations — schema B v0.3 |
| `kamervragen-metadata.csv` | `extract_metadata.py` | Document-level metadata |

### Guidelines and reports (`Analysis - Kamervragen/`)

| File | Contents |
|---|---|
| `annotation-guidelines.md` | Full codebook: schema A (v0.7) + schema B (v0.3) with examples and decision rules |
| `annotation-report.md` | Pilot annotation findings (50 items, schema v0.1) with recommendations |
| `docs.md` | This file |

### Archive (`archive/`)

| File | Contents |
|---|---|
| `annotation-guidelines-v0.3.md` | Schema v0.3 (LEG, +CAB, +MET, 6 modifiers) |
| `annotation-guidelines-v0.5.md` | Schema A v0.5 + schema B v0.1 (superseded by v0.6/v0.2) |
| `annotation-guidelines-b.md` | Schema B (principal-agent framing: AR→FACT / AR→JUDGE / GOV→ACCOUNT / GOV→ACT / AR→EXPAND) |
| `annotate_b.py` | LLM annotation with schema B (principal-agent) |
| `annotations.tsv` | Early manual annotations (schema v0.1) |
| `split_questions.py` | Composite question splitting (no longer used in annotation pipeline) |

### Models (`Analysis - Kamervragen/models/`)

Created by `train_classifier.py`. One triple of files per trained model:

| File pattern | Contents |
|---|---|
| `{schema}_{gold}_{clf}_clf.joblib` | Fitted sklearn classifier |
| `{schema}_{gold}_{clf}_le.joblib` | LabelEncoder |
| `{schema}_{gold}_{clf}_metrics.json` | CV results (macro-F1, per-class F1, confusion matrix) |

### Embedding cache (`Analysis - Kamervragen/.embed_cache/`)

Auto-generated by `train_classifier.py`. SHA-256-keyed `.npy` files; safe to delete.
