#!/usr/bin/env python3
"""
train_classifier.py — Train annotation classifiers using multilingual-e5-large embeddings.

Trains a logistic-regression (or SVM) classifier on top of frozen sentence embeddings
for either schema A (questions: FEIT/CAU/OOR/ADV/AGN) or schema B (answers:
FEIT/OOR/ADV/DEFL).

Gold-truth source can be:
  llm-high  — LLM annotations with confidence H (large, ~1100 questions)
  llm-all   — All LLM annotations including M/L confidence
  manual    — Human annotations only (~117 questions / ~107 answers)

Usage
-----
    # Train question classifier on high-confidence LLM annotations (default)
    uv run python "Analysis - Kamervragen/train_classifier.py" questions

    # Train answer classifier with manual gold truth
    uv run python "Analysis - Kamervragen/train_classifier.py" answers --gold manual

    # Both schemas, custom output dir, SVM
    uv run python "Analysis - Kamervragen/train_classifier.py" questions answers \\
        --gold llm-high --classifier svm --output-dir models/

    # Force re-embed (ignore cache)
    uv run python "Analysis - Kamervragen/train_classifier.py" questions --no-cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from paths import (
    ANNOTATED_A_CSV,
    ANNOTATED_Q_CSV,
    MANUAL_A_CSV,
    MANUAL_Q_CSV,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
MODEL_DIR = Path("/home/rb/Documents/Code/Rekenkamer/_MODELS/sentence-transformers-multilingual-e5-large")
DEFAULT_OUTPUT_DIR = _HERE / "models"
CACHE_DIR = _HERE / ".embed_cache"

# ── Label definitions ──────────────────────────────────────────────────────────

Q_LABELS = ["FEIT", "CAU", "OOR", "ADV"]            # schema A v0.7
A_LABELS = ["FEIT", "CAU", "OOR", "ADV", "DEFL"]   # schema B v0.3

# Old-schema question labels → current schema (v0.7)
OLD_Q_MAP: dict[str, str] = {
    "INF-CIJ": "FEIT",
    "INF-OOR": "OOR",
    "INF":     "FEIT",
    "LEG":     "ADV",
    "POL":     "OOR",
    "VEC":     "OOR",
    "AGN":     "ADV",   # former AGN → ADV (research request = prescriptive)
}

# ── Data loading ───────────────────────────────────────────────────────────────


def _load_llm(schema: str, confidence: str | None) -> pd.DataFrame:
    """Load LLM-annotated data, optionally filtered by confidence level."""
    path = ANNOTATED_Q_CSV if schema == "questions" else ANNOTATED_A_CSV
    df = pd.read_csv(path, sep="\t")
    df = df[df["llm_label"].notna() & ~df["llm_label"].isin(["?", "", "nan"])]
    if confidence == "H":
        df = df[df["llm_confidence"] == "H"]
    elif confidence in ("H", "M"):
        df = df[df["llm_confidence"].isin(["H", "M"])]
    df = df.rename(columns={"llm_label": "label"})
    return df


def _load_manual(schema: str, map_old: bool = True) -> pd.DataFrame:
    """Load manually annotated data. Filters out '?' and empty labels."""
    path = MANUAL_Q_CSV if schema == "questions" else MANUAL_A_CSV
    df = pd.read_csv(path, sep="\t")
    df = df[df["label"].notna() & ~df["label"].isin(["?", "", "nan"])]
    if map_old and schema == "questions":
        df["label"] = df["label"].map(lambda x: OLD_Q_MAP.get(x, x))
    return df


def load_data(schema: str, gold: str) -> pd.DataFrame:
    """
    Load and return a DataFrame with at minimum columns: text, label.

    Parameters
    ----------
    schema : "questions" or "answers"
    gold   : "llm-high", "llm-all", or "manual"
    """
    valid_labels = Q_LABELS if schema == "questions" else A_LABELS

    if gold == "llm-high":
        df = _load_llm(schema, confidence="H")
    elif gold == "llm-all":
        df = _load_llm(schema, confidence=None)
    elif gold == "manual":
        df = _load_manual(schema, map_old=True)
    else:
        raise ValueError(f"Unknown gold source: {gold!r}")

    # Keep only recognised labels
    df = df[df["label"].isin(valid_labels)].copy()

    # Build input text
    if schema == "questions":
        # Prefix with report title for context, then question
        df["text"] = (
            "query: "
            + df["rapport_titel"].fillna("").str.strip()
            + " | "
            + df["vraag"].fillna("").str.strip()
        )
    else:
        # For answers: include the original question as context
        df["text"] = (
            "passage: "
            + df["vraag"].fillna("").str.strip()
            + " | "
            + df["antwoord"].fillna("").str.strip()
        )

    df = df[["src_id", "vraag_nr", "text", "label"]].reset_index(drop=True)
    return df


# ── Embedding ──────────────────────────────────────────────────────────────────


def _cache_key(texts: list[str], model_path: str) -> str:
    h = hashlib.sha256()
    h.update(model_path.encode())
    for t in texts:
        h.update(t.encode())
    return h.hexdigest()[:16]


def embed(
    texts: list[str],
    model_path: Path = MODEL_DIR,
    cache_dir: Path = CACHE_DIR,
    use_cache: bool = True,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode texts with multilingual-e5-large. Results are cached to disk.
    The e5 convention (query: / passage: prefix) must already be applied in texts.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(texts, str(model_path))
    cache_file = cache_dir / f"{key}.npy"

    if use_cache and cache_file.exists():
        print(f"  Loading embeddings from cache ({cache_file.name})")
        return np.load(cache_file)

    print(f"  Encoding {len(texts)} texts with {model_path.name} …")
    model = SentenceTransformer(str(model_path))
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine-ready; good for LogReg
        convert_to_numpy=True,
    )
    if use_cache:
        np.save(cache_file, embeddings)
        print(f"  Cached to {cache_file}")
    return embeddings


# ── Classifier ─────────────────────────────────────────────────────────────────


def _make_classifier(kind: str) -> LogisticRegression | LinearSVC:
    if kind == "logreg":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=42,
        )
    elif kind == "svm":
        return LinearSVC(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown classifier: {kind!r}")


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    classifier: str = "logreg",
    n_splits: int = 5,
) -> tuple[dict, object, LabelEncoder]:
    """
    Stratified k-fold cross-validation + final model trained on all data.

    Returns
    -------
    metrics  : dict with macro-F1, per-class F1, confusion matrix, report
    model    : fitted classifier (trained on all data)
    le       : LabelEncoder mapping label names to ints
    """
    le = LabelEncoder()
    le.fit(labels)
    y_enc = le.transform(y)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = _make_classifier(classifier)

    print(f"\n  Running {n_splits}-fold stratified cross-validation …")
    y_pred = cross_val_predict(clf, X, y_enc, cv=cv)

    y_str   = le.inverse_transform(y_enc)
    yp_str  = le.inverse_transform(y_pred)

    report  = classification_report(y_str, yp_str, labels=labels, zero_division=0)
    macro_f1 = f1_score(y_enc, y_pred, average="macro", zero_division=0)
    per_class = dict(zip(labels, f1_score(y_enc, y_pred, average=None, labels=le.transform(labels), zero_division=0)))

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_enc, y_pred, labels=le.transform(labels))

    metrics = {
        "macro_f1":  float(macro_f1),
        "per_class": {k: float(v) for k, v in per_class.items()},
        "report":    report,
        "confusion_matrix": cm.tolist(),
        "labels":    labels,
        "n":         int(len(y)),
        "n_splits":  n_splits,
    }

    # Train final model on all data
    print("  Training final model on all data …")
    final_clf = _make_classifier(classifier)
    final_clf.fit(X, y_enc)

    return metrics, final_clf, le


# ── Save / load ────────────────────────────────────────────────────────────────


def save_model(
    schema: str,
    gold: str,
    classifier: str,
    model: object,
    le: LabelEncoder,
    metrics: dict,
    output_dir: Path,
) -> Path:
    """Save classifier, label encoder, and metrics to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{schema}_{gold}_{classifier}"

    clf_path     = output_dir / f"{stem}_clf.joblib"
    le_path      = output_dir / f"{stem}_le.joblib"
    metrics_path = output_dir / f"{stem}_metrics.json"

    joblib.dump(model, clf_path)
    joblib.dump(le,    le_path)

    # confusion_matrix is a list-of-lists — JSON serialisable
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in metrics.items() if k != "report"}, f, indent=2)

    print(f"\n  Saved classifier  → {clf_path}")
    print(f"  Saved label enc   → {le_path}")
    print(f"  Saved metrics     → {metrics_path}")
    return clf_path


def load_model(schema: str, gold: str, classifier: str, model_dir: Path):
    """Load a previously saved classifier and label encoder."""
    stem = f"{schema}_{gold}_{classifier}"
    clf = joblib.load(model_dir / f"{stem}_clf.joblib")
    le  = joblib.load(model_dir / f"{stem}_le.joblib")
    return clf, le


# ── Prediction helper ──────────────────────────────────────────────────────────


def predict(
    texts: list[str],
    schema: str,
    gold: str = "llm-high",
    classifier: str = "logreg",
    model_dir: Path = DEFAULT_OUTPUT_DIR,
    embed_model_path: Path = MODEL_DIR,
) -> list[str]:
    """
    Predict labels for a list of raw texts (without e5 prefix).
    The function adds the correct prefix automatically.

    Parameters
    ----------
    texts    : raw question or answer texts
    schema   : "questions" or "answers"
    """
    prefix = "query: " if schema == "questions" else "passage: "
    prefixed = [prefix + t for t in texts]

    X = embed(prefixed, model_path=embed_model_path)
    clf, le = load_model(schema, gold, classifier, model_dir)
    y_enc = clf.predict(X)
    return list(le.inverse_transform(y_enc))


# ── Main ───────────────────────────────────────────────────────────────────────


def run_schema(
    schema: str,
    gold: str,
    classifier: str,
    output_dir: Path,
    use_cache: bool,
    n_splits: int,
) -> None:
    valid_labels = Q_LABELS if schema == "questions" else A_LABELS
    print(f"\n{'='*60}")
    print(f"  Schema    : {schema}")
    print(f"  Gold      : {gold}")
    print(f"  Classifier: {classifier}")
    print(f"{'='*60}")

    # 1. Load data
    df = load_data(schema, gold)
    print(f"\n  Loaded {len(df)} labelled items")
    label_counts = df["label"].value_counts()
    print("  Label distribution:")
    for lbl, n in label_counts.items():
        print(f"    {lbl:8s}  {n:5d}")

    if len(df) < 20:
        print("  WARNING: very few samples — results may be unreliable")

    # 2. Embed
    X = embed(df["text"].tolist(), use_cache=use_cache)

    # 3. Cross-validate + train
    labels_present = [l for l in valid_labels if l in df["label"].values]
    metrics, model, le = train_and_evaluate(
        X, df["label"].values, labels_present, classifier=classifier, n_splits=n_splits
    )

    # 4. Print results
    print(f"\n  {'─'*50}")
    print(f"  CV macro-F1 : {metrics['macro_f1']:.3f}  (n={metrics['n']}, {n_splits}-fold)")
    print(f"\n  Per-label F1:")
    for lbl, f1 in metrics["per_class"].items():
        bar = "█" * int(f1 * 20)
        print(f"    {lbl:8s}  {f1:.3f}  {bar}")
    print(f"\n  Classification report (CV):\n")
    for line in metrics["report"].splitlines():
        print(f"    {line}")

    # Confusion matrix as text table
    labels_for_cm = metrics["labels"]
    cm = np.array(metrics["confusion_matrix"])
    col_w = max(len(l) for l in labels_for_cm) + 2
    print(f"\n  Confusion matrix (manual rows, predicted cols):")
    header = f"  {'':>{col_w}}" + "".join(f"{l:>{col_w}}" for l in labels_for_cm)
    print(header)
    for i, lbl in enumerate(labels_for_cm):
        row = f"  {lbl:>{col_w}}" + "".join(f"{cm[i,j]:>{col_w}}" for j in range(len(labels_for_cm)))
        print(row)

    # 5. Save
    save_model(schema, gold, classifier, model, le, metrics, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train embedding-based classifiers for question/answer annotation"
    )
    parser.add_argument(
        "schemas",
        nargs="+",
        choices=["questions", "answers"],
        help="Which schema(s) to train",
    )
    parser.add_argument(
        "--gold",
        choices=["llm-high", "llm-all", "manual"],
        default="llm-high",
        help=(
            "Gold-truth source: "
            "'llm-high' = LLM with confidence H (default), "
            "'llm-all' = all LLM annotations, "
            "'manual' = human annotations only"
        ),
    )
    parser.add_argument(
        "--classifier",
        choices=["logreg", "svm"],
        default="logreg",
        help="Classifier type: logreg (default) or svm",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for saved models (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help=f"Path to multilingual-e5-large model (default: {MODEL_DIR})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore embedding cache and re-encode all texts",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )

    args = parser.parse_args()

    if not args.model_dir.exists():
        sys.exit(f"Model directory not found: {args.model_dir}")

    for schema in args.schemas:
        run_schema(
            schema=schema,
            gold=args.gold,
            classifier=args.classifier,
            output_dir=args.output_dir,
            use_cache=not args.no_cache,
            n_splits=args.cv_folds,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
