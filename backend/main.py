from __future__ import annotations

import csv
import io
import json
import math
import statistics
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .features import AA, compute_features, validate_sequence

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DIST = ROOT / "dist"

NUMERIC_27 = [
    "MW_kDa",
    "pI",
    "GRAVY",
    "Instability_Index",
    "Aliphatic_Index",
    "Hydrophobic_pct",
    "Charged_pct",
    *[f"{a}_pct" for a in AA],
]
LOGISTIC_NUMERIC = [
    "log_MW",
    "pI",
    "log_Length",
    "GRAVY",
    "Aromaticity",
    "Instability_Index",
    "Aliphatic_Index",
    "Charged_pct",
    "Hydrophobic_pct",
    "Polar_pct",
    "Acidic_pct",
    "Basic_pct",
    *[f"{a}_pct" for a in AA],
    "pI_dist7",
]

app = FastAPI(title="E. coli Protein Solubility Predictor", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SequenceRequest(BaseModel):
    sequence: str = Field(min_length=1)


class PredictRequest(BaseModel):
    name: str = Field(default="Unnamed protein", max_length=200)
    identifier: str | None = Field(default=None, max_length=200)
    parameters: dict[str, float] = Field(default_factory=dict)
    cell_location: str | None = Field(default=None, max_length=200)
    gene_type: str | None = Field(default=None, max_length=200)
    sequence: str | None = None
    edited_fields: list[str] = Field(default_factory=list)

    @field_validator("sequence")
    @classmethod
    def validate_sequence_text(cls, value: str | None) -> str | None:
        if value is not None:
            clean, error = validate_sequence(value)
            if error:
                raise ValueError(error)
            return clean
        return value


class ErrorMessage(BaseModel):
    detail: str


def _read_json(name: str, fallback: Any) -> Any:
    path = ARTIFACTS / name
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return fallback


def _load_artifact(name: str) -> Any:
    path = ARTIFACTS / name
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


MODELS = {
    "lr": _load_artifact("lr.joblib"),
    "svm": _load_artifact("svm.joblib"),
    "rf": _load_artifact("rf.joblib"),
}
META = _load_artifact("meta.joblib")
FEATURE_ORDER = _read_json(
    "feature_order.json",
    {"numeric_27": NUMERIC_27, "logistic_numeric": LOGISTIC_NUMERIC},
)
RANGES = _read_json("training_ranges.json", {})
CATEGORIES = _read_json("categories.json", {"Cell_location": [], "Gene_type": []})
MODEL_CARD = _read_json("model_card.json", {"metrics": {}, "limitations": []})


def _display_name(field: str) -> str:
    return field.replace("_", " ").replace("pct", "percent").replace("MW", "molecular weight")


def _score_band(score: float) -> str:
    if score < 0.35:
        return "Likely insoluble"
    if score < 0.65:
        return "Uncertain"
    return "Likely soluble"


def _agreement(probabilities: dict[str, float]) -> tuple[float, str, str | None]:
    values = list(probabilities.values())
    spread = max(values) - min(values)
    if spread < 0.15:
        label = "Models agree"
    elif spread <= 0.30:
        label = "Models partly disagree"
    else:
        label = "Models disagree"
    median = statistics.median(values)
    outlier_key = max(probabilities, key=lambda key: abs(probabilities[key] - median))
    outlier = outlier_key if spread >= 0.15 else None
    return round(float(spread), 4), label, outlier


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(number):
        return 0.5
    return max(0.0, min(1.0, number))


def _build_features(parameters: dict[str, float], sequence: str | None) -> dict[str, float]:
    computed: dict[str, Any] = {}
    if sequence:
        clean, error = validate_sequence(sequence)
        if error:
            raise HTTPException(status_code=422, detail=error)
        computed = compute_features(clean)
        computed["MW_kDa"] = computed.pop("MW_kDa_calc")
        computed["pI"] = computed.pop("pI_calc")
    computed.update(parameters)
    missing = [field for field in NUMERIC_27 if field not in computed]
    if missing:
        raise HTTPException(
            status_code=422,
            detail="Missing required parameters: " + ", ".join(_display_name(field) for field in missing),
        )
    if "Length" not in computed:
        computed["Length"] = computed.get("log_Length", 0.0)
    computed["log_MW"] = math.log(max(float(computed["MW_kDa"]), 1e-9))
    computed["log_Length"] = math.log(max(float(computed["Length"]), 1.0))
    computed["pI_dist7"] = abs(float(computed["pI"]) - 7.0)
    return {key: float(value) for key, value in computed.items() if isinstance(value, (int, float))}


def _build_frame(features: dict[str, float], cell_location: str | None, gene_type: str | None) -> pd.DataFrame:
    row: dict[str, Any] = dict(features)
    row["Cell_location"] = cell_location or "Unknown"
    row["Gene_type"] = gene_type or "Unknown"
    for field in NUMERIC_27 + LOGISTIC_NUMERIC:
        row.setdefault(field, float("nan"))
    return pd.DataFrame([row])


def _fallback_probability(features: dict[str, float], model_name: str) -> float:
    hydrophobic = float(features.get("Hydrophobic_pct", 40.0))
    charged = float(features.get("Charged_pct", 20.0))
    instability = float(features.get("Instability_Index", 40.0))
    pi_distance = abs(float(features.get("pI", 7.0)) - 7.0)
    base = 0.58 - (hydrophobic - 40.0) * 0.008 + (charged - 20.0) * 0.004 - (instability - 40.0) * 0.003
    if model_name == "svm":
        base += (0.5 - min(pi_distance, 3.0) / 6.0) * 0.03
    if model_name == "rf":
        base += (float(features.get("Aliphatic_Index", 80.0)) - 80.0) * 0.001
    return _clamp(base)


def _predict_model(model_name: str, frame: pd.DataFrame, features: dict[str, float]) -> float:
    model = MODELS.get(model_name)
    if model is None:
        return _fallback_probability(features, model_name)
    try:
        model_frame = frame if model_name == "lr" else frame[NUMERIC_27]
        probability = model.predict_proba(model_frame)[0, 1]
        return _clamp(probability)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"The {model_name} model could not score this input: {exc}") from exc


def _contributions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    model = MODELS.get("lr")
    if model is None or not hasattr(model, "named_steps"):
        return []
    try:
        transformer = model.named_steps["preprocess"]
        classifier = model.named_steps["model"]
        transformed = transformer.transform(frame)
        names = transformer.get_feature_names_out()
        values = transformed.toarray()[0] if hasattr(transformed, "toarray") else transformed[0]
        contributions = []
        for name, coefficient, value in zip(names, classifier.coef_[0], values):
            amount = float(coefficient * value)
            contributions.append({"feature": str(name), "direction": "up" if amount >= 0 else "down", "size": round(abs(amount), 4)})
        return sorted(contributions, key=lambda item: item["size"], reverse=True)[:8]
    except Exception:
        return []


def _importances() -> list[dict[str, Any]]:
    model = MODELS.get("rf")
    if model is None or not hasattr(model, "named_steps"):
        return []
    try:
        classifier = model.named_steps["model"]
        order = FEATURE_ORDER.get("numeric_27", NUMERIC_27)
        return [
            {"feature": feature, "importance": round(float(importance), 4)}
            for feature, importance in sorted(zip(order, classifier.feature_importances_), key=lambda pair: pair[1], reverse=True)[:8]
        ]
    except Exception:
        return []


def _warnings(features: dict[str, float], cell_location: str | None, spread: float) -> list[str]:
    messages: list[str] = []
    for field, bounds in RANGES.items():
        if field not in features or not isinstance(bounds, dict):
            continue
        low, high = bounds.get("min"), bounds.get("max")
        if low is None or high is None:
            continue
        value = features[field]
        if value < low or value > high:
            messages.append(f"{_display_name(field)} is outside the training range of {low:.3g} to {high:.3g}.")
    minimum_length = RANGES.get("Length", {}).get("min")
    if minimum_length is not None and features.get("Length", 0) < minimum_length:
        messages.append(f"Sequence length is shorter than the training minimum of {minimum_length:.0f} residues.")
    if cell_location and "Membrane" in cell_location:
        messages.append("The training data under-represents some membrane-related classes, so interpret this result carefully.")
    if spread > 0.30:
        messages.append("The models disagree strongly. The final score is less reliable.")
    elif spread >= 0.15:
        messages.append("The models partly disagree. Treat the final score as less reliable than an agreement result.")
    if not all(MODELS.values()):
        messages.append("One or more trained artifacts are unavailable. The fallback score is for interface testing only.")
    return messages


def _result(request: PredictRequest) -> dict[str, Any]:
    features = _build_features(request.parameters, request.sequence)
    frame = _build_frame(features, request.cell_location, request.gene_type)
    probabilities = {name: _predict_model(name, frame, features) for name in ("lr", "svm", "rf")}
    spread, agreement, outlier = _agreement(probabilities)
    if META and isinstance(META, dict) and META.get("mode") == "stacked" and META.get("model") is not None:
        try:
            final_score = _clamp(META["model"].predict_proba([[probabilities["lr"], probabilities["svm"], probabilities["rf"]]])[0, 1])
        except Exception:
            final_score = _clamp(sum(probabilities.values()) / 3)
    else:
        final_score = _clamp(sum(probabilities.values()) / 3)
    return {
        "name": request.name,
        "identifier": request.identifier,
        "final_score": round(final_score, 2),
        "band": _score_band(final_score),
        "per_model": {key: round(value, 4) for key, value in probabilities.items()},
        "spread": spread,
        "agreement": agreement,
        "outlier_model": outlier,
        "lr_contributions": _contributions(frame),
        "rf_importances": _importances(),
        "warnings": _warnings(features, request.cell_location, spread),
        "parameters": features,
        "categories": {"cell_location": request.cell_location or "Unknown", "gene_type": request.gene_type or "Unknown"},
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/meta")
def meta() -> dict[str, Any]:
    example: dict[str, Any] | None = None
    data_path = ROOT / "data" / "features_all.csv"
    if data_path.exists():
        try:
            sample = pd.read_csv(data_path, nrows=1).iloc[0]
            sequence = str(sample["Sequence"])
            sequence_features = compute_features(sequence)
            example = {
                "name": str(sample.get("Gene_name", "Example protein")),
                "identifier": str(sample.get("B_number", "")),
                "sequence": sequence,
                "cell_location": str(sample.get("Cell_location", "Unknown")),
                "gene_type": str(sample.get("Gene_type", "Unknown")),
                "parameters": {
                    **{key: float(value) for key, value in sequence_features.items() if key not in {"MW_kDa_calc", "pI_calc"}},
                    "MW_kDa": float(sample["MW_kDa"]),
                    "pI": float(sample["pI"]),
                },
            }
        except Exception:
            example = None
    return {
        "feature_order": FEATURE_ORDER,
        "categories": CATEGORIES,
        "training_ranges": RANGES,
        "model_versions": MODEL_CARD.get("library_versions", {}),
        "model_card": MODEL_CARD,
        "example_protein": example,
        "parameter_descriptions": {
            "MW_kDa": "Molecular weight in kilodaltons.",
            "pI": "Isoelectric point.",
            "Length": "Sequence length in amino acids.",
            "GRAVY": "Grand average of hydropathy.",
            "Aromaticity": "Fraction of aromatic residues.",
            "Instability_Index": "Biopython instability index.",
            "Aliphatic_Index": "Aliphatic residue index including leucine.",
            "Hydrophobic_pct": "Percent of A, I, L, M, F, W, V, and Y.",
            "Charged_pct": "Percent of D, E, K, and R.",
            "Polar_pct": "Percent of S, T, N, and Q.",
            "Acidic_pct": "Percent of D and E.",
            "Basic_pct": "Percent of K and R.",
        },
    }


@app.post("/features")
def features(request: SequenceRequest) -> dict[str, Any]:
    clean, error = validate_sequence(request.sequence)
    if error:
        raise HTTPException(status_code=422, detail=error)
    result = compute_features(clean)
    return {"sequence": clean, "features": result, "calculated_fields": ["MW_kDa", "pI"]}


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    return _result(request)


@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)) -> StreamingResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Upload a CSV file.")
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(text)))
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="The CSV must be UTF-8 encoded.") from exc
    if not rows:
        raise HTTPException(status_code=422, detail="The CSV has no data rows.")
    output = io.StringIO()
    columns = ["name", "identifier", "lr_probability", "svm_probability", "rf_probability", "final_score", "agreement", "spread", "outlier_model"]
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for index, row in enumerate(rows, start=1):
        parameters: dict[str, float] = {}
        for key in NUMERIC_27 + ["Length"]:
            value = row.get(key)
            if value not in (None, ""):
                try:
                    parameters[key] = float(value)
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=f"Row {index}: {key} must be numeric.") from exc
        request = PredictRequest(
            name=row.get("name") or row.get("Gene_name") or row.get("Gene name") or f"Batch row {index}",
            identifier=row.get("identifier") or row.get("B_number") or row.get("B number"),
            sequence=row.get("Sequence") or row.get("sequence") or None,
            parameters=parameters,
            cell_location=row.get("Cell_location") or row.get("Cell location") or None,
            gene_type=row.get("Gene_type") or row.get("Type of gene product") or None,
        )
        result = _result(request)
        writer.writerow(
            {
                "name": result["name"],
                "identifier": result["identifier"] or "",
                "lr_probability": result["per_model"]["lr"],
                "svm_probability": result["per_model"]["svm"],
                "rf_probability": result["per_model"]["rf"],
                "final_score": result["final_score"],
                "agreement": result["agreement"],
                "spread": result["spread"],
                "outlier_model": result["outlier_model"] or "",
            }
        )
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=solubility_predictions.csv"},
    )


if DIST.exists():
    from fastapi.staticfiles import StaticFiles

    @app.get("/", include_in_schema=False)
    def frontend_root() -> FileResponse:
        return FileResponse(DIST / "index.html")

    @app.get("/batch", include_in_schema=False)
    def frontend_batch() -> FileResponse:
        return FileResponse(DIST / "index.html")

    @app.get("/how-this-works", include_in_schema=False)
    def frontend_how() -> FileResponse:
        return FileResponse(DIST / "index.html")

    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
