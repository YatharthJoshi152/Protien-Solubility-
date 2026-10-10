from __future__ import annotations

from fastapi.testclient import TestClient

from backend.features import AA, compute_features
from backend.main import app

client = TestClient(app)


def example_parameters() -> dict[str, float]:
    sequence = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE"
    features = compute_features(sequence)
    return {
        **{key: float(value) for key, value in features.items() if key.endswith("_pct") or key in {"Length", "GRAVY", "Aromaticity", "Instability_Index", "Aliphatic_Index", "Hydrophobic_pct", "Charged_pct", "Polar_pct", "Acidic_pct", "Basic_pct"}},
        "MW_kDa": float(features["MW_kDa_calc"]),
        "pI": float(features["pI_calc"]),
    }


def test_feature_percentages_and_aliphatic_index() -> None:
    features = compute_features("MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE")
    assert abs(sum(features[f"{aa}_pct"] for aa in AA) - 100.0) < 1e-9
    expected = features["A_pct"] + 2.9 * features["V_pct"] + 3.9 * (features["I_pct"] + features["L_pct"])
    assert abs(features["Aliphatic_Index"] - expected) < 1e-9


def test_invalid_sequence_is_rejected() -> None:
    response = client.post("/features", json={"sequence": "ACDEFGHIKX"})
    assert response.status_code == 422
    assert "standard amino acids" in response.json()["detail"]


def test_predict_contract_and_bounds() -> None:
    response = client.post(
        "/predict",
        json={
            "name": "Test protein",
            "parameters": example_parameters(),
            "cell_location": "Cytoplasmic",
            "gene_type": "e",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["final_score"] <= 1.0
    assert set(body["per_model"]) == {"lr", "svm", "rf"}
    assert all(0.0 <= value <= 1.0 for value in body["per_model"].values())
    assert body["agreement"]


def test_meta_contract() -> None:
    response = client.get("/meta")
    assert response.status_code == 200
    assert "feature_order" in response.json()
    assert "categories" in response.json()
    assert "training_ranges" in response.json()
