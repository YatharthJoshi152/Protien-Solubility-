from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.features import AA, compute_features, validate_sequence

DATA = ROOT / "data" / "features_all.csv"
ARTIFACTS = ROOT / "artifacts"

NUMERIC_27 = ["MW_kDa", "pI", "GRAVY", "Instability_Index", "Aliphatic_Index", "Hydrophobic_pct", "Charged_pct", *[f"{a}_pct" for a in AA]]
LOGISTIC_NUMERIC = ["log_MW", "pI", "log_Length", "GRAVY", "Aromaticity", "Instability_Index", "Aliphatic_Index", "Charged_pct", "Hydrophobic_pct", "Polar_pct", "Acidic_pct", "Basic_pct", *[f"{a}_pct" for a in AA], "pI_dist7"]


def model_frame(table: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in table.iterrows():
        sequence, error = validate_sequence(row["Sequence"])
        if error:
            raise ValueError(f"{row['B_number']}: {error}")
        values = compute_features(sequence)
        values["MW_kDa"] = float(row["MW_kDa"])
        values["pI"] = float(row["pI"])
        values["log_MW"] = float(np.log(max(values["MW_kDa"], 1e-9)))
        values["log_Length"] = float(np.log(max(values["Length"], 1.0)))
        values["pI_dist7"] = abs(values["pI"] - 7.0)
        values["Cell_location"] = str(row["Cell_location"])
        values["Gene_type"] = str(row["Gene_type"])
        records.append(values)
    return pd.DataFrame(records)


def grouped_categories(values: pd.Series, minimum: int = 30) -> tuple[pd.Series, list[str]]:
    counts = values.value_counts()
    allowed = sorted(counts[counts >= minimum].index.astype(str).tolist())
    mapped = values.astype(str).where(values.astype(str).isin(allowed), "Other")
    return mapped, allowed + (["Other"] if "Other" in mapped.values else [])


def apply_categories(values: pd.Series, allowed: list[str]) -> pd.Series:
    return values.astype(str).where(values.astype(str).isin(set(allowed)), "Other")


def lr_pipeline() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    spline = Pipeline([("impute", SimpleImputer(strategy="median")), ("spline", SplineTransformer(n_knots=4, degree=3)), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, LOGISTIC_NUMERIC),
            ("spline", spline, ["log_MW", "pI"]),
            ("categorical", categorical, ["Cell_location", "Gene_type"]),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])


def svm_pipeline() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))])


def rf_pipeline() -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced", max_features="sqrt", random_state=42, n_jobs=-1))])


def fit_predictions(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> dict[str, np.ndarray]:
    lr = lr_pipeline().fit(train_x, train_y)
    svm = svm_pipeline().fit(train_x[NUMERIC_27], train_y)
    rf = rf_pipeline().fit(train_x[NUMERIC_27], train_y)
    return {
        "lr": lr.predict_proba(test_x)[..., 1],
        "svm": svm.predict_proba(test_x[NUMERIC_27])[..., 1],
        "rf": rf.predict_proba(test_x[NUMERIC_27])[..., 1],
    }, {"lr": lr, "svm": svm, "rf": rf}


def metrics(y_true: pd.Series, probability: np.ndarray) -> dict[str, float]:
    predicted = probability >= 0.5
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    table = pd.read_csv(DATA)
    labeled_mask = table["has_solubility_label"].astype(str).str.lower().eq("true")
    labeled = table.loc[labeled_mask].reset_index(drop=True)
    raw_x = model_frame(labeled)
    y = labeled["Soluble"].astype(int).reset_index(drop=True)
    groups = labeled["B_number"].astype(str).reset_index(drop=True)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_index, test_index = next(splitter.split(raw_x, y, groups=groups))
    train_x = raw_x.iloc[train_index].reset_index(drop=True)
    test_x = raw_x.iloc[test_index].reset_index(drop=True)
    train_x["Cell_location"], cell_categories = grouped_categories(train_x["Cell_location"])
    train_x["Gene_type"], gene_categories = grouped_categories(train_x["Gene_type"])
    test_x["Cell_location"] = apply_categories(test_x["Cell_location"], cell_categories)
    test_x["Gene_type"] = apply_categories(test_x["Gene_type"], gene_categories)
    train_y, test_y = y.iloc[train_index].reset_index(drop=True), y.iloc[test_index].reset_index(drop=True)
    train_groups = groups.iloc[train_index].reset_index(drop=True)

    oof = {name: np.zeros(len(train_x), dtype=float) for name in ("lr", "svm", "rf")}
    fold_metrics: dict[str, list[dict[str, float]]] = {name: [] for name in ("lr", "svm", "rf")}
    group_kfold = GroupKFold(n_splits=5)
    for fold_train, fold_valid in group_kfold.split(train_x, train_y, groups=train_groups):
        fold_predictions, _ = fit_predictions(train_x.iloc[fold_train], train_y.iloc[fold_train], train_x.iloc[fold_valid])
        for name, values in fold_predictions.items():
            oof[name][fold_valid] = values
            fold_metrics[name].append(metrics(train_y.iloc[fold_valid], values))

    best_base_auc = max(roc_auc_score(train_y, values) for values in oof.values())
    meta_model = LogisticRegression(max_iter=1000, random_state=42).fit(np.column_stack([oof["lr"], oof["svm"], oof["rf"]]), train_y)
    stacked_oof = meta_model.predict_proba(np.column_stack([oof["lr"], oof["svm"], oof["rf"]]))[:, 1]
    average_oof = np.mean(np.column_stack([oof["lr"], oof["svm"], oof["rf"]]), axis=1)
    mode = "stacked" if roc_auc_score(train_y, stacked_oof) > best_base_auc else "average"
    chosen_oof = stacked_oof if mode == "stacked" else average_oof

    test_predictions, _ = fit_predictions(train_x, train_y, test_x)
    test_matrix = np.column_stack([test_predictions["lr"], test_predictions["svm"], test_predictions["rf"]])
    test_final = meta_model.predict_proba(test_matrix)[:, 1] if mode == "stacked" else test_matrix.mean(axis=1)

    production_x = model_frame(labeled)
    production_x["Cell_location"], production_cell_categories = grouped_categories(production_x["Cell_location"])
    production_x["Gene_type"], production_gene_categories = grouped_categories(production_x["Gene_type"])
    production_predictions, production_models = fit_predictions(production_x, y, production_x)
    del production_predictions
    joblib.dump(production_models["lr"], ARTIFACTS / "lr.joblib")
    joblib.dump(production_models["svm"], ARTIFACTS / "svm.joblib")
    joblib.dump(production_models["rf"], ARTIFACTS / "rf.joblib")
    joblib.dump({"mode": mode, "model": meta_model}, ARTIFACTS / "meta.joblib")

    numeric_ranges = {}
    for field in sorted(set(NUMERIC_27 + LOGISTIC_NUMERIC + ["Length"])):
        numeric_ranges[field] = {"min": float(production_x[field].min()), "max": float(production_x[field].max())}
    (ARTIFACTS / "training_ranges.json").write_text(json.dumps(numeric_ranges, indent=2))
    (ARTIFACTS / "categories.json").write_text(json.dumps({"Cell_location": production_cell_categories, "Gene_type": production_gene_categories}, indent=2))
    (ARTIFACTS / "feature_order.json").write_text(json.dumps({"numeric_27": NUMERIC_27, "logistic_numeric": LOGISTIC_NUMERIC}, indent=2))

    test_metrics = {name: metrics(test_y, values) for name, values in test_predictions.items()}
    test_metrics["ensemble"] = metrics(test_y, test_final)
    cv_metrics = {name: metrics(train_y, values) for name, values in oof.items()}
    cv_metrics["ensemble"] = metrics(train_y, chosen_oof)
    card = {
        "dataset_sizes": {
            "total_rows": int(len(table)),
            "labeled_rows": int(len(labeled)),
            "unlabeled_rows": int(len(table) - len(labeled)),
            "training_rows": int(len(train_x)),
            "held_out_rows": int(len(test_x)),
            "training_genes": int(train_groups.nunique()),
            "held_out_genes": int(groups.iloc[test_index].nunique()),
        },
        "metrics": {"held_out_test": test_metrics, "grouped_cross_validation": cv_metrics, "folds": fold_metrics},
        "ensemble": {"mode": mode, "selection_rule": "Stacked logistic regression is used only when its grouped cross-validation ROC-AUC beats every base model, otherwise the simple average is used."},
        "limitations": [
            "Trained on E. coli K-12 proteins only.",
            "Related proteins can sit on both sides of the gene-grouped split.",
            "The unlabeled set contains more membrane proteins than the labeled training set, so those scores are less reliable.",
        ],
        "library_versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (ARTIFACTS / "model_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps({"mode": mode, "held_out": test_metrics, "cross_validation": cv_metrics}, indent=2))


if __name__ == "__main__":
    main()
