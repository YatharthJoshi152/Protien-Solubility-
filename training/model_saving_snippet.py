"""Notebook cell to persist production models after grouped evaluation.

The training script in this project is the executable source of truth. This snippet
shows the artifact contract to add after the final refit in a training notebook.
"""
from pathlib import Path
import json
import joblib

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

# `lr_model`, `svm_model`, and `rf_model` must already be refit on all labeled rows.
# `meta_model` must be fit only on out-of-fold probabilities.
# `ensemble_mode` is either "stacked" or "average" based on grouped CV ROC-AUC.
joblib.dump(lr_model, ARTIFACTS / "lr.joblib")
joblib.dump(svm_model, ARTIFACTS / "svm.joblib")
joblib.dump(rf_model, ARTIFACTS / "rf.joblib")
joblib.dump({"mode": ensemble_mode, "model": meta_model}, ARTIFACTS / "meta.joblib")
(ARTIFACTS / "feature_order.json").write_text(json.dumps(feature_order, indent=2))
(ARTIFACTS / "training_ranges.json").write_text(json.dumps(training_ranges, indent=2))
(ARTIFACTS / "categories.json").write_text(json.dumps(categories, indent=2))
(ARTIFACTS / "model_card.json").write_text(json.dumps(model_card, indent=2))
