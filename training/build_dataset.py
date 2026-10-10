from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.features import AA, compute_features, validate_sequence

DATA = ROOT / "data" / "features_all.csv"
REPORT = ROOT / "data" / "feature_parity_report.json"


def main() -> None:
    table = pd.read_csv(DATA)
    required = {"B_number", "Sequence", "has_solubility_label", "Cell_location", "Gene_type", "MW_kDa", "pI"}
    missing = required.difference(table.columns)
    if missing:
        raise SystemExit("Missing required columns: " + ", ".join(sorted(missing)))
    checks = []
    for _, row in table.sample(n=min(20, len(table)), random_state=42).iterrows():
        clean, error = validate_sequence(row["Sequence"])
        if error:
            checks.append({"B_number": row["B_number"], "ok": False, "error": error})
            continue
        calculated = compute_features(clean)
        diffs = {}
        for field in [*[f"{a}_pct" for a in AA], "Length", "GRAVY", "Aromaticity", "Instability_Index", "Aliphatic_Index", "Hydrophobic_pct", "Charged_pct", "Polar_pct", "Acidic_pct", "Basic_pct"]:
            csv_field = field
            if csv_field not in row:
                continue
            diffs[field] = abs(float(calculated[field]) - float(row[csv_field]))
        checks.append({"B_number": row["B_number"], "ok": max(diffs.values(), default=0.0) <= 0.0001, "max_abs_difference": max(diffs.values(), default=0.0)})
    result = {
        "rows": int(len(table)),
        "labeled_rows": int(table["has_solubility_label"].astype(str).str.lower().eq("true").sum()),
        "unlabeled_rows": int(table["has_solubility_label"].astype(str).str.lower().eq("false").sum()),
        "duplicated_b_numbers": int(table["B_number"].duplicated().sum()),
        "parity_checks": checks,
    }
    REPORT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
