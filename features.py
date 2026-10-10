"""
Single source of truth for sequence features. Use this SAME function for training and in the website backend.
Requires: biopython
"""
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AA = "ACDEFGHIKLMNPQRSTVWY"
MIN_LENGTH = 10


def validate_sequence(seq):
    """Return (clean_sequence, error_message). error_message is None when valid."""
    s = "".join(str(seq).split()).upper()
    if len(s) < MIN_LENGTH:
        return None, f"Sequence is too short (minimum {MIN_LENGTH} amino acids)."
    bad = sorted({c for c in s if c not in AA})
    if bad:
        return None, "Sequence contains letters that are not standard amino acids: " + ", ".join(bad)
    return s, None


def compute_features(seq):
    """Features from one validated sequence. Raises ValueError if the sequence is not valid."""
    s, err = validate_sequence(seq)
    if err:
        raise ValueError(err)
    n = len(s)
    pa = ProteinAnalysis(s)
    pct = {a: 100.0 * s.count(a) / n for a in AA}
    f = {f"{a}_pct": pct[a] for a in AA}
    f["Length"] = n
    f["GRAVY"] = pa.gravy()
    f["Aromaticity"] = pa.aromaticity()
    f["Instability_Index"] = pa.instability_index()
    f["Aliphatic_Index"] = pct["A"] + 2.9 * pct["V"] + 3.9 * (pct["I"] + pct["L"])   # includes leucine
    f["Hydrophobic_pct"] = sum(pct[a] for a in "AILMFWVY")
    f["Charged_pct"] = sum(pct[a] for a in "DEKR")
    f["Polar_pct"] = sum(pct[a] for a in "STNQ")
    f["Acidic_pct"] = pct["D"] + pct["E"]
    f["Basic_pct"] = pct["K"] + pct["R"]
    # Calculated from the sequence, for proteins typed in by a user (training rows use the table's MW and pI)
    f["MW_kDa_calc"] = pa.molecular_weight() / 1000.0
    f["pI_calc"] = pa.isoelectric_point()
    return f


if __name__ == "__main__":
    # self-test
    seq = "MKLLVAAGLLVVAAVLAAEEKK"
    f = compute_features(seq)
    assert abs(sum(f[f"{a}_pct"] for a in AA) - 100) < 1e-9
    n = len(seq)
    expect = 100*seq.count("A")/n + 2.9*100*seq.count("V")/n + 3.9*100*(seq.count("I")+seq.count("L"))/n
    assert abs(f["Aliphatic_Index"] - expect) < 1e-9 and seq.count("L") > 0
    for bad in ["ACDX" * 5, "ACD"]:
        try:
            compute_features(bad); raise SystemExit("should have failed")
        except ValueError:
            pass
    print("features.py self-test passed")
