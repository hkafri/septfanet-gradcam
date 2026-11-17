"""Pair selection for the per-frame decision-map figures (Step 2 of the per-frame
analysis). The rule below is DECLARED AND COMMITTED BEFORE ANY FIGURE IS LOOKED AT;
a figure chosen after looking at it is an illustration, not evidence.

SELECTION RULE (fixed in advance)
---------------------------------
Universe: the 65 held-out pairs only (rows with set == "heldout" in
results/slot_matching/matched_analysis_per_pair.csv; pair_index 35..99 in the
resolve_all_pairs() order), SI-SDR slot matching throughout.

1. Six pairs at the 5/20/40/60/80/95-th percentile of pair-level block-23 AUC
   (column b23_auc): sort the 65 pairs by b23_auc ascending and take sorted
   position round(q*(n-1)) for q in {.05, .20, .40, .60, .80, .95}. This
   includes the worst pairs on purpose.
2. Two pairs with the largest min(|E_s|, |E_o|), where E_s = v_s & ~v_o and
   E_o = ~v_s & v_o are the exclusive-activity frame sets of the two Silero
   reference VADs (computed from the clean sources) -- the only situation where
   speaker-specificity can be seen by eye. If such a pair is already among the
   six percentile pairs, the next-largest pair is taken instead, so exactly 8
   distinct pairs result.
3. Thesis figure (figures/per_frame_maps_thesis.pdf): of the six AUC-percentile
   pairs, the one whose b23_auc is closest to the held-out median b23_auc
   (ties: the lower AUC). No other criterion.

Output: results/per_frame_maps/selection.json (pair indices + this rule).
Compute-only; edits nothing else.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from silero_vad import load_silero_vad

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

from scripts.evaluate_baselines import resolve_all_pairs, load_and_pad, silero_mask

PERCENTILES = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]
MATCHED_CSV = gradcam_root / "results" / "slot_matching" / "matched_analysis_per_pair.csv"


def main():
    rows = [r for r in csv.DictReader(open(MATCHED_CSV)) if r["set"] == "heldout"]
    assert len(rows) == 65, f"expected 65 held-out rows, got {len(rows)}"
    held_pairs = [p for p in resolve_all_pairs() if p[2] == "heldout"]
    assert len(held_pairs) == 65

    auc = np.array([float(r["b23_auc"]) for r in rows])
    order = np.argsort(auc, kind="stable")
    n = len(rows)

    selected = {}  # heldout-local index -> record
    for q in PERCENTILES:
        k = int(order[int(round(q * (n - 1)))])
        selected.setdefault(k, {"roles": []})["roles"].append(f"auc_p{int(q*100)}")

    # exclusive-activity counts from the Silero masks of the clean sources
    vad_model = load_silero_vad()
    min_excl = np.zeros(n, int)
    for i, (p0, p1, _) in enumerate(held_pairs):
        v0 = silero_mask(vad_model, load_and_pad(p0))
        v1 = silero_mask(vad_model, load_and_pad(p1))
        e_s = int((v0 & ~v1).sum())
        e_o = int((~v0 & v1).sum())
        min_excl[i] = min(e_s, e_o)
        print(f"  heldout {i}: min(|E_s|,|E_o|) = {min_excl[i]}")

    taken = 0
    for k in np.argsort(-min_excl, kind="stable"):
        k = int(k)
        if taken == 2:
            break
        if k in selected:
            continue  # already a percentile pair; take the next largest
        selected.setdefault(k, {"roles": []})["roles"].append("max_exclusive")
        taken += 1

    assert len(selected) == 8, f"expected 8 distinct pairs, got {len(selected)}"

    # thesis pair: percentile pair with b23_auc closest to the held-out median
    median = float(np.median(auc))
    perc_keys = [k for k, v in selected.items() if any(r.startswith("auc_") for r in v["roles"])]
    thesis = min(perc_keys, key=lambda k: (abs(auc[k] - median), auc[k]))
    selected[thesis]["roles"].append("thesis_figure")

    out = {
        "rule": __doc__.strip(),
        "declared": "2026-11-17, before any per-frame figure was generated or viewed",
        "n_heldout": n,
        "median_b23_auc_heldout": median,
        "pairs": [],
    }
    for k in sorted(selected):
        p0, p1, _ = held_pairs[k]
        out["pairs"].append({
            "heldout_index": k,
            "pair_index": int(rows[k]["pair_index"]),
            "b23_auc": float(auc[k]),
            "min_exclusive_frames": int(min_excl[k]),
            "utt_0": p0.name, "utt_1": p1.name,
            "roles": selected[k]["roles"],
        })
    out_dir = gradcam_root / "results" / "per_frame_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selection.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["pairs"], indent=2))
    print("[+] saved results/per_frame_maps/selection.json")


if __name__ == "__main__":
    main()
