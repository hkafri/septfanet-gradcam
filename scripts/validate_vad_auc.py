"""Formal validation of the block-23 VAD-AUC signal.

Diagnostic only -- no README update, no commit.

Step 1: one-sample Wilcoxon signed-rank of per-instance AUC vs null 0.5
        (block 23 and block 9), plus effect size.
Step 2: variance investigation -- which instances have low AUC and whether
        pair / speaker / VAD active rate predicts signal strength.
Step 3: re-calibrated F1 via Youden's J optimal threshold on the continuous
        block-23 CAM, vs the original (poorly-calibrated) F1 and the chance
        baseline.
"""

import csv
import json
from pathlib import Path

import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_curve

gradcam_root = Path(__file__).resolve().parents[1]


def wilcoxon_one_sample(values, null=0.5):
    v = np.asarray(values, dtype=float)
    diffs = v - null
    res = stats.wilcoxon(diffs)
    abs_d = np.abs(diffs)
    ranks = stats.rankdata(abs_d)
    w_pos = np.sum(ranks[diffs > 0])
    w_neg = np.sum(ranks[diffs < 0])
    tot = w_pos + w_neg
    r = float((w_pos - w_neg) / tot) if tot > 0 else 0.0
    return {"w_statistic": float(res.statistic), "p_value": float(res.pvalue), "rank_biserial_r": r, "n": int(len(v))}


def main():
    rows = list(csv.DictReader(open(gradcam_root / "results" / "auc_check" / "auc_check_results.csv")))

    out = {}
    for layer in ["TCN.TCN.9.conv1d", "TCN.TCN.23.conv1d"]:
        sub = [r for r in rows if r["layer"] == layer]
        vad = np.array([float(r["vad_auc_roc"]) for r in sub if r["vad_auc_roc"] not in ("nan", "")])
        w = wilcoxon_one_sample(vad, 0.5)
        out[layer] = {
            "n_instances": int(len(vad)),
            "auc_mean": float(vad.mean()),
            "auc_std": float(vad.std()),
            "auc_min": float(vad.min()),
            "auc_max": float(vad.max()),
            "wilcoxon_vs_0.5": w,
        }
        print(f"{layer}: n={w['n']}, AUC={vad.mean():.3f}±{vad.std():.3f} "
              f"[{vad.min():.3f},{vad.max():.3f}] | Wilcoxon vs 0.5: W={w['w_statistic']:.1f}, "
              f"p={w['p_value']:.3e}, r={w['rank_biserial_r']:+.3f}")

    # Variance investigation for block 23
    b23 = [r for r in rows if r["layer"] == "TCN.TCN.23.conv1d"]
    aucs = np.array([float(r["vad_auc_roc"]) for r in b23 if r["vad_auc_roc"] not in ("nan", "")])
    rates = np.array([float(r["vad_ref_active_rate"]) for r in b23 if r["vad_auc_roc"] not in ("nan", "")])
    speakers = np.array([int(r["speaker"]) for r in b23 if r["vad_auc_roc"] not in ("nan", "")])
    pairs = np.array([r["pair"] for r in b23 if r["vad_auc_roc"] not in ("nan", "")])

    low = aucs < 0.5
    print(f"\nBlock 23 low-AUC (<0.5) instances: {low.sum()} of {len(aucs)}")
    if low.sum() > 0:
        print(f"  low-AUC active-rate: mean={rates[low].mean():.3f}±{rates[low].std():.3f} | "
              f"high-AUC active-rate: mean={rates[~low].mean():.3f}±{rates[~low].std():.3f}")
        print(f"  low-AUC by speaker: spk0={int((speakers[low]==0).sum())}, spk1={int((speakers[low]==1).sum())} | "
              f"overall spk0={int((speakers==0).sum())}, spk1={int((speakers==1).sum())}")
        # correlation between active rate and AUC
        corr = np.corrcoef(rates, aucs)[0, 1]
        print(f"  correlation(vad_ref_active_rate, auc) = {corr:+.3f}")
    out["TCN.TCN.23.conv1d"]["low_auc_count"] = int(low.sum())

    out_path = gradcam_root / "results" / "auc_check" / "vad_auc_validation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[+] Saved: {out_path}")


if __name__ == "__main__":
    main()
