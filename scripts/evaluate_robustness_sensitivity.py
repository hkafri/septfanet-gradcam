"""Sensitivity analysis for the AUC-based robustness verdicts (NO new simulations).

This script recomputes the per-pair CAM AUCs using the SAME seeded degradation
pipeline as scripts/evaluate_robustness_auc.py (identical seeds, RIRs, babble,
and target selection), saves the per-pair arrays, and then:

  Step 1: three-way non-inferiority classification of each condition from the
      pre-registered margin (-0.06) and the 95% pair-bootstrap CI of
      (AUC_condition - AUC_clean): robust (lower > -0.06), inferior (upper < -0.06),
      inconclusive (otherwise; borderline when a bound is within 0.01 of margin).

  Step 2 (SENSITIVITY ANALYSIS, not a change to the primary verdicts): recompute
      the CI of (AUC_condition - AUC_clean) at Bonferroni-adjusted coverage
      1 - 0.05/7 (99.2857%) with the same pair bootstrap (10,000 resamples of
      pairs), and report whether any verdict changes -- in particular T60=0.4 s
      (unadjusted lower bound -0.049) and T60=0.2 s.

Primary (unadjusted 95%) verdicts remain primary; the Bonferroni column is a
sensitivity check only.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from sklearn.metrics import roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

# reuse the exact degradation + model machinery from the robustness script
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "robust_mod", gradcam_root / "scripts" / "evaluate_robustness_auc.py")
robust_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(robust_mod)

MARGIN = -0.06
N_BOOT = 10000
BONF_COVERAGE = 1 - 0.05 / 7  # ~0.992857
ALPHA_MULT = 1 - BONF_COVERAGE   # 0.007143 -> two-sided percentiles

BLOCK_23 = "TCN.TCN.23.conv1d"
CONDITIONS = robust_mod.CONDITIONS
DEGRADED = robust_mod.DEGRADED


def pair_boot_ci_diff(cond_vals, clean_vals, coverage, n_boot=N_BOOT, seed=0):
    """Bootstrap CI for the mean paired difference, resampling PAIRS."""
    cond = np.asarray(cond_vals, float)
    clean = np.asarray(clean_vals, float)
    diff = cond - clean
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    tail = (1 - coverage) / 2 * 100
    lo, hi = np.percentile(boots, [tail, 100 - tail])
    return float(lo), float(hi)


def classify(lo, hi, margin=MARGIN):
    if lo > margin:
        verdict = "robust"
    elif hi < margin:
        verdict = "inferior"
    else:
        verdict = "inconclusive"
    borderline = (abs(lo - margin) <= 0.01) or (abs(hi - margin) <= 0.01)
    return verdict, borderline


def compute_per_pair_auc(subset):
    """Recompute per-pair CAM AUC (block 23) for clean + all conditions, using the
    exact same pipeline/seeds as evaluate_robustness_auc.py."""
    device = "cpu"
    import network.model as module_arch
    from gradcam import GradCAM  # noqa: F401 (used via robust_mod.compute_cam)
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    ckpt = torch.load(gradcam_root / "weights" / "model_with_vad.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    vad_model = load_silero_vad()

    pairs = robust_mod.resolve_pairs(subset)
    used_names = set().union(*[p["used_names"] for p in pairs])
    unused = [f for r in (robust_mod.SAMPLES_DIR, robust_mod.HOLDOUT_DIR)
              for f in r.glob("*.flac") if f.name not in used_names]
    babble = robust_mod.build_babble_track(unused, np.random.default_rng(42))

    per_pair = {c: [] for c in CONDITIONS}
    for idx, p in enumerate(pairs):
        sig0 = robust_mod.load_and_pad(p["path_0"])
        sig1 = robust_mod.load_and_pad(p["path_1"])
        v0 = robust_mod.silero_mask(vad_model, sig0, robust_mod.NUM_FRAMES)
        v1 = robust_mod.silero_mask(vad_model, sig1, robust_mod.NUM_FRAMES)
        refs = (v0, v1)
        for cond in CONDITIONS:
            mix = robust_mod.build_mixture(sig0, sig1, cond, babble, idx)
            audio = torch.from_numpy(mix).unsqueeze(0)
            with torch.no_grad():
                _ = model(audio.to(device))
                vad_logits = model.vad_logits.detach().cpu().numpy()[0]
            aucs = []
            for speaker in range(2):
                cam = robust_mod.minmax(robust_mod.compute_cam(
                    model, audio, BLOCK_23, speaker,
                    int(np.argmax(np.abs(vad_logits[speaker]))), device))
                cam = cam[:robust_mod.NUM_FRAMES] if len(cam) >= robust_mod.NUM_FRAMES else np.pad(
                    cam, (0, robust_mod.NUM_FRAMES - len(cam)))
                v_s = refs[speaker]
                if v_s.sum() > 0 and v_s.sum() < len(v_s):
                    aucs.append(float(roc_auc_score(v_s, cam)))
            per_pair[cond].append(float(np.mean(aucs)) if aucs else float("nan"))
        if (idx + 1) % 10 == 0:
            print(f"  {subset}: {idx+1}/{len(pairs)} pairs")
    return per_pair


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="heldout", choices=["heldout", "dev"])
    ap.add_argument("--load-cache", action="store_true",
                    help="load previously saved per-pair AUCs instead of recomputing")
    args = ap.parse_args()

    out_dir = gradcam_root / "results" / "robustness_auc"
    cache = out_dir / f"per_pair_auc_{args.subset}.json"

    if args.load_cache and cache.exists():
        per_pair = {k: v for k, v in json.loads(cache.read_text()).items()}
        print(f"[*] loaded cached per-pair AUCs for subset={args.subset}")
    else:
        print(f"[*] recomputing per-pair AUCs for subset={args.subset}")
        per_pair = compute_per_pair_auc(args.subset)
        cache.write_text(json.dumps(per_pair))

    clean = np.array(per_pair["clean"], float)
    rows = []
    summary = {"subset": args.subset, "margin": MARGIN, "n_pairs": int(np.sum(~np.isnan(clean))),
               "bonferroni_coverage": BONF_COVERAGE, "conditions": {}}
    for cond in CONDITIONS:
        a = np.array(per_pair[cond], float)
        rec = {"condition": cond, "mean_auc": float(np.nanmean(a))}
        if cond != "clean":
            lo95, hi95 = pair_boot_ci_diff(a, clean, 0.95)
            loB, hiB = pair_boot_ci_diff(a, clean, BONF_COVERAGE)
            v95, b95 = classify(lo95, hi95)
            vB, bB = classify(loB, hiB)
            rec.update({
                "diff_mean": float(np.nanmean(a - clean)),
                "ci95_diff": [lo95, hi95], "verdict_95": v95, "borderline_95": b95,
                "ciBonf_diff": [loB, hiB], "verdict_bonf": vB, "borderline_bonf": bB,
                "verdict_changed": v95 != vB,
            })
        rows.append(rec)
        summary["conditions"][cond] = rec

    (out_dir / f"robustness_sensitivity_{args.subset}.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / f"robustness_sensitivity_{args.subset}.csv", "w", newline="") as f:
        fieldnames = ["condition", "mean_auc", "diff_mean", "ci95_lo", "ci95_hi", "verdict_95",
                      "borderline_95", "ciBonf_lo", "ciBonf_hi", "verdict_bonf", "verdict_changed"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {"condition": r["condition"], "mean_auc": r["mean_auc"]}
            if r["condition"] != "clean":
                row.update({"diff_mean": r["diff_mean"],
                            "ci95_lo": r["ci95_diff"][0], "ci95_hi": r["ci95_diff"][1],
                            "verdict_95": r["verdict_95"], "borderline_95": r["borderline_95"],
                            "ciBonf_lo": r["ciBonf_diff"][0], "ciBonf_hi": r["ciBonf_diff"][1],
                            "verdict_bonf": r["verdict_bonf"], "verdict_changed": r["verdict_changed"]})
            w.writerow(row)

    print("\n" + "=" * 100)
    print(f"SENSITIVITY (subset={args.subset}) — three-way non-inferiority, margin={MARGIN}")
    print("=" * 100)
    print(f"{'condition':22s} {'Δmean':>7s} {'95% CI':>18s} {'verdict95':>14s} {'99.3% CI (Bonf)':>18s} {'verdictBonf':>12s} {'changed?':>9s}")
    for r in rows:
        c = r["condition"]
        if c == "clean":
            print(f"{c:22s} {'—':>7s} {'—':>18s} {'—':>14s} {'—':>18s} {'—':>12s} {'—':>9s}")
        else:
            bl95 = "*" if r["borderline_95"] else ""
            print(f"{c:22s} {r['diff_mean']:+.3f} "
                  f"[{r['ci95_diff'][0]:+.3f},{r['ci95_diff'][1]:+.3f}] "
                  f"{(r['verdict_95']+bl95):>14s} "
                  f"[{r['ciBonf_diff'][0]:+.3f},{r['ciBonf_diff'][1]:+.3f}] "
                  f"{r['verdict_bonf']:>12s} {str(r['verdict_changed']):>9s}")
    print(f"\n[+] Saved to {out_dir}")


if __name__ == "__main__":
    main()
