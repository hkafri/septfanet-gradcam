"""Pair-level clustered re-analysis of the block-23 / block-9 VAD-AUC statistics and
the 2B exclusive-activity AUC.

The per-instance tests treat the two speakers of a pair as independent, but they
share a mixture and are correlated. This script clusters at the PAIR level:

  (1) average the two per-instance AUCs within each pair, run a one-sample
      Wilcoxon vs. 0.5 across PAIRS (not instances), and report p + effect size;
  (2) compute the 95% CI with a bootstrap that resamples PAIRS, not instances.

Sources (all from the exact-saved-utterance run, results/mae_null_controls/):
  - union_auc_check.csv     : per-instance AUC_own / AUC_union for block 23
                              (exact pairs, dev 35 = pair_index 1-35, held-out 65 = 36-100)
  - speaker_specificity_2b_block23.csv : per-instance exclusive-activity AUC
                              (block 23, eligible flag), for the 2B pair-level test

Note: block-9 per-instance AUCs vs own reference were not separately saved in the
exact-utterance run; the pair-level block-9 VAD result is taken from the held-out
replication script's per-instance output (recomputed below from the same exact
pairs). The 2B exclusive-activity AUC is block-23 only (its pre-registered layer).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from sklearn.metrics import roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import load_utterance
from gradcam import GradCAM

RES = gradcam_root / "results" / "mae_null_controls"
FULLSCALE_DIR = gradcam_root / "data" / "librispeech_fullscale"
SAMPLES_DIR = gradcam_root / "data" / "librispeech_samples"
HOLDOUT_DIR = gradcam_root / "data" / "librispeech_holdout"
SELECTION_CSV = gradcam_root / "results" / "librispeech_gradcam" / "multi_pair_results.csv"
HOLDOUT_CSV = gradcam_root / "results" / "librispeech_gradcam" / "holdout_results.csv"
SAMPLE_RATE = 16000
TARGET_LENGTH = int(3.0 * SAMPLE_RATE)
HOP = 256
NUM_FRAMES = 1 + TARGET_LENGTH // HOP
BLOCK_9 = "TCN.TCN.9.conv1d"
BLOCK_23 = "TCN.TCN.23.conv1d"
N_BOOT = 10000


def wilcoxon_one_sample(values, null=0.5):
    v = np.asarray(values, float)
    d = v - null
    res = stats.wilcoxon(d)
    ranks = stats.rankdata(np.abs(d))
    w_pos = np.sum(ranks[d > 0]); w_neg = np.sum(ranks[d < 0])
    tot = w_pos + w_neg
    return {"w": float(res.statistic), "p": float(res.pvalue),
            "rank_biserial_r": float((w_pos - w_neg) / tot) if tot > 0 else 0.0,
            "n": int(len(v))}


def pair_cluster(rows, value_key):
    """Average the two speakers' per-instance values within each pair.
    Returns {pair_index: mean_value} for pairs where at least one value is finite."""
    by_pair = {}
    for r in rows:
        v = r[value_key]
        if v in ("nan", "", None):
            continue
        by_pair.setdefault(int(r["pair_index"]), []).append(float(v))
    return {k: float(np.mean(vs)) for k, vs in by_pair.items() if len(vs) > 0}


def bootstrap_ci_pairs(pair_means, n_boot=N_BOOT, seed=0):
    """Bootstrap CI for the mean, resampling PAIRS (keys) with replacement."""
    keys = np.array(sorted(pair_means.keys()))
    vals = np.array([pair_means[k] for k in keys], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    boots = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def report(name, pair_means):
    pm = np.array([pair_means[k] for k in sorted(pair_means)], float)
    w = wilcoxon_one_sample(pm, 0.5)
    lo, hi = bootstrap_ci_pairs(pair_means)
    return {
        "name": name, "n_pairs": int(len(pm)),
        "pair_mean_of_mean_auc": float(pm.mean()), "pair_std": float(pm.std()),
        "ci95_pair_bootstrap": [lo, hi],
        "wilcoxon_vs_0.5_pairs": w,
    }


def main():
    union_rows = list(csv.DictReader(open(RES / "union_auc_check.csv")))
    excl_rows = list(csv.DictReader(open(RES / "speaker_specificity_2b_block23.csv")))

    # split dev (pair_index 1-35) vs held-out (36-100)
    def split(rows):
        dev = [r for r in rows if int(r["pair_index"]) <= 35]
        hold = [r for r in rows if int(r["pair_index"]) > 35]
        return dev, hold

    out = {"note": "pair-level clustered analysis; pairs resampled, not instances",
           "n_boot": N_BOOT, "results": {}}

    # --- block 23 own-reference VAD AUC (from union_auc_check, exact pairs) ---
    dev_u, hold_u = split(union_rows)
    out["results"]["block23_dev35"] = report("block23 own-VAD AUC, dev 35 pairs",
                                             pair_cluster(dev_u, "auc_own"))
    out["results"]["block23_hold65"] = report("block23 own-VAD AUC, held-out 65 pairs",
                                              pair_cluster(hold_u, "auc_own"))

    # --- 2B exclusive-activity AUC (block 23), pair-level where BOTH speakers eligible ---
    elig = [r for r in excl_rows if r["eligible"] == "True" and r["excl_auc"] not in ("nan", "")]
    by_pair = {}
    for r in elig:
        by_pair.setdefault(int(r["pair_index"]), []).append(float(r["excl_auc"]))
    both_elig = {k: float(np.mean(v)) for k, v in by_pair.items() if len(v) == 2}
    # fall back to any eligible instance in the pair if no pair has both eligible
    if len(both_elig) < 5:
        both_elig = {k: float(np.mean(v)) for k, v in by_pair.items()}
        tag = "block23 exclusive-activity AUC, pairs with >=1 eligible speaker"
    else:
        tag = "block23 exclusive-activity AUC, pairs with BOTH speakers eligible"
    out["results"]["block23_excl_pair"] = report(tag, both_elig)

    # --- block 9 own-reference VAD AUC: recompute per-instance from exact pairs ---
    print("[*] Recomputing block-9 per-instance AUCs from exact pairs (model inference)")
    device = "cpu"
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    ckpt = torch.load(gradcam_root / "weights" / "model_with_vad.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    vad_model = load_silero_vad()

    def resolve_pairs():
        pairs = []
        for cp, base in ((SELECTION_CSV, SAMPLES_DIR), (HOLDOUT_CSV, HOLDOUT_DIR)):
            for row in csv.DictReader(open(cp)):
                pairs.append({"path_0": base / row["utt_0"], "path_1": base / row["utt_1"]})
        by = {}
        for f in sorted(FULLSCALE_DIR.glob("*.flac")):
            by.setdefault(f.name.split("-")[0], []).append(f)
        sids = sorted(by)
        for i in range(0, min(65 * 2, len(sids) - 1), 2):
            pairs.append({"path_0": by[sids[i]][0], "path_1": by[sids[i + 1]][0]})
        return pairs

    def silero_mask(clean):
        ts = get_speech_timestamps(torch.from_numpy(clean.astype(np.float32)), vad_model,
                                   sampling_rate=SAMPLE_RATE, return_seconds=False)
        m = np.zeros(NUM_FRAMES, dtype=bool)
        for seg in ts:
            s = seg["start"] // HOP; e = min(NUM_FRAMES, seg["end"] // HOP + 1)
            m[s:e] = True
        return m

    def compute_cam(audio, speaker, frame):
        inp = audio.clone().detach().to(device).float().requires_grad_(True)
        gc = GradCAM(model, BLOCK_9, device)
        try:
            with torch.enable_grad():
                _ = model(inp)
                model.vad_logits[0, speaker, frame].backward()
                act, grad = gc.hook.activations, gc.hook.gradients
                w = grad.mean(dim=tuple(range(2, grad.ndim)), keepdim=True)
                return torch.relu((w * act).sum(dim=1)).detach().cpu().numpy()[0]
        finally:
            gc.hook.remove_hooks()

    def minmax(v):
        v = np.asarray(v, np.float64); sp = v.max() - v.min()
        return (v - v.min()) / sp if sp > 1e-12 else np.zeros_like(v)

    pairs = resolve_pairs()
    b9_rows = []
    for i, p in enumerate(pairs):
        s0 = load_utterance(p["path_0"])[:TARGET_LENGTH]
        s1 = load_utterance(p["path_1"])[:TARGET_LENGTH]
        sig0 = np.zeros(TARGET_LENGTH, np.float32); sig0[:len(s0)] = s0
        sig1 = np.zeros(TARGET_LENGTH, np.float32); sig1[:len(s1)] = s1
        mix = (sig0 + sig1).astype(np.float32)
        mix = mix / max(np.abs(mix).max(), 1e-8) * 0.9
        audio = torch.from_numpy(mix).unsqueeze(0)
        with torch.no_grad():
            _ = model(audio.to(device))
            vl = model.vad_logits.detach().cpu().numpy()[0]
        v0, v1 = silero_mask(sig0), silero_mask(sig1)
        for speaker in range(2):
            cam = minmax(compute_cam(audio, speaker, int(np.argmax(np.abs(vl[speaker])))))
            cam = cam[:NUM_FRAMES] if len(cam) >= NUM_FRAMES else np.pad(cam, (0, NUM_FRAMES - len(cam)))
            v_s = v0 if speaker == 0 else v1
            if v_s.sum() > 0 and v_s.sum() < len(v_s):
                b9_rows.append({"pair_index": i + 1, "speaker": speaker,
                                "auc_own": float(roc_auc_score(v_s, cam))})
        if (i + 1) % 20 == 0:
            print(f"    block9: {i+1}/{len(pairs)} pairs")

    dev_b9 = [r for r in b9_rows if r["pair_index"] <= 35]
    hold_b9 = [r for r in b9_rows if r["pair_index"] > 35]
    out["results"]["block9_dev35"] = report("block9 own-VAD AUC, dev 35 pairs", pair_cluster(dev_b9, "auc_own"))
    out["results"]["block9_hold65"] = report("block9 own-VAD AUC, held-out 65 pairs", pair_cluster(hold_b9, "auc_own"))

    out_dir = gradcam_root / "results" / "pair_level"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pair_level_summary.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 84)
    print("PAIR-LEVEL CLUSTERED ANALYSIS (pairs resampled, not instances)")
    print("=" * 84)
    for k, r in out["results"].items():
        w = r["wilcoxon_vs_0.5_pairs"]
        print(f"{r['name']}")
        print(f"    n_pairs={r['n_pairs']}  pair-mean AUC={r['pair_mean_of_mean_auc']:.4f} ± {r['pair_std']:.4f}")
        print(f"    95% CI (pair bootstrap) [{r['ci95_pair_bootstrap'][0]:.4f}, {r['ci95_pair_bootstrap'][1]:.4f}]")
        print(f"    Wilcoxon vs 0.5 across pairs: W={w['w']:.1f} p={w['p']:.3e} r={w['rank_biserial_r']:+.3f} (n={w['n']})")
    print(f"\n[+] Saved to {out_dir / 'pair_level_summary.json'}")


if __name__ == "__main__":
    main()
