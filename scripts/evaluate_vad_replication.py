"""Held-out replication of the block-23 VAD-timing ranking signal + 2B bootstrap CI.

PRE-REGISTRATION DECLARATION (written 2026-10-12, before any computation runs):

  PRIMARY: block 23 (`TCN.TCN.23.conv1d`), the 65 NEW pairs (130 instances) that
      were never used in any layer or method investigation (the layer choice and
      the VAD-AUC hypothesis were developed on the original 35 pairs). Per-instance
      AUC(CAM_s, v_s) against the speaker's own Silero reference VAD, one-sample
      Wilcoxon vs. 0.5, alpha = 0.05. Report mean +/- std and a 95% bootstrap CI
      (10,000 resamples) for the mean. Fixed orientation: CAM high => active. No
      flipping. Exact saved utterances. Layer, target (pre-sigmoid VAD logit at
      argmax|vad_logit|), and min-max normalization are unchanged.

  SECONDARY (labelled as such): block 9 on the same 65 new pairs; block 23 on the
      original 35 pairs (the development set, expected ~0.723) for comparison.

Also computes (Step 3): 95% bootstrap CI (10,000 resamples) for the mean
exclusive-activity AUC over eligible instances at block 23, from the corrected
exact-utterance 2B run (results/mae_null_controls/speaker_specificity_2b_block23.csv).
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

SAMPLE_RATE = 16000
TARGET_SECONDS = 3.0
TARGET_LENGTH = int(TARGET_SECONDS * SAMPLE_RATE)
HOP = 256
NUM_FRAMES = 1 + TARGET_LENGTH // HOP
ALPHA = 0.05
N_BOOT = 10000

FULLSCALE_DIR = gradcam_root / "data" / "librispeech_fullscale"
SAMPLES_DIR = gradcam_root / "data" / "librispeech_samples"
HOLDOUT_DIR = gradcam_root / "data" / "librispeech_holdout"
SELECTION_CSV = gradcam_root / "results" / "librispeech_gradcam" / "multi_pair_results.csv"
HOLDOUT_CSV = gradcam_root / "results" / "librispeech_gradcam" / "holdout_results.csv"
EXCL_2B_CSV = gradcam_root / "results" / "mae_null_controls" / "speaker_specificity_2b_block23.csv"

BLOCK_23 = "TCN.TCN.23.conv1d"
BLOCK_9 = "TCN.TCN.9.conv1d"


def normalize_audio(a):
    a = a.astype(np.float32)
    return a / max(np.abs(a).max(), 1e-8) * 0.9


def load_and_pad(path):
    s = load_utterance(path)[:TARGET_LENGTH]
    pad = np.zeros(TARGET_LENGTH, dtype=np.float32)
    pad[: len(s)] = s
    return pad


def minmax(v):
    v = np.asarray(v, dtype=np.float64)
    span = v.max() - v.min()
    return (v - v.min()) / span if span > 1e-12 else np.zeros_like(v)


def compute_cam(model, audio, layer, speaker, frame, device):
    inp = audio.clone().detach().to(device).float().requires_grad_(True)
    gc = GradCAM(model, layer, device)
    try:
        with torch.enable_grad():
            _ = model(inp)
            model.vad_logits[0, speaker, frame].backward()
            act, grad = gc.hook.activations, gc.hook.gradients
            w = grad.mean(dim=tuple(range(2, grad.ndim)), keepdim=True)
            raw = torch.relu((w * act).sum(dim=1)).detach().cpu().numpy()[0]
            return raw
    finally:
        gc.hook.remove_hooks()


def silero_mask(vad_model, clean, num_frames):
    ts = get_speech_timestamps(torch.from_numpy(clean.astype(np.float32)), vad_model,
                               sampling_rate=SAMPLE_RATE, return_seconds=False)
    mask = np.zeros(num_frames, dtype=bool)
    for seg in ts:
        s = seg["start"] // HOP
        e = min(num_frames, seg["end"] // HOP + 1)
        mask[s:e] = True
    return mask


def resolve_existing_pairs():
    pairs = []
    for csv_path, base in ((SELECTION_CSV, SAMPLES_DIR), (HOLDOUT_CSV, HOLDOUT_DIR)):
        for row in csv.DictReader(open(csv_path)):
            pairs.append({"path_0": base / row["utt_0"], "path_1": base / row["utt_1"]})
    return pairs


def resolve_new_pairs():
    by_speaker = {}
    for f in sorted(FULLSCALE_DIR.glob("*.flac")):
        by_speaker.setdefault(f.name.split("-")[0], []).append(f)
    sids = sorted(by_speaker)
    pairs = []
    for i in range(0, min(65 * 2, len(sids) - 1), 2):
        pairs.append({"path_0": by_speaker[sids[i]][0], "path_1": by_speaker[sids[i + 1]][0]})
    return pairs


def wilcoxon_one_sample(values, null=0.5):
    v = np.asarray(values, float)
    d = v - null
    res = stats.wilcoxon(d)
    ranks = stats.rankdata(np.abs(d))
    w_pos = np.sum(ranks[d > 0]); w_neg = np.sum(ranks[d < 0])
    tot = w_pos + w_neg
    return {"w": float(res.statistic), "p": float(res.pvalue),
            "rank_biserial_r": float((w_pos - w_neg) / tot) if tot > 0 else 0.0, "n": int(len(v))}


def bootstrap_ci_mean(values, n_boot=N_BOOT, seed=0):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def eval_pairs(model, vad_model, pairs, layer, device):
    aucs = []
    for p in pairs:
        sig0, sig1 = load_and_pad(p["path_0"]), load_and_pad(p["path_1"])
        mix = normalize_audio(sig0 + sig1)
        audio = torch.from_numpy(mix).unsqueeze(0)
        with torch.no_grad():
            _ = model(audio.to(device))
            vad_logits = model.vad_logits.detach().cpu().numpy()[0]
        v0 = silero_mask(vad_model, sig0, NUM_FRAMES)
        v1 = silero_mask(vad_model, sig1, NUM_FRAMES)
        for speaker in range(2):
            cam_raw = compute_cam(model, audio, layer, speaker,
                                  int(np.argmax(np.abs(vad_logits[speaker]))), device)
            cam = minmax(cam_raw)
            cam = cam[:NUM_FRAMES] if len(cam) >= NUM_FRAMES else np.pad(cam, (0, NUM_FRAMES - len(cam)))
            v_s = v0 if speaker == 0 else v1
            if v_s.sum() > 0 and v_s.sum() < len(v_s):
                aucs.append(float(roc_auc_score(v_s, cam)))
    return np.array(aucs, float)


def main():
    device = "cpu"
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    ckpt = torch.load(gradcam_root / "weights" / "model_with_vad.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    vad_model = load_silero_vad()

    new_pairs = resolve_new_pairs()
    existing_pairs = resolve_existing_pairs()
    print(f"[*] {len(new_pairs)} new (held-out) pairs, {len(existing_pairs)} original pairs")

    out = {"pre_registration": {
        "primary": "block 23, 65 NEW pairs (130 instances), per-instance AUC(CAM_s, v_s) vs own reference VAD, one-sample Wilcoxon vs 0.5, alpha=0.05, 95% bootstrap CI (10000)",
        "secondary": "block 9 same instances; block 23 on original 35 pairs (development set)",
        "declared": "2026-10-12, before computation",
    }, "step3_bootstrap_2b": {}, "replication": {}}

    # ---- Step 3: bootstrap CI for the 2B exclusive-activity AUC (corrected run) ----
    excl = []
    for row in csv.DictReader(open(EXCL_2B_CSV)):
        if row["eligible"] == "True" and row["excl_auc"] not in ("nan", ""):
            excl.append(float(row["excl_auc"]))
    excl = np.array(excl, float)
    lo, hi = bootstrap_ci_mean(excl)
    out["step3_bootstrap_2b"] = {
        "n_eligible": int(len(excl)), "mean": float(excl.mean()), "std": float(excl.std()),
        "ci95": [lo, hi],
    }
    print(f"\n[Step 3] 2B exclusive-AUC block23: mean={excl.mean():.3f} 95% CI [{lo:.3f}, {hi:.3f}] (n={len(excl)})")

    # ---- Step 4: held-out replication ----
    print("\n[Step 4] Held-out replication on 65 new pairs")
    for layer, tag in ((BLOCK_23, "block23_new65_PRIMARY"), (BLOCK_9, "block9_new65_secondary")):
        aucs = eval_pairs(model, vad_model, new_pairs, layer, device)
        lo, hi = bootstrap_ci_mean(aucs)
        w = wilcoxon_one_sample(aucs, 0.5)
        out["replication"][tag] = {
            "n": int(len(aucs)), "mean": float(aucs.mean()), "std": float(aucs.std()),
            "ci95": [lo, hi], "wilcoxon_vs_0.5": w,
        }
        print(f"  {tag}: AUC={aucs.mean():.3f}+-{aucs.std():.3f} 95% CI [{lo:.3f},{hi:.3f}] "
              f"W={w['w']:.1f} p={w['p']:.3e} r={w['rank_biserial_r']:+.3f} (n={len(aucs)})")

    # development-set comparison (original 35 pairs, block 23)
    aucs_dev = eval_pairs(model, vad_model, existing_pairs, BLOCK_23, device)
    lo, hi = bootstrap_ci_mean(aucs_dev)
    w = wilcoxon_one_sample(aucs_dev, 0.5)
    out["replication"]["block23_dev35_secondary"] = {
        "n": int(len(aucs_dev)), "mean": float(aucs_dev.mean()), "std": float(aucs_dev.std()),
        "ci95": [lo, hi], "wilcoxon_vs_0.5": w,
    }
    print(f"  block23_dev35_secondary: AUC={aucs_dev.mean():.3f}+-{aucs_dev.std():.3f} 95% CI [{lo:.3f},{hi:.3f}] "
          f"p={w['p']:.3e} (n={len(aucs_dev)})")

    out_dir = gradcam_root / "results" / "replication"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "replication_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\n[+] Saved to {out_dir / 'replication_summary.json'}")


if __name__ == "__main__":
    main()
