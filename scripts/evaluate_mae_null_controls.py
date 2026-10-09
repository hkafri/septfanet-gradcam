"""Pre-registered null-control and speaker-specificity tests for the MAE
"discriminability" finding.

PRE-REGISTRATION DECLARATION (written 2026-10-09, before any computation runs):

  Primary test 2A: block 9, N=100 pairs. Paired Wilcoxon (two-sided, alpha=0.05)
      of real speaker-vs-speaker MAE vs. the CIRCULAR-SHIFT null (N1, defined
      below) -- the null that preserves each map's value distribution and
      temporal smoothness, breaking only alignment between the two maps.

  Primary test 2B: block 23, eligible instances only. One-sample Wilcoxon of
      EXCLUSIVE-ACTIVITY AUC vs. 0.5 -- restricting to frames where exactly one
      speaker is active, does the CAM rank "speaker s active, other not" frames
      above "other active, speaker s not" frames. Fixed orientation: CAM high =>
      active. No flipping, no best-of-two.

  Everything else (full temporal permutation null N2, constant-map baseline N3,
  block-23 MAE, block-9 own-vs-other) is SECONDARY/EXPLORATORY and labelled as such.
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

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import SpeakerSampler, load_utterance
from gradcam import GradCAM

SAMPLE_RATE = 16000
TARGET_SECONDS = 3.0
TARGET_LENGTH = int(TARGET_SECONDS * SAMPLE_RATE)
HOP = 256
NUM_FRAMES = 1 + TARGET_LENGTH // HOP  # 188
ALPHA = 0.05

FULLSCALE_CSV = gradcam_root / "results" / "fullscale" / "fullscale_results.csv"
SAMPLES_DIR = gradcam_root / "data" / "librispeech_samples"
HOLDOUT_DIR = gradcam_root / "data" / "librispeech_holdout"
FULLSCALE_DIR = gradcam_root / "data" / "librispeech_fullscale"

BLOCK_9 = "TCN.TCN.9.conv1d"
BLOCK_23 = "TCN.TCN.23.conv1d"


def normalize_audio(a):
    a = a.astype(np.float32)
    return a / max(np.abs(a).max(), 1e-8) * 0.9


def load_and_pad(path):
    s = load_utterance(path)[:TARGET_LENGTH]
    pad = np.zeros(TARGET_LENGTH, dtype=np.float32)
    pad[:len(s)] = s
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


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def wilcoxon_paired(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    res = stats.wilcoxon(a, b)
    diffs = a - b
    abs_d = np.abs(diffs)
    ranks = stats.rankdata(abs_d)
    w_pos = np.sum(ranks[diffs > 0])
    w_neg = np.sum(ranks[diffs < 0])
    tot = w_pos + w_neg
    r = float((w_pos - w_neg) / tot) if tot > 0 else 0.0
    return {"w": float(res.statistic), "p": float(res.pvalue), "rank_biserial_r": r, "n": int(len(a))}


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
    return {"w": float(res.statistic), "p": float(res.pvalue), "rank_biserial_r": r, "n": int(len(v))}


def load_fullscale_pairs():
    """Reconstruct the exact 100 full-scale pairs from the saved results CSV.

    The CSV stores origin (existing/new) and speaker IDs. Existing pairs' audio
    lives in data/librispeech_samples (selection) or data/librispeech_holdout
    (held-out); new pairs come from data/librispeech_fullscale. Since the CSV
    does not store exact utterance filenames, we resolve each speaker's first
    matching .flac by speaker-ID prefix in the appropriate directory.
    """
    pairs = []

    def first_utt(spk, dirs):
        for d in dirs:
            fs = sorted(d.glob(f"{spk}-*.flac"))
            if fs:
                return fs[0]
        return None

    for row in csv.DictReader(open(FULLSCALE_CSV)):
        origin = row["origin"]
        if origin == "existing":
            dirs = (SAMPLES_DIR, HOLDOUT_DIR)
        else:
            dirs = (FULLSCALE_DIR,)
        p0 = first_utt(row["speaker_0"], dirs)
        p1 = first_utt(row["speaker_1"], dirs)
        if p0 is not None and p1 is not None:
            pairs.append({"path_0": p0, "path_1": p1, "origin": origin})
        else:
            raise FileNotFoundError(
                f"Could not resolve audio for pair {row.get('pair_index')}: "
                f"speakers {row['speaker_0']}/{row['speaker_1']} in {[str(d) for d in dirs]}")
    return pairs


def main():
    device = "cpu"
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    ckpt = torch.load(gradcam_root / "weights" / "model_with_vad.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    vad_model = load_silero_vad()

    pairs = load_fullscale_pairs()
    print(f"[*] Loaded {len(pairs)} full-scale pairs from {FULLSCALE_CSV.name}")

    out_dir = gradcam_root / "results" / "mae_null_controls"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 2A: MAE null controls (block 9 primary, block 23 secondary) ----
    print("\n[2A] MAE null controls (real vs circular-shift / permutation / constant / uniform)")
    records_2a = {BLOCK_9: [], BLOCK_23: []}
    for i, p in enumerate(pairs):
        sig0, sig1 = load_and_pad(p["path_0"]), load_and_pad(p["path_1"])
        mix = normalize_audio(sig0 + sig1)
        audio = torch.from_numpy(mix).unsqueeze(0)
        with torch.no_grad():
            _ = model(audio.to(device))
            vad_logits = model.vad_logits.detach().cpu().numpy()[0]
        for layer in (BLOCK_9, BLOCK_23):
            cams = {}
            for speaker in range(2):
                frame = int(np.argmax(np.abs(vad_logits[speaker])))
                raw = compute_cam(model, audio, layer, speaker, frame, device)
                cams[speaker] = minmax(raw)
            c0, c1 = cams[0], cams[1]
            c0 = c0[:NUM_FRAMES] if len(c0) >= NUM_FRAMES else np.pad(c0, (0, NUM_FRAMES - len(c0)))
            c1 = c1[:NUM_FRAMES] if len(c1) >= NUM_FRAMES else np.pad(c1, (0, NUM_FRAMES - len(c1)))

            real = mae(c0, c1)
            # N1 circular shift null
            rng = np.random.default_rng(3000 + i)
            ks = rng.integers(10, NUM_FRAMES - 10, size=200)
            n1 = float(np.mean([mae(c0, np.roll(c1, int(k))) for k in ks]))
            # N2 full permutation null
            n2 = float(np.mean([mae(c0, c1[rng.permutation(NUM_FRAMES)]) for _ in range(200)]))
            # N3 constant-map baseline
            n3 = 0.5 * (mae(c0, np.median(c1)) + mae(c1, np.median(c0)))
            # uniform-noise control (reference)
            noise = rng.random(NUM_FRAMES)
            uni = mae(c0, noise)
            records_2a[layer].append({"pair_index": i + 1, "real": real, "n1_shift": n1,
                                      "n2_perm": n2, "n3_const": n3, "uniform": uni})
        if (i + 1) % 10 == 0:
            print(f"  2A: {i+1}/{len(pairs)} pairs done")

    # ---- 2B: speaker-specificity (block 23 primary, block 9 secondary) ----
    print("\n[2B] Speaker-specificity: exclusive-activity AUC (block 23 primary) + own-vs-other")
    records_2b = {BLOCK_23: [], BLOCK_9: []}
    for i, p in enumerate(pairs):
        sig0, sig1 = load_and_pad(p["path_0"]), load_and_pad(p["path_1"])
        mix = normalize_audio(sig0 + sig1)
        audio = torch.from_numpy(mix).unsqueeze(0)
        with torch.no_grad():
            _ = model(audio.to(device))
            vad_logits = model.vad_logits.detach().cpu().numpy()[0]
        v0 = silero_mask(vad_model, sig0, NUM_FRAMES)
        v1 = silero_mask(vad_model, sig1, NUM_FRAMES)
        agree = float(np.mean(v0 == v1))
        for layer in (BLOCK_23, BLOCK_9):
            for speaker in range(2):
                cam_raw = compute_cam(model, audio, layer, speaker,
                                      int(np.argmax(np.abs(vad_logits[speaker]))), device)
                cam = minmax(cam_raw)
                cam = cam[:NUM_FRAMES] if len(cam) >= NUM_FRAMES else np.pad(cam, (0, NUM_FRAMES - len(cam)))
                v_s = v0 if speaker == 0 else v1
                v_o = v1 if speaker == 0 else v0
                # exclusive-activity AUC
                E_s = np.nonzero(v_s & ~v_o)[0]
                E_o = np.nonzero(~v_s & v_o)[0]
                if len(E_s) >= 10 and len(E_o) >= 10:
                    scores = cam[np.concatenate([E_s, E_o])]
                    labels = np.concatenate([np.ones(len(E_s)), np.zeros(len(E_o))])
                    excl_auc = float(roc_auc_score(labels, scores))
                    eligible = True
                else:
                    excl_auc = float("nan")
                    eligible = False
                # own-vs-other
                auc_own = float(roc_auc_score(v_s, cam)) if (v_s.sum() > 0 and v_s.sum() < len(v_s)) else float("nan")
                auc_other = float(roc_auc_score(v_o, cam)) if (v_o.sum() > 0 and v_o.sum() < len(v_o)) else float("nan")
                records_2b[layer].append({
                    "pair_index": i + 1, "speaker": speaker, "eligible": eligible,
                    "excl_auc": excl_auc, "auc_own": auc_own, "auc_other": auc_other,
                    "frame_agreement": agree,
                })
        if (i + 1) % 10 == 0:
            print(f"  2B: {i+1}/{len(pairs)} pairs done")

    # ---- aggregate ----
    out = {"pre_registration": {
        "primary_2a": "block 9, N=100, paired Wilcoxon real MAE vs circular-shift null, alpha=0.05, two-sided",
        "primary_2b": "block 23, eligible instances, one-sample Wilcoxon exclusive-activity AUC vs 0.5, alpha=0.05",
        "declared": "2026-10-09, before computation",
    }, "2a": {}, "2b": {}}

    print("\n" + "=" * 90)
    print("2A — MAE null controls")
    print("=" * 90)
    for layer in (BLOCK_9, BLOCK_23):
        recs = records_2a[layer]
        role = "PRIMARY" if layer == BLOCK_9 else "secondary"
        real = [r["real"] for r in recs]
        for null_name in ("n1_shift", "n2_perm", "n3_const", "uniform"):
            null_vals = [r[null_name] for r in recs]
            w = wilcoxon_paired(real, null_vals)
            frac_below = float(np.mean(np.array(real) < np.array(null_vals)))
            out["2a"].setdefault(layer, {})[null_name] = {
                "role": role, "real_mean": float(np.mean(real)), "real_std": float(np.std(real)),
                "null_mean": float(np.mean(null_vals)), "null_std": float(np.std(null_vals)),
                "wilcoxon": w, "frac_real_below_null": frac_below,
            }
            print(f"{layer} [{role}] vs {null_name}: real={np.mean(real):.4f}±{np.std(real):.4f} "
                  f"null={np.mean(null_vals):.4f}±{np.std(null_vals):.4f} "
                  f"W={w['w']:.1f} p={w['p']:.2e} r={w['rank_biserial_r']:+.3f} frac_real<null={frac_below:.3f}")

    print("\n" + "=" * 90)
    print("2B — Speaker-specificity")
    print("=" * 90)
    for layer in (BLOCK_23, BLOCK_9):
        recs = records_2b[layer]
        role = "PRIMARY" if layer == BLOCK_23 else "secondary"
        elig = [r for r in recs if r["eligible"]]
        excl = [r["excl_auc"] for r in elig]
        w = wilcoxon_one_sample(excl, 0.5)
        agree = np.mean([r["frame_agreement"] for r in recs])
        # own-vs-other (secondary)
        own = np.array([r["auc_own"] for r in recs if not np.isnan(r["auc_own"])], dtype=float)
        oth = np.array([r["auc_other"] for r in recs if not np.isnan(r["auc_other"])], dtype=float)
        d_auc = own - oth
        w2 = wilcoxon_one_sample(d_auc, 0.0) if len(d_auc) > 1 else None
        out["2b"][layer] = {
            "role": role, "n_eligible": len(elig), "n_total": len(recs),
            "excl_auc_mean": float(np.mean(excl)) if excl else float("nan"),
            "excl_auc_std": float(np.std(excl)) if excl else float("nan"),
            "excl_auc_wilcoxon_vs_0.5": w,
            "mean_frame_agreement": float(agree),
            "own_vs_other_delta_auc_mean": float(np.mean(d_auc)) if len(d_auc) else float("nan"),
            "own_vs_other_wilcoxon_vs_0": w2,
        }
        print(f"{layer} [{role}]: eligible={len(elig)}/{len(recs)} | "
              f"excl-AUC={np.mean(excl):.3f}±{np.std(excl):.3f} vs 0.5: W={w['w']:.1f} p={w['p']:.2e} r={w['rank_biserial_r']:+.3f}")
        if w2:
            print(f"    own-vs-other ΔAUC={np.mean(d_auc):+.4f} (W={w2['w']:.1f} p={w2['p']:.2e} r={w2['rank_biserial_r']:+.3f}) | "
                  f"frame-agreement={agree:.3f}")

    (out_dir / "mae_null_controls_summary.json").write_text(json.dumps(out, indent=2))

    for layer, recs in records_2a.items():
        tag = "block9" if layer == BLOCK_9 else "block23"
        with open(out_dir / f"mae_null_controls_2a_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader(); w.writerows(recs)
    for layer, recs in records_2b.items():
        tag = "block23" if layer == BLOCK_23 else "block9"
        with open(out_dir / f"speaker_specificity_2b_{tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader(); w.writerows(recs)

    print(f"\n[+] Saved to {out_dir}")


if __name__ == "__main__":
    main()
