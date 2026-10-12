"""Exploratory union-AUC check (Step 3) + 2B eligible-frame distribution + power.

LABEL: EXPLORATORY (not pre-registered). Added 2026-10-12 after the pre-registered
2A/2B results were seen, to disambiguate what the validated block-23 VAD ranking
signal (AUC 0.723) is about.

Question: does the block-23 VAD-logit CAM track the TARGET speaker's activity, or
the OVERALL speech activity of the mixture? Compute, per instance:
    AUC_own  = AUC(CAM_s, v_s)            (speaker s active vs not)
    AUC_union = AUC(CAM_s, v_s OR v_o)    (either speaker active vs neither)
If AUC_union >= AUC_own, the map tracks overall mixture activity rather than the
target speaker specifically. Paired Wilcoxon on (AUC_union - AUC_own).

Also reports (Step 2 leftovers): the distribution of |E_s| / |E_o| exclusive-activity
frame counts per instance, and a power analysis for the 2B exclusive-activity test.

Pair reconstruction is EXACT (same utterances as the full-scale run), unlike the
first null-control pass which resolved speakers to their alphabetically-first
utterance (that caused the 0.233 -> 0.243 baseline shift).
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
NUM_FRAMES = 1 + TARGET_LENGTH // HOP  # 188

SAMPLES_DIR = gradcam_root / "data" / "librispeech_samples"
HOLDOUT_DIR = gradcam_root / "data" / "librispeech_holdout"
FULLSCALE_DIR = gradcam_root / "data" / "librispeech_fullscale"
SELECTION_CSV = gradcam_root / "results" / "librispeech_gradcam" / "multi_pair_results.csv"
HOLDOUT_CSV = gradcam_root / "results" / "librispeech_gradcam" / "holdout_results.csv"

BLOCK_23 = "TCN.TCN.23.conv1d"


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


def resolve_exact_pairs():
    """Rebuild the exact 100 full-scale pairs (same utterances as the original run)."""
    pairs = []
    for csv_path, base in ((SELECTION_CSV, SAMPLES_DIR), (HOLDOUT_CSV, HOLDOUT_DIR)):
        for row in csv.DictReader(open(csv_path)):
            pairs.append({"path_0": base / row["utt_0"], "path_1": base / row["utt_1"]})
    num_new = 100 - len(pairs)
    by_speaker = {}
    for f in sorted(FULLSCALE_DIR.glob("*.flac")):
        by_speaker.setdefault(f.name.split("-")[0], []).append(f)
    sids = sorted(by_speaker)
    for i in range(0, min(num_new * 2, len(sids) - 1), 2):
        pairs.append({"path_0": by_speaker[sids[i]][0], "path_1": by_speaker[sids[i + 1]][0]})
    return pairs


def wilcoxon_paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = stats.wilcoxon(a, b)
    d = a - b
    ranks = stats.rankdata(np.abs(d))
    w_pos = np.sum(ranks[d > 0]); w_neg = np.sum(ranks[d < 0])
    tot = w_pos + w_neg
    return {"w": float(res.statistic), "p": float(res.pvalue),
            "rank_biserial_r": float((w_pos - w_neg) / tot) if tot > 0 else 0.0, "n": int(len(a))}


def auc(scores, labels):
    pos = int(np.sum(labels)); n = len(labels)
    if pos == 0 or pos == n:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def main():
    device = "cpu"
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    ckpt = torch.load(gradcam_root / "weights" / "model_with_vad.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    vad_model = load_silero_vad()

    pairs = resolve_exact_pairs()
    print(f"[*] Reconstructed {len(pairs)} exact pairs")

    records = []
    e_s_counts, e_o_counts = [], []
    for i, p in enumerate(pairs):
        sig0, sig1 = load_and_pad(p["path_0"]), load_and_pad(p["path_1"])
        mix = normalize_audio(sig0 + sig1)
        audio = torch.from_numpy(mix).unsqueeze(0)
        with torch.no_grad():
            _ = model(audio.to(device))
            vad_logits = model.vad_logits.detach().cpu().numpy()[0]
        v0 = silero_mask(vad_model, sig0, NUM_FRAMES)
        v1 = silero_mask(vad_model, sig1, NUM_FRAMES)
        v_union = v0 | v1
        for speaker in range(2):
            cam_raw = compute_cam(model, audio, BLOCK_23, speaker,
                                  int(np.argmax(np.abs(vad_logits[speaker]))), device)
            cam = minmax(cam_raw)
            cam = cam[:NUM_FRAMES] if len(cam) >= NUM_FRAMES else np.pad(cam, (0, NUM_FRAMES - len(cam)))
            v_s = v0 if speaker == 0 else v1
            v_o = v1 if speaker == 0 else v0
            E_s = int(np.sum(v_s & ~v_o)); E_o = int(np.sum(~v_s & v_o))
            e_s_counts.append(E_s); e_o_counts.append(E_o)
            records.append({
                "pair_index": i + 1, "speaker": speaker,
                "auc_own": auc(cam, v_s),
                "auc_union": auc(cam, v_union),
                "prev_own": float(v_s.mean()),
                "prev_union": float(v_union.mean()),
                "E_s": E_s, "E_o": E_o,
                "eligible": (E_s >= 10 and E_o >= 10),
            })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(pairs)} pairs done")

    own = np.array([r["auc_own"] for r in records], float)
    union = np.array([r["auc_union"] for r in records], float)
    valid = ~(np.isnan(own) | np.isnan(union))
    own_v, union_v = own[valid], union[valid]
    w = wilcoxon_paired(union_v, own_v)
    frac_union_ge = float(np.mean(union_v >= own_v))

    n_elig = int(sum(r["eligible"] for r in records))
    e_s_arr = np.array(e_s_counts); e_o_arr = np.array(e_o_counts)

    # Power analysis for the 2B exclusive-activity one-sample Wilcoxon/t (H0: AUC=0.5)
    # observed effect
    elig_recs = [r for r in records if r["eligible"]]
    # note: excl_auc recomputed in the prior script; here we report the standardized
    # effect detectable at n=n_elig with 80% power using a normal approximation for
    # the one-sample signed-rank / t (dz), then convert to AUC units using the
    # observed SD of exclusive AUC (0.310 from the pre-registered run).
    from scipy.stats import norm
    alpha, power = 0.05, 0.80
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    dz = z / np.sqrt(n_elig)  # detectable standardized effect (t)
    sd_excl = 0.310  # observed SD of exclusive-activity AUC (pre-registered run, block 23)
    detectable_auc_shift = dz * sd_excl

    out = {
        "label": "EXPLORATORY (not pre-registered), 2026-10-12",
        "block": BLOCK_23,
        "n_instances": len(records),
        "union_vs_own": {
            "auc_own_mean": float(np.nanmean(own)), "auc_own_std": float(np.nanstd(own)),
            "auc_union_mean": float(np.nanmean(union)), "auc_union_std": float(np.nanstd(union)),
            "delta_union_minus_own_mean": float(np.nanmean(union_v - own_v)),
            "wilcoxon_union_vs_own": w,
            "frac_union_ge_own": frac_union_ge,
            "n_valid": int(valid.sum()),
        },
        "prevalence": {
            "own_mean": float(np.mean([r["prev_own"] for r in records])),
            "union_mean": float(np.mean([r["prev_union"] for r in records])),
        },
        "eligible_frames": {
            "n_eligible_instances": n_elig,
            "E_s_mean": float(e_s_arr.mean()), "E_s_median": float(np.median(e_s_arr)),
            "E_s_min": int(e_s_arr.min()), "E_s_max": int(e_s_arr.max()),
            "E_o_mean": float(e_o_arr.mean()), "E_o_median": float(np.median(e_o_arr)),
            "E_o_min": int(e_o_arr.min()), "E_o_max": int(e_o_arr.max()),
        },
        "power_2b": {
            "n_eligible": n_elig,
            "dz_80pct": float(dz),
            "sd_excl_auc": sd_excl,
            "detectable_auc_shift_80pct": float(detectable_auc_shift),
            "detectable_auc_80pct": float(0.5 + detectable_auc_shift),
        },
    }

    out_dir = gradcam_root / "results" / "mae_null_controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "union_auc_check.json").write_text(json.dumps(out, indent=2))
    with open(out_dir / "union_auc_check.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        wr.writeheader(); wr.writerows(records)

    print("\n" + "=" * 78)
    print("STEP 3 (EXPLORATORY) — union-AUC check, block 23")
    print("=" * 78)
    print(f"AUC_own   = {out['union_vs_own']['auc_own_mean']:.4f} ± {out['union_vs_own']['auc_own_std']:.4f} "
          f"(prevalence {out['prevalence']['own_mean']:.3f})")
    print(f"AUC_union = {out['union_vs_own']['auc_union_mean']:.4f} ± {out['union_vs_own']['auc_union_std']:.4f} "
          f"(prevalence {out['prevalence']['union_mean']:.3f})")
    print(f"Δ(union−own) = {out['union_vs_own']['delta_union_minus_own_mean']:+.4f} | "
          f"frac(union>=own)={frac_union_ge:.3f} | Wilcoxon p={w['p']:.3e} r={w['rank_biserial_r']:+.3f} (n={w['n']})")
    print("\nSTEP 2 leftover — 2B eligible frames")
    print(f"eligible instances: {n_elig}/{len(records)}")
    print(f"|E_s| mean {e_s_arr.mean():.1f} median {np.median(e_s_arr):.0f} min {e_s_arr.min()} max {e_s_arr.max()}")
    print(f"|E_o| mean {e_o_arr.mean():.1f} median {np.median(e_o_arr):.0f} min {e_o_arr.min()} max {e_o_arr.max()}")
    print("\nSTEP 4 — 2B power")
    print(f"n_eligible = {n_elig}; 80%-power detectable standardized effect dz={dz:.3f}")
    print(f"=> detectable exclusive-AUC shift ≈ {detectable_auc_shift:.3f} (i.e. AUC ≥ {0.5+detectable_auc_shift:.3f})")
    print(f"\n[+] Saved to {out_dir}")


if __name__ == "__main__":
    main()
