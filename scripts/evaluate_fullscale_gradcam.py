"""Full-scale (N>=100) Grad-CAM validation with gender/pitch subgroup breakdown.

Builds on the existing 35 pairs (20 selection + 15 held-out) rather than
replacing them: those exact pairs are re-loaded from their saved CSVs (same
speakers, same utterances) and extended with new train-clean-100 pairs to reach
the target pair count. For every speaker it joins gender (from SPEAKERS.TXT)
and computes mean F0 (librosa.pyin), then reports MAE/Wilcoxon statistics
overall and split by gender and by pitch tertile.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import scipy.stats as stats
import soundfile as sf
import torch

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import load_utterance, utterance_duration_s
from data.speaker_meta import load_speaker_genders
from gradcam import GradCAM

SAMPLE_RATE = 16000
TARGET_SECONDS = 3.0

SAMPLES_DIR = gradcam_root / "data" / "librispeech_samples"
HOLDOUT_DIR = gradcam_root / "data" / "librispeech_holdout"
FULLSCALE_DIR = gradcam_root / "data" / "librispeech_fullscale"
SELECTION_CSV = gradcam_root / "results" / "librispeech_gradcam" / "multi_pair_results.csv"
HOLDOUT_CSV = gradcam_root / "results" / "librispeech_gradcam" / "holdout_results.csv"


def load_selected_layer():
    p = gradcam_root / "results" / "layer_selection" / "layer_scores.json"
    if p.exists():
        return json.loads(p.read_text()).get("winning_layer", "TCN.TCN.9.conv1d")
    return "TCN.TCN.9.conv1d"


def normalize_audio(audio):
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    return audio / max(peak, 1e-8) * 0.9


def load_source(path):
    target_length = int(TARGET_SECONDS * SAMPLE_RATE)
    signal = load_utterance(path)[:target_length]
    padded = np.zeros(target_length, dtype=np.float32)
    padded[: len(signal)] = signal
    return padded


def prepare_mixture(paths):
    signals = [load_source(p) for p in paths]
    mixture = normalize_audio(signals[0] + signals[1])
    return torch.from_numpy(mixture).unsqueeze(0)


def compute_cam(model, audio, target_layer, target_kind, speaker, index, device):
    input_audio = audio.clone().detach().to(device).float().requires_grad_(True)
    gradcam = GradCAM(model, target_layer, device)
    try:
        with torch.enable_grad():
            output = model(input_audio)
            separated = output[0]
            if target_kind == "vad_logit":
                target = model.vad_logits[0, speaker, index]
            else:
                target = separated[0, speaker, index].abs()
            target.backward()
            activations = gradcam.hook.activations
            gradients = gradcam.hook.gradients
            if activations is None or gradients is None:
                raise RuntimeError("Grad-CAM hook did not capture activations and gradients")
            weights = gradients.mean(dim=tuple(range(2, gradients.ndim)), keepdim=True)
            raw = torch.relu((weights * activations).sum(dim=1)).detach().cpu().numpy()[0]
            if not np.isfinite(raw).all() or raw.max() <= 0 or np.ptp(raw) <= 1e-12:
                raise RuntimeError("CAM is zero, non-finite, or constant")
            return raw
    finally:
        gradcam.hook.remove_hooks()


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values)
    span = values.max() - values.min()
    if span > 1e-12:
        result = (values - values.min()) / span
    return result


def pair_mae(left, right, pair_idx, seed_offset):
    ind_left, ind_right = minmax(left), minmax(right)
    rng = np.random.default_rng(seed=seed_offset + pair_idx)
    random_map = rng.random(left.shape)
    return {
        "vad_real_mae": float(np.mean(np.abs(ind_left - ind_right))),
        "vad_random_mae": float(np.mean(np.abs(ind_left - random_map))),
    }


def compute_wilcoxon(real, rand):
    real, rand = np.array(real), np.array(rand)
    res = stats.wilcoxon(real, rand)
    diffs = rand - real
    ranks = stats.rankdata(np.abs(diffs))
    w_pos, w_neg = np.sum(ranks[diffs > 0]), np.sum(ranks[diffs < 0])
    tot = w_pos + w_neg
    r_rb = float((w_pos - w_neg) / tot) if tot > 0 else 0.0
    return {
        "n": len(real),
        "real_mean": float(np.mean(real)),
        "real_std": float(np.std(real)),
        "rand_mean": float(np.mean(rand)),
        "rand_std": float(np.std(rand)),
        "w": float(res.statistic),
        "p": float(res.pvalue),
        "rank_biserial_r": r_rb,
    }


def mean_f0(path):
    """Mean voiced F0 (Hz) of an utterance via librosa.pyin."""
    y = load_utterance(path)
    f0, voiced, _ = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=SAMPLE_RATE,
        frame_length=2048,
    )
    voiced_f0 = f0[~np.isnan(f0)]
    return float(np.mean(voiced_f0)) if len(voiced_f0) else float("nan")


def resolve_existing_pairs():
    """Rebuild the exact 35 existing pairs (same speakers + utterances) from CSVs."""
    pairs = []
    for csv_path, base in ((SELECTION_CSV, SAMPLES_DIR), (HOLDOUT_CSV, HOLDOUT_DIR)):
        for row in csv.DictReader(open(csv_path)):
            p0 = base / row["utt_0"]
            p1 = base / row["utt_1"]
            if not p0.exists() or not p1.exists():
                raise FileNotFoundError(f"Missing audio for existing pair: {p0} / {p1}")
            pairs.append({
                "speaker_0": row["speaker_0"], "speaker_1": row["speaker_1"],
                "path_0": p0, "path_1": p1, "origin": "existing",
            })
    return pairs


def resolve_new_pairs(num_new):
    """Pair up new fullscale speakers (deterministic order), 2 utterances each."""
    by_speaker = {}
    for f in sorted(FULLSCALE_DIR.glob("*.flac")):
        by_speaker.setdefault(f.name.split("-")[0], []).append(f)
    speaker_ids = sorted(by_speaker)
    pairs = []
    for i in range(0, min(num_new * 2, len(speaker_ids) - 1), 2):
        s0, s1 = speaker_ids[i], speaker_ids[i + 1]
        p0 = by_speaker[s0][0]
        p1 = by_speaker[s1][0]
        pairs.append({
            "speaker_0": s0, "speaker_1": s1,
            "path_0": p0, "path_1": p1, "origin": "new",
        })
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--target-pairs", type=int, default=100)
    ap.add_argument("--seed-offset", type=int, default=5000)
    ap.add_argument("--target-layer", default=None)
    ap.add_argument("--speakers-txt", default=None)
    ap.add_argument("--results-dir", type=Path, default=gradcam_root / "results" / "fullscale")
    ap.add_argument("--f0-cache", type=Path, default=gradcam_root / "results" / "fullscale" / "speaker_f0.json")
    args = ap.parse_args()

    target_layer = args.target_layer or load_selected_layer()
    genders = load_speaker_genders(args.speakers_txt)
    print(f"[*] Full-scale validation, target layer {target_layer}, target pairs={args.target_pairs}")

    # ----- build pair list: existing 35 + new -----
    existing = resolve_existing_pairs()
    num_new_needed = args.target_pairs - len(existing)
    new = resolve_new_pairs(num_new_needed)
    pairs = existing + new
    print(f"[*] pairs: {len(existing)} existing + {len(new)} new = {len(pairs)} total "
          f"({len(pairs)*2} speakers)")

    all_speakers = sorted({p["speaker_0"] for p in pairs} | {p["speaker_1"] for p in pairs})
    missing_gender = [s for s in all_speakers if s not in genders]
    if missing_gender:
        raise RuntimeError(f"No gender metadata for speakers: {missing_gender[:10]}")
    print(f"[*] gender resolved for all {len(all_speakers)} speakers")

    # ----- per-speaker F0 (cached) -----
    args.results_dir.mkdir(parents=True, exist_ok=True)
    f0_cache = json.loads(args.f0_cache.read_text()) if args.f0_cache.exists() else {}
    for i, p in enumerate(pairs):
        for key in ("speaker_0", "speaker_1"):
            sid, path = p[key], p["path_0" if key == "speaker_0" else "path_1"]
            if sid not in f0_cache:
                f0_cache[sid] = mean_f0(path)
                args.f0_cache.write_text(json.dumps(f0_cache, indent=2))
        if (i + 1) % 10 == 0:
            print(f"    F0 computed for pairs up to {i+1}/{len(pairs)}")
    print(f"[*] F0 computed for {len(f0_cache)} speakers")

    # pitch tertiles over the full speaker set
    f0_vals = np.array([f0_cache[s] for s in all_speakers], dtype=float)
    t1, t2 = np.nanpercentile(f0_vals, [100 / 3, 200 / 3])

    def tertile(sid):
        v = f0_cache[sid]
        return "low" if v <= t1 else ("mid" if v <= t2 else "high")

    # ----- model -----
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(args.device).eval()
    ckpt = torch.load(gradcam_root / "weights" / "model_with_vad.pth", map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)

    # ----- CAM evaluation -----
    records = []
    for i, p in enumerate(pairs):
        audio = prepare_mixture([p["path_0"], p["path_1"]])
        with torch.no_grad():
            output = model(audio.to(args.device))
            vad_logits = model.vad_logits.detach().cpu().numpy()[0]
            separated = output[0].detach().cpu().numpy()[0]
        for speaker in range(2):
            idx = int(np.argmax(np.abs(vad_logits[speaker])))
            vad_cam = compute_cam(model, audio, target_layer, "vad_logit", speaker, idx, args.device)
            if speaker == 0:
                vad_cam_0 = vad_cam
            else:
                vad_cam_1 = vad_cam
        m = pair_mae(vad_cam_0, vad_cam_1, i, args.seed_offset)
        s0, s1 = p["speaker_0"], p["speaker_1"]
        records.append({
            "pair_index": i + 1, "origin": p["origin"],
            "speaker_0": s0, "speaker_1": s1,
            "gender_0": genders[s0], "gender_1": genders[s1],
            "f0_0": f0_cache[s0], "f0_1": f0_cache[s1],
            "tertile_0": tertile(s0), "tertile_1": tertile(s1),
            **m,
        })
        if (i + 1) % 5 == 0:
            print(f"    evaluated {i+1}/{len(pairs)} pairs (VAD MAE={m['vad_real_mae']:.4f})")

    # ----- save per-pair CSV -----
    csv_path = args.results_dir / "fullscale_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    # ----- aggregate + subgroup stats -----
    def stats_for(subset):
        return compute_wilcoxon([r["vad_real_mae"] for r in subset], [r["vad_random_mae"] for r in subset])

    summary = {
        "target_layer": target_layer,
        "num_pairs": len(records),
        "num_speakers": len(all_speakers),
        "pitch_tertile_edges_hz": [float(t1), float(t2)],
        "overall": stats_for(records),
        "by_gender_pairtype": {},
        "by_pitch_tertile_0": {},
    }
    # gender subgroup: pair composition (F/F, M/M, mixed)
    for label, pred in (("F/F", lambda g: g == ("F", "F")),
                        ("M/M", lambda g: g == ("M", "M")),
                        ("mixed", lambda g: g[0] != g[1])):
        subset = [r for r in records if pred((r["gender_0"], r["gender_1"]))]
        if len(subset) >= 3:
            summary["by_gender_pairtype"][label] = stats_for(subset)
    # pitch tertile of the first speaker in each pair
    for t in ("low", "mid", "high"):
        subset = [r for r in records if r["tertile_0"] == t]
        if len(subset) >= 3:
            summary["by_pitch_tertile_0"][t] = stats_for(subset)

    (args.results_dir / "fullscale_summary.json").write_text(json.dumps(summary, indent=2))

    # ----- console report -----
    def fmt(d):
        return (f"N={d['n']:3d}  real MAE={d['real_mean']:.4f}±{d['real_std']:.4f}  "
                f"rand MAE={d['rand_mean']:.4f}±{d['rand_std']:.4f}  "
                f"W={d['w']:.1f}  p={d['p']:.2e}  r={d['rank_biserial_r']:.3f}")

    print("\n" + "=" * 78)
    print("FULL-SCALE VALIDATION (VAD-logit CAM)")
    print("=" * 78)
    print("  OVERALL      " + fmt(summary["overall"]))
    for label, d in summary["by_gender_pairtype"].items():
        print(f"  gender {label:6s} " + fmt(d))
    for t, d in summary["by_pitch_tertile_0"].items():
        print(f"  pitch {t:6s}  " + fmt(d))
    print("=" * 78)
    print(f"[+] Saved: {csv_path}\n[+] Saved: {args.results_dir / 'fullscale_summary.json'}")


if __name__ == "__main__":
    main()
