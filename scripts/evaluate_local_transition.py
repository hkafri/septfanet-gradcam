"""Local per-transition evaluation of VAD-logit attribution.

Diagnostic/decision-stage: tests whether attribution concentrates near true VAD
transitions when targeted AT each transition, using a matched control window --
instead of the global frame-by-frame F1 that both Grad-CAM (6 layers) and IG
failed on.

For each Silero-reference VAD transition (onset or offset) in a clip:
  - compute a FRESH attribution backpropagating from the VAD logit at that
    transition frame (not one global peak-|logit| frame),
  - measure magnitude within +/-W frames around the transition (W in {5, 10}),
  - measure the same attribution's magnitude in a matched control window
    elsewhere in the same clip (same width, away from all transitions),
  - paired comparison (near-transition vs control) per transition.

Layers tested: TCN.TCN.9.conv1d and TCN.TCN.23.conv1d.
Statistical test: paired Wilcoxon signed-rank across all transitions.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import SpeakerSampler, load_utterance
from gradcam import GradCAM

SAMPLE_RATE = 16000
TARGET_SECONDS = 3.0
TARGET_LENGTH = int(TARGET_SECONDS * SAMPLE_RATE)
HOP = 256
NUM_FRAMES = 1 + TARGET_LENGTH // HOP
LAYERS = ["TCN.TCN.9.conv1d", "TCN.TCN.23.conv1d"]
WINDOWS = [5, 10]
NUM_PAIRS = 35  # few existing example pairs first, per the discipline instruction

# PRE-REGISTERED PRIMARY TEST (declared 2026-10-05, BEFORE running the full
# 35-pair pool): the single confirmatory hypothesis is
#     TCN.TCN.23.conv1d, window W=10  ->  near-transition attribution magnitude
#     > matched-control window magnitude (paired Wilcoxon, one primary cell).
# The other three cells (block 9 W=5, block 9 W=10, block 23 W=5) are
# secondary/exploratory and are not used to claim significance if the primary
# test does not cross alpha=0.05. This declaration is written into the script
# before any full-pool computation, so it cannot have been chosen after
# seeing the larger dataset.
PRIMARY_LAYER = "TCN.TCN.23.conv1d"
PRIMARY_WINDOW = 10
ALPHA = 0.05


def normalize_audio(a):
    a = a.astype(np.float32)
    return a / max(np.abs(a).max(), 1e-8) * 0.9


def load_and_pad(path):
    s = load_utterance(path)[:TARGET_LENGTH]
    pad = np.zeros(TARGET_LENGTH, dtype=np.float32)
    pad[:len(s)] = s
    return pad


def silero_mask(vad_model, clean, num_frames):
    ts = get_speech_timestamps(torch.from_numpy(clean.astype(np.float32)), vad_model,
                               sampling_rate=SAMPLE_RATE, return_seconds=False)
    mask = np.zeros(num_frames, dtype=bool)
    for seg in ts:
        s = seg["start"] // HOP
        e = min(num_frames, seg["end"] // HOP + 1)
        mask[s:e] = True
    return mask


def find_transitions(mask):
    """Frame indices where the binary mask changes value (onsets and offsets)."""
    diff = np.diff(mask.astype(int))
    return list(np.nonzero(diff != 0)[0] + 1)


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


def minmax(v):
    v = np.asarray(v, dtype=np.float64)
    span = v.max() - v.min()
    return (v - v.min()) / span if span > 1e-12 else np.zeros_like(v)


def window_mean(curve, center, half_width):
    lo = max(0, center - half_width)
    hi = min(len(curve), center + half_width + 1)
    if lo >= hi:
        return 0.0
    return float(np.mean(curve[lo:hi]))


def pick_control_center(center, half_width, num_frames, transitions, rng):
    """A matched control window center, same width, away from all transitions."""
    margin = half_width
    lo, hi = margin, num_frames - margin - 1
    if hi <= lo:
        return None
    for _ in range(200):
        cand = int(rng.integers(lo, hi + 1))
        if all(abs(cand - t) > 2 * half_width + 1 for t in transitions):
            return cand
    return None


def wilcoxon(near, ctrl):
    near = np.asarray(near, dtype=float)
    ctrl = np.asarray(ctrl, dtype=float)
    if len(near) < 2 or np.allclose(near, ctrl):
        return {"w_statistic": float("nan"), "p_value": float("nan"),
                "rank_biserial_r": float("nan"), "n": int(len(near))}
    res = stats.wilcoxon(near, ctrl)
    diffs = near - ctrl
    abs_diffs = np.abs(diffs)
    ranks = stats.rankdata(abs_diffs)
    w_pos = np.sum(ranks[diffs > 0])
    w_neg = np.sum(ranks[diffs < 0])
    tot = w_pos + w_neg
    r = float((w_pos - w_neg) / tot) if tot > 0 else 0.0
    return {"w_statistic": float(res.statistic), "p_value": float(res.pvalue),
            "rank_biserial_r": r, "n": int(len(near))}


def load_pairs():
    sel = SpeakerSampler("data/librispeech_samples", seed=123)
    sel_ids = sel.speaker_ids
    pairs = []
    n_sel = min(NUM_PAIRS, len(sel_ids) // 2)
    for i in range(n_sel):
        s1, s2 = sel_ids[2 * i], sel_ids[2 * i + 1]
        pairs.append((sel.sample_utterance(s1), sel.sample_utterance(s2), f"sel{i+1}"))

    hold = SpeakerSampler("data/librispeech_holdout", seed=123)
    hold_ids = hold.speaker_ids
    remaining = NUM_PAIRS - len(pairs)
    n_hold = min(len(hold_ids) // 2, remaining)
    for i in range(n_hold):
        s1, s2 = hold_ids[2 * i], hold_ids[2 * i + 1]
        pairs.append((hold.sample_utterance(s1), hold.sample_utterance(s2), f"hold{i+1}"))
    return pairs


def main():
    device = "cpu"
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    ckpt = torch.load(gradcam_root / "weights" / "model_with_vad.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    vad_model = load_silero_vad()

    pairs = load_pairs()
    rng = np.random.default_rng(7)

    records = []  # one row per (pair, speaker, transition, layer, window)
    for p_idx, (p1, p2, tag) in enumerate(pairs):
        sig0, sig1 = load_and_pad(p1), load_and_pad(p2)
        mix = normalize_audio(sig0 + sig1)
        audio = torch.from_numpy(mix).unsqueeze(0)
        for speaker in range(2):
            ref = silero_mask(vad_model, (sig0, sig1)[speaker], NUM_FRAMES)
            transitions = find_transitions(ref)
            transitions = [t for t in transitions if t > 0 and t < NUM_FRAMES - 1]
            if not transitions:
                continue
            for layer in LAYERS:
                for t in transitions:
                    raw = compute_cam(model, audio, layer, speaker, t, device)
                    curve = minmax(raw)
                    curve = curve[:NUM_FRAMES] if len(curve) >= NUM_FRAMES else np.pad(curve, (0, NUM_FRAMES - len(curve)))
                    for W in WINDOWS:
                        near = window_mean(curve, t, W)
                        ctrl_center = pick_control_center(t, W, NUM_FRAMES, transitions, rng)
                        if ctrl_center is None:
                            continue
                        ctrl = window_mean(curve, ctrl_center, W)
                        records.append({
                            "pair": tag, "speaker": speaker, "transition_frame": t,
                            "layer": layer, "window": W,
                            "near_mean": near, "control_mean": ctrl, "near_minus_control": near - ctrl,
                        })
        print(f"  pair {p_idx+1} ({tag}) done")

    out_dir = gradcam_root / "results" / "local_transition"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "local_transition_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)

    summary = {
        "pre_registration": {
            "primary_layer": PRIMARY_LAYER,
            "primary_window": PRIMARY_WINDOW,
            "alpha": ALPHA,
            "declared_before_full_pool_run": True,
        },
        "num_pairs": len(pairs),
    }
    cells = {}
    for layer in LAYERS:
        for W in WINDOWS:
            sub = [r for r in records if r["layer"] == layer and r["window"] == W]
            near = [r["near_mean"] for r in sub]
            ctrl = [r["control_mean"] for r in sub]
            wres = wilcoxon(near, ctrl)
            key = f"{layer} W={W}"
            role = "primary" if (layer == PRIMARY_LAYER and W == PRIMARY_WINDOW) else "secondary"
            cells[key] = {
                "role": role,
                "n_transitions": len(sub),
                "near_mean": float(np.mean(near)), "near_std": float(np.std(near)),
                "control_mean": float(np.mean(ctrl)), "control_std": float(np.std(ctrl)),
                "mean_diff": float(np.mean(near) - np.mean(ctrl)),
                "wilcoxon": wres,
            }
    summary["cells"] = cells

    primary = cells[f"{PRIMARY_LAYER} W={PRIMARY_WINDOW}"]
    summary["primary_result"] = {
        "n_transitions": primary["n_transitions"],
        "p_value": primary["wilcoxon"]["p_value"],
        "rank_biserial_r": primary["wilcoxon"]["rank_biserial_r"],
        "significant_at_alpha": bool(primary["wilcoxon"]["p_value"] < ALPHA),
    }

    (out_dir / "local_transition_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 88)
    print("LOCAL PER-TRANSITION EVALUATION (near-transition vs matched control, paired)")
    print(f"Pre-registered primary test: {PRIMARY_LAYER} at W={PRIMARY_WINDOW}, alpha={ALPHA}")
    print("=" * 88)
    for layer in LAYERS:
        for W in WINDOWS:
            s = summary["cells"][f"{layer} W={W}"]
            w = s["wilcoxon"]
            tag = "PRIMARY (pre-registered)" if s["role"] == "primary" else "secondary/exploratory"
            print(f"{layer} W={W:2d} [{tag}]: n={s['n_transitions']:3d} transitions | "
                  f"near={s['near_mean']:.4f}±{s['near_std']:.4f} ctrl={s['control_mean']:.4f}±{s['control_std']:.4f} "
                  f"diff={s['mean_diff']:+.4f} | W={w['w_statistic']:.1f} p={w['p_value']:.3e} r={w['rank_biserial_r']:+.3f}")
    print("=" * 88)
    pr = summary["primary_result"]
    verdict = "SIGNIFICANT" if pr["significant_at_alpha"] else "NOT SIGNIFICANT"
    print(f"PRIMARY TEST ({PRIMARY_LAYER} W={PRIMARY_WINDOW}): n={pr['n_transitions']}, "
          f"p={pr['p_value']:.3e}, r={pr['rank_biserial_r']:+.3f} -> {verdict} at alpha={ALPHA}")
    print(f"[+] Saved: {csv_path}")


if __name__ == "__main__":
    main()
