"""Integrated Gradients (IG) attribution for VAD-logit alignment.

Diagnostic/decision-stage: tests whether input-space attribution (IG) beats the
chance baseline that Grad-CAM could not, for VAD-timing alignment.

- Input space: raw mixture waveform (STFT is differentiable inside the autograd
  graph, confirmed -- gradients flow to the waveform, so attribute there).
- Baseline: all-zero (silence) waveform.
- Path integral: m interpolation steps, gradients wrt interpolated input,
  backprop from the same VAD-logit target (same pair/speaker/frame as elsewhere).
- Final attribution = (x - x') * mean(accumulated grads), elementwise.
- Completeness check: sum(attributions) ~= F(x) - F(x').
- Aggregation to 1D: for the waveform IG map, sum |attribution| over 512-sample
  non-overlapping windows (matching the STFT frame hop of 256 -> but the
  alignment pipeline uses 188 frames for 3s/16kHz; we resample the per-sample
  IG curve to 188 frames to match the existing CAM/reference format).
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import SpeakerSampler, load_utterance

SAMPLE_RATE = 16000
TARGET_SECONDS = 3.0
TARGET_LENGTH = int(TARGET_SECONDS * SAMPLE_RATE)
HOP = 256
NUM_FRAMES = 1 + TARGET_LENGTH // HOP
CHANCE_SEEDS = range(20)


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


def integrated_gradients(model, x, speaker, frame, m, device):
    """IG from all-zero baseline to input x, backprop from VAD logit [speaker, frame].

    x: (1, T) waveform tensor. Returns attribution (T,) as float64 numpy, plus
    completeness residual (sum_attrib - (F(x) - F(baseline))).
    """
    x = x.to(device).float()
    baseline = torch.zeros_like(x)

    def target_for(inp):
        with torch.enable_grad():
            _ = model(inp)
            return model.vad_logits[0, speaker, frame]

    # F(x) and F(baseline)
    with torch.no_grad():
        _ = model(x)
        f_x = float(model.vad_logits[0, speaker, frame])
        _ = model(baseline)
        f_b = float(model.vad_logits[0, speaker, frame])

    total_grad = torch.zeros_like(x)
    for k in range(1, m + 1):
        alpha = k / m
        xk = baseline + alpha * (x - baseline)
        xk = xk.clone().detach().requires_grad_(True)
        t = target_for(xk)
        t.backward()
        total_grad = total_grad + xk.grad.detach()
    avg_grad = total_grad / m
    attribution = (x - baseline) * avg_grad  # (1, T)
    attr = attribution.detach().cpu().numpy()[0]

    sum_attrib = float(attr.sum())
    residual = sum_attrib - (f_x - f_b)
    return attr, sum_attrib, f_x, f_b, residual


def aggregate_to_frames(attr):
    """Aggregate per-sample attribution to per-frame importance (188 frames).

    Sum of absolute attribution within each 256-sample hop window, then trim/pad
    to NUM_FRAMES to match the existing CAM/reference format.
    """
    abs_attr = np.abs(attr)
    frames = []
    for i in range(NUM_FRAMES):
        start = i * HOP
        end = min(len(abs_attr), start + HOP)
        frames.append(abs_attr[start:end].sum())
    frames = np.array(frames, dtype=np.float64)
    if len(frames) < NUM_FRAMES:
        frames = np.pad(frames, (0, NUM_FRAMES - len(frames)))
    return frames


def silero_mask(vad_model, clean, num_frames):
    ts = get_speech_timestamps(torch.from_numpy(clean.astype(np.float32)), vad_model,
                               sampling_rate=SAMPLE_RATE, return_seconds=False)
    mask = np.zeros(num_frames, dtype=bool)
    for seg in ts:
        s = seg["start"] // HOP
        e = min(num_frames, seg["end"] // HOP + 1)
        mask[s:e] = True
    return mask


def iou_f1_pr(pred, ref):
    pred = pred.astype(bool); ref = ref.astype(bool)
    tp = np.sum(pred & ref); fp = np.sum(pred & ~ref); fn = np.sum(~pred & ref)
    union = np.sum(pred | ref)
    iou = float(tp/union) if union > 0 else 0.0
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f1 = float(2*p*r/(p+r)) if p+r else 0.0
    return iou, p, r, f1


def chance_metrics(ref_mask, seeds=CHANCE_SEEDS):
    base_rate = float(np.mean(ref_mask))
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rows.append(iou_f1_pr(rng.random(len(ref_mask)) < base_rate, ref_mask))
    rows = np.asarray(rows, dtype=np.float64)
    return {"iou": float(rows[:,0].mean()), "precision": float(rows[:,1].mean()),
            "recall": float(rows[:,2].mean()), "f1": float(rows[:,3].mean())}


def best_f1_threshold(scores, ref_mask):
    best_thr, best = 0.5, (0.0, 0.0, 0.0, -1.0)
    for thr in np.linspace(0.05, 0.95, 19):
        m = iou_f1_pr(scores >= thr, ref_mask)
        if m[3] > best[3]:
            best = m
            best_thr = float(thr)
    return best_thr, best  # best = (iou, precision, recall, f1)


def main():
    device = "cpu"
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    ckpt = torch.load(gradcam_root / "weights" / "model_with_vad.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    vad_model = load_silero_vad()

    sampler = SpeakerSampler("data/librispeech_samples", seed=0)
    spk_ids = sampler.sample_two_speakers()
    paths = [sampler.sample_utterance(s) for s in spk_ids]
    sig0, sig1 = load_and_pad(paths[0]), load_and_pad(paths[1])
    mix = normalize_audio(sig0 + sig1)
    audio = torch.from_numpy(mix).unsqueeze(0)

    with torch.no_grad():
        _ = model(audio.to(device))
        vad_logits = model.vad_logits.detach().cpu().numpy()[0]

    # Sanity-check the all-zero baseline first.
    with torch.no_grad():
        _ = model(torch.zeros_like(audio))
        base_logits = model.vad_logits.detach().cpu().numpy()[0]
    print(f"Baseline (silence) VAD logits range: [{base_logits.min():.3f}, {base_logits.max():.3f}] "
          f"(sensible/low expected, not NaN: {not np.isnan(base_logits).any()})")

    out_dir = gradcam_root / "results" / "ig_vad_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for speaker in range(2):
        ref = silero_mask(vad_model, (sig0, sig1)[speaker], NUM_FRAMES)
        if ref.sum() == 0 or ref.sum() == len(ref):
            print(f"speaker {speaker}: SKIPPED (degenerate reference mask)")
            continue
        frame = int(np.argmax(np.abs(vad_logits[speaker])))

        # convergence check: m=20 vs m=50 on this speaker
        for m in (20, 50):
            t0 = time.time()
            attr, sum_attr, f_x, f_b, residual = integrated_gradients(model, audio, speaker, frame, m, device)
            dt = time.time() - t0
            frames = aggregate_to_frames(attr)
            curve = minmax(frames)
            best_thr, (iou, prec, rec, f1) = best_f1_threshold(curve, ref)
            chance = chance_metrics(ref)
            row = {
                "speaker": speaker, "m": m, "frame": frame, "seconds": dt,
                "sum_attributions": sum_attr, "F(x)": f_x, "F(baseline)": f_b,
                "completeness_residual": residual,
                "completeness_rel_err": abs(residual) / (abs(f_x - f_b) + 1e-9),
                "best_thr": best_thr, "iou": iou, "precision": prec, "recall": rec, "f1": f1,
                "chance_f1": chance["f1"], "chance_precision": chance["precision"],
                "chance_recall": chance["recall"], "chance_iou": chance["iou"],
                "ref_active_rate": float(ref.mean()),
                "mask_active_rate": float((curve >= best_thr).mean()),
            }
            rows.append(row)
            print(f"speaker {speaker} m={m}: dt={dt:.1f}s, completeness sum_attr={sum_attr:.4f} vs "
                  f"F(x)-F(b)={f_x - f_b:.4f} (rel err {row['completeness_rel_err']:.3f}), "
                  f"best-F1={f1:.3f} (thr={best_thr:.2f}, mask rate={row['mask_active_rate']:.3f}) vs chance F1={chance['f1']:.3f}")

    csv_path = out_dir / "ig_vad_alignment_example.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out_dir / "ig_vad_alignment_example.json").write_text(json.dumps(rows, indent=2))
    print(f"\n[+] Saved: {csv_path}")


if __name__ == "__main__":
    main()
