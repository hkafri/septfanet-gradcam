"""Orientation-invariant check: AUC-ROC/AUC-PR of continuous CAM vs ground truth.

Diagnostic only -- settles whether the flip-test result reflects a real (possibly
mis-oriented) signal or a thresholding/base-rate artifact. AUC-ROC flips as
AUC -> 1 - AUC under sign inversion, so it directly measures discriminative
signal regardless of orientation. No fix applied, no prior conclusion revised.

Tasks:
  (a) Separation: continuous CAM vs permutation-corrected IBM, 70 instances,
      both layers.
  (b) VAD: continuous CAM vs Silero reference, the existing pairs, blocks 9 & 23.

Also reports, per instance, the CAM's own thresholded active rate vs. the true
ground-truth active rate (the base-rate/threshold-calibration mismatch check).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from sklearn.metrics import roc_auc_score, average_precision_score

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import SpeakerSampler, load_utterance
from gradcam import GradCAM

SAMPLE_RATE = 16000
TARGET_SECONDS = 3.0
TARGET_LENGTH = int(TARGET_SECONDS * SAMPLE_RATE)
N_FFT = 512
HOP = 256
NUM_FRAMES = 1 + TARGET_LENGTH // HOP
NUM_FREQ = N_FFT // 2 + 1
LAYERS = ["TCN.TCN.9.conv1d", "TCN.TCN.23.conv1d"]
NUM_PAIRS = 35


def normalize_audio(a):
    a = a.astype(np.float32)
    return a / max(np.abs(a).max(), 1e-8) * 0.9


def load_and_pad(path):
    s = load_utterance(path)[:TARGET_LENGTH]
    pad = np.zeros(TARGET_LENGTH, dtype=np.float32)
    pad[:len(s)] = s
    return pad


def stft_mag(sig):
    t = torch.from_numpy(sig.astype(np.float32))
    window = torch.hann_window(N_FFT)
    return torch.abs(torch.stft(t, N_FFT, HOP, window=window, return_complex=True)).numpy()


def ideal_binary_mask(a, b):
    return (stft_mag(a) > stft_mag(b)).astype(bool)


def compute_cam_1d(model, audio, layer, speaker, freq, time, device, task):
    inp = audio.clone().detach().to(device).float().requires_grad_(True)
    gc = GradCAM(model, layer, device)
    try:
        with torch.enable_grad():
            _ = model(inp)
            if task == "separation":
                target = model.mask_per_speaker[0, speaker, freq, time]
            else:
                target = model.vad_logits[0, speaker, time]
            target.backward()
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


def cam_to_tf_2d(cam_1d):
    if len(cam_1d) < NUM_FRAMES:
        cam_1d = np.pad(cam_1d, (0, NUM_FRAMES - len(cam_1d)))
    cam_1d = cam_1d[:NUM_FRAMES]
    return np.tile(cam_1d, (NUM_FREQ, 1))


def silero_mask(vad_model, clean, num_frames):
    ts = get_speech_timestamps(torch.from_numpy(clean.astype(np.float32)), vad_model,
                               sampling_rate=SAMPLE_RATE, return_seconds=False)
    mask = np.zeros(num_frames, dtype=bool)
    for seg in ts:
        s = seg["start"] // HOP
        e = min(num_frames, seg["end"] // HOP + 1)
        mask[s:e] = True
    return mask


def load_pairs():
    sel = SpeakerSampler("data/librispeech_samples", seed=123)
    ids = sel.speaker_ids
    pairs = []
    n_sel = min(NUM_PAIRS, len(ids) // 2)
    for i in range(n_sel):
        s1, s2 = ids[2 * i], ids[2 * i + 1]
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
    out_dir = gradcam_root / "results" / "auc_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for p_idx, (p1, p2, tag) in enumerate(pairs):
        sig0, sig1 = load_and_pad(p1), load_and_pad(p2)
        mix = normalize_audio(sig0 + sig1)
        audio = torch.from_numpy(mix).unsqueeze(0)

        with torch.no_grad():
            _ = model(audio.to(device))
            pred_mask = model.mask_per_speaker.detach().cpu().numpy()[0]
            vad_logits = model.vad_logits.detach().cpu().numpy()[0]

        for speaker in range(2):
            self_sig = sig0 if speaker == 0 else sig1
            other_sig = sig1 if speaker == 0 else sig0

            # --- Separation (AUC vs IBM, permutation-corrected) ---
            ibm_s = ideal_binary_mask(self_sig, other_sig)
            ibm_w = ideal_binary_mask(other_sig, self_sig)
            pred_bin = pred_mask[speaker] >= 0.5
            f1_s = _f1(pred_bin, ibm_s)
            f1_w = _f1(pred_bin, ibm_w)
            ibm = ibm_w if f1_w >= f1_s else ibm_s
            ref_self, ref_other = (other_sig, self_sig) if f1_w >= f1_s else (self_sig, other_sig)
            diff = np.abs(stft_mag(ref_self) - stft_mag(ref_other))
            f_star, t_star = np.unravel_index(np.argmax(diff), diff.shape)
            t_star = min(t_star, NUM_FRAMES - 1)
            f_star = min(f_star, NUM_FREQ - 1)

            # --- VAD (AUC vs Silero reference) ---
            ref_mask = silero_mask(vad_model, self_sig, NUM_FRAMES)
            vad_frame = int(np.argmax(np.abs(vad_logits[speaker])))

            for layer in LAYERS:
                # separation CAM
                cam_1d = compute_cam_1d(model, audio, layer, speaker, f_star, t_star, device, "separation")
                cam_2d = cam_to_tf_2d(cam_1d)
                sep_auc = roc_auc_score(ibm.ravel(), cam_2d.ravel())
                sep_ap = average_precision_score(ibm.ravel(), cam_2d.ravel())
                # base-rate/threshold check
                cam_thr = cam_2d >= 0.5
                cam_rate = float(cam_thr.mean())

                # VAD CAM (same layer, existing target frame)
                vcam_1d = compute_cam_1d(model, audio, layer, speaker, 0, vad_frame, device, "vad")
                vcam = minmax(vcam_1d)
                vcam = vcam[:NUM_FRAMES] if len(vcam) >= NUM_FRAMES else np.pad(vcam, (0, NUM_FRAMES - len(vcam)))
                vad_auc = roc_auc_score(ref_mask, vcam) if (ref_mask.sum() > 0 and ref_mask.sum() < len(ref_mask)) else float("nan")
                vad_ap = average_precision_score(ref_mask, vcam) if (ref_mask.sum() > 0 and ref_mask.sum() < len(ref_mask)) else float("nan")
                vcam_thr = vcam >= 0.5
                vcam_rate = float(vcam_thr.mean())

                records.append({
                    "pair": tag, "speaker": speaker, "layer": layer,
                    "sep_auc_roc": float(sep_auc), "sep_auc_pr": float(sep_ap),
                    "sep_cam_active_rate": cam_rate, "sep_ibm_active_rate": float(ibm.mean()),
                    "vad_auc_roc": float(vad_auc), "vad_auc_pr": float(vad_ap),
                    "vad_cam_active_rate": vcam_rate, "vad_ref_active_rate": float(ref_mask.mean()),
                })
        print(f"pair {p_idx+1} ({tag}) done")

    csv_path = out_dir / "auc_check_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    (out_dir / "auc_check_results.json").write_text(json.dumps(records, indent=2))

    print("\n" + "=" * 100)
    print("ORIENTATION-INVARIANT CHECK: AUC-ROC / AUC-PR of continuous CAM vs ground truth")
    print("=" * 100)
    for layer in LAYERS:
        sub = [r for r in records if r["layer"] == layer]
        sep_auc = [r["sep_auc_roc"] for r in sub]
        sep_ap = [r["sep_auc_pr"] for r in sub]
        sep_cam_rate = [r["sep_cam_active_rate"] for r in sub]
        sep_ibm_rate = [r["sep_ibm_active_rate"] for r in sub]
        vad_auc = [r["vad_auc_roc"] for r in sub if not np.isnan(r["vad_auc_roc"])]
        vad_ap = [r["vad_auc_pr"] for r in sub if not np.isnan(r["vad_auc_pr"])]
        print(f"{layer}:")
        print(f"  Separation: AUC-ROC={np.mean(sep_auc):.3f}±{np.std(sep_auc):.3f} (range [{min(sep_auc):.3f},{max(sep_auc):.3f}]) "
              f"AUC-PR={np.mean(sep_ap):.3f} | CAM rate={np.mean(sep_cam_rate):.3f} vs IBM rate={np.mean(sep_ibm_rate):.3f}")
        if vad_auc:
            print(f"  VAD:        AUC-ROC={np.mean(vad_auc):.3f}±{np.std(vad_auc):.3f} (range [{min(vad_auc):.3f},{max(vad_auc):.3f}]) "
                  f"AUC-PR={np.mean(vad_ap):.3f} (n={len(vad_auc)})")
    print("=" * 100)
    print(f"[+] Saved: {csv_path}")


def _f1(pred, ref):
    pred = pred.astype(bool); ref = ref.astype(bool)
    tp = np.sum(pred & ref); fp = np.sum(pred & ~ref); fn = np.sum(~pred & ref)
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    return float(2*p*r/(p+r)) if p+r else 0.0


if __name__ == "__main__":
    main()
