"""Flip test: does inverting the thresholded CAM prediction fix the below-chance result?

Diagnostic only. For each instance in the full-scale separation-alignment run,
recompute F1/accuracy for the thresholded CAM and its inversion (~CAM) against
the same permutation-corrected IBM ground truth. If the flipped version is well
above chance, the below-chance result is an inversion bug; if not, the result is
a genuine anti-correlated finding.

Also probes where any inversion could originate: whether the Grad-CAM CAM
weights (global-mean-pooled gradients per channel) are predominantly negative,
which under the ReLU(weighted-sum) produces high CAM where activations are LOW.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

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


def ideal_binary_mask(sig_self, sig_other):
    return (stft_mag(sig_self) > stft_mag(sig_other)).astype(bool)


def binary_metrics(pred, ref):
    pred = pred.astype(bool); ref = ref.astype(bool)
    tp = np.sum(pred & ref); fp = np.sum(pred & ~ref)
    fn = np.sum(~pred & ref); tn = np.sum(~pred & ~ref)
    acc = float((tp + tn) / pred.size)
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f1 = float(2*p*r/(p+r)) if p+r else 0.0
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def compute_cam_with_weights(model, audio, layer, speaker, freq, time, device):
    inp = audio.clone().detach().to(device).float().requires_grad_(True)
    gc = GradCAM(model, layer, device)
    try:
        with torch.enable_grad():
            _ = model(inp)
            target = model.mask_per_speaker[0, speaker, freq, time]
            target.backward()
            act, grad = gc.hook.activations, gc.hook.gradients
            w = grad.mean(dim=tuple(range(2, grad.ndim)), keepdim=True)
            raw = torch.relu((w * act).sum(dim=1)).detach().cpu().numpy()[0]
            # diagnostics: sign composition of the channel weights
            w_flat = w.detach().cpu().numpy().flatten()
            frac_neg = float(np.mean(w_flat < 0))
            return raw, frac_neg
    finally:
        gc.hook.remove_hooks()


def cam_to_tf_2d(cam_1d):
    if len(cam_1d) < NUM_FRAMES:
        cam_1d = np.pad(cam_1d, (0, NUM_FRAMES - len(cam_1d)))
    cam_1d = cam_1d[:NUM_FRAMES]
    return np.tile(cam_1d, (NUM_FREQ, 1))


def minmax(v):
    v = np.asarray(v, dtype=np.float64)
    span = v.max() - v.min()
    return (v - v.min()) / span if span > 1e-12 else np.zeros_like(v)


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

    pairs = load_pairs()
    out_dir = gradcam_root / "results" / "separation_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for p_idx, (p1, p2, tag) in enumerate(pairs):
        sig0, sig1 = load_and_pad(p1), load_and_pad(p2)
        mix = normalize_audio(sig0 + sig1)
        audio = torch.from_numpy(mix).unsqueeze(0)

        with torch.no_grad():
            _ = model(audio.to(device))
            pred_mask = model.mask_per_speaker.detach().cpu().numpy()[0]

        for speaker in range(2):
            self_sig = sig0 if speaker == 0 else sig1
            other_sig = sig1 if speaker == 0 else sig0
            ibm_straight = ideal_binary_mask(self_sig, other_sig)
            ibm_swapped = ideal_binary_mask(other_sig, self_sig)
            pred_bin = pred_mask[speaker] >= 0.5
            f1_straight = binary_metrics(pred_bin, ibm_straight)["f1"]
            f1_swapped = binary_metrics(pred_bin, ibm_swapped)["f1"]
            ibm = ibm_swapped if f1_swapped >= f1_straight else ibm_straight

            # target bin: where the (corrected) reference speaker dominates most
            if f1_swapped >= f1_straight:
                ref_self, ref_other = other_sig, self_sig
            else:
                ref_self, ref_other = self_sig, other_sig
            diff = np.abs(stft_mag(ref_self) - stft_mag(ref_other))
            f_star, t_star = np.unravel_index(np.argmax(diff), diff.shape)
            t_star = min(t_star, NUM_FRAMES - 1)
            f_star = min(f_star, NUM_FREQ - 1)

            for layer in LAYERS:
                cam_1d, frac_neg = compute_cam_with_weights(model, audio, layer, speaker, f_star, t_star, device)
                cam_2d = minmax(cam_to_tf_2d(cam_1d))
                cam_bin = cam_2d >= 0.5
                orig = binary_metrics(cam_bin, ibm)
                flipped = binary_metrics(~cam_bin, ibm)
                records.append({
                    "pair": tag, "speaker": speaker, "layer": layer,
                    "orig_f1": orig["f1"], "orig_acc": orig["accuracy"],
                    "flip_f1": flipped["f1"], "flip_acc": flipped["accuracy"],
                    "frac_negative_cam_weights": frac_neg,
                    "mask_active_rate": float(cam_bin.mean()),
                    "ibm_active_rate": float(ibm.mean()),
                })
        print(f"pair {p_idx+1} ({tag}) done")

    print("\n" + "=" * 100)
    print("FLIP TEST (thresholded CAM vs its inversion, against permutation-corrected IBM)")
    print("=" * 100)
    for layer in LAYERS:
        sub = [r for r in records if r["layer"] == layer]
        orig_f1 = [r["orig_f1"] for r in sub]
        flip_f1 = [r["flip_f1"] for r in sub]
        orig_acc = [r["orig_acc"] for r in sub]
        flip_acc = [r["flip_acc"] for r in sub]
        frac_neg = [r["frac_negative_cam_weights"] for r in sub]
        print(f"{layer}: orig F1={np.mean(orig_f1):.3f}±{np.std(orig_f1):.3f} acc={np.mean(orig_acc):.3f} | "
              f"FLIPPED F1={np.mean(flip_f1):.3f}±{np.std(flip_f1):.3f} acc={np.mean(flip_acc):.3f} | "
              f"mean frac negative CAM weights={np.mean(frac_neg):.3f}")
    print("=" * 100)

    csv_path = out_dir / "separation_alignment_flip_test.csv"
    import csv as _csv
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    (out_dir / "separation_alignment_flip_test.json").write_text(json.dumps(records, indent=2))
    print(f"[+] Saved: {csv_path}")


if __name__ == "__main__":
    main()
