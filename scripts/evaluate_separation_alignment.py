"""Pilot: does Grad-CAM explain the separation decision (IBM ground truth)?

Pilot/diagnostic-stage: tests whether attribution aligns with the Ideal Binary
Mask (IBM) ground truth from clean sources -- the separation analog of the
(closed) VAD-alignment investigation.

Carries over the VAD lessons: real ceiling check, properly defined target,
matched chance baseline, small pilot before scaling, honest reporting either way.

- Ground truth: IBM(f,t) = 1 for speaker s if |S_s(f,t)| > |S_other(f,t)|, from
  the two clean sources' STFT magnitudes.
- Ceiling: network's own predicted mask (sigmoid output) vs. IBM, accuracy/F1
  per T-F bin.
- Target: Grad-CAM backpropagating from the network's predicted mask value at a
  chosen (freq, time) bin, for a chosen speaker.
- Chance baseline: class-balance-matched Bernoulli matched to the IBM active
  rate, averaged over seeds.
- Pilot: 1-2 existing example pairs, both block 9 and block 23.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
import torchaudio

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
NUM_FREQ = N_FFT // 2 + 1  # 257
LAYERS = ["TCN.TCN.9.conv1d", "TCN.TCN.23.conv1d"]
CHANCE_SEEDS = range(20)
NUM_PILOT_PAIRS = 35  # full existing pool: 20 selection + 15 holdout (both already fetched)


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
    stft = torch.stft(t, N_FFT, HOP, window=window, return_complex=True)
    return torch.abs(stft).numpy()  # (257, 188)


def ideal_binary_mask(sig_self, sig_other):
    """IBM(f,t) = 1 if |S_self(f,t)| > |S_other(f,t)| else 0."""
    return (stft_mag(sig_self) > stft_mag(sig_other)).astype(bool)  # (257, 188)


def binary_metrics(pred, ref):
    pred = pred.astype(bool); ref = ref.astype(bool)
    tp = np.sum(pred & ref); fp = np.sum(pred & ~ref)
    fn = np.sum(~pred & ref); tn = np.sum(~pred & ~ref)
    total = pred.size
    acc = float((tp + tn) / total) if total > 0 else 0.0
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f1 = float(2*p*r/(p+r)) if p+r else 0.0
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def chance_metrics(ref_mask, seeds=CHANCE_SEEDS):
    base_rate = float(np.mean(ref_mask))
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rows.append(binary_metrics(rng.random(ref_mask.shape) < base_rate, ref_mask))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def compute_cam_2d(model, audio, layer, speaker, freq, time, device):
    """Grad-CAM targeting the predicted mask value at (freq, time) for speaker."""
    inp = audio.clone().detach().to(device).float().requires_grad_(True)
    gc = GradCAM(model, layer, device)
    try:
        with torch.enable_grad():
            _ = model(inp)
            # model.mask_per_speaker: (B, num_spk, 257, T), post-sigmoid
            target = model.mask_per_speaker[0, speaker, freq, time]
            target.backward()
            act, grad = gc.hook.activations, gc.hook.gradients
            w = grad.mean(dim=tuple(range(2, grad.ndim)), keepdim=True)
            raw = torch.relu((w * act).sum(dim=1)).detach().cpu().numpy()[0]  # (T,)
            return raw
    finally:
        gc.hook.remove_hooks()


def cam_to_tf_2d(cam_1d):
    """Broadcast a 1D time CAM to a (257, 188) T-F map to compare against IBM."""
    if len(cam_1d) < NUM_FRAMES:
        cam_1d = np.pad(cam_1d, (0, NUM_FRAMES - len(cam_1d)))
    cam_1d = cam_1d[:NUM_FRAMES]
    return np.tile(cam_1d, (NUM_FREQ, 1))


def minmax(v):
    v = np.asarray(v, dtype=np.float64)
    span = v.max() - v.min()
    return (v - v.min()) / span if span > 1e-12 else np.zeros_like(v)


def load_pilot_pairs():
    sel = SpeakerSampler("data/librispeech_samples", seed=123)
    ids = sel.speaker_ids
    pairs = []
    n_sel = min(NUM_PILOT_PAIRS, len(ids) // 2)
    for i in range(n_sel):
        s1, s2 = ids[2 * i], ids[2 * i + 1]
        pairs.append((sel.sample_utterance(s1), sel.sample_utterance(s2), f"sel{i+1}"))

    hold = SpeakerSampler("data/librispeech_holdout", seed=123)
    hold_ids = hold.speaker_ids
    remaining = NUM_PILOT_PAIRS - len(pairs)
    n_hold = min(len(hold_ids) // 2, remaining)
    for i in range(n_hold):
        s1, s2 = hold_ids[2 * i], hold_ids[2 * i + 1]
        pairs.append((hold.sample_utterance(s1), hold.sample_utterance(s2), f"hold{i+1}"))
    return pairs


def wilcoxon_vs_chance(cam_f1, chance_f1):
    cam_f1 = np.asarray(cam_f1, dtype=float)
    chance_f1 = np.asarray(chance_f1, dtype=float)
    if len(cam_f1) < 2 or np.allclose(cam_f1, chance_f1):
        return {"w_statistic": float("nan"), "p_value": float("nan"), "rank_biserial_r": float("nan"), "n": int(len(cam_f1))}
    res = stats.wilcoxon(cam_f1, chance_f1)
    diffs = cam_f1 - chance_f1
    abs_diffs = np.abs(diffs)
    ranks = stats.rankdata(abs_diffs)
    w_pos = np.sum(ranks[diffs > 0])
    w_neg = np.sum(ranks[diffs < 0])
    tot = w_pos + w_neg
    r = float((w_pos - w_neg) / tot) if tot > 0 else 0.0
    return {"w_statistic": float(res.statistic), "p_value": float(res.pvalue), "rank_biserial_r": r, "n": int(len(cam_f1))}


def main():
    device = "cpu"
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    ckpt = torch.load(gradcam_root / "weights" / "model_with_vad.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)

    pairs = load_pilot_pairs()
    out_dir = gradcam_root / "results" / "separation_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p_idx, (p1, p2, tag) in enumerate(pairs):
        sig0, sig1 = load_and_pad(p1), load_and_pad(p2)
        mix = normalize_audio(sig0 + sig1)
        audio = torch.from_numpy(mix).unsqueeze(0)

        with torch.no_grad():
            _ = model(audio.to(device))
            pred_mask = model.mask_per_speaker.detach().cpu().numpy()[0]  # (2, 257, 188), post-sigmoid

        for speaker in range(2):
            self_sig = sig0 if speaker == 0 else sig1
            other_sig = sig1 if speaker == 0 else sig0
            ibm = ideal_binary_mask(self_sig, other_sig)  # (257, 188)

            # Step 2 ceiling: network's own predicted mask vs IBM, under PIT-style
            # best-permutation assignment. The model's output slot 0/1 does not
            # consistently correspond to "true speaker 0/1" (permutation-invariant
            # training), so we evaluate both assignments and keep whichever
            # maximizes F1 per instance.
            pred_bin_self = pred_mask[speaker] >= 0.5
            straight = binary_metrics(pred_bin_self, ibm)
            # "swapped": this output slot's predicted mask vs the OTHER speaker's IBM
            ibm_other = ideal_binary_mask(other_sig, self_sig)
            swapped = binary_metrics(pred_bin_self, ibm_other)
            # choose the assignment that maximizes F1 for THIS output slot
            ceiling = swapped if swapped["f1"] >= straight["f1"] else straight

            # The Grad-CAM target and the IBM reference must use the SAME corrected
            # assignment. If swapped is better for the network's output at this slot,
            # the attribution is explaining that output's reconstruction of the OTHER
            # true speaker; so the IBM reference (and the dominant-bin target) must
            # follow the swapped assignment too. We keep both the network output slot
            # (for the CAM target) and the matching ground truth consistent.
            if swapped["f1"] >= straight["f1"]:
                ref_self_sig, ref_other_sig = other_sig, self_sig  # corrected: this slot maps to other true speaker
            else:
                ref_self_sig, ref_other_sig = self_sig, other_sig

            ibm = ideal_binary_mask(ref_self_sig, ref_other_sig)  # corrected reference

            # Step 4 chance baseline (matched to corrected IBM active rate)
            chance = chance_metrics(ibm)

            # Step 3: Grad-CAM from predicted mask value at a chosen (freq, time) bin.
            # Choose the bin where the (corrected) reference speaker dominates most.
            diff = np.abs(stft_mag(ref_self_sig) - stft_mag(ref_other_sig))
            f_star, t_star = np.unravel_index(np.argmax(diff), diff.shape)
            t_star = min(t_star, NUM_FRAMES - 1)
            f_star = min(f_star, NUM_FREQ - 1)

            for layer in LAYERS:
                cam_1d = compute_cam_2d(model, audio, layer, speaker, f_star, t_star, device)
                cam_2d = minmax(cam_to_tf_2d(cam_1d))
                cam_bin = cam_2d >= 0.5
                cam_metrics = binary_metrics(cam_bin, ibm)

                rows.append({
                    "pair": tag, "speaker": speaker, "layer": layer,
                    "target_freq": int(f_star), "target_time": int(t_star),
                    "assignment": "swapped" if swapped["f1"] >= straight["f1"] else "straight",
                    "ibm_active_rate": float(ibm.mean()),
                    "ceiling_accuracy": ceiling["accuracy"], "ceiling_f1": ceiling["f1"],
                    "ceiling_precision": ceiling["precision"], "ceiling_recall": ceiling["recall"],
                    "cam_accuracy": cam_metrics["accuracy"], "cam_f1": cam_metrics["f1"],
                    "cam_precision": cam_metrics["precision"], "cam_recall": cam_metrics["recall"],
                    "chance_accuracy": chance["accuracy"], "chance_f1": chance["f1"],
                    "chance_precision": chance["precision"], "chance_recall": chance["recall"],
                })
                print(f"  {tag} spk{speaker} {layer} ({'swapped' if swapped['f1'] >= straight['f1'] else 'straight'}): "
                      f"IBM rate={ibm.mean():.3f} | "
                      f"ceiling F1={ceiling['f1']:.3f} acc={ceiling['accuracy']:.3f} | "
                      f"CAM F1={cam_metrics['f1']:.3f} acc={cam_metrics['accuracy']:.3f} | "
                      f"chance F1={chance['f1']:.3f} acc={chance['accuracy']:.3f}")
        print(f"pair {p_idx+1} ({tag}) done")

    csv_path = out_dir / "separation_alignment_pilot.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out_dir / "separation_alignment_pilot.json").write_text(json.dumps(rows, indent=2))

    # Full-scale summary + paired Wilcoxon (CAM F1 vs per-instance chance F1), per layer
    summary = {"num_pairs": len(pairs), "num_instances": len(rows) // len(LAYERS), "layers": {}}
    print("\n" + "=" * 100)
    print(f"SEPARATION-ALIGNMENT (mask-value-target CAM vs IBM, best-permutation) - FULL POOL (N={len(pairs)} pairs, {len(rows)//len(LAYERS)} instances)")
    print("=" * 100)
    for layer in LAYERS:
        sub = [r for r in rows if r["layer"] == layer]
        cam_f1 = [r["cam_f1"] for r in sub]
        cam_acc = [r["cam_accuracy"] for r in sub]
        ceil_f1 = [r["ceiling_f1"] for r in sub]
        ceil_acc = [r["ceiling_accuracy"] for r in sub]
        ch_f1 = [r["chance_f1"] for r in sub]
        ch_acc = [r["chance_accuracy"] for r in sub]
        wres = wilcoxon_vs_chance(cam_f1, ch_f1)
        summary["layers"][layer] = {
            "cam_f1_mean": float(np.mean(cam_f1)), "cam_f1_std": float(np.std(cam_f1)),
            "cam_accuracy_mean": float(np.mean(cam_acc)), "cam_accuracy_std": float(np.std(cam_acc)),
            "ceiling_f1_mean": float(np.mean(ceil_f1)), "ceiling_f1_std": float(np.std(ceil_f1)),
            "ceiling_accuracy_mean": float(np.mean(ceil_acc)), "ceiling_accuracy_std": float(np.std(ceil_acc)),
            "chance_f1_mean": float(np.mean(ch_f1)), "chance_f1_std": float(np.std(ch_f1)),
            "chance_accuracy_mean": float(np.mean(ch_acc)), "chance_accuracy_std": float(np.std(ch_acc)),
            "wilcoxon_cam_vs_chance_f1": wres,
        }
        print(f"{layer}: CAM F1={np.mean(cam_f1):.3f}±{np.std(cam_f1):.3f} acc={np.mean(cam_acc):.3f}±{np.std(cam_acc):.3f} | "
              f"ceiling F1={np.mean(ceil_f1):.3f}±{np.std(ceil_f1):.3f} acc={np.mean(ceil_acc):.3f} | "
              f"chance F1={np.mean(ch_f1):.3f}±{np.std(ch_f1):.3f} acc={np.mean(ch_acc):.3f} | "
              f"CAM vs chance: W={wres['w_statistic']:.1f} p={wres['p_value']:.3e} r={wres['rank_biserial_r']:+.3f} (n={wres['n']})")
    (out_dir / "separation_alignment_summary.json").write_text(json.dumps(summary, indent=2))
    print("=" * 100)
    print(f"[+] Saved: {csv_path} and {out_dir / 'separation_alignment_summary.json'}")


if __name__ == "__main__":
    main()
