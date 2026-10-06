"""Step 3: re-calibrated F1 via Youden's J optimal threshold on the continuous
block-23 VAD CAM, against the Silero reference. Diagnostic only."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from sklearn.metrics import roc_curve

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
LAYER = "TCN.TCN.23.conv1d"
NUM_PAIRS = 35


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


def compute_cam_1d(model, audio, speaker, frame, device):
    inp = audio.clone().detach().to(device).float().requires_grad_(True)
    gc = GradCAM(model, LAYER, device)
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


def youden_f1(scores, ref_mask):
    fpr, tpr, thr = roc_curve(ref_mask.astype(int), scores)
    # sklearn prepends a threshold=inf point (TPR=FPR=0). Exclude it so the
    # Youden-optimal threshold is a real, usable cutoff.
    finite = np.isfinite(thr)
    fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    best_thr = float(thr[best_idx])
    pred = scores >= best_thr
    tp = np.sum(pred & ref_mask); fp = np.sum(pred & ~ref_mask); fn = np.sum(~pred & ref_mask)
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    f1 = float(2*p*r/(p+r)) if p+r else 0.0
    return best_thr, p, r, f1


def chance_f1(ref_mask, seeds=range(20)):
    base = float(ref_mask.mean())
    vals = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rnd = rng.random(len(ref_mask)) < base
        tp = np.sum(rnd & ref_mask); fp = np.sum(rnd & ~ref_mask); fn = np.sum(~rnd & ref_mask)
        p = tp/(tp+fp) if tp+fp else 0.0
        r = tp/(tp+fn) if tp+fn else 0.0
        vals.append(2*p*r/(p+r) if p+r else 0.0)
    return float(np.mean(vals))


def load_pairs():
    sel = SpeakerSampler("data/librispeech_samples", seed=123)
    ids = sel.speaker_ids
    pairs = []
    n_sel = min(NUM_PAIRS, len(ids)//2)
    for i in range(n_sel):
        s1, s2 = ids[2*i], ids[2*i+1]
        pairs.append((sel.sample_utterance(s1), sel.sample_utterance(s2), f"sel{i+1}"))
    hold = SpeakerSampler("data/librispeech_holdout", seed=123)
    hold_ids = hold.speaker_ids
    remaining = NUM_PAIRS - len(pairs)
    for i in range(min(len(hold_ids)//2, remaining)):
        s1, s2 = hold_ids[2*i], hold_ids[2*i+1]
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
            vad_logits = model.vad_logits.detach().cpu().numpy()[0]
        for speaker in range(2):
            ref = silero_mask(vad_model, (sig0, sig1)[speaker], NUM_FRAMES)
            if ref.sum() == 0 or ref.sum() == len(ref):
                continue
            frame = int(np.argmax(np.abs(vad_logits[speaker])))
            cam = compute_cam_1d(model, audio, speaker, frame, device)
            cam = minmax(cam)
            cam = cam[:NUM_FRAMES] if len(cam) >= NUM_FRAMES else np.pad(cam, (0, NUM_FRAMES - len(cam)))
            thr, p, r, f1 = youden_f1(cam, ref)
            records.append({
                "pair": tag, "speaker": speaker, "frame": frame,
                "youden_threshold": thr, "precision": p, "recall": r, "f1": f1,
                "ref_active_rate": float(ref.mean()), "chance_f1": chance_f1(ref),
            })
        print(f"pair {p_idx+1} ({tag}) done")

    csv_path = out_dir / "vad_block23_youden_f1.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    (out_dir / "vad_block23_youden_f1.json").write_text(json.dumps(records, indent=2))

    f1s = np.array([r["f1"] for r in records])
    ch = np.array([r["chance_f1"] for r in records])
    thrs = np.array([r["youden_threshold"] for r in records])
    print("\n" + "=" * 80)
    print(f"Block 23 re-calibrated F1 (Youden's J): mean={f1s.mean():.3f}±{f1s.std():.3f} "
          f"(avg optimal thr={thrs.mean():.2f})")
    print(f"  vs. original poorly-calibrated F1 = 0.706 (block 23, thr~0.06)")
    print(f"  vs. chance baseline F1 = {ch.mean():.3f}±{ch.std():.3f}")
    print(f"  Beats chance: {'YES' if f1s.mean() > ch.mean() else 'NO'} "
          f"(diff={f1s.mean()-ch.mean():+.3f})")
    print("=" * 80)
    print(f"[+] Saved: {csv_path}")


if __name__ == "__main__":
    main()
