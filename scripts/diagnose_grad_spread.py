"""Diagnostic: measure backward gradient spread from a single VAD-logit target
at several hook depths, and plot gradient-magnitude-vs-time curves.

Diagnostic only -- no evaluation changes, no layer swap, no new attribution method.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import SpeakerSampler, load_utterance

SAMPLE_RATE = 16000
TARGET_SECONDS = 3.0
HOOK_BLOCKS = [2, 6, 9, 15, 20, 22]
PLOT_BLOCKS = [2, 22]  # shallowest and deepest candidates


def normalize_audio(audio):
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    return audio / max(peak, 1e-8) * 0.9


def prepare_mixture(paths):
    target_length = int(TARGET_SECONDS * SAMPLE_RATE)
    signals = []
    for path in paths:
        signal = load_utterance(path)[:target_length]
        padded = np.zeros(target_length, dtype=np.float32)
        padded[:len(signal)] = signal
        signals.append(padded)
    mixture = normalize_audio(signals[0] + signals[1])
    return torch.from_numpy(mixture).unsqueeze(0)


def get_module_by_name(model, name):
    parts = name.split('.')
    m = model
    for p in parts:
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    return m


def measure_gradient_spread(model, audio, hook_layer_name, speaker, frame, device):
    """Forward+backward from a single VAD-logit target, capture gradient at hook layer.

    Uses retain_grad on the hooked activation so the gradient is taken at the
    same activation tensor the existing CAM hooks use (conv1d output).
    Returns per-frame gradient magnitude (L2 over channels), and the activation shape.
    """
    input_audio = audio.clone().detach().to(device).float().requires_grad_(True)
    target_module = get_module_by_name(model, hook_layer_name)

    captured = {}

    def fwd_hook(module, inp, out):
        out.retain_grad()
        captured['act'] = out

    handle = target_module.register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            _ = model(input_audio)
            target = model.vad_logits[0, speaker, frame]
            target.backward()
        act = captured['act']
        grad = act.grad  # (B, C, T) — gradient of target wrt this activation
        # per-frame gradient magnitude: L2 over channels, per frame
        grad_mag = grad.detach().abs().sum(dim=(0, 1)).cpu().numpy()  # (T,)
        return grad_mag, tuple(act.shape)
    finally:
        handle.remove()


def effective_spread(grad_mag, rel_threshold):
    """Number of frames with grad magnitude >= rel_threshold * max."""
    mx = grad_mag.max()
    if mx <= 0:
        return 0
    return int(np.sum(grad_mag >= rel_threshold * mx))


def main():
    device = "cpu"
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    checkpoint = torch.load(gradcam_root / "weights" / "model_with_vad.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)

    # Same example pair as the existing example figure (SpeakerSampler seed=0).
    sampler = SpeakerSampler("data/librispeech_samples", seed=0)
    spk_ids = sampler.sample_two_speakers()
    paths = [sampler.sample_utterance(s) for s in spk_ids]
    audio = prepare_mixture(paths)

    # Same target frame/speaker the existing CAM uses: speaker 0, argmax |logit| frame.
    with torch.no_grad():
        _ = model(audio.to(device))
        vad_logits = model.vad_logits.detach().cpu().numpy()[0]
    speaker = 0
    frame = int(np.argmax(np.abs(vad_logits[speaker])))
    print(f"[*] Example: {paths[0].name} + {paths[1].name}; target = speaker {speaker}, frame {frame}")

    # Zero out model grads between iterations.
    spread_rows = []
    curves = {}
    for block in HOOK_BLOCKS:
        model.zero_grad(set_to_none=True)
        layer = f"TCN.TCN.{block}.conv1d"
        grad_mag, act_shape = measure_gradient_spread(model, audio, layer, speaker, frame, device)
        curves[block] = grad_mag
        row = {"block": block, "blocks_remaining_to_output": 23 - block, "act_shape": act_shape}
        for thr in [0.01, 0.05, 0.10]:
            n = effective_spread(grad_mag, thr)
            row[f"spread_{int(thr*100)}pct"] = n
            row[f"spread_{int(thr*100)}pct_clip_fraction"] = n / 188.0
        spread_rows.append(row)
        print(f"  Block {block:2d} ({layer}, act_shape={act_shape}): "
              f"spread@1%={row['spread_1pct']}, @5%={row['spread_5pct']}, @10%={row['spread_10pct']} of 188")

    # Table
    print("\n" + "=" * 78)
    print(f"{'Block':>6} {'Remaining':>10} {'@1%':>6} {'@5%':>6} {'@10%':>6} {'@1% %clip':>10} {'@5% %clip':>10} {'@10% %clip':>11}")
    for r in spread_rows:
        print(f"{r['block']:>6} {r['blocks_remaining_to_output']:>10} {r['spread_1pct']:>6} {r['spread_5pct']:>6} {r['spread_10pct']:>6} "
              f"{r['spread_1pct_clip_fraction']*100:>9.1f}% {r['spread_5pct_clip_fraction']*100:>9.1f}% {r['spread_10pct_clip_fraction']*100:>10.1f}%")
    print("=" * 78)

    # Correlation between remaining depth and spread
    for thr_name in ["spread_1pct", "spread_5pct", "spread_10pct"]:
        remaining = np.array([r["blocks_remaining_to_output"] for r in spread_rows], dtype=float)
        spread = np.array([r[thr_name] for r in spread_rows], dtype=float)
        corr = np.corrcoef(remaining, spread)[0, 1]
        print(f"Correlation(blocks remaining to output, {thr_name}) = {corr:+.3f}")

    # Plots for shallowest and deepest candidates
    output_dir = gradcam_root / "results" / "grad_spread"
    output_dir.mkdir(parents=True, exist_ok=True)
    for block in PLOT_BLOCKS:
        g = curves[block]
        g_norm = g / (g.max() + 1e-12)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(g_norm, color="darkorange")
        ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.8, label="5% of max")
        ax.axhline(0.10, color="dimgray", linestyle=":", linewidth=0.8, label="10% of max")
        ax.set_title(f"Backward gradient magnitude vs. time frame at TCN.TCN.{block}.conv1d\n"
                     f"(target: speaker {speaker}, frame {frame}; blocks remaining to output: {23 - block})")
        ax.set_xlabel("Time frame (of 188)")
        ax.set_ylabel("Gradient magnitude (normalized)")
        ax.legend()
        out_path = output_dir / f"grad_spread_block{block}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
