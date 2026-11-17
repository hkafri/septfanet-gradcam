"""Per-frame Grad-CAM maps for the VAD logit: one map per decision frame.

The existing analyses explain ONE decision per speaker per clip (the logit at
t* = argmax_t |z_c(t)|) and compare that single map against all active frames.
This module computes, for every requested decision frame tau, a separate map
with target z_c(tau), so the maps can be stacked into a decision-time x
map-position matrix M[tau, t].

Variants:
  - "gap" (the thesis method): alpha_k = mean_t grad[k, t],
    M[tau, :] = ReLU(sum_k alpha_k * act[k, :]).  Identical to compute_cam of
    scripts/evaluate_local_transition.py for a single frame (verified below).
  - "gradact" (exploratory): no pooling over time,
    M[tau, t] = ReLU(sum_k grad[k, t] * act[k, t]).  The thesis method pools
    the gradient over time (alpha_k is a mean), so WHERE in time the gradient
    lives is discarded; this variant keeps it.

sign = +1 explains decisions "active" (z_c(tau) > 0); sign = -1 explains
decisions "inactive" (z_c(tau) < 0) by backpropagating the NEGATED logit (the
--negate-loss idea of the KHCNN regression Grad-CAM port).

Rows are returned RAW (no normalization); per-row min-max and any global
scaling are derived by the callers.

Run as a script, this file performs the mandatory equivalence check:
variant="gap", sign=+1 must reproduce compute_cam of
scripts/evaluate_local_transition.py for 5 random (pair, slot, tau) with
np.allclose(atol=1e-6). The result is written to
results/per_frame_maps/equivalence_check.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

NUM_FRAMES = 188


def _frames_tensor(frames):
    return [int(f) for f in frames]


def per_frame_grads(model, audio, layer, slot, frames, sign, device="cpu",
                    try_batched=False):
    # try_batched=False by default: on CPU the per-frame autograd.grad loop is
    # faster than the vmap-batched path (the backward subgraph from the block-23
    # activation to the logit is small), measured 0.75 s vs 6.9 s for 188 frames.
    """One forward pass; d(sign * z[slot, tau]) / d(activation) for each tau.

    Returns (grads, act): grads (len(frames), C, T) and act (C, T), both
    detached torch tensors on CPU.
    """
    model.eval()
    module = dict(model.named_modules())[layer]
    store = {}

    def fwd_hook(mod, inp, out):
        store["act"] = out  # kept WITHOUT .detach() so autograd.grad can reach it

    handle = module.register_forward_hook(fwd_hook)
    try:
        inp = audio.clone().detach().to(device).float().requires_grad_(True)
        with torch.enable_grad():
            model(inp)
        act = store["act"]  # (1, C, T)
        z = model.vad_logits[0, slot]  # (T,)
        frames = _frames_tensor(frames)
        grads = None
        if try_batched and len(frames) > 1:
            try:
                outs = sign * z[torch.tensor(frames, dtype=torch.long)]
                eye = torch.eye(len(frames), dtype=outs.dtype)
                g = torch.autograd.grad(outs, act, grad_outputs=eye,
                                        is_grads_batched=True, retain_graph=True)[0]
                grads = g[:, 0].detach()
            except Exception:
                grads = None
        if grads is None:
            rows = []
            for tau in frames:
                g = torch.autograd.grad(sign * z[tau], act, retain_graph=True)[0]
                rows.append(g[0].detach())
            grads = torch.stack(rows) if rows else torch.zeros((0,) + act.shape[1:])
        return grads.cpu(), act[0].detach().cpu()
    finally:
        handle.remove()


def rows_from_grads(grads, act, variant):
    """Raw CAM rows (len(frames), T) from per-frame gradients and the activation."""
    if variant == "gap":
        alpha = grads.mean(dim=2)                      # (n, C)
        rows = torch.relu(torch.einsum("nc,ct->nt", alpha, act))
    elif variant == "gradact":
        rows = torch.relu((grads * act.unsqueeze(0)).sum(dim=1))
    else:
        raise ValueError(f"unknown variant {variant!r}")
    rows = rows.numpy().astype(np.float64)
    if rows.shape[1] >= NUM_FRAMES:
        rows = rows[:, :NUM_FRAMES]
    else:
        rows = np.pad(rows, ((0, 0), (0, NUM_FRAMES - rows.shape[1])))
    return rows


def per_frame_cam_matrix(model, audio, layer, slot, frames, sign, variant,
                         device="cpu"):
    """Matrix M of shape (len(frames), NUM_FRAMES): raw per-frame CAM rows."""
    grads, act = per_frame_grads(model, audio, layer, slot, frames, sign, device)
    return rows_from_grads(grads, act, variant)


def per_frame_cam_both_variants(model, audio, layer, slot, frames, signs,
                                device="cpu"):
    """Both variants from ONE forward pass, with a per-frame sign vector.

    signs: array of +1/-1, one per frame (the gradient of -z is -grad(z), so
    mixed signs cost nothing extra: compute grads at sign=+1, flip rows).
    Returns dict {"gap": M, "gradact": M}.
    """
    grads, act = per_frame_grads(model, audio, layer, slot, frames, +1, device)
    s = torch.as_tensor(np.asarray(signs, dtype=np.float32)).view(-1, 1, 1)
    grads = grads * s
    return {"gap": rows_from_grads(grads, act, "gap"),
            "gradact": rows_from_grads(grads, act, "gradact")}


# ---------------------------------------------------------------------------
# Mandatory equivalence check (run as a script)
# ---------------------------------------------------------------------------

def _run_equivalence_check():
    import network.model as module_arch
    from scripts.evaluate_baselines import (resolve_all_pairs, load_and_pad,
                                            normalize_audio)
    from scripts.evaluate_local_transition import compute_cam

    device = "cpu"
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    ckpt = torch.load(gradcam_root / "weights" / "model_with_vad.pth",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)

    held = [p for p in resolve_all_pairs() if p[2] == "heldout"]
    rng = np.random.default_rng(20261117)
    layer = "TCN.TCN.23.conv1d"
    checks = []
    for _ in range(5):
        k = int(rng.integers(len(held)))
        slot = int(rng.integers(2))
        tau = int(rng.integers(NUM_FRAMES))
        p0, p1, _ = held[k]
        mix = normalize_audio(load_and_pad(p0) + load_and_pad(p1))
        audio = torch.from_numpy(mix).unsqueeze(0)
        ref = compute_cam(model, audio, layer, slot, tau, device)
        ref = ref[:NUM_FRAMES] if len(ref) >= NUM_FRAMES else np.pad(ref, (0, NUM_FRAMES - len(ref)))
        mine = per_frame_cam_matrix(model, audio, layer, slot, [tau], +1, "gap", device)[0]
        ok = bool(np.allclose(mine, ref, atol=1e-6))
        checks.append({"heldout_index": k, "slot": slot, "tau": tau, "allclose": ok,
                       "max_abs_diff": float(np.max(np.abs(mine - ref)))})
        print(f"  pair#{k} slot={slot} tau={tau}: allclose={ok} "
              f"max|diff|={checks[-1]['max_abs_diff']:.2e}")
    all_ok = all(c["allclose"] for c in checks)
    out_dir = gradcam_root / "results" / "per_frame_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "equivalence_check.json").write_text(json.dumps({
        "description": "per_frame_cam_matrix(variant='gap', sign=+1) vs "
                       "evaluate_local_transition.compute_cam, 5 random (pair, slot, tau), "
                       "np.allclose(atol=1e-6)",
        "layer": layer, "all_passed": all_ok, "checks": checks}, indent=2))
    print(f"[{'+' if all_ok else '!'}] equivalence check "
          f"{'PASSED' if all_ok else 'FAILED'} -> results/per_frame_maps/equivalence_check.json")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    _run_equivalence_check()
