"""Grad-CAM variants porting the KHCNN regression Grad-CAM principles to Sep-TFAnet-VAD.

One entry point, `cam_variant(model, audio, layer, score, slot, ...)`:

  P1 (score):  "single_frame" z_c(t*), t* = argmax_t |z_c(t)|  (the current method, V0)
               "sum_active"   sum_{tau in A_c} z_c(tau), A_c = {tau : z_c(tau) > 0}
                              (skipped, and counted, when A_c is empty)
               "mean_all"     mean_tau z_c(tau)
  P2 (layer):  any named module; the three pre-registered cells use
               "TCN.TCN.23.conv1d", "TCN.output.0", "vad.common.relu_1"
  P3 (map):    alpha_k = mean_t(d score / d A_k), map = ReLU(sum_k alpha_k A_k), returned RAW;
               apply scaling afterwards with `apply_scale(cam, scale)`, scale in
               {"minmax", "max", "none"}.
  P5:          `info["all_zero"]` marks maps that are entirely zero after the ReLU.

Implementation notes (pre-registered):
  - forward hook stores the activation WITHOUT .detach() (the graph is needed);
  - gradients via torch.autograd.grad(score_scalar, act, retain_graph=True);
  - model.eval() is enforced;
  - `self.vad` is called once per speaker inside one forward pass, so a hook on
    `vad.common.*` fires twice. The hook stores a list and the call index equals the
    slot. Each call asserts that the gradient of z_slot w.r.t. the OTHER call's
    activation is exactly zero (graph-disconnected) and w.r.t. its own call is not.

New file only: gradcam/cam.py is not modified.
"""

import numpy as np
import torch

SCORES = ("single_frame", "sum_active", "mean_all")
LAYERS = ("TCN.TCN.23.conv1d", "TCN.output.0", "vad.common.relu_1")


def get_module_by_name(model, name):
    """Resolve a dotted module path; integer parts index into Sequential/ModuleList."""
    module = model
    for part in name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def apply_scale(cam, scale):
    """Scale a raw map: 'minmax' to [0,1], 'max' divides by the max, 'none' returns it raw."""
    cam = np.asarray(cam, dtype=np.float64)
    if scale == "none":
        return cam
    if scale == "max":
        m = cam.max()
        return cam / m if m > 1e-12 else np.zeros_like(cam)
    if scale == "minmax":
        span = cam.max() - cam.min()
        return (cam - cam.min()) / span if span > 1e-12 else np.zeros_like(cam)
    raise ValueError(f"unknown scale {scale!r}")


def cam_variant(model, audio, layer, score, slot, device="cpu"):
    """Compute one Grad-CAM variant map.

    Args:
        model: SeparationModel exposing `model.vad_logits` of shape (B, 2, T).
        audio: (1, samples) tensor, the network input (normalized mixture).
        layer: dotted module name of the target layer (P2).
        score: one of SCORES (P1).
        slot:  PIT output slot (0 or 1) whose VAD logits define the score.

    Returns:
        (cam, info): cam is the RAW (unscaled) 1-D numpy map of length T, or None when
        the score is "sum_active" and the active set A_c is empty (info records it).
        info carries t_star / n_active / score_value / all_zero / n_hook_calls.
    """
    if score not in SCORES:
        raise ValueError(f"unknown score {score!r}")
    model.eval()
    module = get_module_by_name(model, layer)
    acts = []

    def forward_hook(mod, inp, out):
        acts.append(out)  # no .detach(): the graph is needed for autograd.grad

    handle = module.register_forward_hook(forward_hook)
    info = {"layer": layer, "score": score, "slot": int(slot)}
    try:
        inp = audio.clone().detach().to(device).float()
        with torch.enable_grad():
            _ = model(inp)
            z = model.vad_logits[0, slot]  # (T,)

            if score == "single_frame":
                t_star = int(torch.argmax(torch.abs(z)))
                score_scalar = z[t_star]
                info["t_star"] = t_star
            elif score == "sum_active":
                active = z > 0  # the network's own "active" decisions, not the reference
                info["n_active"] = int(active.sum())
                if info["n_active"] == 0:
                    info["skipped_empty_active"] = True
                    return None, info
                score_scalar = z[active].sum()
            else:  # mean_all
                score_scalar = z.mean()
            info["score_value"] = float(score_scalar)

            n_calls = len(acts)
            info["n_hook_calls"] = n_calls
            if n_calls == 1:
                act = acts[0]
            else:
                # per-speaker VAD branch: call index == slot
                assert n_calls == 2, f"expected 1 or 2 hook calls, got {n_calls}"
                act = acts[slot]
                g_other = torch.autograd.grad(score_scalar, acts[1 - slot],
                                              retain_graph=True, allow_unused=True)[0]
                assert g_other is None or bool(torch.all(g_other == 0)), (
                    f"gradient of z_slot{slot} leaked into the other call's activation")

            grad = torch.autograd.grad(score_scalar, act, retain_graph=True)[0]
            if n_calls > 1:
                assert bool(torch.any(grad != 0)), (
                    f"gradient of z_slot{slot} w.r.t. its own call's activation is identically zero")

            alpha = grad.mean(dim=tuple(range(2, grad.ndim)), keepdim=True)  # mean over time
            cam = torch.relu((alpha * act).sum(dim=1)).detach().cpu().numpy()[0]
            info["all_zero"] = bool(cam.max() <= 0.0)
            return cam, info
    finally:
        handle.remove()
