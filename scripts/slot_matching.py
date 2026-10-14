"""Shared PIT slot-to-source matching.

The network's VAD head runs on mask_per_speaker[:, s] and the separation output
out_separation[:, s] — both are PIT slots with NO fixed assignment to true source
0/1. Any analysis that scores slot s against source s's reference WITHOUT matching
is wrong for roughly half the pairs.

This module provides the single `match_slots` function that every analysis must
call. No analysis script may index an output slot as a source directly.

Criteria (network outputs only, never the CAMs):
  - PRIMARY "sisdr": permutation that maximizes the sum of SI-SDR between the
    separated output waveforms and the clean sources.
  - SECONDARY "vad_auc": permutation that maximizes the sum of the network's own
    VAD-probability AUC against each source's reference VAD (the audit criterion).
"""

import numpy as np
import torch


def si_sdr(estimate, reference, eps=1e-8):
    """Scale-invariant SDR (dB) between two 1-D waveforms."""
    est = np.asarray(estimate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    ref = ref - ref.mean()
    est = est - est.mean()
    ref_energy = np.sum(ref ** 2) + eps
    alpha = np.dot(est, ref) / ref_energy
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10(np.sum(target ** 2) / (np.sum(noise ** 2) + eps) + eps))


def _si_sdr_sum(out_sep, sig0, sig1, swap):
    """out_sep: (2, samples) numpy. swap=False -> slot0->src0, slot1->src1."""
    o = out_sep.detach().cpu().numpy()[0] if torch.is_tensor(out_sep) else np.asarray(out_sep)
    if not swap:
        return si_sdr(o[0], sig0) + si_sdr(o[1], sig1)
    return si_sdr(o[0], sig1) + si_sdr(o[1], sig0)


def _vad_auc_sum(vad_probs, v0, v1, swap):
    """vad_probs: (2, T) numpy post-sigmoid per slot. Higher = more active."""
    from sklearn.metrics import roc_auc_score

    def auc(labels, scores):
        labels = np.asarray(labels)
        if labels.sum() == 0 or labels.sum() == len(labels):
            return 0.5
        return float(roc_auc_score(labels, scores))

    vp = vad_probs.detach().cpu().numpy()[0] if torch.is_tensor(vad_probs) else np.asarray(vad_probs)
    if not swap:
        return auc(v0, vp[0]) + auc(v1, vp[1])
    return auc(v1, vp[0]) + auc(v0, vp[1])


def match_slots(criterion, out_sep=None, vad_probs=None, sig0=None, sig1=None,
                v0=None, v1=None):
    """Return (slot_to_source, swapped, margin).

    slot_to_source: dict mapping slot index -> true source index (0 or 1).
    swapped: True if the chosen permutation is the swap.
    margin: score(chosen) - score(other); larger = more confident.

    criterion: "sisdr" (primary; needs out_sep, sig0, sig1) or
               "vad_auc" (secondary; needs vad_probs, v0, v1).
    """
    if criterion == "sisdr":
        assert out_sep is not None and sig0 is not None and sig1 is not None
        s_straight = _si_sdr_sum(out_sep, sig0, sig1, swap=False)
        s_swapped = _si_sdr_sum(out_sep, sig0, sig1, swap=True)
    elif criterion == "vad_auc":
        assert vad_probs is not None and v0 is not None and v1 is not None
        s_straight = _vad_auc_sum(vad_probs, v0, v1, swap=False)
        s_swapped = _vad_auc_sum(vad_probs, v0, v1, swap=True)
    else:
        raise ValueError(f"unknown criterion {criterion}")
    swapped = bool(s_swapped > s_straight)
    margin = float(max(s_straight, s_swapped) - min(s_straight, s_swapped))
    slot_to_source = {s: (1 - s if swapped else s) for s in range(2)}
    return slot_to_source, swapped, margin
