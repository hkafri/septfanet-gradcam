"""Port of the KHCNN regression Grad-CAM principles to Sep-TFAnet-VAD — pre-registered evaluation.

================================ PRE-REGISTRATION DECLARATION =================================
(Written 2026-11-17 and committed BEFORE any held-out computation; the commit hash of HEAD at
run time is recorded in every result file. Only the development grid, the Step-0 reproduction
and the Step-1 verification may run before the pre-registration commit.)

Development grid (selection only, 35 development pairs, SI-SDR matched, own Silero reference):
  scores (P1):  single_frame  z_c(t*), t* = argmax_t |z_c(t)|          (V0, the current method)
                sum_active    sum over {tau : z_c(tau) > 0} of z_c(tau) (skip + count if empty)
                mean_all      mean_tau z_c(tau)
  layers (P2):  TCN.TCN.23.conv1d (current), TCN.output.0 (the PReLU on the output branch, the
                last activation before the mask head), vad.common.relu_1 (the last activation
                of the VAD branch, 4 channels)
  V0 = (single_frame, TCN.TCN.23.conv1d). 3 x 3 = 9 cells.
  V* = the non-V0 cell with the highest development mean pair-level AUC.
  No other tuning, no second selection round.

Declared confirmatory tests, both on the 65 held-out pairs, two-sided, each at alpha = 0.025
(Bonferroni for two):

  PRIMARY A (ranking): paired Wilcoxon of pair-level AUC(V*) - AUC(V0) against the matched
    speaker's own Silero reference, with a 10,000-resample pair bootstrap CI of the mean
    difference. "Improves" means p < 0.025 AND the CI lower bound > 0. The paired differences
    to the frame-energy baseline and to the positional prior are reported descriptively only.

  PRIMARY B (faithfulness, the Eq. 7 analogue): for each matched slot take the min-max map
    H(t) of V*, upsample to samples by linear interpolation from the frame centres
    (np.interp(np.arange(n), np.arange(T)*256, H)), multiply the mixture waveform BEFORE the
    usual peak normalisation (no re-normalisation after the gating, no floor), run the network.
    Metric: Pearson r between the logits z_slot(x .* H) and z_slot(x) over all frames, averaged
    over the two slots, per pair. Control: the SAME map circularly shifted by a random offset of
    at least 20 frames, 20 shifts per pair, r averaged over shifts. Paired Wilcoxon of
    r(map) - r(shift).

Fixed implementation decisions, declared here before any held-out run:
  - x in the gate test is the raw (un-normalised) mixture sig0 + sig1, for BOTH z(x) and
    z(x .* H); the maps themselves are computed on the usual peak-normalised mixture, exactly
    as in the existing pipeline.
  - Shift offsets: integers drawn uniformly from [20, T-20] (T = 188 frames) with
    rng = np.random.default_rng(9000 + pair_index * 10 + slot), 20 draws per (pair, slot).
  - Zero maps (P5): AUC is computed on the min-max map (an all-zero map gives a constant score,
    hence AUC 0.5 by the roc_auc_score tie convention); the zero-map fraction is reported
    separately. Instances skipped by sum_active (empty A_c) contribute NaN and are counted.
  - Pearson r is NaN when either logit sequence is constant; NaN instances are counted and
    excluded by nanmean, like the existing AUC convention.
  - E4 (literal P1, USES GROUND TRUTH, labelled so everywhere): score = the training loss of
    this network per configs/config_with_vad.json, 0.99 * PIT(-SI-SDR) + 0.01 * BCE of the VAD
    outputs under the matched (PIT) permutation against the clean sources and the Silero
    reference, with and without negation. PIT via network/model/pit_wrapper.PITLossWrapper;
    BCE via BinaryCrossEntropyLoss_Mean copied verbatim from the upstream repo
    MordehayM/Sep-TFAnet-VAD (model/combined_loss.py) — note it operates on post-sigmoid
    probabilities, as in training. The E4 map is computed at the layer of V*; if that layer
    fires once per speaker, the E4 map is the mean of the two calls' maps. E4 is evaluated with
    the gate test only; its AUC is circular and NOT comparable with V0 or V*.
  - E5 (mean_of_relu_rows): mean over tau in A_c of the per-frame single-frame maps, each
    ReLU'd before averaging; computed at the layers of V0 and of V* on the held-out pairs.
  - Random-network control: SeparationModel re-initialised with torch.manual_seed(0), no
    checkpoint; slots matched with the random network's OWN separated outputs (same SI-SDR
    rule); gate logits come from the random network itself.
  - Exploratory E1 p-values (each non-V0 cell vs V0, paired Wilcoxon on held-out pair AUCs)
    are Holm-corrected over the 8 non-V0 cells. No claim from E1 alone.

Everything outside PRIMARY A and PRIMARY B is exploratory and labelled so.

Caveat carried from the source implementation (khcnn-gradcam-nah README): its own faithful
re-implementation of the KHCNN Grad-CAM (pretrained weights, noisy test set) reaches mean
NCC = 0.442 and NMSE = -2.14 dB (-0.98 dB linear) over 7266 samples, versus 0.98 / -13.93 dB
reported in the ICASSP 2023 paper, with about a third of the heatmaps entirely zero after the
ReLU. These principles are a HYPOTHESIS TO TEST, not a recipe known to work.
===============================================================================================

New files only (gradcam/cam_variants.py, this script, results/khcnn_principles/); imports the
shared helpers from scripts/evaluate_baselines and scripts/slot_matching instead of copying
them. CPU only. Stages are resumable: each writes one JSON and is skipped when it exists
(--force re-runs).

Usage:
  python scripts/evaluate_khcnn_principles.py --stage all
  python scripts/evaluate_khcnn_principles.py --stage step0|verify|dev|heldout|gate|random|e5|report
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
from torch import nn

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from gradcam.cam_variants import LAYERS, SCORES, apply_scale, cam_variant, get_module_by_name
from network.model.pit_wrapper import PITLossWrapper
from scripts.evaluate_baselines import (HOP, NUM_FRAMES, TARGET_LENGTH, boot_ci_mean,
                                        frame_log_energy, load_and_pad, minmax,
                                        normalize_audio, resolve_all_pairs, safe_auc,
                                        silero_mask, wilcoxon_paired)
from scripts.slot_matching import match_slots

OUT_DIR = gradcam_root / "results" / "khcnn_principles"
V0_CELL = ("single_frame", "TCN.TCN.23.conv1d")
N_SHIFTS = 20
ALPHA_CONF = 0.025  # per confirmatory test (Bonferroni for two)

EXPECTED_BASELINES = {
    "b23_heldout_auc": 0.644,
    "b23_heldout_ci": [0.613, 0.676],
    "energy_own": 0.858,
    "posprior_own": 0.881,
}
TOLERANCE = 0.005


# --------------------------------------------------------------------------------------------
# E4 loss pieces (uses ground truth; labelled so everywhere)
# --------------------------------------------------------------------------------------------

class BinaryCrossEntropyLoss_Mean(nn.Module):
    """Copied verbatim from the upstream repo MordehayM/Sep-TFAnet-VAD
    (model/combined_loss.py), as the E4 specification requires ("use it as is").
    Weighted BCE on post-sigmoid probabilities, summed over speakers and frames,
    mean over the batch."""

    def __init__(self):
        super().__init__()
        self.BCE_none = nn.BCELoss(reduction='none')

    def forward(self, pred_vad, target_vad):
        """
        pred_csd.shape = [B, num_frames]
        target_csd.shape = [B, num_frames]
        """

        target_vad = target_vad.to(torch.float32)
        target_vad[target_vad == 2] = 1
        out = self.BCE_none(pred_vad, target_vad)  # shape = [B, num_frames]
        ### For weighting
        weight = target_vad * 0.36 + 0.64 * (1 - target_vad)  # Here we give more weight to the 0's since they are less frequent
        out = out * weight
        ###
        output = torch.mean(torch.sum(out, dim=(-1, -2)).to(torch.float32))  # mean over the batch, sum over the frames and speakers
        return output


def neg_sisdr_pt(est, target, eps=None):
    """Differentiable negative SI-SDR for PITLossWrapper(pit_from='pw_pt').

    est, target: (B, time). Returns (B,). Matches the upstream calc_sisdr
    (zero-mean, alpha = (<est,target>+eps)/(||target||^2+eps))."""
    if eps is None:
        eps = torch.finfo(est.dtype).eps
    target = target - target.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    alpha = (torch.sum(est * target, dim=-1, keepdim=True) + eps) / (
        torch.sum(target ** 2, dim=-1, keepdim=True) + eps)
    scaled = alpha * target
    noise = scaled - est
    val = (torch.sum(scaled ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps)
    return -10 * torch.log10(val)


# --------------------------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------------------------

def git_state():
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=gradcam_root).stdout.strip()
    head = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain",
                "scripts/evaluate_khcnn_principles.py", "gradcam/cam_variants.py")
    return {"head": head, "new_files_dirty_vs_head": bool(dirty)}


def load_model(device="cpu"):
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    ckpt = torch.load(gradcam_root / "weights" / "model_with_vad.pth",
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    return model


def random_model(device="cpu"):
    """Same config, seed 0, no checkpoint (Step 5 sanity control)."""
    torch.manual_seed(0)
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    return module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()


def fit_frames(v):
    v = np.asarray(v, dtype=np.float64)
    return v[:NUM_FRAMES] if len(v) >= NUM_FRAMES else np.pad(v, (0, NUM_FRAMES - len(v)))


def pair_context(model, vad_model, p0, p1, device="cpu"):
    sig0, sig1 = load_and_pad(p0), load_and_pad(p1)
    mix_raw = (sig0 + sig1).astype(np.float32)  # BEFORE peak normalisation (gate-test input)
    mix = normalize_audio(sig0 + sig1)          # the usual network input (maps)
    audio = torch.from_numpy(mix).unsqueeze(0)
    with torch.no_grad():
        out_sep, _, _ = model(audio.to(device))
        vad_logits = model.vad_logits.detach().cpu().numpy()[0]
    v0 = silero_mask(vad_model, sig0)
    v1 = silero_mask(vad_model, sig1)
    slot_to_source, swapped, margin = match_slots(
        "sisdr", out_sep=out_sep.detach().cpu().numpy()[0], sig0=sig0, sig1=sig1)
    return {"sig0": sig0, "sig1": sig1, "mix_raw": mix_raw, "mix": mix, "audio": audio,
            "vad_logits": vad_logits, "v_refs": [v0, v1],
            "slot_to_source": slot_to_source, "swapped": swapped, "margin": margin}


def logits_for(model, wave, device="cpu"):
    with torch.no_grad():
        model(torch.from_numpy(np.asarray(wave, dtype=np.float32)).unsqueeze(0).to(device))
        return model.vad_logits.detach().cpu().numpy()[0]  # (2, T)


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(stats.pearsonr(a, b)[0])


def gate_logits(model, mix_raw, H_frames, device="cpu"):
    """Gate the raw mixture with a frame map (upsampled per the declaration) and run the net."""
    H = np.asarray(H_frames, dtype=np.float64)
    H_up = np.interp(np.arange(len(mix_raw)), np.arange(len(H)) * HOP, H)
    return logits_for(model, mix_raw * H_up, device)


def gate_metrics(z_gated_slot, z_base_slot):
    """Pearson r, KHCNN-style NCC, NMSE components, unchanged hard decisions."""
    a = np.asarray(z_gated_slot, float)
    b = np.asarray(z_base_slot, float)
    err = float(np.sum((a - b) ** 2))
    den = float(np.sum(b ** 2))
    ncc = float(np.abs(np.dot(a, b)) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return {"r": pearson(a, b), "ncc": ncc,
            "nmse_db": float(10 * np.log10(max(err, 1e-300) / max(den, 1e-300))),
            "err_sum": err, "den_sum": den,
            "unchanged_decision_frac": float(np.mean((a > 0) == (b > 0)))}


def cell_key(score, layer):
    return f"{score}@{layer}"


def mean_ci(values, seed=0):
    v = np.asarray(values, float)
    return {"mean": float(np.nanmean(v)) if np.any(~np.isnan(v)) else float("nan"),
            "ci95": boot_ci_mean(v, seed=seed),
            "n": int(np.sum(~np.isnan(v)))}


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    print(f"[+] wrote {path}")


def save_csv(path, rows):
    if not rows:
        return
    keys = sorted(set().union(*(r.keys() for r in rows)), key=str)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[+] wrote {path}")


def get_pairs():
    pairs = resolve_all_pairs()
    dev = [p for p in pairs if p[2] == "dev"]
    held = [p for p in pairs if p[2] == "heldout"]
    assert len(dev) == 35 and len(held) == 65, (len(dev), len(held))
    return dev, held


# --------------------------------------------------------------------------------------------
# Step 0 — baseline reproduction (stop and report if it fails)
# --------------------------------------------------------------------------------------------

def stage_step0(force=False):
    out_path = OUT_DIR / "baseline_reproduction.json"
    if out_path.exists() and not force:
        print("[=] step0 exists, skipping")
        return
    matched = gradcam_root / "results" / "slot_matching" / "matched_analysis_summary.json"
    basel = gradcam_root / "results" / "baselines" / "baselines_summary.json"
    if not matched.exists():
        print("[*] running scripts/evaluate_matched_analysis.py ...")
        subprocess.run([sys.executable, str(gradcam_root / "scripts" / "evaluate_matched_analysis.py")],
                       check=True, cwd=gradcam_root)
    if not basel.exists():
        print("[*] running scripts/evaluate_baselines.py ...")
        subprocess.run([sys.executable, str(gradcam_root / "scripts" / "evaluate_baselines.py")],
                       check=True, cwd=gradcam_root)
    m = json.loads(matched.read_text())
    b = json.loads(basel.read_text())
    got = {
        "b23_heldout_auc": m["results"]["heldout"]["b23_auc"]["mean"],
        "b23_heldout_ci": m["results"]["heldout"]["b23_auc"]["ci95"],
        "energy_own": b["heldout"]["energy_own"]["mean"],
        "posprior_own": b["heldout"]["posprior_own"]["mean"],
    }
    diffs = {k: abs(got[k] - EXPECTED_BASELINES[k])
             for k in ("b23_heldout_auc", "energy_own", "posprior_own")}
    ok = all(d <= TOLERANCE for d in diffs.values())
    out = {"git": git_state(), "expected": EXPECTED_BASELINES, "got": got,
           "abs_diffs": diffs, "tolerance": TOLERANCE, "reproduced": ok,
           "note": ("Thesis values are the SI-SDR-matched pair-level numbers; the README's "
                    "older 0.681 is the unmatched number and is not the comparison target.")}
    save_json(out_path, out)
    if not ok:
        print("[!] STEP 0 FAILED: baseline numbers differ by more than 0.005 — stopping. "
              f"diffs={diffs}. Check pairs/seeds/matching/aggregation before proceeding.")
        sys.exit(1)
    print(f"[OK] step0 reproduced: {got}")


# --------------------------------------------------------------------------------------------
# Step 1 check — equality with the existing compute_cam, and the per-call hook assertion
# --------------------------------------------------------------------------------------------

def stage_verify(force=False):
    out_path = OUT_DIR / "verification.json"
    if out_path.exists() and not force:
        print("[=] verify exists, skipping")
        return
    from silero_vad import load_silero_vad
    from scripts.evaluate_matched_analysis import compute_cam
    device = "cpu"
    model = load_model(device)
    vad_model = load_silero_vad()
    dev, _ = get_pairs()
    rng = np.random.default_rng(42)
    picks = [(int(rng.integers(0, len(dev))), int(rng.integers(0, 2))) for _ in range(5)]
    checks = []
    for pair_idx, slot in picks:
        p0, p1, _ = dev[pair_idx]
        ctx = pair_context(model, vad_model, p0, p1, device)
        frame = int(np.argmax(np.abs(ctx["vad_logits"][slot])))
        ref = compute_cam(model, ctx["audio"], "TCN.TCN.23.conv1d", slot, frame, device)
        ours, info = cam_variant(model, ctx["audio"], "TCN.TCN.23.conv1d", "single_frame",
                                 slot, device)
        same = bool(np.allclose(ours, ref, atol=1e-6))
        checks.append({"pair_idx": pair_idx, "slot": slot, "frame_existing": frame,
                       "frame_variant": info["t_star"], "allclose_1e-6": same,
                       "max_abs_diff": float(np.max(np.abs(ours - ref)))})
    # per-call hook assertion exercised on all three layers, both slots, one pair
    p0, p1, _ = dev[0]
    ctx = pair_context(model, vad_model, p0, p1, device)
    hook_checks = []
    for layer in LAYERS:
        for slot in range(2):
            cam, info = cam_variant(model, ctx["audio"], layer, "mean_all", slot, device)
            hook_checks.append({"layer": layer, "slot": slot,
                                "n_hook_calls": info["n_hook_calls"],
                                "all_zero": info["all_zero"]})
    ok = all(c["allclose_1e-6"] for c in checks)
    out = {"git": git_state(), "equality_checks": checks, "hook_checks": hook_checks,
           "passed": ok}
    save_json(out_path, out)
    if not ok:
        print("[!] STEP 1 equality check FAILED — stopping.")
        sys.exit(1)
    print("[OK] verify: equality with compute_cam holds; per-call hook assertions passed")


# --------------------------------------------------------------------------------------------
# Grid evaluation (dev = Step 2 selection; heldout = Step 3 + E1)
# --------------------------------------------------------------------------------------------

def run_grid(model, vad_model, pairs, device="cpu", label=""):
    recs = []
    zero_counts = {cell_key(s, l): [0, 0] for s in SCORES for l in LAYERS}  # [zero, computed]
    skipped = {cell_key(s, l): 0 for s in SCORES for l in LAYERS}
    for idx, (p0, p1, _) in enumerate(pairs):
        ctx = pair_context(model, vad_model, p0, p1, device)
        rec = {"pair_index": idx, "swapped": ctx["swapped"]}
        for layer in LAYERS:
            for score in SCORES:
                key = cell_key(score, layer)
                slot_aucs = []
                for slot in range(2):
                    cam, info = cam_variant(model, ctx["audio"], layer, score, slot, device)
                    if cam is None:
                        skipped[key] += 1
                        slot_aucs.append(float("nan"))
                        continue
                    zero_counts[key][1] += 1
                    zero_counts[key][0] += int(info["all_zero"])
                    v_own = ctx["v_refs"][ctx["slot_to_source"][slot]]
                    slot_aucs.append(safe_auc(v_own, fit_frames(apply_scale(cam, "minmax"))))
                rec[key] = float(np.nanmean(slot_aucs)) if np.any(~np.isnan(slot_aucs)) else float("nan")
        recs.append(rec)
        if (idx + 1) % 10 == 0:
            print(f"  {label} grid {idx + 1}/{len(pairs)}")
    summary = {}
    for s in SCORES:
        for l in LAYERS:
            key = cell_key(s, l)
            vals = np.array([r[key] for r in recs], float)
            z, n = zero_counts[key]
            summary[key] = {**mean_ci(vals),
                            "zero_map_frac": (z / n) if n else float("nan"),
                            "n_zero_maps": z, "n_maps": n,
                            "n_skipped_empty_active": skipped[key]}
    return recs, summary


def stage_dev(force=False):
    out_path = OUT_DIR / "dev_grid.json"
    if out_path.exists() and not force:
        print("[=] dev grid exists, skipping")
        return
    from silero_vad import load_silero_vad
    model = load_model()
    vad_model = load_silero_vad()
    dev, _ = get_pairs()
    recs, summary = run_grid(model, vad_model, dev, label="dev")
    v0_key = cell_key(*V0_CELL)
    candidates = {k: v["mean"] for k, v in summary.items() if k != v0_key}
    v_star = max(candidates, key=lambda k: candidates[k])
    out = {"git": git_state(), "set": "dev (35 pairs, selection only)",
           "cells": summary, "V0": v0_key, "V_star": v_star,
           "V_star_rule": "non-V0 cell with the highest development mean AUC; no other tuning"}
    save_json(out_path, out)
    save_csv(OUT_DIR / "dev_grid_per_pair.csv", recs)
    print(f"[OK] dev grid: V0={v0_key} auc={summary[v0_key]['mean']:.4f}; "
          f"V*={v_star} auc={summary[v_star]['mean']:.4f}")


def read_vstar():
    dev = json.loads((OUT_DIR / "dev_grid.json").read_text())
    v_star = dev["V_star"]
    score, layer = v_star.split("@")
    return (score, layer), dev


def require_preregistration():
    """Held-out stages refuse to run if this script is not committed (pre-registration)."""
    g = git_state()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "scripts/evaluate_khcnn_principles.py"],
        capture_output=True, text=True, cwd=gradcam_root).returncode == 0
    if not tracked or g["new_files_dirty_vs_head"]:
        print("[!] Pre-registration violation: commit scripts/evaluate_khcnn_principles.py and "
              "gradcam/cam_variants.py before any held-out computation.")
        sys.exit(1)
    return g


def stage_heldout(force=False):
    out_path = OUT_DIR / "heldout_grid.json"
    if out_path.exists() and not force:
        print("[=] heldout grid exists, skipping")
        return
    g = require_preregistration()
    from silero_vad import load_silero_vad
    model = load_model()
    vad_model = load_silero_vad()
    dev, held = get_pairs()

    # frame-energy baseline and positional prior per held-out pair (descriptive comparators)
    dev_ref_sum = np.zeros(NUM_FRAMES)
    for (p0, p1, _) in dev:
        dev_ref_sum += silero_mask(vad_model, load_and_pad(p0)).astype(float)
        dev_ref_sum += silero_mask(vad_model, load_and_pad(p1)).astype(float)
    pos_prior = dev_ref_sum / (2 * len(dev))

    recs, summary = run_grid(model, vad_model, held, label="heldout")
    for idx, (p0, p1, _) in enumerate(held):
        ctx = pair_context(model, vad_model, p0, p1)
        energy = frame_log_energy(ctx["mix"])
        e_aucs, p_aucs = [], []
        for slot in range(2):
            v_own = ctx["v_refs"][ctx["slot_to_source"][slot]]
            e_aucs.append(safe_auc(v_own, energy))
            p_aucs.append(safe_auc(v_own, pos_prior))
        recs[idx]["energy_own"] = float(np.nanmean(e_aucs))
        recs[idx]["posprior_own"] = float(np.nanmean(p_aucs))

    out = {"git": g, "set": "heldout (65 pairs)", "cells": summary,
           "label": "E1 (all 9 cells) is EXPLORATORY; only V0 and V* feed the confirmatory tests"}
    save_json(out_path, out)
    save_csv(OUT_DIR / "heldout_grid_per_pair.csv", recs)
    print("[OK] heldout grid computed")


# --------------------------------------------------------------------------------------------
# Gate tests (PRIMARY B + V0 descriptive + E2 + E3 + E4)
# --------------------------------------------------------------------------------------------

def cam_e4_loss(model, ctx, layer, negate, device="cpu"):
    """E4: Grad-CAM with the training-loss score (USES GROUND TRUTH). One map per mixture."""
    module = get_module_by_name(model, layer)
    acts = []
    handle = module.register_forward_hook(lambda m, i, o: acts.append(o))
    try:
        inp = ctx["audio"].clone().detach().to(device).float()
        with torch.enable_grad():
            out_sep, _, _ = model(inp)
            targets = torch.from_numpy(
                np.stack([ctx["sig0"], ctx["sig1"]]).astype(np.float32)).unsqueeze(0)
            pit = PITLossWrapper(loss_func=neg_sisdr_pt, pit_from="pw_pt")
            sep_loss, batch_idx = pit(out_sep, targets, return_incides=True)
            perm = batch_idx[0].tolist()  # reordered[i] = est[perm[i]] aligned to source i
            pit_matches = all(perm[src] == slot for slot, src in ctx["slot_to_source"].items())
            probs = torch.sigmoid(model.vad_logits)
            probs_r = PITLossWrapper.reorder_source(probs, batch_idx)
            target_vad = torch.from_numpy(
                np.stack([fit_frames(v) for v in ctx["v_refs"]]).astype(np.float32)).unsqueeze(0)
            bce = BinaryCrossEntropyLoss_Mean()(probs_r, target_vad)
            loss = 0.99 * sep_loss + 0.01 * bce
            score = -loss if negate else loss
            cams = []
            for act in acts:  # mean over calls when the layer fires per speaker
                grad = torch.autograd.grad(score, act, retain_graph=True, allow_unused=True)[0]
                if grad is None:
                    continue
                alpha = grad.mean(dim=tuple(range(2, grad.ndim)), keepdim=True)
                cams.append(torch.relu((alpha * act).sum(dim=1)).detach().cpu().numpy()[0])
            cam = np.mean(cams, axis=0)
            return cam, {"sep_loss": float(sep_loss), "bce": float(bce),
                         "loss": float(loss), "pit_matches_sisdr_matching": bool(pit_matches),
                         "all_zero": bool(cam.max() <= 0.0)}
    finally:
        handle.remove()


def stage_gate(force=False):
    out_path = OUT_DIR / "gate_tests.json"
    if out_path.exists() and not force:
        print("[=] gate tests exist, skipping")
        return
    g = require_preregistration()
    from silero_vad import load_silero_vad
    (vs_score, vs_layer), dev_info = read_vstar()
    model = load_model()
    vad_model = load_silero_vad()
    _, held = get_pairs()
    variants = {"V0": V0_CELL, "Vstar": (vs_score, vs_layer)}

    recs = []
    for idx, (p0, p1, _) in enumerate(held):
        ctx = pair_context(model, vad_model, p0, p1)
        z_base = logits_for(model, ctx["mix_raw"])
        rec = {"pair_index": idx}
        H_by = {}  # (variant, slot) -> min-max map (or None)
        raw_by = {}
        for vname, (score, layer) in variants.items():
            for slot in range(2):
                cam, info = cam_variant(model, ctx["audio"], layer, score, slot)
                if cam is None:
                    H_by[(vname, slot)] = None
                    raw_by[(vname, slot)] = None
                    continue
                raw_by[(vname, slot)] = fit_frames(cam)
                H_by[(vname, slot)] = fit_frames(apply_scale(cam, "minmax"))

        # PRIMARY B (Vstar) and the same gate test for V0 (descriptive)
        for vname in variants:
            r_maps, r_shifts = [], []
            for slot in range(2):
                H = H_by[(vname, slot)]
                if H is None:
                    continue
                z_g = gate_logits(model, ctx["mix_raw"], H)
                r_maps.append(pearson(z_g[slot], z_base[slot]))
                rng = np.random.default_rng(9000 + idx * 10 + slot)
                rs = []
                for _ in range(N_SHIFTS):
                    off = int(rng.integers(20, NUM_FRAMES - 20 + 1))
                    z_s = gate_logits(model, ctx["mix_raw"], np.roll(H, off))
                    rs.append(pearson(z_s[slot], z_base[slot]))
                r_shifts.append(float(np.nanmean(rs)))
            rec[f"{vname}_r_map"] = float(np.nanmean(r_maps)) if r_maps else float("nan")
            rec[f"{vname}_r_shift"] = float(np.nanmean(r_shifts)) if r_shifts else float("nan")

        # E2 (exploratory): scale variants for Vstar, full KHCNN-style metrics
        for scale in ("minmax", "max", "none"):
            mets = []
            for slot in range(2):
                raw = raw_by[("Vstar", slot)]
                if raw is None:
                    continue
                z_g = gate_logits(model, ctx["mix_raw"], apply_scale(raw, scale))
                mets.append(gate_metrics(z_g[slot], z_base[slot]))
            for m in ("r", "ncc", "nmse_db", "unchanged_decision_frac", "err_sum", "den_sum"):
                rec[f"E2_{scale}_{m}"] = (float(np.nanmean([x[m] for x in mets]))
                                          if mets else float("nan"))

        # E3 (exploratory): comparator maps, same gate test
        energy_map = fit_frames(minmax(frame_log_energy(ctx["mix"])))
        z_e = gate_logits(model, ctx["mix_raw"], energy_map)
        rec["E3_energy_r"] = float(np.nanmean([pearson(z_e[s], z_base[s]) for s in range(2)]))
        for comp in ("constant", "inverted", "permuted"):
            rvals = []
            for slot in range(2):
                H = H_by[("Vstar", slot)]
                if H is None:
                    continue
                if comp == "constant":
                    M = np.full(NUM_FRAMES, float(np.mean(H)))
                elif comp == "inverted":
                    M = 1.0 - H
                else:
                    M = np.random.default_rng(8000 + idx * 10 + slot).permutation(H)
                z_c = gate_logits(model, ctx["mix_raw"], M)
                rvals.append(pearson(z_c[slot], z_base[slot]))
            rec[f"E3_{comp}_r"] = float(np.nanmean(rvals)) if rvals else float("nan")

        # E4 (exploratory, USES GROUND TRUTH): training-loss score at the Vstar layer
        for negate in (False, True):
            tag = "E4_neg" if negate else "E4"
            cam, info = cam_e4_loss(model, ctx, vs_layer, negate)
            H = fit_frames(apply_scale(cam, "minmax"))
            z_g = gate_logits(model, ctx["mix_raw"], H)
            rec[f"{tag}_r_map"] = float(np.nanmean([pearson(z_g[s], z_base[s]) for s in range(2)]))
            rng = np.random.default_rng(9500 + idx)
            rs = []
            for _ in range(N_SHIFTS):
                off = int(rng.integers(20, NUM_FRAMES - 20 + 1))
                z_s = gate_logits(model, ctx["mix_raw"], np.roll(H, off))
                rs.append(float(np.nanmean([pearson(z_s[s], z_base[s]) for s in range(2)])))
            rec[f"{tag}_r_shift"] = float(np.nanmean(rs))
            aucs = [safe_auc(ctx["v_refs"][ctx["slot_to_source"][s]], H) for s in range(2)]
            rec[f"{tag}_auc_circular"] = float(np.nanmean(aucs))
            rec[f"{tag}_all_zero"] = info["all_zero"]
            rec[f"{tag}_pit_matches"] = info["pit_matches_sisdr_matching"]
        recs.append(rec)
        if (idx + 1) % 5 == 0:
            print(f"  gate {idx + 1}/{len(held)}")

    def col(name):
        return np.array([r.get(name, float("nan")) for r in recs], float)

    out = {"git": g, "V_star": dev_info["V_star"], "n_pairs": len(held),
           "PRIMARY_B": {
               "r_map": mean_ci(col("Vstar_r_map")),
               "r_shift": mean_ci(col("Vstar_r_shift")),
               "diff": mean_ci(col("Vstar_r_map") - col("Vstar_r_shift")),
               "paired_wilcoxon": wilcoxon_paired(col("Vstar_r_map"), col("Vstar_r_shift")),
               "alpha": ALPHA_CONF},
           "V0_gate_descriptive": {
               "r_map": mean_ci(col("V0_r_map")),
               "r_shift": mean_ci(col("V0_r_shift")),
               "diff": mean_ci(col("V0_r_map") - col("V0_r_shift")),
               "paired_wilcoxon": wilcoxon_paired(col("V0_r_map"), col("V0_r_shift"))},
           "E2_exploratory_scales": {
               scale: {m: mean_ci(col(f"E2_{scale}_{m}"))
                       for m in ("r", "ncc", "nmse_db", "unchanged_decision_frac")}
               for scale in ("minmax", "max", "none")},
           "E2_nmse_linear_ratio_db": {
               scale: float(10 * np.log10(np.nansum(col(f"E2_{scale}_err_sum"))
                                          / np.nansum(col(f"E2_{scale}_den_sum"))))
               for scale in ("minmax", "max", "none")},
           "E3_exploratory_comparators": {
               "energy_r": mean_ci(col("E3_energy_r")),
               "constant_r": mean_ci(col("E3_constant_r")),
               "inverted_r": mean_ci(col("E3_inverted_r")),
               "permuted_r": mean_ci(col("E3_permuted_r")),
               "Vstar_vs_energy_paired": wilcoxon_paired(col("Vstar_r_map"), col("E3_energy_r"))},
           "E4_exploratory_uses_ground_truth": {
               "note": ("the score uses the reference and the clean sources, so this AUC is "
                        "circular and is NOT comparable with V0 or V*; the map explains the "
                        "error, not the decision"),
               "loss": {"r_map": mean_ci(col("E4_r_map")), "r_shift": mean_ci(col("E4_r_shift")),
                        "auc_circular": mean_ci(col("E4_auc_circular")),
                        "zero_map_frac": float(np.mean(col("E4_all_zero")))},
               "negated_loss": {"r_map": mean_ci(col("E4_neg_r_map")),
                                "r_shift": mean_ci(col("E4_neg_r_shift")),
                                "auc_circular": mean_ci(col("E4_neg_auc_circular")),
                                "zero_map_frac": float(np.mean(col("E4_neg_all_zero")))},
               "pit_perm_matches_sisdr_matching_frac": float(np.mean(col("E4_pit_matches")))}}
    save_json(out_path, out)
    save_csv(OUT_DIR / "gate_tests_per_pair.csv", recs)
    print("[OK] gate tests computed")


# --------------------------------------------------------------------------------------------
# Step 5 — random-network control
# --------------------------------------------------------------------------------------------

def stage_random(force=False):
    out_path = OUT_DIR / "random_network.json"
    if out_path.exists() and not force:
        print("[=] random-network control exists, skipping")
        return
    g = require_preregistration()
    from silero_vad import load_silero_vad
    (vs_score, vs_layer), dev_info = read_vstar()
    model = random_model()
    vad_model = load_silero_vad()
    _, held = get_pairs()
    variants = {"V0": V0_CELL, "Vstar": (vs_score, vs_layer)}
    recs = []
    for idx, (p0, p1, _) in enumerate(held):
        ctx = pair_context(model, vad_model, p0, p1)  # matching uses the random net's outputs
        z_base = logits_for(model, ctx["mix_raw"])
        rec = {"pair_index": idx}
        for vname, (score, layer) in variants.items():
            aucs, r_maps, r_shifts, zeros, computed = [], [], [], 0, 0
            for slot in range(2):
                cam, info = cam_variant(model, ctx["audio"], layer, score, slot)
                if cam is None:
                    aucs.append(float("nan"))
                    continue
                computed += 1
                zeros += int(info["all_zero"])
                H = fit_frames(apply_scale(cam, "minmax"))
                v_own = ctx["v_refs"][ctx["slot_to_source"][slot]]
                aucs.append(safe_auc(v_own, H))
                z_g = gate_logits(model, ctx["mix_raw"], H)
                r_maps.append(pearson(z_g[slot], z_base[slot]))
                rng = np.random.default_rng(9000 + idx * 10 + slot)
                rs = []
                for _ in range(N_SHIFTS):
                    off = int(rng.integers(20, NUM_FRAMES - 20 + 1))
                    z_s = gate_logits(model, ctx["mix_raw"], np.roll(H, off))
                    rs.append(pearson(z_s[slot], z_base[slot]))
                r_shifts.append(float(np.nanmean(rs)))
            rec[f"{vname}_auc"] = float(np.nanmean(aucs)) if np.any(~np.isnan(aucs)) else float("nan")
            rec[f"{vname}_r_map"] = float(np.nanmean(r_maps)) if r_maps else float("nan")
            rec[f"{vname}_r_shift"] = float(np.nanmean(r_shifts)) if r_shifts else float("nan")
            rec[f"{vname}_zero"] = zeros
            rec[f"{vname}_computed"] = computed
        recs.append(rec)
        if (idx + 1) % 10 == 0:
            print(f"  random-net {idx + 1}/{len(held)}")

    def col(name):
        return np.array([r.get(name, float("nan")) for r in recs], float)

    out = {"git": g, "V_star": dev_info["V_star"],
           "note": ("SeparationModel re-initialised with torch.manual_seed(0), no checkpoint; "
                    "slots matched with the random network's own outputs; gate logits from the "
                    "random network itself. If V* looks the same here, any improvement reflects "
                    "the input, not the learned decision."),
           "V0": {"auc": mean_ci(col("V0_auc")), "r_map": mean_ci(col("V0_r_map")),
                  "r_shift": mean_ci(col("V0_r_shift")),
                  "zero_map_frac": float(np.sum(col("V0_zero")) / max(np.sum(col("V0_computed")), 1))},
           "Vstar": {"auc": mean_ci(col("Vstar_auc")), "r_map": mean_ci(col("Vstar_r_map")),
                     "r_shift": mean_ci(col("Vstar_r_shift")),
                     "zero_map_frac": float(np.sum(col("Vstar_zero")) / max(np.sum(col("Vstar_computed")), 1))}}
    save_json(out_path, out)
    save_csv(OUT_DIR / "random_network_per_pair.csv", recs)
    print("[OK] random-network control computed")


# --------------------------------------------------------------------------------------------
# E5 — mean_of_relu_rows (exploratory; many backward passes)
# --------------------------------------------------------------------------------------------

def cam_mean_of_relu_rows(model, audio, layer, slot, device="cpu"):
    """Mean over tau in A_c of the per-frame maps, each ReLU'd BEFORE averaging."""
    model.eval()
    module = get_module_by_name(model, layer)
    acts = []
    handle = module.register_forward_hook(lambda m, i, o: acts.append(o))
    try:
        inp = audio.clone().detach().to(device).float()
        with torch.enable_grad():
            _ = model(inp)
            z = model.vad_logits[0, slot]
            active = torch.nonzero(z > 0).flatten().tolist()
            if not active:
                return None, {"skipped_empty_active": True}
            act = acts[slot] if len(acts) > 1 else acts[0]
            total = None
            for tau in active:
                grad = torch.autograd.grad(z[tau], act, retain_graph=True)[0]
                alpha = grad.mean(dim=tuple(range(2, grad.ndim)), keepdim=True)
                row = torch.relu((alpha * act).sum(dim=1)).detach()
                total = row if total is None else total + row
            cam = (total / len(active)).cpu().numpy()[0]
            return cam, {"n_active": len(active), "all_zero": bool(cam.max() <= 0.0)}
    finally:
        handle.remove()


def stage_e5(force=False):
    out_path = OUT_DIR / "e5_mean_of_relu_rows.json"
    if out_path.exists() and not force:
        print("[=] E5 exists, skipping")
        return
    g = require_preregistration()
    from silero_vad import load_silero_vad
    (vs_score, vs_layer), dev_info = read_vstar()
    layers = sorted({V0_CELL[1], vs_layer})
    model = load_model()
    vad_model = load_silero_vad()
    _, held = get_pairs()
    recs = []
    for idx, (p0, p1, _) in enumerate(held):
        ctx = pair_context(model, vad_model, p0, p1)
        rec = {"pair_index": idx}
        for layer in layers:
            aucs, zeros, computed = [], 0, 0
            for slot in range(2):
                cam, info = cam_mean_of_relu_rows(model, ctx["audio"], layer, slot)
                if cam is None:
                    aucs.append(float("nan"))
                    continue
                computed += 1
                zeros += int(info["all_zero"])
                v_own = ctx["v_refs"][ctx["slot_to_source"][slot]]
                aucs.append(safe_auc(v_own, fit_frames(apply_scale(cam, "minmax"))))
            rec[f"{layer}_auc"] = float(np.nanmean(aucs)) if np.any(~np.isnan(aucs)) else float("nan")
            rec[f"{layer}_zero"] = zeros
            rec[f"{layer}_computed"] = computed
        recs.append(rec)
        if (idx + 1) % 5 == 0:
            print(f"  E5 {idx + 1}/{len(held)}")
    out = {"git": g, "label": "E5 EXPLORATORY: mean of ReLU'd per-frame maps over A_c",
           "layers": {}}
    for layer in layers:
        vals = np.array([r[f"{layer}_auc"] for r in recs], float)
        out["layers"][layer] = {
            "auc": mean_ci(vals),
            "zero_map_frac": float(sum(r[f"{layer}_zero"] for r in recs)
                                   / max(sum(r[f"{layer}_computed"] for r in recs), 1))}
    save_json(out_path, out)
    save_csv(OUT_DIR / "e5_per_pair.csv", recs)
    print("[OK] E5 computed")


# --------------------------------------------------------------------------------------------
# Report — confirmatory results (Step 3), E1 Holm, power (Step 6)
# --------------------------------------------------------------------------------------------

def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def stage_report(force=False):
    out_path = OUT_DIR / "summary.json"
    if out_path.exists() and not force:
        print("[=] report exists, skipping")
        return
    (vs_score, vs_layer), dev_info = read_vstar()
    v0_key = cell_key(*V0_CELL)
    vs_key = dev_info["V_star"]
    held = json.loads((OUT_DIR / "heldout_grid.json").read_text())
    gate = json.loads((OUT_DIR / "gate_tests.json").read_text())
    rand = json.loads((OUT_DIR / "random_network.json").read_text())
    rows = list(csv.DictReader(open(OUT_DIR / "heldout_grid_per_pair.csv")))

    def col(name):
        return np.array([float(r[name]) if r.get(name, "") != "" else float("nan")
                         for r in rows], float)

    auc_v0 = col(v0_key)
    auc_vs = col(vs_key)
    diff = auc_vs - auc_v0

    primary_a = {
        "V0": {"cell": v0_key, **held["cells"][v0_key]},
        "V_star": {"cell": vs_key, **held["cells"][vs_key]},
        "diff": mean_ci(diff),
        "paired_wilcoxon": wilcoxon_paired(auc_vs, auc_v0),
        "alpha": ALPHA_CONF,
    }
    p_a = primary_a["paired_wilcoxon"]["p"]
    ci_lo = primary_a["diff"]["ci95"][0]
    primary_a["improves"] = bool(p_a < ALPHA_CONF and ci_lo > 0)
    primary_a["descriptive_vs_energy"] = {
        "V_star_minus_energy": mean_ci(auc_vs - col("energy_own")),
        "V0_minus_energy": mean_ci(auc_v0 - col("energy_own")),
        "energy_own": mean_ci(col("energy_own"))}
    primary_a["descriptive_vs_posprior"] = {
        "V_star_minus_posprior": mean_ci(auc_vs - col("posprior_own")),
        "posprior_own": mean_ci(col("posprior_own"))}

    p_b = gate["PRIMARY_B"]["paired_wilcoxon"]["p"]
    b_lo = gate["PRIMARY_B"]["diff"]["ci95"][0]
    primary_b_passes = bool(p_b < ALPHA_CONF and b_lo > 0)

    # E1 Holm over the 8 non-V0 cells (exploratory)
    e1 = {}
    nonv0 = [k for k in held["cells"] if k != v0_key]
    pvals = []
    for k in nonv0:
        w = wilcoxon_paired(col(k), auc_v0)
        e1[k] = {"auc": held["cells"][k], "p_vs_V0_raw": w["p"],
                 "diff_vs_V0": mean_ci(col(k) - auc_v0)}
        pvals.append(w["p"] if not np.isnan(w["p"]) else 1.0)
    for k, p_adj in zip(nonv0, holm(np.array(pvals))):
        e1[k]["p_vs_V0_holm"] = float(p_adj)

    # Step 6 power
    d = diff[~np.isnan(diff)]
    sd = float(np.std(d, ddof=1))
    mde = float(2.8 * sd / np.sqrt(65))
    power = {
        "sd_of_paired_auc_diffs": sd,
        "min_detectable_auc_diff_80pct_power": mde,
        "approximation": "2.8*SD/sqrt(65), normal approximation",
        "limits": [
            "a non-significant primary A means 'not shown to improve', not 'cannot improve'",
            "the reference is itself a model (Silero VAD), not hand-labelled ground truth",
            "about 82% of frames are speech-active",
            "one dataset (LibriSpeech), 3-second clips",
            "the maps are time-only (no frequency axis)",
            "the gate test changes the input distribution: zeroed segments are digital silence",
        ]}

    outcome = []
    outcome.append(f"PRIMARY A ({'significant' if primary_a['improves'] else 'not significant'}): "
                   f"AUC(V*) - AUC(V0) = {primary_a['diff']['mean']:+.4f} "
                   f"CI {primary_a['diff']['ci95']}, p = {p_a:.3g} (alpha 0.025)")
    outcome.append(f"PRIMARY B ({'passes' if primary_b_passes else 'does not pass'}): "
                   f"r(map) - r(shift) = {gate['PRIMARY_B']['diff']['mean']:+.4f} "
                   f"CI {gate['PRIMARY_B']['diff']['ci95']}, p = {p_b:.3g} (alpha 0.025)")

    summary = {"git": git_state(),
               "preregistration_commit_of_dev_selection": dev_info["git"],
               "V0": v0_key, "V_star": vs_key,
               "dev_grid": dev_info["cells"],
               "PRIMARY_A": primary_a,
               "PRIMARY_B": {**gate["PRIMARY_B"], "passes": primary_b_passes},
               "V0_gate_descriptive": gate["V0_gate_descriptive"],
               "E1_exploratory_holm": e1,
               "E2_exploratory": {"scales": gate["E2_exploratory_scales"],
                                   "nmse_linear_ratio_db": gate["E2_nmse_linear_ratio_db"]},
               "E3_exploratory": gate["E3_exploratory_comparators"],
               "E4_exploratory_uses_ground_truth": gate["E4_exploratory_uses_ground_truth"],
               "random_network_control": rand,
               "power_and_limits": power,
               "declared_outcome": outcome,
               "khcnn_caveat": ("the source implementation's own re-run reached NCC 0.442 / "
                                "NMSE -2.14 dB (-0.98 dB linear) vs 0.98 / -13.93 dB in the "
                                "paper, with ~1/3 of heatmaps entirely zero — these principles "
                                "were a hypothesis to test, not a recipe known to work")}
    e5p = OUT_DIR / "e5_mean_of_relu_rows.json"
    if e5p.exists():
        summary["E5_exploratory"] = json.loads(e5p.read_text())

    save_json(out_path, summary)
    print("\n" + "=" * 90)
    print(f"V0 = {v0_key}   V* = {vs_key}")
    print("=" * 90)
    for line in outcome:
        print(line)
    print(f"V0 held-out AUC    = {held['cells'][v0_key]['mean']:.4f} CI {held['cells'][v0_key]['ci95']} "
          f"zero-maps {held['cells'][v0_key]['zero_map_frac']:.3f}")
    print(f"V* held-out AUC    = {held['cells'][vs_key]['mean']:.4f} CI {held['cells'][vs_key]['ci95']} "
          f"zero-maps {held['cells'][vs_key]['zero_map_frac']:.3f}")
    print(f"energy baseline    = {primary_a['descriptive_vs_energy']['energy_own']['mean']:.4f} (descriptive)")
    print(f"positional prior   = {primary_a['descriptive_vs_posprior']['posprior_own']['mean']:.4f} (descriptive)")
    print(f"power: SD(diff) = {sd:.4f}, min detectable diff (80% power) = {mde:.4f}")


STAGES = {"step0": stage_step0, "verify": stage_verify, "dev": stage_dev,
          "heldout": stage_heldout, "gate": stage_gate, "random": stage_random,
          "e5": stage_e5, "report": stage_report}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    ap.add_argument("--force", action="store_true", help="re-run the stage even if its output exists")
    ap.add_argument("--skip-e5", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    if args.stage == "all":
        for name, fn in STAGES.items():
            if name == "e5" and args.skip_e5:
                continue
            fn(force=args.force)
    else:
        STAGES[args.stage](force=args.force)


if __name__ == "__main__":
    main()
