"""Pre-registered baselines for the block-23 VAD-logit CAM AUC (SI-SDR matched).

PRE-REGISTRATION DECLARATION (written 2026-10-14, before any computation runs):

  PRIMARY TEST: held-out pairs (65), pair-level, paired Wilcoxon of
      AUC(CAM_block23) - AUC(energy_baseline) against the matched speaker's OWN
      reference VAD. alpha = 0.05, two-sided. Null: the CAM does not beat a
      per-frame log-energy baseline computed from the mixture alone (no network).
      This asks whether the block-23 CAM carries more voice-activity-timing
      information than the raw mixture energy envelope.

  ALL OTHER ANALYSES BELOW ARE EXPLORATORY (labelled as such):
      (1) energy baseline vs own and vs union references;
      (2) positional prior: mean reference VAD per frame index from the
          DEVELOPMENT pairs only, scored on held-out; plus a centered-bump score;
      (3) cross-pair shuffle: AUC of CAM_i against the reference of a different
          pair j (200 random j, matched on zero-padding status where possible);
      (4) padding audit: fraction of pairs/frames zero-padded; CAM AUC restricted
          to non-padded frames, and to no-padding pairs;
      (5) per-pair look: 3 held-out pairs, plotting CAM_0, CAM_1, v_0, v_1,
          v_union, and frame energy on one time axis (to see why own-vs-other is
          negative).

  Matching: SI-SDR (primary) via scripts/slot_matching.match_slots, throughout.
  Pair-level 95% bootstrap CIs (10,000 resamples of pairs) for every number.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from sklearn.metrics import roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import load_utterance
from gradcam import GradCAM
from scripts.slot_matching import match_slots

SAMPLE_RATE = 16000
TARGET_LENGTH = int(3.0 * SAMPLE_RATE)
HOP = 256
N_FFT = 512
NUM_FRAMES = 1 + TARGET_LENGTH // HOP
N_BOOT = 10000
ALPHA = 0.05

FULLSCALE_DIR = gradcam_root / "data" / "librispeech_fullscale"
SAMPLES_DIR = gradcam_root / "data" / "librispeech_samples"
HOLDOUT_DIR = gradcam_root / "data" / "librispeech_holdout"
SELECTION_CSV = gradcam_root / "results" / "librispeech_gradcam" / "multi_pair_results.csv"
HOLDOUT_CSV = gradcam_root / "results" / "librispeech_gradcam" / "holdout_results.csv"

BLOCK_23 = "TCN.TCN.23.conv1d"


def normalize_audio(a):
    a = a.astype(np.float32)
    return a / max(np.abs(a).max(), 1e-8) * 0.9


def load_and_pad(path):
    s = load_utterance(path)[:TARGET_LENGTH]
    pad = np.zeros(TARGET_LENGTH, dtype=np.float32)
    pad[: len(s)] = s
    return pad


def minmax(v):
    v = np.asarray(v, dtype=np.float64)
    span = v.max() - v.min()
    return (v - v.min()) / span if span > 1e-12 else np.zeros_like(v)


def compute_cam(model, audio, layer, slot, frame, device):
    inp = audio.clone().detach().to(device).float().requires_grad_(True)
    gc = GradCAM(model, layer, device)
    try:
        with torch.enable_grad():
            _ = model(inp)
            model.vad_logits[0, slot, frame].backward()
            act, grad = gc.hook.activations, gc.hook.gradients
            w = grad.mean(dim=tuple(range(2, grad.ndim)), keepdim=True)
            return torch.relu((w * act).sum(dim=1)).detach().cpu().numpy()[0]
    finally:
        gc.hook.remove_hooks()


def silero_mask(vad_model, clean):
    ts = get_speech_timestamps(torch.from_numpy(clean.astype(np.float32)), vad_model,
                               sampling_rate=SAMPLE_RATE, return_seconds=False)
    m = np.zeros(NUM_FRAMES, dtype=bool)
    for seg in ts:
        s = seg["start"] // HOP
        e = min(NUM_FRAMES, seg["end"] // HOP + 1)
        m[s:e] = True
    return m


def frame_log_energy(mix):
    """Per-frame log-energy of the mixture at the STFT frame rate (no network)."""
    x = torch.from_numpy(mix.astype(np.float32)).unsqueeze(0)
    stft = torch.stft(x, n_fft=N_FFT, hop_length=HOP, window=torch.hann_window(N_FFT),
                      return_complex=True, center=True, pad_mode="constant")
    power = (stft.abs() ** 2).sum(dim=1).numpy()[0]  # (T,)
    power = power[:NUM_FRAMES] if len(power) >= NUM_FRAMES else np.pad(power, (0, NUM_FRAMES - len(power)))
    return np.log(power + 1e-10)


def resolve_all_pairs():
    pairs = []
    for cp, base in ((SELECTION_CSV, SAMPLES_DIR), (HOLDOUT_CSV, HOLDOUT_DIR)):
        for row in csv.DictReader(open(cp)):
            pairs.append((base / row["utt_0"], base / row["utt_1"], "dev"))
    by = {}
    for f in sorted(FULLSCALE_DIR.glob("*.flac")):
        by.setdefault(f.name.split("-")[0], []).append(f)
    sids = sorted(by)
    for i in range(0, min(65 * 2, len(sids) - 1), 2):
        pairs.append((by[sids[i]][0], by[sids[i + 1]][0], "heldout"))
    return pairs


def safe_auc(labels, scores):
    labels = np.asarray(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


def boot_ci_mean(values, n_boot=N_BOOT, seed=0):
    v = np.asarray(values, float); v = v[~np.isnan(v)]
    if len(v) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return [float(lo), float(hi)]


def wilcoxon_one(values, null=0.5):
    v = np.asarray(values, float); v = v[~np.isnan(v)]
    if len(v) < 2:
        return {"p": float("nan"), "n": int(len(v))}
    d = v - null
    res = stats.wilcoxon(d)
    ranks = stats.rankdata(np.abs(d))
    w_pos = np.sum(ranks[d > 0]); w_neg = np.sum(ranks[d < 0])
    tot = w_pos + w_neg
    return {"p": float(res.pvalue), "rank_biserial_r": float((w_pos - w_neg) / tot) if tot > 0 else 0.0,
            "n": int(len(v))}


def wilcoxon_paired(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 2:
        return {"p": float("nan"), "n": int(len(a))}
    res = stats.wilcoxon(a, b)
    d = a - b
    ranks = stats.rankdata(np.abs(d))
    w_pos = np.sum(ranks[d > 0]); w_neg = np.sum(ranks[d < 0])
    tot = w_pos + w_neg
    return {"p": float(res.pvalue), "rank_biserial_r": float((w_pos - w_neg) / tot) if tot > 0 else 0.0,
            "n": int(len(a))}


def main():
    device = "cpu"
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    ckpt = torch.load(gradcam_root / "weights" / "model_with_vad.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    vad_model = load_silero_vad()

    pairs = resolve_all_pairs()
    dev_pairs = [p for p in pairs if p[2] == "dev"]
    held_pairs = [p for p in pairs if p[2] == "heldout"]
    print(f"[*] {len(dev_pairs)} dev + {len(held_pairs)} held-out pairs")

    # positional prior from DEV pairs only (mean matched reference VAD per frame)
    dev_ref_sum = np.zeros(NUM_FRAMES); dev_ref_cnt = 0
    for (p0, p1, _) in dev_pairs:
        sig0, sig1 = load_and_pad(p0), load_and_pad(p1)
        dev_ref_sum += silero_mask(vad_model, sig0).astype(float)
        dev_ref_sum += silero_mask(vad_model, sig1).astype(float)
        dev_ref_cnt += 2
    pos_prior = dev_ref_sum / max(dev_ref_cnt, 1)  # (NUM_FRAMES,) mean active rate
    bump = np.exp(-0.5 * ((np.arange(NUM_FRAMES) - NUM_FRAMES / 2) / (NUM_FRAMES / 6)) ** 2)  # centered bump

    out = {"pre_registration": "see docstring", "matching": "SI-SDR", "heldout": {}}
    cam_aucs, energy_own, energy_union, posprior_own, bump_own = [], [], [], [], []
    n_pairs_padded = 0
    total_frames_padded = 0
    recs = []
    held_cam_slot0 = []   # CAM for slot 0 per pair (for cross-pair shuffle)
    held_refs = []        # matched reference per slot per pair
    held_energy = []      # frame energy per pair (for per-pair look)
    held_cams = []        # both slots' CAMs per pair (for per-pair look)
    held_v = []           # (v0, v1) per pair (for per-pair look)

    for idx, (p0, p1, _) in enumerate(held_pairs):
        sig0_raw = load_utterance(p0)
        sig1_raw = load_utterance(p1)
        sig0, sig1 = load_and_pad(p0), load_and_pad(p1)
        pad0 = max(0, TARGET_LENGTH - len(sig0_raw))  # zero-padded samples
        pad1 = max(0, TARGET_LENGTH - len(sig1_raw))
        if pad0 > 0 or pad1 > 0:
            n_pairs_padded += 1
        mix = normalize_audio(sig0 + sig1)
        audio = torch.from_numpy(mix).unsqueeze(0)
        with torch.no_grad():
            out_sep, _, _ = model(audio.to(device))
            vad_logits = model.vad_logits.detach().cpu().numpy()[0]
        v0 = silero_mask(vad_model, sig0)
        v1 = silero_mask(vad_model, sig1)
        v_refs = [v0, v1]
        out_sep_np = out_sep.detach().cpu().numpy()[0]
        slot_to_source, swapped, _ = match_slots("sisdr", out_sep=out_sep_np, sig0=sig0, sig1=sig1)

        energy = frame_log_energy(mix)
        v_union = v0 | v1
        last_real_frame = min(TARGET_LENGTH - max(pad0, pad1), TARGET_LENGTH) // HOP
        np_mask = np.zeros(NUM_FRAMES, bool)
        np_mask[:max(last_real_frame, 1)] = True
        total_frames_padded += int((~np_mask).sum())

        cam_pair, energy_own_pair, posp_pair, bump_pair, energy_union_pair = [], [], [], [], []
        cam_np_pair = []
        slot_cams = []
        for slot in range(2):
            src = slot_to_source[slot]
            v_own = v_refs[src]
            frame = int(np.argmax(np.abs(vad_logits[slot])))
            cam = minmax(compute_cam(model, audio, BLOCK_23, slot, frame, device))
            cam = cam[:NUM_FRAMES] if len(cam) >= NUM_FRAMES else np.pad(cam, (0, NUM_FRAMES - len(cam)))
            slot_cams.append(cam)
            cam_pair.append(safe_auc(v_own, cam))
            energy_own_pair.append(safe_auc(v_own, energy))
            energy_union_pair.append(safe_auc(v_union, energy))
            posp_pair.append(safe_auc(v_own, pos_prior))
            bump_pair.append(safe_auc(v_own, bump))
            cam_np_pair.append(safe_auc(v_own[np_mask], cam[np_mask]))
        cam_aucs.append(np.nanmean(cam_pair))
        energy_own.append(np.nanmean(energy_own_pair))
        energy_union.append(np.nanmean(energy_union_pair))
        posprior_own.append(np.nanmean(posp_pair))
        bump_own.append(np.nanmean(bump_pair))
        held_cam_slot0.append(slot_cams[0])
        held_refs.append(v_refs)
        held_energy.append(energy)
        held_cams.append(slot_cams)
        held_v.append((v0, v1))
        recs.append({"pair_index": idx, "swapped": swapped,
                     "cam_auc": float(np.nanmean(cam_pair)),
                     "cam_auc_nopad": float(np.nanmean(cam_np_pair)),
                     "energy_own": float(np.nanmean(energy_own_pair)),
                     "energy_union": float(np.nanmean(energy_union_pair)),
                     "posprior_own": float(np.nanmean(posp_pair)),
                     "bump_own": float(np.nanmean(bump_pair)),
                     "pad0_samples": int(pad0), "pad1_samples": int(pad1)})
        if (idx + 1) % 10 == 0:
            print(f"  heldout {idx+1}/{len(held_pairs)}")

    # ---- (3) cross-pair shuffle: CAM of pair i (slot 0) vs reference of pair j ----
    # 200 random j per i, matched on zero-padding status (padded vs not) where possible.
    shuffle_vals = []
    padded_flags = np.array([1 if (r["pad0_samples"] > 0 or r["pad1_samples"] > 0) else 0 for r in recs])
    for i in range(len(held_pairs)):
        rng = np.random.default_rng(7000 + i)
        # candidate js with the same padding flag as i, excluding i
        same_pad = np.nonzero((padded_flags == padded_flags[i]) & (np.arange(len(held_pairs)) != i))[0]
        pool = same_pad if len(same_pad) >= 5 else np.arange(len(held_pairs))
        js = rng.choice(pool, size=min(200, len(pool)), replace=True)
        cam_i = held_cam_slot0[i]
        v0_i = held_v[i][0]
        for j in js:
            v0_j = held_v[j][0]
            a = safe_auc(v0_j, cam_i)
            if not np.isnan(a):
                shuffle_vals.append(a)
    shuffle_vals = np.array(shuffle_vals, float)

    cam_aucs = np.array(cam_aucs); energy_own = np.array(energy_own)
    energy_union = np.array(energy_union); posprior_own = np.array(posprior_own); bump_own = np.array(bump_own)

    # PRIMARY: CAM vs energy (own reference), paired Wilcoxon
    primary = wilcoxon_paired(cam_aucs, energy_own)

    out["heldout"] = {
        "n_pairs": len(held_pairs),
        "cam_auc": {"mean": float(np.nanmean(cam_aucs)), "ci95": boot_ci_mean(cam_aucs)},
        "PRIMARY_cam_vs_energy_own": {"cam_mean": float(np.nanmean(cam_aucs)),
                                       "energy_mean": float(np.nanmean(energy_own)),
                                       "diff_mean": float(np.nanmean(cam_aucs - energy_own)),
                                       "paired_wilcoxon": primary},
        "energy_own": {"mean": float(np.nanmean(energy_own)), "ci95": boot_ci_mean(energy_own)},
        "energy_union": {"mean": float(np.nanmean(energy_union)), "ci95": boot_ci_mean(energy_union)},
        "posprior_own": {"mean": float(np.nanmean(posprior_own)), "ci95": boot_ci_mean(posprior_own)},
        "bump_own": {"mean": float(np.nanmean(bump_own)), "ci95": boot_ci_mean(bump_own)},
        "padding": {"n_pairs_padded": n_pairs_padded, "frac_pairs_padded": n_pairs_padded / len(held_pairs),
                    "total_frames_padded": total_frames_padded,
                    "frac_frames_padded": total_frames_padded / (len(held_pairs) * NUM_FRAMES)},
        "cam_auc_nopad_frames": {"mean": float(np.nanmean([r["cam_auc_nopad"] for r in recs])),
                                  "ci95": boot_ci_mean(np.array([r["cam_auc_nopad"] for r in recs]))},
    }
    # CAM AUC restricted to pairs with NO padding
    nopad = [r["cam_auc"] for r in recs if r["pad0_samples"] == 0 and r["pad1_samples"] == 0]
    out["heldout"]["cam_auc_nopad_pairs"] = {"mean": float(np.nanmean(nopad)) if nopad else float("nan"),
                                              "ci95": boot_ci_mean(np.array(nopad)) if nopad else [float("nan"), float("nan")],
                                              "n_pairs": len(nopad)}

    # (3) cross-pair shuffle summary
    out["heldout"]["crosspair_shuffle"] = {"mean": float(np.mean(shuffle_vals)),
                                            "ci95": boot_ci_mean(shuffle_vals),
                                            "n": int(len(shuffle_vals))}

    # ---- (5) per-pair look: 3 held-out pairs, plot CAM_0, CAM_1, v_0, v_1, v_union, energy ----
    fig_dir = out_dir = gradcam_root / "results" / "baselines"
    for k in range(3):
        v0, v1 = held_v[k]
        cam0, cam1 = held_cams[k][0], held_cams[k][1]
        energy = held_energy[k]
        energy_n = minmax(energy)
        t = np.arange(NUM_FRAMES) * HOP / SAMPLE_RATE
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(t, cam0, label="CAM slot0", lw=1.4)
        ax.plot(t, cam1, label="CAM slot1", lw=1.4, alpha=0.8)
        ax.plot(t, energy_n, label="frame log-energy (norm)", lw=1.0, color="gray", alpha=0.7)
        ax.step(t, v0.astype(float) * 1.05, where="mid", label="v0 (ref)", color="C0", ls="--", alpha=0.6)
        ax.step(t, v1.astype(float) * 1.05, where="mid", label="v1 (ref)", color="C1", ls="--", alpha=0.6)
        ax.step(t, (v0 | v1).astype(float) * 1.1, where="mid", label="v_union", color="k", ls=":", alpha=0.5)
        ax.set_ylim(-0.05, 1.2)
        ax.set_xlabel("time (s)"); ax.set_ylabel("normalized value / mask")
        ax.set_title(f"held-out pair {k} (block 23, SI-SDR matched)")
        ax.legend(ncol=3, fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(fig_dir / f"per_pair_look_{k}.png", dpi=130)
        plt.close(fig)
    out["heldout"]["per_pair_look_figures"] = [f"per_pair_look_{k}.png" for k in range(3)]

    out_dir = gradcam_root / "results" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baselines_summary.json").write_text(json.dumps(out, indent=2))
    with open(out_dir / "baselines_per_pair.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)

    print("\n" + "=" * 88)
    print("BASELINES (held-out, block 23, SI-SDR matched, pair-level)")
    print("=" * 88)
    h = out["heldout"]
    print(f"CAM AUC                = {h['cam_auc']['mean']:.4f} CI{h['cam_auc']['ci95']}")
    print(f"PRIMARY CAM vs energy (own): CAM {h['PRIMARY_cam_vs_energy_own']['cam_mean']:.4f} vs "
          f"energy {h['PRIMARY_cam_vs_energy_own']['energy_mean']:.4f} | "
          f"Δ={h['PRIMARY_cam_vs_energy_own']['diff_mean']:+.4f} paired p={h['PRIMARY_cam_vs_energy_own']['paired_wilcoxon']['p']:.3e}")
    print(f"energy (own)   = {h['energy_own']['mean']:.4f} CI{h['energy_own']['ci95']}")
    print(f"energy (union) = {h['energy_union']['mean']:.4f} CI{h['energy_union']['ci95']}")
    print(f"positional prior (own, from DEV) = {h['posprior_own']['mean']:.4f} CI{h['posprior_own']['ci95']}")
    print(f"centered-bump (own)              = {h['bump_own']['mean']:.4f} CI{h['bump_own']['ci95']}")
    print(f"padding: {h['padding']['n_pairs_padded']}/{h['n_pairs']} pairs padded "
          f"({h['padding']['frac_pairs_padded']*100:.1f}%), frames padded {h['padding']['frac_frames_padded']*100:.2f}%")
    print(f"CAM AUC non-padded frames = {h['cam_auc_nopad_frames']['mean']:.4f} CI{h['cam_auc_nopad_frames']['ci95']}")
    print(f"CAM AUC no-pad pairs (n={h['cam_auc_nopad_pairs']['n_pairs']}) = {h['cam_auc_nopad_pairs']['mean']:.4f} CI{h['cam_auc_nopad_pairs']['ci95']}")
    print(f"[+] Saved to {out_dir}")


if __name__ == "__main__":
    main()
