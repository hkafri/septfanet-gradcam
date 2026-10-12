"""Pre-registered robustness analysis of the validated block-23 VAD-logit CAM
ranking signal, under simulated reverberation and babble noise.

PRE-REGISTRATION DECLARATION (written 2026-10-12, before any computation runs):

  STATISTIC: pair-level AUC-ROC of the block-23 VAD-logit CAM against the
      speaker's own reference VAD (Silero on the clean, dry sources), averaged
      over the two speakers of each pair, as in scripts/evaluate_pair_level.py.
      Fixed orientation (CAM high => active). No flipping, no best-of-two, no
      per-condition layer or threshold selection.

  DATA (PRIMARY): the 65 held-out pairs. SECONDARY: the 35 development pairs,
      for continuity.

  CONDITIONS (7 + clean): reverberation T60 in {0.2, 0.4, 0.6} s; babble noise
      SNR in {10, 5, 0} dB; one combined condition (T60=0.4 s with SNR=5 dB).

  PRIMARY TESTS, each Holm-corrected across the 7 degraded conditions at
      alpha = 0.05:
      (a) one-sample Wilcoxon across pairs of the condition's AUC vs. 0.5
          (does the signal persist);
      (b) paired Wilcoxon across pairs of (AUC_clean - AUC_condition)
          (is there degradation).

  NON-INFERIORITY MARGIN: a condition is called "robust" only if the lower bound
      of the 95% pair-bootstrap CI of (AUC_condition - AUC_clean) is above
      -0.06 (one third of the clean-vs-chance gap, 0.681 - 0.5). Report the CIs
      so a reader can apply another margin.

  Block 9 is SECONDARY/EXPLORATORY (contrast only).

  REFERENCE VAD: Silero VAD on the CLEAN, DRY sources (ground truth for when
      each speaker talks, independent of the degradation). Consequence: in the
      reverberation conditions, reverberant tails extend acoustic energy beyond
      the dry-source activity, so the reference may under-mark the tail frames.

  No MAE and no thresholded F1 are used anywhere in this analysis.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
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

SAMPLE_RATE = 16000
TARGET_SECONDS = 3.0
TARGET_LENGTH = int(TARGET_SECONDS * SAMPLE_RATE)
HOP = 256
NUM_FRAMES = 1 + TARGET_LENGTH // HOP
N_BOOT = 10000
ALPHA = 0.05
MARGIN = -0.06

FULLSCALE_DIR = gradcam_root / "data" / "librispeech_fullscale"
SAMPLES_DIR = gradcam_root / "data" / "librispeech_samples"
HOLDOUT_DIR = gradcam_root / "data" / "librispeech_holdout"
SELECTION_CSV = gradcam_root / "results" / "librispeech_gradcam" / "multi_pair_results.csv"
HOLDOUT_CSV = gradcam_root / "results" / "librispeech_gradcam" / "holdout_results.csv"

BLOCK_23 = "TCN.TCN.23.conv1d"
BLOCK_9 = "TCN.TCN.9.conv1d"
T60_VALUES = [0.2, 0.4, 0.6]
SNR_VALUES = [10, 5, 0]
CONDITIONS = (["clean"] + [f"reverb_t60_{t}" for t in T60_VALUES]
              + [f"noise_snr_{s}" for s in SNR_VALUES] + ["combo_t60_0.4_snr_5"])
DEGRADED = CONDITIONS[1:]


def normalize_audio(a, target_peak=0.9):
    a = a.astype(np.float32)
    return a / max(np.abs(a).max(), 1e-8) * target_peak


def load_and_pad(path):
    s = load_utterance(path)[:TARGET_LENGTH]
    pad = np.zeros(TARGET_LENGTH, dtype=np.float32)
    pad[: len(s)] = s
    return pad


def minmax(v):
    v = np.asarray(v, dtype=np.float64)
    span = v.max() - v.min()
    return (v - v.min()) / span if span > 1e-12 else np.zeros_like(v)


def compute_cam(model, audio, layer, speaker, frame, device):
    inp = audio.clone().detach().to(device).float().requires_grad_(True)
    gc = GradCAM(model, layer, device)
    try:
        with torch.enable_grad():
            _ = model(inp)
            model.vad_logits[0, speaker, frame].backward()
            act, grad = gc.hook.activations, gc.hook.gradients
            w = grad.mean(dim=tuple(range(2, grad.ndim)), keepdim=True)
            return torch.relu((w * act).sum(dim=1)).detach().cpu().numpy()[0]
    finally:
        gc.hook.remove_hooks()


def silero_mask(vad_model, clean, num_frames):
    ts = get_speech_timestamps(torch.from_numpy(clean.astype(np.float32)), vad_model,
                               sampling_rate=SAMPLE_RATE, return_seconds=False)
    mask = np.zeros(num_frames, dtype=bool)
    for seg in ts:
        s = seg["start"] // HOP
        e = min(num_frames, seg["end"] // HOP + 1)
        mask[s:e] = True
    return mask


def simulate_reverb(signal, t60, src_pos, mic_pos, room_dim, fs=SAMPLE_RATE):
    e_absorption, max_order = pra.inverse_sabine(t60, list(room_dim))
    room = pra.ShoeBox(list(room_dim), fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source(list(src_pos), signal=signal.astype(np.float64))
    room.add_microphone(list(mic_pos))
    room.simulate()
    out = room.mic_array.signals[0][:len(signal)]
    if len(out) < len(signal):
        out = np.pad(out, (0, len(signal) - len(out)))
    return out.astype(np.float32)


def apply_reverb_to_pair(sig0, sig1, t60, pair_seed):
    rng = np.random.default_rng(pair_seed)
    room_dim = (6.0 + rng.uniform(-0.5, 0.5), 5.0 + rng.uniform(-0.5, 0.5), 3.0)
    mic_pos = (room_dim[0] / 2, room_dim[1] / 2, 1.5)
    src0 = (rng.uniform(1.0, room_dim[0] - 1.0), rng.uniform(1.0, room_dim[1] - 1.0), 1.5)
    src1 = (rng.uniform(1.0, room_dim[0] - 1.0), rng.uniform(1.0, room_dim[1] - 1.0), 1.5)
    rev0 = simulate_reverb(sig0, t60, src0, mic_pos, room_dim)
    rev1 = simulate_reverb(sig1, t60, src1, mic_pos, room_dim)
    for rev, orig in ((rev0, sig0), (rev1, sig1)):
        op, rp = np.max(np.abs(orig)), np.max(np.abs(rev))
        if rp > 1e-8:
            rev *= (op / rp)
    return rev0, rev1


def build_babble_track(unused_paths, rng, n_sources=8, length=TARGET_LENGTH):
    chosen = rng.choice(len(unused_paths), size=min(n_sources, len(unused_paths)), replace=False)
    babble = np.zeros(length, dtype=np.float32)
    for idx in chosen:
        babble += load_and_pad(unused_paths[idx])
    return normalize_audio(babble, target_peak=1.0)


def add_noise_at_snr(signal, noise, snr_db):
    sig_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    target_noise_power = sig_power / (10 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / max(noise_power, 1e-12))
    return signal + noise * scale


def build_mixture(sig0, sig1, condition, babble_track, pair_idx):
    s0, s1 = sig0, sig1
    if condition.startswith("reverb") or condition.startswith("combo"):
        t60 = float(condition.split("t60_")[1].split("_")[0])
        s0, s1 = apply_reverb_to_pair(s0, s1, t60, pair_seed=9000 + pair_idx)
    mixture = normalize_audio(s0 + s1)
    if condition.startswith("noise") or condition.startswith("combo"):
        snr = float(condition.split("snr_")[1].split("_")[0])
        mixture = normalize_audio(add_noise_at_snr(mixture, babble_track, snr))
    return mixture


def resolve_pairs(subset):
    """subset in {'dev','heldout','all'}. Returns list of dicts with exact saved
    utterances (existing 35) and resolve_new_pairs-replicated 65."""
    pairs = []
    existing = []
    for cp, base in ((SELECTION_CSV, SAMPLES_DIR), (HOLDOUT_CSV, HOLDOUT_DIR)):
        for row in csv.DictReader(open(cp)):
            existing.append({"path_0": base / row["utt_0"], "path_1": base / row["utt_1"],
                             "used_names": {row["utt_0"], row["utt_1"]}})
    by = {}
    for f in sorted(FULLSCALE_DIR.glob("*.flac")):
        by.setdefault(f.name.split("-")[0], []).append(f)
    sids = sorted(by)
    new = []
    for i in range(0, min(65 * 2, len(sids) - 1), 2):
        new.append({"path_0": by[sids[i]][0], "path_1": by[sids[i + 1]][0],
                    "used_names": {by[sids[i]][0].name, by[sids[i + 1]][0].name}})
    if subset == "dev":
        return existing
    if subset == "heldout":
        return new
    return existing + new


def wilcoxon_one_sample(values, null=0.5):
    v = np.asarray(values, float); d = v - null
    res = stats.wilcoxon(d)
    ranks = stats.rankdata(np.abs(d))
    w_pos = np.sum(ranks[d > 0]); w_neg = np.sum(ranks[d < 0])
    tot = w_pos + w_neg
    return {"w": float(res.statistic), "p": float(res.pvalue),
            "rank_biserial_r": float((w_pos - w_neg) / tot) if tot > 0 else 0.0, "n": int(len(v))}


def wilcoxon_paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = stats.wilcoxon(a, b)
    d = a - b
    ranks = stats.rankdata(np.abs(d))
    w_pos = np.sum(ranks[d > 0]); w_neg = np.sum(ranks[d < 0])
    tot = w_pos + w_neg
    return {"w": float(res.statistic), "p": float(res.pvalue),
            "rank_biserial_r": float((w_pos - w_neg) / tot) if tot > 0 else 0.0, "n": int(len(a))}


def boot_ci_mean(values, n_boot=N_BOOT, seed=0):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def holm_adjust(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="heldout", choices=["heldout", "dev", "all"])
    ap.add_argument("--smoke", type=int, default=0, help="run only N pairs for a smoke test")
    args = ap.parse_args()

    device = "cpu"
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    ckpt = torch.load(gradcam_root / "weights" / "model_with_vad.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    vad_model = load_silero_vad()

    pairs = resolve_pairs(args.subset)
    if args.smoke:
        pairs = pairs[: args.smoke]
    used_names = set().union(*[p["used_names"] for p in pairs])
    unused = [f for r in (SAMPLES_DIR, HOLDOUT_DIR) for f in r.glob("*.flac") if f.name not in used_names]
    # babble from speakers not in the mixture
    babble = build_babble_track(unused, np.random.default_rng(42))
    print(f"[*] subset={args.subset}, {len(pairs)} pairs, {len(unused)} leftover utts for babble")

    # per-pair per-condition pair-level AUC for CAM (block23, block9) and network VAD
    cam23 = {c: [] for c in CONDITIONS}
    cam9 = {c: [] for c in CONDITIONS}
    net = {c: [] for c in CONDITIONS}
    for idx, p in enumerate(pairs):
        sig0, sig1 = load_and_pad(p["path_0"]), load_and_pad(p["path_1"])
        v0 = silero_mask(vad_model, sig0, NUM_FRAMES)
        v1 = silero_mask(vad_model, sig1, NUM_FRAMES)
        refs = (v0, v1)
        for cond in CONDITIONS:
            mix = build_mixture(sig0, sig1, cond, babble, idx)
            audio = torch.from_numpy(mix).unsqueeze(0)
            with torch.no_grad():
                _ = model(audio.to(device))
                vad_logits = model.vad_logits.detach().cpu().numpy()[0]
                vad_probs = torch.sigmoid(model.vad_logits).detach().cpu().numpy()[0]
            for layer, store in ((BLOCK_23, cam23), (BLOCK_9, cam9)):
                aucs = []
                for speaker in range(2):
                    cam = minmax(compute_cam(model, audio, layer, speaker,
                                             int(np.argmax(np.abs(vad_logits[speaker]))), device))
                    cam = cam[:NUM_FRAMES] if len(cam) >= NUM_FRAMES else np.pad(cam, (0, NUM_FRAMES - len(cam)))
                    v_s = refs[speaker]
                    if v_s.sum() > 0 and v_s.sum() < len(v_s):
                        aucs.append(float(roc_auc_score(v_s, cam)))
                store[cond].append(float(np.mean(aucs)) if aucs else float("nan"))
            # network's own VAD AUC (post-sigmoid probabilities)
            aucs_net = []
            for speaker in range(2):
                probs = vad_probs[speaker][:NUM_FRAMES]
                v_s = refs[speaker]
                if v_s.sum() > 0 and v_s.sum() < len(v_s):
                    aucs_net.append(float(roc_auc_score(v_s, probs)))
            net[cond].append(float(np.mean(aucs_net)) if aucs_net else float("nan"))
        if (idx + 1) % 5 == 0:
            print(f"  {idx+1}/{len(pairs)} pairs done")

    clean = np.array(cam23["clean"], float)
    out = {"pre_registration": "see module docstring", "subset": args.subset,
           "n_pairs": len(pairs), "margin": MARGIN, "conditions": {}}
    rows = []
    for cond in CONDITIONS:
        a23 = np.array(cam23[cond], float)
        a9 = np.array(cam9[cond], float)
        an = np.array(net[cond], float)
        rec = {"condition": cond,
               "cam23_mean": float(np.nanmean(a23)), "cam23_std": float(np.nanstd(a23)),
               "cam23_ci": boot_ci_mean(a23),
               "cam9_mean": float(np.nanmean(a9)), "cam9_std": float(np.nanstd(a9)),
               "net_mean": float(np.nanmean(an)), "net_std": float(np.nanstd(an)),
               "net_ci": boot_ci_mean(an)}
        if cond != "clean":
            # test (a): persists vs 0.5
            ta = wilcoxon_one_sample(a23, 0.5)
            # test (b): degradation vs clean
            tb = wilcoxon_paired(clean, a23)
            # non-inferiority: CI of (cond - clean)
            diff = a23 - clean
            dlo, dhi = boot_ci_mean(diff)
            rec.update({"test_a_vs_0.5": ta, "test_b_vs_clean": tb,
                        "diff_cond_minus_clean_mean": float(np.nanmean(diff)),
                        "diff_ci": [dlo, dhi],
                        "robust_noninferior": bool(dlo > MARGIN)})
        out["conditions"][cond] = rec
        rows.append(rec)

    # Holm correction across the 7 degraded conditions for tests (a) and (b)
    pa = [out["conditions"][c]["test_a_vs_0.5"]["p"] for c in DEGRADED]
    pb = [out["conditions"][c]["test_b_vs_clean"]["p"] for c in DEGRADED]
    adj_a = holm_adjust(pa)
    adj_b = holm_adjust(pb)
    for i, c in enumerate(DEGRADED):
        out["conditions"][c]["test_a_vs_0.5"]["p_holm"] = float(adj_a[i])
        out["conditions"][c]["test_b_vs_clean"]["p_holm"] = float(adj_b[i])

    out_dir = gradcam_root / "results" / "robustness_auc"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"robustness_auc_{args.subset}_summary.json").write_text(json.dumps(out, indent=2))
    with open(out_dir / f"robustness_auc_{args.subset}.csv", "w", newline="") as f:
        fieldnames = ["condition", "cam23_mean", "cam23_std", "cam23_ci_lo", "cam23_ci_hi",
                      "cam9_mean", "net_mean", "net_ci_lo", "net_ci_hi",
                      "test_a_p", "test_a_p_holm", "test_b_p", "test_b_p_holm",
                      "diff_mean", "diff_ci_lo", "diff_ci_hi", "robust_noninferior"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in rows:
            c = rec["condition"]
            row = {"condition": c, "cam23_mean": rec["cam23_mean"], "cam23_std": rec["cam23_std"],
                   "cam23_ci_lo": rec["cam23_ci"][0], "cam23_ci_hi": rec["cam23_ci"][1],
                   "cam9_mean": rec["cam9_mean"], "net_mean": rec["net_mean"],
                   "net_ci_lo": rec["net_ci"][0], "net_ci_hi": rec["net_ci"][1]}
            if c != "clean":
                row.update({"test_a_p": rec["test_a_vs_0.5"]["p"],
                            "test_a_p_holm": rec["test_a_vs_0.5"]["p_holm"],
                            "test_b_p": rec["test_b_vs_clean"]["p"],
                            "test_b_p_holm": rec["test_b_vs_clean"]["p_holm"],
                            "diff_mean": rec["diff_cond_minus_clean_mean"],
                            "diff_ci_lo": rec["diff_ci"][0], "diff_ci_hi": rec["diff_ci"][1],
                            "robust_noninferior": rec["robust_noninferior"]})
            w.writerow(row)

    print("\n" + "=" * 96)
    print(f"ROBUSTNESS (AUC-based), subset={args.subset}, margin={MARGIN}")
    print("=" * 96)
    print(f"{'condition':22s} {'CAM23 AUC':>16s} {'persist?':>10s} {'vs clean Δ[CI]':>22s} {'robust?':>8s} {'netVAD AUC':>12s}")
    for c in CONDITIONS:
        rec = out["conditions"][c]
        ci = rec["cam23_ci"]
        net_ci = rec["net_ci"]
        if c == "clean":
            print(f"{c:22s} {rec['cam23_mean']:.3f} [{ci[0]:.3f},{ci[1]:.3f}] {'—':>10s} {'—':>22s} {'—':>8s} {rec['net_mean']:.3f} [{net_ci[0]:.3f},{net_ci[1]:.3f}]")
        else:
            pa = rec["test_a_vs_0.5"]["p_holm"]
            persist = "yes" if pa < ALPHA else "no"
            d = rec["diff_cond_minus_clean_mean"]; dl, dh = rec["diff_ci"]
            rob = "yes" if rec["robust_noninferior"] else "no"
            print(f"{c:22s} {rec['cam23_mean']:.3f} [{ci[0]:.3f},{ci[1]:.3f}] {persist:>10s} {d:+.3f} [{dl:+.3f},{dh:+.3f}] {rob:>8s} {rec['net_mean']:.3f} [{net_ci[0]:.3f},{net_ci[1]:.3f}]")
    print(f"\n[+] Saved to {out_dir}")


if __name__ == "__main__":
    main()
