"""Per-frame decision maps: matrix cache + descriptive numbers (Steps 4 and 5).

ALL NUMBERS PRODUCED HERE ARE EXPLORATORY. There is no confirmatory test in
this analysis. Pair-level means over the 65 held-out pairs with 95%
pair-bootstrap CIs (boot_ci_mean imported from scripts/evaluate_baselines).

For every held-out pair, both PIT slots (SI-SDR matched to sources), layer
TCN.TCN.23.conv1d: the full 188-row matrix M[tau, t] of per-frame Grad-CAM maps
(scripts/per_frame_cam.py), one row per decision frame tau, with sign(z(tau))
as the backprop sign (active decisions explained with +z, inactive with -z).
Variants "gap" (thesis method) and "gradact" (exploratory, no time pooling).

Descriptive numbers (active-decision rows only, i.e. z(tau) > 0):
 1. Locality L_W(tau) = sum_{|t-tau|<=W} M[tau,t] / sum_t M[tau,t], W in {5,10},
    vs the edge-corrected uniform value and vs the same rows circularly shifted
    by a random offset. Reported for gap and gradact.
 2. Row-level AUC of M[tau,:] against v_c (own), v_c' (other), v_c|v_c' (union);
    pair means; compared with the single-frame b23 AUC of the same pairs
    (results/slot_matching/matched_analysis_per_pair.csv).
 3. Pearson correlation between the images M_c and M_c' of the two matched
    speakers (on the union of active rows), vs the same correlation between
    M_c of one pair and M_c' of a DIFFERENT pair (200 random partners, matched
    on zero-padding status as in evaluate_baselines.py; common active frames).
 4. Locality and row-AUC split by |z_c(tau)| terciles (within-pair terciles).

Step 5 sanity control: everything above is recomputed with a randomly
re-initialised SeparationModel (same config, torch.manual_seed(0), no
checkpoint). If the random network shows the same locality, the diagonal
concentration reflects the architecture, not the learned decision.

Caching: one npz per (network, pair) in results/per_frame_maps/cache/ so the
run can resume; metrics are recomputed from the cache. The cache is
reproducible and is NOT committed (see cache/.gitignore).

Compute-only: writes only new files under results/per_frame_maps/.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
from silero_vad import load_silero_vad

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import load_utterance
from scripts.evaluate_baselines import (resolve_all_pairs, silero_mask, load_and_pad,
                                        normalize_audio, safe_auc, boot_ci_mean,
                                        TARGET_LENGTH, NUM_FRAMES)
from scripts.slot_matching import match_slots
from scripts.per_frame_cam import per_frame_cam_both_variants

LAYER = "TCN.TCN.23.conv1d"
WINDOWS = [5, 10]
VARIANTS = ["gap", "gradact"]
OUT_DIR = gradcam_root / "results" / "per_frame_maps"
CACHE_DIR = OUT_DIR / "cache"
MATCHED_CSV = gradcam_root / "results" / "slot_matching" / "matched_analysis_per_pair.csv"
N_CROSS_PARTNERS = 200


def build_model(net, device="cpu"):
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    if net == "random":
        torch.manual_seed(0)  # fixed seed, no checkpoint: architecture-only control
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    if net == "real":
        ckpt = torch.load(gradcam_root / "weights" / "model_with_vad.pth",
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    return model


def compute_pair_cache(model, vad_model, pair, net, heldout_idx, device="cpu"):
    """Full per-frame matrices + references for one pair; cached as npz."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{net}_pair{heldout_idx:02d}.npz"
    if path.exists():
        return dict(np.load(path, allow_pickle=False))
    p0, p1, _ = pair
    sig0, sig1 = load_and_pad(p0), load_and_pad(p1)
    pad0 = max(0, TARGET_LENGTH - len(load_utterance(p0)))
    pad1 = max(0, TARGET_LENGTH - len(load_utterance(p1)))
    mix = normalize_audio(sig0 + sig1)
    audio = torch.from_numpy(mix).unsqueeze(0)
    with torch.no_grad():
        out_sep, _, _ = model(audio.to(device))
        z = model.vad_logits.detach().cpu().numpy()[0]  # (2, T)
        spectrum = model.spectrum.detach().cpu().numpy()[0]  # (257, T)
    slot_to_source, swapped, _ = match_slots(
        "sisdr", out_sep=out_sep.detach().cpu().numpy()[0], sig0=sig0, sig1=sig1)
    v0 = silero_mask(vad_model, sig0)
    v1 = silero_mask(vad_model, sig1)
    data = {
        "z": z[:, :NUM_FRAMES].astype(np.float32),
        "v0": v0, "v1": v1,
        "slot_to_source": np.array([slot_to_source[0], slot_to_source[1]], int),
        "swapped": np.array(swapped),
        "pad0": np.array(pad0), "pad1": np.array(pad1),
        "spectrum": spectrum[:, :NUM_FRAMES].astype(np.float32),
    }
    frames = list(range(NUM_FRAMES))
    for slot in range(2):
        signs = np.where(z[slot, :NUM_FRAMES] > 0, 1.0, -1.0)
        mats = per_frame_cam_both_variants(model, audio, LAYER, slot, frames, signs, device)
        data[f"gap{slot}"] = mats["gap"].astype(np.float32)
        data[f"gradact{slot}"] = mats["gradact"].astype(np.float32)
    np.savez_compressed(path, **data)
    return data


def locality(row, tau, W):
    T = len(row)
    tot = row.sum()
    if tot <= 1e-12:
        return np.nan, np.nan
    lo, hi = max(0, tau - W), min(T, tau + W + 1)
    return float(row[lo:hi].sum() / tot), float((hi - lo) / T)


def rank_auc(labels, scores):
    """AUC-ROC via the tie-aware Mann-Whitney rank statistic. Numerically
    identical to evaluate_baselines.safe_auc (checked at startup), used here
    because ~10^5 sklearn roc_auc_score calls crash with a native access
    violation (numpy errstate) on this Windows setup."""
    labels = np.asarray(labels, bool)
    npos = int(labels.sum())
    nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = stats.rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def check_rank_auc(n=50, seed=123):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        labels = rng.random(NUM_FRAMES) < rng.uniform(0.1, 0.9)
        scores = rng.random(NUM_FRAMES) * rng.choice([1.0, 0.0])  # include all-tie rows
        a, b = rank_auc(labels, scores), safe_auc(labels, scores)
        assert (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12, (a, b)
    return True


def pair_metrics(data, rng):
    """Pair-level means of the Step 4 quantities, from one cached pair."""
    z = data["z"]
    v = [data["v0"].astype(bool), data["v1"].astype(bool)]
    s2s = data["slot_to_source"]
    out = {}
    # pooled per-row records across both slots
    loc = {(var, W): [] for var in VARIANTS for W in WINDOWS}
    loc_unif = {(var, W): [] for var in VARIANTS for W in WINDOWS}
    loc_shift = {(var, W): [] for var in VARIANTS for W in WINDOWS}
    auc_own, auc_other, auc_union = {v_: [] for v_ in VARIANTS}, {v_: [] for v_ in VARIANTS}, {v_: [] for v_ in VARIANTS}
    zero_rows, n_rows = 0, 0
    conf_rows = []  # (|z|, loc_gap_W10, auc_own_gap)
    for slot in range(2):
        v_own = v[s2s[slot]]
        v_other = v[1 - s2s[slot]]
        v_union = v_own | v_other
        active = np.nonzero(z[slot] > 0)[0]
        for var in VARIANTS:
            M = data[f"{var}{slot}"].astype(np.float64)
            for tau in active:
                row = M[tau]
                if var == "gap":
                    n_rows += 1
                    if row.sum() <= 1e-12:
                        zero_rows += 1
                if row.sum() <= 1e-12:
                    continue
                shift = int(rng.integers(1, NUM_FRAMES))
                row_shifted = np.roll(row, shift)
                for W in WINDOWS:
                    L, U = locality(row, tau, W)
                    Ls, _ = locality(row_shifted, tau, W)
                    loc[(var, W)].append(L)
                    loc_unif[(var, W)].append(U)
                    loc_shift[(var, W)].append(Ls)
                a_own = rank_auc(v_own, row)
                a_oth = rank_auc(v_other, row)
                a_uni = rank_auc(v_union, row)
                auc_own[var].append(a_own)
                auc_other[var].append(a_oth)
                auc_union[var].append(a_uni)
                if var == "gap":
                    L10, _ = locality(row, tau, 10)
                    conf_rows.append((abs(float(z[slot, tau])), L10, a_own))
    for var in VARIANTS:
        for W in WINDOWS:
            out[f"loc_{var}_W{W}"] = float(np.nanmean(loc[(var, W)])) if loc[(var, W)] else np.nan
            out[f"loc_{var}_W{W}_uniform"] = float(np.nanmean(loc_unif[(var, W)])) if loc_unif[(var, W)] else np.nan
            out[f"loc_{var}_W{W}_shifted"] = float(np.nanmean(loc_shift[(var, W)])) if loc_shift[(var, W)] else np.nan
        out[f"rowauc_{var}_own"] = float(np.nanmean(auc_own[var])) if auc_own[var] else np.nan
        out[f"rowauc_{var}_other"] = float(np.nanmean(auc_other[var])) if auc_other[var] else np.nan
        out[f"rowauc_{var}_union"] = float(np.nanmean(auc_union[var])) if auc_union[var] else np.nan
    out["n_active_rows"] = n_rows
    out["frac_zero_rows"] = zero_rows / n_rows if n_rows else np.nan
    out["frac_active_frames"] = float((z > 0).mean())
    # |z| terciles (within pair, pooled over both slots, gap variant)
    conf_rows.sort(key=lambda r: r[0])
    k = len(conf_rows)
    for t_i, name in enumerate(["t1", "t2", "t3"]):
        seg = conf_rows[t_i * k // 3:(t_i + 1) * k // 3]
        out[f"loc_gap_W10_{name}"] = float(np.nanmean([r[1] for r in seg])) if seg else np.nan
        out[f"rowauc_gap_own_{name}"] = float(np.nanmean([r[2] for r in seg])) if seg else np.nan
    return out


def active_union_image(data, var, frames):
    """Stack rows at the given frames for both slots -> (img_slot0, img_slot1)."""
    return (data[f"{var}0"].astype(np.float64)[frames],
            data[f"{var}1"].astype(np.float64)[frames])


def pearson(a, b):
    a, b = a.ravel(), b.ravel()
    if a.std() <= 1e-12 or b.std() <= 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", nargs="+", default=["real", "random"], choices=["real", "random"])
    args = ap.parse_args()
    device = "cpu"

    vad_model = load_silero_vad()
    check_rank_auc()
    print("[*] rank_auc == safe_auc check passed (50 random rows)")
    held = [p for p in resolve_all_pairs() if p[2] == "heldout"]
    assert len(held) == 65

    caches = {}
    for net in args.nets:
        model = build_model(net, device)
        for i, pair in enumerate(held):
            caches[(net, i)] = compute_pair_cache(model, vad_model, pair, net, i, device)
            if (i + 1) % 10 == 0:
                print(f"  [{net}] cached {i + 1}/65")
        del model

    # ---- per-pair metrics ----
    rows = []
    for i in range(65):
        rec = {"heldout_index": i, "pair_index": 35 + i}
        for net in args.nets:
            rng = np.random.default_rng(1000 + i)
            m = pair_metrics(caches[(net, i)], rng)
            rec.update({f"{net}_{k}": v for k, v in m.items()})
        rows.append(rec)

    # ---- (3) speaker-map correlation, real network ----
    if "real" in args.nets:
        pad_flag = np.array([
            1 if (int(caches[("real", i)]["pad0"]) > 0 or int(caches[("real", i)]["pad1"]) > 0) else 0
            for i in range(65)])
        act_union = [np.union1d(np.nonzero(caches[("real", i)]["z"][0] > 0)[0],
                                np.nonzero(caches[("real", i)]["z"][1] > 0)[0]) for i in range(65)]
        for var in VARIANTS:
            within, between = [], []
            for i in range(65):
                U = act_union[i]
                if len(U) >= 5:
                    img0, img1 = active_union_image(caches[("real", i)], var, U)
                    w = pearson(img0, img1)
                else:
                    w = np.nan
                within.append(w)
                rows[i][f"real_speakercorr_{var}_within"] = w
                rng = np.random.default_rng(7000 + i)
                same_pad = np.nonzero((pad_flag == pad_flag[i]) & (np.arange(65) != i))[0]
                pool = same_pad if len(same_pad) >= 5 else np.arange(65)
                js = rng.choice(pool, size=min(N_CROSS_PARTNERS, len(pool)), replace=True)
                vals = []
                for j in js:
                    Uij = np.intersect1d(U, act_union[j])
                    if len(Uij) < 5:
                        continue
                    a0, _ = active_union_image(caches[("real", i)], var, Uij)
                    _, b1 = active_union_image(caches[("real", j)], var, Uij)
                    c = pearson(a0, b1)
                    if not np.isnan(c):
                        vals.append(c)
                b = float(np.mean(vals)) if vals else np.nan
                between.append(b)
                rows[i][f"real_speakercorr_{var}_between"] = b

    # ---- single-frame b23 AUC for comparison (same pairs) ----
    b23 = {int(r["pair_index"]): float(r["b23_auc"])
           for r in csv.DictReader(open(MATCHED_CSV)) if r["set"] == "heldout"}
    for r in rows:
        r["singleframe_b23_auc"] = b23.get(r["pair_index"], np.nan)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(OUT_DIR / "per_pair_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    # ---- pair-level summary with bootstrap CIs ----
    def col(name):
        return np.array([r.get(name, np.nan) for r in rows], float)

    summary = {"EXPLORATORY": "No confirmatory test. Pair-level means over 65 held-out "
                              "pairs, 95% pair-bootstrap CIs (boot_ci_mean).",
               "layer": LAYER, "matching": "SI-SDR", "reference": "Silero",
               "nets": args.nets, "metrics": {}}
    names = []
    for net in args.nets:
        for var in VARIANTS:
            for W in WINDOWS:
                names += [f"{net}_loc_{var}_W{W}", f"{net}_loc_{var}_W{W}_uniform",
                          f"{net}_loc_{var}_W{W}_shifted"]
            names += [f"{net}_rowauc_{var}_own", f"{net}_rowauc_{var}_other",
                      f"{net}_rowauc_{var}_union"]
        names += [f"{net}_frac_zero_rows", f"{net}_frac_active_frames", f"{net}_n_active_rows"]
        names += [f"{net}_loc_gap_W10_{t}" for t in ("t1", "t2", "t3")]
        names += [f"{net}_rowauc_gap_own_{t}" for t in ("t1", "t2", "t3")]
    if "real" in args.nets:
        for var in VARIANTS:
            names += [f"real_speakercorr_{var}_within", f"real_speakercorr_{var}_between"]
    names += ["singleframe_b23_auc"]
    for name in names:
        v = col(name)
        summary["metrics"][name] = {"mean": float(np.nanmean(v)), "ci95": boot_ci_mean(v),
                                    "n_pairs": int(np.sum(~np.isnan(v)))}
    # paired difference: mean row-AUC (own, gap) minus single-frame AUC, per pair
    if "real" in args.nets:
        d = col("real_rowauc_gap_own") - col("singleframe_b23_auc")
        summary["metrics"]["rowauc_minus_singleframe_gap_own"] = {
            "mean": float(np.nanmean(d)), "ci95": boot_ci_mean(d),
            "n_pairs": int(np.sum(~np.isnan(d)))}
        d = col("real_speakercorr_gap_within") - col("real_speakercorr_gap_between")
        summary["metrics"]["speakercorr_gap_within_minus_between"] = {
            "mean": float(np.nanmean(d)), "ci95": boot_ci_mean(d),
            "n_pairs": int(np.sum(~np.isnan(d)))}
    if "real" in args.nets and "random" in args.nets:
        for var in VARIANTS:
            for W in WINDOWS:
                d = col(f"real_loc_{var}_W{W}") - col(f"random_loc_{var}_W{W}")
                summary["metrics"][f"loc_{var}_W{W}_real_minus_random"] = {
                    "mean": float(np.nanmean(d)), "ci95": boot_ci_mean(d),
                    "n_pairs": int(np.sum(~np.isnan(d)))}

    (OUT_DIR / "per_frame_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 100)
    print("PER-FRAME DECISION MAPS -- descriptive numbers (ALL EXPLORATORY), 65 held-out pairs, block 23")
    print("=" * 100)
    for name in summary["metrics"]:
        m = summary["metrics"][name]
        print(f"{name:48s} mean={m['mean']:+.4f}  CI95=[{m['ci95'][0]:+.4f}, {m['ci95'][1]:+.4f}]  n={m['n_pairs']}")
    print(f"[+] saved {OUT_DIR / 'per_pair_metrics.csv'} and per_frame_summary.json")


if __name__ == "__main__":
    main()
