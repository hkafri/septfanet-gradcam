"""Step 2 + Step 3a: matched re-analysis of the VAD AUC and thresholded alignment.

Uses the shared scripts/slot_matching.match_slots with the PRIMARY SI-SDR criterion
(separated waveforms vs clean sources) to assign PIT slots to true sources.

Step 2 (re-test the surprising signs), block 23 and block 9, pair-level, dev and
held-out, under SI-SDR matching:
  - VAD-AUC (pair mean of the two slots' CAM AUC vs the matched source's reference)
  - own-vs-other delta-AUC
  - exclusive-activity AUC (with 95% pair-bootstrap CI and number of eligible pairs)
  - union-vs-own AUC
  - fraction of instances whose target logit z(t*) is POSITIVE vs NEGATIVE
    (the target frame is argmax|logit|, which can be a confidently-INACTIVE frame),
    and the AUC / delta-AUC split by that sign
  - state whether the negative delta-AUC persists (no interpretation)

Step 3a (thresholded VAD alignment), block 9 and block 23, under SI-SDR matching:
  - best-F1, Youden's-J F1, precision, recall (CAM thresholded vs matched reference)
  - the chance baseline recomputed at the matched prevalence
  - the network's own VAD ceiling (post-sigmoid probability vs matched reference)

Pair-level 95% bootstrap CIs (10,000 resamples of pairs) for every matched number.
Compute-only: edits nothing.
"""

import csv
import json
import sys
from pathlib import Path

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
NUM_FRAMES = 1 + TARGET_LENGTH // HOP
N_BOOT = 10000

FULLSCALE_DIR = gradcam_root / "data" / "librispeech_fullscale"
SAMPLES_DIR = gradcam_root / "data" / "librispeech_samples"
HOLDOUT_DIR = gradcam_root / "data" / "librispeech_holdout"
SELECTION_CSV = gradcam_root / "results" / "librispeech_gradcam" / "multi_pair_results.csv"
HOLDOUT_CSV = gradcam_root / "results" / "librispeech_gradcam" / "holdout_results.csv"

BLOCK_23 = "TCN.TCN.23.conv1d"
BLOCK_9 = "TCN.TCN.9.conv1d"


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


def f1_pr(pred, ref):
    pred = pred.astype(bool); ref = ref.astype(bool)
    tp = int(np.sum(pred & ref)); fp = int(np.sum(pred & ~ref)); fn = int(np.sum(~pred & ref))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec


def best_f1(scores, ref):
    best = (0.0, 0.5, 0.0, 0.0)
    for thr in np.linspace(0.05, 0.95, 19):
        f1, p, r = f1_pr(scores >= thr, ref)
        if f1 > best[0]:
            best = (f1, float(thr), p, r)
    return best


def youden_f1(scores, ref):
    fpr, tpr, thr = stats_roc(scores, ref)
    j = tpr - fpr
    k = int(np.argmax(j))
    f1, p, r = f1_pr(scores >= thr[k], ref)
    return f1, p, r, float(thr[k])


def stats_roc(scores, ref):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(ref, scores)
    m = np.isfinite(thr)
    return fpr[m], tpr[m], thr[m]


def chance_f1(prevalence, seeds=range(20)):
    """Chance F1: random scores thresholded at the prevalence base rate."""
    vals = []
    for s in seeds:
        rng = np.random.default_rng(s)
        scores = rng.random(NUM_FRAMES)
        thr = np.quantile(scores, 1 - prevalence)
        pred = scores >= thr
        # expected F1 against a mask of that prevalence
        ref = rng.random(NUM_FRAMES) < prevalence
        f1, _, _ = f1_pr(pred, ref)
        vals.append(f1)
    return float(np.mean(vals))


def boot_ci_mean(values, n_boot=N_BOOT, seed=0):
    v = np.asarray(values, float); v = v[~np.isnan(v)]
    if len(v) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return [float(lo), float(hi)]


def wilcoxon_one_sample(values, null=0.5):
    v = np.asarray(values, float); v = v[~np.isnan(v)]
    if len(v) < 2:
        return {"p": float("nan"), "rank_biserial_r": float("nan"), "n": int(len(v))}
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
        return {"p": float("nan"), "rank_biserial_r": float("nan"), "n": int(len(a))}
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
    print(f"[*] {len(pairs)} pairs, PRIMARY matching = SI-SDR")

    per_pair = []
    for idx, (p0, p1, tag) in enumerate(pairs):
        sig0, sig1 = load_and_pad(p0), load_and_pad(p1)
        mix = normalize_audio(sig0 + sig1)
        audio = torch.from_numpy(mix).unsqueeze(0)
        with torch.no_grad():
            out_sep, _, _ = model(audio.to(device))
            vad_logits = model.vad_logits.detach().cpu().numpy()[0]
            vad_probs = torch.sigmoid(model.vad_logits).detach().cpu().numpy()[0]
        v0 = silero_mask(vad_model, sig0)
        v1 = silero_mask(vad_model, sig1)
        v_refs = [v0, v1]
        out_sep_np = out_sep.detach().cpu().numpy()[0]
        slot_to_source, swapped, margin = match_slots("sisdr", out_sep=out_sep_np, sig0=sig0, sig1=sig1)

        rec = {"pair_index": idx, "set": tag, "swapped": swapped, "sisdr_margin": margin}
        for layer, lname in ((BLOCK_23, "b23"), (BLOCK_9, "b9")):
            aucs, owns, others, unions, excls, deltas = [], [], [], [], [], []
            best_f1s, youden_f1s, precs, recs, net_f1s = [], [], [], [], []
            zsigns = []
            for slot in range(2):
                src = slot_to_source[slot]
                v_own = v_refs[src]
                v_other = v_refs[1 - src]
                v_union = v_own | v_other
                frame = int(np.argmax(np.abs(vad_logits[slot])))
                zsign = 1 if vad_logits[slot][frame] > 0 else -1
                zsigns.append(zsign)
                cam = minmax(compute_cam(model, audio, layer, slot, frame, device))
                cam = cam[:NUM_FRAMES] if len(cam) >= NUM_FRAMES else np.pad(cam, (0, NUM_FRAMES - len(cam)))
                # AUCs
                a_own = safe_auc(v_own, cam)
                a_other = safe_auc(v_other, cam)
                a_union = safe_auc(v_union, cam)
                aucs.append(a_own); owns.append(a_own); others.append(a_other); unions.append(a_union)
                if not (np.isnan(a_own) or np.isnan(a_other)):
                    deltas.append(a_own - a_other)
                # exclusive-activity
                E_s = np.nonzero(v_own & ~v_other)[0]
                E_o = np.nonzero(~v_own & v_other)[0]
                if len(E_s) >= 10 and len(E_o) >= 10:
                    sc = cam[np.concatenate([E_s, E_o])]
                    lb = np.concatenate([np.ones(len(E_s)), np.zeros(len(E_o))])
                    excls.append(float(roc_auc_score(lb, sc)))
                # thresholded alignment (Step 3a)
                bf1, bthr, bprec, brec = best_f1(cam, v_own)
                yf1, yprec, yrec, ythr = youden_f1(cam, v_own)
                best_f1s.append(bf1); youden_f1s.append(yf1); precs.append(bprec); recs.append(brec)
                # network ceiling (own VAD probs thresholded at 0.5)
                nf1, _, _ = f1_pr(vad_probs[slot][:NUM_FRAMES] >= 0.5, v_own)
                net_f1s.append(nf1)
            # pair-level aggregates
            def nm(x):
                return float(np.nanmean(x)) if len(x) else float("nan")
            rec[f"{lname}_auc"] = nm(aucs)
            rec[f"{lname}_delta"] = nm(deltas)
            rec[f"{lname}_union"] = nm(unions)
            rec[f"{lname}_own"] = nm(owns)
            rec[f"{lname}_excl"] = nm(excls)
            rec[f"{lname}_best_f1"] = nm(best_f1s)
            rec[f"{lname}_youden_f1"] = nm(youden_f1s)
            rec[f"{lname}_precision"] = nm(precs)
            rec[f"{lname}_recall"] = nm(recs)
            rec[f"{lname}_net_f1"] = nm(net_f1s)
            rec[f"{lname}_prevalence"] = float(np.mean([v_refs[slot_to_source[s]].mean() for s in range(2)]))
            rec[f"{lname}_zpos_frac"] = float(np.mean([1 if z > 0 else 0 for z in zsigns]))
            # split by z-sign
            pos = [aucs[k] for k in range(2) if zsigns[k] > 0]
            neg = [aucs[k] for k in range(2) if zsigns[k] < 0]
            posd = [deltas[k] for k in range(len(deltas))]
            rec[f"{lname}_auc_zpos"] = nm(pos)
            rec[f"{lname}_auc_zneg"] = nm(neg)
        per_pair.append(rec)
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(pairs)} pairs")

    # ---- aggregate per set ----
    out = {"n_pairs": len(per_pair), "matching": "SI-SDR (primary)", "results": {}}
    for tag in ("dev", "heldout"):
        sub = [r for r in per_pair if r["set"] == tag]
        res = {}
        for lname in ("b23", "b9"):
            auc = np.array([r[f"{lname}_auc"] for r in sub], float)
            res[f"{lname}_auc"] = {"mean": float(np.nanmean(auc)), "std": float(np.nanstd(auc)),
                                   "ci95": boot_ci_mean(auc), "wilcoxon_vs_0.5": wilcoxon_one_sample(auc, 0.5),
                                   "n_pairs": int(np.sum(~np.isnan(auc)))}
            dl = np.array([r[f"{lname}_delta"] for r in sub], float)
            res[f"{lname}_delta"] = {"mean": float(np.nanmean(dl)), "ci95": boot_ci_mean(dl),
                                     "wilcoxon_vs_0": wilcoxon_one_sample(dl, 0.0)}
            un = np.array([r[f"{lname}_union"] for r in sub], float)
            ow = np.array([r[f"{lname}_own"] for r in sub], float)
            res[f"{lname}_union_vs_own"] = {"union_mean": float(np.nanmean(un)), "own_mean": float(np.nanmean(ow)),
                                            "paired": wilcoxon_paired(un, ow)}
            ex = np.array([r[f"{lname}_excl"] for r in sub], float)
            ex = ex[~np.isnan(ex)]
            res[f"{lname}_excl"] = {"mean": float(np.mean(ex)) if len(ex) else float("nan"),
                                    "ci95": boot_ci_mean(ex), "wilcoxon_vs_0.5": wilcoxon_one_sample(ex, 0.5),
                                    "n_eligible_pairs": int(len(ex))}
            res[f"{lname}_zpos_frac"] = float(np.nanmean([r[f"{lname}_zpos_frac"] for r in sub]))
            res[f"{lname}_auc_zpos"] = float(np.nanmean([r[f"{lname}_auc_zpos"] for r in sub]))
            res[f"{lname}_auc_zneg"] = float(np.nanmean([r[f"{lname}_auc_zneg"] for r in sub]))
            # Step 3a thresholded
            for mname in ("best_f1", "youden_f1", "precision", "recall", "net_f1"):
                arr = np.array([r[f"{lname}_{mname}"] for r in sub], float)
                res[f"{lname}_{mname}"] = {"mean": float(np.nanmean(arr)), "ci95": boot_ci_mean(arr)}
            prev = float(np.nanmean([r[f"{lname}_prevalence"] for r in sub]))
            res[f"{lname}_prevalence"] = prev
            res[f"{lname}_chance_f1"] = chance_f1(prev)
        out["results"][tag] = res

    out_dir = gradcam_root / "results" / "slot_matching"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matched_analysis_summary.json").write_text(json.dumps(out, indent=2))
    with open(out_dir / "matched_analysis_per_pair.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_pair[0].keys()))
        w.writeheader(); w.writerows(per_pair)

    # ---- console report ----
    print("\n" + "=" * 90)
    print("STEP 2 + 3a — SI-SDR-matched analysis (pair-level)")
    print("=" * 90)
    for tag in ("dev", "heldout"):
        res = out["results"][tag]
        print(f"\n=== {tag} ===")
        for lname, layer in (("b23", "block 23"), ("b9", "block 9")):
            a = res[f"{lname}_auc"]
            print(f"  {layer} AUC = {a['mean']:.4f} CI{a['ci95']} p={a['wilcoxon_vs_0.5']['p']:.2e} (n={a['n_pairs']})")
            d = res[f"{lname}_delta"]
            print(f"    own-vs-other Δ = {d['mean']:+.4f} CI{d['ci95']} p={d['wilcoxon_vs_0']['p']:.3f}")
            u = res[f"{lname}_union_vs_own"]
            print(f"    union {u['union_mean']:.4f} vs own {u['own_mean']:.4f} paired p={u['paired']['p']:.3f}")
            e = res[f"{lname}_excl"]
            print(f"    exclusive AUC = {e['mean']:.4f} CI{e['ci95']} p={e['wilcoxon_vs_0.5']['p']:.3f} n_elig_pairs={e['n_eligible_pairs']}")
            print(f"    z>0 frac = {res[f'{lname}_zpos_frac']:.3f} | AUC z>0 {res[f'{lname}_auc_zpos']:.3f} z<0 {res[f'{lname}_auc_zneg']:.3f}")
            print(f"    [3a] bestF1={res[f'{lname}_best_f1']['mean']:.3f} youdenF1={res[f'{lname}_youden_f1']['mean']:.3f} "
                  f"prec={res[f'{lname}_precision']['mean']:.3f} rec={res[f'{lname}_recall']['mean']:.3f} "
                  f"chanceF1={res[f'{lname}_chance_f1']:.3f} netF1={res[f'{lname}_net_f1']['mean']:.3f}")
    print(f"\n[+] Saved to {out_dir}")


if __name__ == "__main__":
    main()
