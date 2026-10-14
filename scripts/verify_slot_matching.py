"""Step 1: verify the shared slot-matching function.

Compares the PRIMARY criterion (SI-SDR between separated output waveforms and the
clean sources) against the SECONDARY criterion (network VAD-AUC, used in the audit),
on all 100 clean pairs. Reports per set (dev 35 / held-out 65): swap rate under each
criterion, per-pair agreement between the two, and the per-pair margin (score
difference between the two permutations). If they disagree on >10% of pairs, this
script's output says so explicitly (the pipeline must stop before any rerun).

Compute-only; edits nothing.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

import network.model as module_arch
from data.librispeech import load_utterance
from scripts.slot_matching import match_slots

import csv

SAMPLE_RATE = 16000
TARGET_LENGTH = int(3.0 * SAMPLE_RATE)
HOP = 256
NUM_FRAMES = 1 + TARGET_LENGTH // HOP

FULLSCALE_DIR = gradcam_root / "data" / "librispeech_fullscale"
SAMPLES_DIR = gradcam_root / "data" / "librispeech_samples"
HOLDOUT_DIR = gradcam_root / "data" / "librispeech_holdout"
SELECTION_CSV = gradcam_root / "results" / "librispeech_gradcam" / "multi_pair_results.csv"
HOLDOUT_CSV = gradcam_root / "results" / "librispeech_gradcam" / "holdout_results.csv"


def normalize_audio(a):
    a = a.astype(np.float32)
    return a / max(np.abs(a).max(), 1e-8) * 0.9


def load_and_pad(path):
    s = load_utterance(path)[:TARGET_LENGTH]
    pad = np.zeros(TARGET_LENGTH, dtype=np.float32)
    pad[: len(s)] = s
    return pad


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


def main():
    device = "cpu"
    config = json.loads((gradcam_root / "configs" / "config_with_vad.json").read_text())
    model = module_arch.SeparationModel(**config["arch"]["args"]).to(device).eval()
    ckpt = torch.load(gradcam_root / "weights" / "model_with_vad.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    vad_model = load_silero_vad()

    pairs = resolve_all_pairs()
    print(f"[*] {len(pairs)} pairs")

    recs = []
    for idx, (p0, p1, tag) in enumerate(pairs):
        sig0, sig1 = load_and_pad(p0), load_and_pad(p1)
        mix = normalize_audio(sig0 + sig1)
        audio = torch.from_numpy(mix).unsqueeze(0)
        with torch.no_grad():
            out_sep, _, _ = model(audio.to(device))
            vad_probs = torch.sigmoid(model.vad_logits).detach().cpu().numpy()[0]
        v0 = silero_mask(vad_model, sig0)
        v1 = silero_mask(vad_model, sig1)
        out_sep_np = out_sep.detach().cpu().numpy()[0]  # (2, samples)

        _, sw_sisdr, mg_sisdr = match_slots("sisdr", out_sep=out_sep_np, sig0=sig0, sig1=sig1)
        _, sw_vad, mg_vad = match_slots("vad_auc", vad_probs=vad_probs, v0=v0, v1=v1)
        recs.append({"pair_index": idx, "set": tag,
                     "sisdr_swapped": sw_sisdr, "sisdr_margin": mg_sisdr,
                     "vad_swapped": sw_vad, "vad_margin": mg_vad,
                     "agree": sw_sisdr == sw_vad})
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(pairs)}")

    out = {"n_pairs": len(recs), "per_set": {}}
    disagree_total = 0
    for tag in ("dev", "heldout", "all"):
        sub = [r for r in recs if (r["set"] == tag or tag == "all")]
        n = len(sub)
        sw_sisdr = sum(r["sisdr_swapped"] for r in sub)
        sw_vad = sum(r["vad_swapped"] for r in sub)
        agree = sum(r["agree"] for r in sub)
        disagree = n - agree
        disagree_total += (disagree if tag != "all" else 0)
        out["per_set"][tag] = {
            "n": n,
            "sisdr_swap_rate": sw_sisdr / n, "vad_swap_rate": sw_vad / n,
            "criteria_agree_frac": agree / n, "criteria_disagree_frac": disagree / n,
            "sisdr_margin_mean": float(np.mean([r["sisdr_margin"] for r in sub])),
            "sisdr_margin_median": float(np.median([r["sisdr_margin"] for r in sub])),
            "vad_margin_mean": float(np.mean([r["vad_margin"] for r in sub])),
            "vad_margin_median": float(np.median([r["vad_margin"] for r in sub])),
        }

    out["disagree_over_10pct_all"] = bool(out["per_set"]["all"]["criteria_disagree_frac"] > 0.10)

    out_dir = gradcam_root / "results" / "slot_matching"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "match_verification.json").write_text(json.dumps(out, indent=2))
    with open(out_dir / "match_verification_per_pair.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader(); w.writerows(recs)

    print("\n" + "=" * 80)
    print("STEP 1 — matching-criteria verification (PRIMARY=SI-SDR, SECONDARY=VAD-AUC)")
    print("=" * 80)
    for tag in ("dev", "heldout", "all"):
        s = out["per_set"][tag]
        print(f"{tag:9s} n={s['n']:3d} | swap rate SI-SDR={s['sisdr_swap_rate']*100:.1f}% "
              f"VAD={s['vad_swap_rate']*100:.1f}% | criteria agree {s['criteria_agree_frac']*100:.1f}% "
              f"(disagree {s['criteria_disagree_frac']*100:.1f}%)")
        print(f"          margin: SI-SDR mean {s['sisdr_margin_mean']:.2f} dB (median {s['sisdr_margin_median']:.2f}) | "
              f"VAD mean {s['vad_margin_mean']:.3f} (median {s['vad_margin_median']:.3f})")
    flag = out["disagree_over_10pct_all"]
    print(f"\nCriteria disagree on >10% of all pairs? {flag}")
    print("STOP: criteria disagree too much — do not rerun." if flag else "OK: criteria agree enough to rerun under PRIMARY (SI-SDR) matching.")
    print(f"[+] Saved to {out_dir}")


if __name__ == "__main__":
    main()
