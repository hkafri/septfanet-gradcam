# Sep-TFAnet-VAD Grad-CAM

Grad-CAM interpretability for [Sep-TFAnet-VAD](https://github.com/MordehayM/Sep-TFAnet-VAD), a PyTorch
speaker-separation/VAD network — adapted from image-based Grad-CAM and validated on real LibriSpeech audio.

**Full investigation narrative** (every hypothesis, bug fix, and pre-registration decision, in order) is preserved in
[docs/INVESTIGATION_LOG.md](docs/INVESTIGATION_LOG.md). This README is the scannable, result-first summary.

## What Grad-CAM successfully explains

**VAD timing, at the ranking level (validated, and replicated on held-out pairs).** At the late TCN layer
`TCN.TCN.23.conv1d`, the block-23 VAD-logit map ranks speech-active frames above inactive ones (continuous-CAM
AUC-ROC = **0.723 ± 0.149**, p ≈ 3×10⁻¹¹, n=70, rank-biserial r = +0.914). This **replicated on 65 held-out pairs
never used in layer selection**: AUC = 0.681 ± 0.192, 95% CI [0.647, 0.712] (excludes 0.5), p ≈ 8×10⁻¹⁵, n=130.
An exploratory check found own-speaker and union references give **statistically indistinguishable** AUCs (0.695 vs.
0.709, p = 0.124), so the ranking signal is **not shown to be specific to the target speaker**; it is consistent with
tracking overall speech activity. The signal does **not** beat a majority-class-matched threshold decision, because
~82% of frames are genuinely speech-active (re-calibrated F1 = 0.679 vs. a chance baseline of 0.815).


- **Speaker discriminability (MAE, pre-registered null controls)**: the speaker-vs-speaker MAE is *not*
  significantly different from value-distribution-preserving nulls (circular-shift null N1: p=0.487; full temporal
  permutation null N2: p=0.462) at the pre-registered layer `TCN.TCN.9.conv1d` on N=100 pairs. The earlier
  "discriminability" result (real MAE 0.233 vs. uniform-noise 0.416, p=9e-17) is **not sufficient** to claim the
  CAMs carry speaker-discriminative structure — that difference is explained by the maps' value distributions
  (sparsity), not by any genuine speaker-dependent temporal alignment. The claim is removed from the positive
  findings above and placed here as a null result. (Pre-registered script: `scripts/evaluate_mae_null_controls.py`.)
- **Speaker-specificity (exclusive-activity AUC, pre-registered)**: at `TCN.TCN.23.conv1d`, restricting to frames
  where exactly one speaker is active, the CAM does not rank "speaker s active, other not" frames above "other
  active, speaker s not" frames above chance (exclusive-activity AUC = 0.482 vs. 0.5, 95% CI [0.389, 0.576],
  p=0.856, n=40 eligible — **underpowered**, see below). Speaker specificity **not shown**; small effects cannot be
  excluded.
- **VAD timing at block 9** (the layer selected for the discriminability result): no positive signal (AUC = 0.417,
  significantly *below* 0.5, p=0.002).
- **VAD timing via thresholded F1**: below the chance baseline at every layer tested (best 0.706 vs. 0.815) — the
  ranking-level signal above does not survive conversion to a hard binary decision against a high-base-rate reference.
- **Integrated Gradients**: below chance, and failed its completeness axiom for one speaker (a phase-wrapping
  non-smoothness issue from `noisy_phase=True`).
- **Local transition-based evaluation**: a pre-registered primary test reversed sign at full scale (pilot r=+0.582 →
  full-scale r=−0.230, p=0.040) — attribution does not concentrate near true VAD transitions.
- **Separation (IBM) alignment**: no real signal in either orientation (AUC-ROC ≈ 0.497–0.498 at both layers), after
  catching and fixing a PIT permutation bug that initially made the ceiling look broken, and ruling out a CAM sign
  inversion via the orientation-invariant check.

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\activate | Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

Download `model_with_vad.pth` from [MordehayM/Sep-TFAnet-VAD](https://github.com/MordehayM/Sep-TFAnet-VAD) and place
it at `weights/model_with_vad.pth` (`configs/config_with_vad.json` is included as a small text file).

Fetch LibriSpeech samples (streamed, no full-corpus download; never falls back to synthetic audio):

```bash
python scripts/fetch_librispeech_samples.py
```

## Usage

```bash
# Core verification figure + multi-pair validation
python scripts/verify_librispeech_gradcam.py --librispeech-root data/librispeech_samples --device cpu
python scripts/evaluate_multi_pair_gradcam.py --librispeech-root data/librispeech_samples --num-pairs 20 --seed-offset 1000

# Layer selection (scores all 24 TCN conv1d blocks)
python scripts/select_target_layer.py

# Held-out set + full-scale (N=100) validation
python scripts/fetch_librispeech_holdout.py
python scripts/evaluate_multi_pair_gradcam.py --librispeech-root data/librispeech_holdout --num-pairs 15 --seed-offset 2000 --csv-output results/librispeech_gradcam/holdout_results.csv --summary-output results/librispeech_gradcam/holdout_summary.json --plot-output results/librispeech_gradcam/holdout_paired_comparison_plot.png
python scripts/fetch_librispeech_fullscale.py --num-speakers 130 --source parquet
python scripts/evaluate_fullscale_gradcam.py --device cpu --target-pairs 100

# MAE null-control and speaker-specificity diagnostics
python scripts/evaluate_mae_null_controls.py
python scripts/evaluate_union_auc_check.py   # exploratory: mixture-activity vs target-speaker check
python scripts/evaluate_pair_level.py        # pair-level clustered re-analysis (resamples pairs, not instances)
python scripts/evaluate_vad_replication.py   # held-out replication of the block-23 VAD signal
python scripts/evaluate_robustness_auc.py --subset heldout   # AUC-based robustness (noise + reverberation)
python scripts/evaluate_robustness_auc.py --subset dev       # same, on the development pairs (secondary)

# VAD-alignment and separation-alignment diagnostics
python scripts/evaluate_vad_alignment.py --target-layer TCN.TCN.23.conv1d --output-suffix _block23
python scripts/diagnose_grad_spread.py
python scripts/evaluate_ig_vad_alignment.py
python scripts/evaluate_local_transition.py
python scripts/evaluate_separation_alignment.py
python scripts/evaluate_auc_check.py
python scripts/validate_vad_auc.py
python scripts/validate_vad_auc_f1.py
python scripts/make_vad_auc_figure.py
```

## Results summary

### MAE discriminability vs. null controls (pre-registered, N=100 pairs, block 9)

The original finding (real MAE 0.233 vs. uniform-noise 0.416, p=9e-17) is reproduced below, followed by the
pre-registered null controls that preserve the maps' value distributions and temporal smoothness. All numbers are
computed on the **exact saved utterances** (real MAE reproduces at 0.2333); an earlier version of the null script that
resolved speakers to the alphabetically-first utterance gave 0.243 — the same 200 speakers, a slightly different
utterance set — with identical conclusions.

| Comparison | Real MAE | Null MAE | Wilcoxon p | Rank-biserial r |
|---|---|---|---|---|
| Real vs. uniform-noise control (original, insufficient) | 0.233 ± 0.091 | 0.416 ± 0.051 | 9.0e-17 | −0.953 |
| Real vs. **N1 circular-shift null** (PRIMARY) | 0.233 ± 0.091 | 0.246 ± 0.081 | 0.202 | −0.147 |
| Real vs. **N2 full-permutation null** (secondary) | 0.233 ± 0.091 | 0.246 ± 0.080 | 0.207 | −0.145 |
| Real vs. **N3 constant-map baseline** (secondary) | 0.233 ± 0.091 | 0.201 ± 0.101 | 7.7e-08 | +0.619 |

The MAE is *not* significantly below the value-distribution-preserving nulls (N1, N2). The apparent discriminability
is an artifact of the maps' sparsity (value distribution), not speaker-dependent temporal alignment. The N3 result
(real > constant) is a secondary check that the maps are not simply degenerate. (Block 23, secondary: N1 p=0.103,
N2 p=0.136, N3 p=1.5e-06.)

### Speaker-specificity (exclusive-activity AUC, pre-registered, block 23)

Restricting to frames where exactly one speaker is active. With ~80%-active references, exclusive frames are scarce:
median |E_s| = |E_o| = 12 across all instances (23 among eligible); mean frame agreement between the two reference
VADs = 0.782. **n = 40 eligible instances** (exact saved utterances).

| Test | AUC | 95% CI (bootstrap) | Null | Wilcoxon p | Rank-biserial r |
|---|---|---|---|---|---|
| Exclusive-activity AUC vs. 0.5 (PRIMARY, instance-level, n=40) | 0.482 ± 0.301 | [0.389, 0.576] | 0.5 | 0.856 | −0.026 |
| Exclusive-activity AUC vs. 0.5 (**pair-level**, both speakers eligible, n=20 pairs) | 0.482 ± 0.169 | [0.406, 0.554] | 0.5 | 0.748 | −0.081 |
| Own-vs-other ΔAUC block 23 (secondary) | +0.015 | — | 0 | 0.295 | — |
| Own-vs-other ΔAUC block 9 (secondary) | −0.003 | — | 0 | 0.642 | — |

**Power**: with n=40 eligible instances (n=20 fully-eligible pairs) the test had 80% power only for an exclusive-AUC
shift ≥ ~0.14 (AUC ≳ 0.64). The pair-level 95% CI [0.406, 0.554] **excludes a speaker-specific effect as large as the
overall ranking effect** (an AUC shift of ~0.2), so **large speaker-specific effects are excluded; small effects
cannot be excluded** — the test is **underpowered** and the correct reading is "speaker specificity **not shown**",
**not** "not speaker-specific".

### VAD-timing ranking signal (continuous-CAM AUC-ROC)

Instance-level (treating the two speakers of each pair as independent) and **pair-level** (averaging the two speakers
within each pair; Wilcoxon across pairs; bootstrap resamples pairs) — shown side by side. Pair-level is the more
conservative, assumption-light test because the two instances of a pair share a mixture.

| Set | Layer | Instance AUC | Instance 95% CI / p | **Pair-level AUC** | **Pair 95% CI / p** |
|---|---|---|---|---|---|
| Development (35 pairs) | Block 9 | 0.417 ± 0.192 (n=70) | — / p=0.0023 | 0.417 ± 0.105 (n=35) | [0.384, 0.452] / p=8.4e-05 |
| Development (35 pairs) | **Block 23** | **0.723 ± 0.149** (n=70) | [0.687, 0.757] / p=3.0e-11 | **0.723 ± 0.093** (n=35) | **[0.692, 0.753] / p=5.8e-11** |
| **Held-out (65 pairs)** | **Block 23** | **0.681 ± 0.192** (n=130) | [0.647, 0.712] / p=8.1e-15 | **0.681 ± 0.130** (n=65) | **[0.649, 0.713] / p=7.4e-11** |
| Held-out (65 pairs) | Block 9 | 0.443 ± 0.174 (n=130) | [0.412, 0.473] / p=1.2e-03 | 0.443 ± 0.129 (n=65) | [0.412, 0.474] / p=1.5e-03 |

The block-23 VAD-timing ranking signal **replicates on 65 pairs never used in layer or method selection** — AUC
clearly above 0.5 with a CI that excludes 0.5. The pair-level clustering leaves the conclusion unchanged (held-out
block 23: p=7.4e-11, r=+0.929 across pairs; block 9 stays below 0.5).

### Separation (IBM) alignment (orientation-invariant AUC-ROC, n=70)

| Layer | AUC-ROC | Interpretation |
|---|---|---|
| Block 9 | 0.497 ± 0.068 | No signal (at chance) |
| Block 23 | 0.498 ± 0.099 | No signal (at chance) |

### Robustness of the block-23 VAD-timing signal (AUC-based, pair-level, held-out 65 pairs)

Pre-registered (`scripts/evaluate_robustness_auc.py`). Pair-level AUC-ROC of the block-23 VAD-logit CAM against the
speaker's own clean-source reference VAD, under simulated reverberation (independent RIR per speaker) and babble
noise (LibriSpeech leftovers, speakers not in the mixture). "Persists" = condition AUC > 0.5 (Holm-adj across the 7
degraded conditions). Non-inferiority uses a three-way classification against the pre-registered margin (−0.06) on the
95% pair-bootstrap CI of (AUC_condition − AUC_clean): **robust** (lower bound > −0.06), **inferior** (upper bound
< −0.06), **inconclusive** (otherwise; *borderline* when a bound is within 0.01 of the margin). The Bonferroni column
is a **sensitivity analysis** (coverage 1 − 0.05/7 ≈ 99.3%) — the primary verdicts use the unadjusted 95% CIs.

| Condition | CAM AUC [95% CI] | Persists? | Δ vs clean [95% CI] | Verdict (95%) | Bonferroni Δ lower | Network-VAD AUC |
|---|---|---|---|---|---|---|
| clean (reference) | 0.681 [0.649, 0.713] | — | — | — | — | 0.827 |
| reverb T60=0.2 s | 0.687 [0.652, 0.720] | yes (p=9e-10) | +0.006 [−0.026, +0.037] | **robust** | −0.038 (robust) | 0.845 |
| reverb T60=0.4 s | 0.665 [0.630, 0.699] | yes (p=1e-09) | −0.016 [−0.049, +0.017] | **robust** | −0.062 (inconclusive) † | 0.830 |
| reverb T60=0.6 s | 0.651 [0.623, 0.679] | yes (p=3e-10) | −0.030 [−0.064, +0.003] | inconclusive (borderline) | −0.077 | 0.843 |
| noise SNR=10 dB | 0.644 [0.620, 0.667] | yes (p=3e-10) | −0.037 [−0.069, −0.004] | inconclusive (borderline) | −0.080 | 0.855 |
| noise SNR=5 dB | 0.614 [0.591, 0.635] | yes (p=1e-09) | −0.067 [−0.103, −0.031] | inconclusive | −0.114 | 0.844 |
| noise SNR=0 dB | 0.577 [0.552, 0.604] | yes (p=1e-06) | −0.103 [−0.139, −0.067] | **inferior** (borderline) | −0.152 (inconclusive) † | 0.815 |
| combo T60=0.4 + SNR=5 | 0.625 [0.601, 0.649] | yes (p=7e-10) | −0.056 [−0.090, −0.021] | inconclusive | −0.102 | 0.836 |

† *Sensitivity: the verdict changes under Bonferroni-adjusted coverage (99.3%). T60=0.4 s moves robust → inconclusive
(adjusted lower bound −0.062 crosses the margin); SNR=0 moves inferior → inconclusive (adjusted upper bound −0.053
crosses back inside the margin). T60=0.2 s stays robust at both coverages.*

**What can be claimed (held-out, block 23), condition by condition:**

- **Persists** (CI above 0.5) in **every** condition, including 0 dB babble.
- **Robust within the margin** at T60 = 0.2 s (stable under the multiplicity sensitivity check), and at T60 = 0.4 s
  **with the multiplicity caveat** (robust on the primary 95% CI; inconclusive under Bonferroni).
- **Inconclusive** at T60 = 0.6 s, SNR = 10 dB, SNR = 5 dB, and the combined condition — for SNR = 5 dB and the
  combined condition the degradation is significant (CI excludes 0) but degradation *beyond the margin* is not
  established; robustness is not demonstrated there.
- **Inferior (beyond the margin)** at SNR = 0 dB on the primary CI (upper bound −0.067 < −0.06); note this verdict is
  itself borderline and flips to inconclusive under the Bonferroni sensitivity check.

Reverberation numbers use dry-source references, which under-mark reverberant tails, so they are **conservative**. The
network's own VAD output stays high in all conditions (0.81–0.86), so the CAM-AUC drop under strong noise is not
simply the network failing — the explanation degrades faster than the network's VAD. **Development set (35 pairs,
secondary)** shows the same direction: robust at all three T60 values and SNR=10 dB, inconclusive at SNR=5 dB and the
combined condition, inferior at SNR=0 dB — consistent with the held-out set.

*(Superseded: an earlier MAE-vs-uniform-noise robustness table is no longer used — that metric was later shown to be
uninformative, because sparse maps beat uniform noise trivially. See the investigation log.)*

## Methodology (key decisions)

- **Pre-sigmoid VAD logit** (not the saturating post-sigmoid probability) as the Grad-CAM target, exposed as
  `model.vad_logits` in [network/model/model.py](network/model/model.py) without changing the model's normal output.
- **Pre-registered null controls** for the MAE discriminability claim: circular-shift (N1), full temporal
  permutation (N2), and constant-map (N3) baselines that preserve the maps' value distributions and temporal
  smoothness. The original uniform-noise control was found insufficient on its own.
- **MAE vs. a random-noise control**, not max-diff (max-diff reads ~1.0 for any two sparse maps, so it's uninformative).
- **Non-monotonic entropy penalty** in layer selection (penalize deviation from target entropy 0.65), so neither
  diffuse-uniform nor single-pixel-spike maps are rewarded. Selected `TCN.TCN.9.conv1d` for the discriminability result.
- A **multi-layer ensemble** (blocks 0, 9, 19) was evaluated and explicitly rejected — it didn't widen the
  discriminability gap. Single-layer block 9 was kept for that result.
- **Orientation-invariant AUC** used to rule out sign inversions before trusting any thresholded metric.

## Attribution and license

- **Base network and weights**: [MordehayM/Sep-TFAnet-VAD](https://github.com/MordehayM/Sep-TFAnet-VAD). This repo
  references and adapts that architecture (`network/model/model.py`, with the `vad_logits` addition) but does **not**
  redistribute the original repository's code or checkpoints.
- **Paper**:
  > Opochinsky, R., Moradi, M., Gannot, S. "Single-microphone speaker separation and voice activity detection in noisy and
  > reverberant environments." *EURASIP Journal on Audio, Speech, and Music Processing* (2025). DOI 10.1186/s13636-025-00404-7.
  > (R. Opochinsky and M. Moradi contributed equally, per the arXiv version.)
- **License note**: as of writing, the `Sep-TFAnet-VAD` repository does not publish a `LICENSE` file. No permissive
  reuse rights are assumed. This repo only references/attributes that network's architecture and weights for
  research/interpretability purposes; it does not redistribute them. You must obtain the weights directly from the
  original repository and are responsible for complying with whatever terms the original author sets.

## Limitations

- **Accent diversity (open)**: LibriSpeech has no accent labels; a different corpus (Common Voice, VCTK) is needed.
- **Noise proxy**: the robustness test uses a babble-sum of leftover LibriSpeech utterances, not a real ambient-noise
  corpus such as [WHAM!](http://wham.whisper.ai/).
- **VAD reference**: the alignment reference is Silero VAD output, not hand-labeled ground truth.
- **CPU-only verification**: GPU execution is supported (`--device cuda`) but not separately validated here.
