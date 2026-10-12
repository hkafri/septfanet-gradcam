# Full Investigation Log

This document preserves the complete chronological narrative of the Grad-CAM interpretability investigation —
every hypothesis tried, every bug caught and fixed, every pre-registration decision. The main
[README.md](../README.md) is the scannable, result-first summary; this log is for anyone who wants the full story.

---

## Setup

Grad-CAM adapted from image-based Grad-CAM to the PyTorch [Sep-TFAnet-VAD](https://github.com/MordehayM/Sep-TFAnet-VAD)
speaker-separation/VAD network, on real LibriSpeech mixtures. Two initial decisions: target the pre-sigmoid VAD logit
(not the saturating post-sigmoid probability), and report MAE against a random-noise control rather than max-diff
(which reads ~1.0 for any two sparse maps, so it's uninformative).

## Phase 1 — Layer selection (MAE-based discriminability)

All 24 TCN conv1d blocks scored on three metrics: gradient strength (log + z), activation variance (log + z), and
CAM entropy — with a **non-monotonic entropy penalty** penalizing deviation from a target entropy of 0.65
(`-abs(H - 0.65)`), since monotonic entropy rewards both diffuse uninformative maps and single-pixel spikes.
Winner: **`TCN.TCN.9.conv1d`** (combined +0.8931). Artifacts: `results/layer_selection/`.

A multi-layer ensemble (blocks 0, 9, 19 — top scorer per depth third) was **tested and rejected**: it didn't widen
the real-vs-random separation gap relative to single-layer block 9. Artifacts: `results/ensemble_cam/`.

## Phase 2 — Discriminability validation (the positive result)

The speaker-vs-speaker MAE is far lower than a random-noise control, validated across selection (N=20), a
zero-overlap held-out set (N=15), pooled (N=35), and full-scale (N=100 pairs, 200 speakers, with gender/pitch
subgroup breakdowns). **(Superseded: this MAE "discriminability" finding was later shown to be an artifact of the
maps' value distributions — see Phase 4. The robustness test referenced here used the same MAE metric and is likewise
superseded — see Phase 8.)**

## Phase 3 — VAD ground-truth alignment (the long investigation)

Question: does the VAD-logit CAM align with true voice-activity timing? Reference: Silero VAD on the clean pre-mix
source (a second automatic VAD model, not hand-labeled ground truth).

### 3.1 Original finding (below chance)

Best-F1 = 0.522 (block 9) vs. a class-balance-matched chance baseline F1 ≈ 0.815 (the reference is ~82% active, so
a coin flip at that base rate scores that high). Below chance.

### 3.2 Hypothesis: wrong layer

- A forward receptive-field calculation initially suggested an earlier layer — but that answers the wrong question
  (Grad-CAM gradients flow *backward* from the target).
- A **gradient-spread diagnostic** ([scripts/diagnose_grad_spread.py](../scripts/diagnose_grad_spread.py)) measured
  backward gradient spread vs. hook depth: spread shrinks toward the output (block 22 is a sharp 7-frame spike on the
  target frame; block 2 is a 52-frame smear). This correctly predicted that *later* layers would localize better,
  overturning the earlier forward-RF implication.
- A 6-layer sweep (blocks 9, 15, 18, 20, 22, 23) found block 23 best at F1 = 0.706 — a real improvement over block 9,
  but still below chance. A mask-inspection check confirmed block 23's win is a genuine partial match, not a
  floor-dominance artifact. Artifacts: `results/vad_alignment/vad_alignment_*_block*.csv/json`.

### 3.3 Hypothesis: wrong attribution method

Integrated Gradients (input-space, no layer choice; STFT confirmed differentiable in-graph) also came out below
chance on both speakers, and **failed its completeness axiom** for one speaker (residual error 96–122%) — attributed
to `noisy_phase=True` making the target depend on the non-smooth STFT angle (phase wraps). A genuine methodological
finding about IG on phase-dependent targets. Artifacts: `results/ig_vad_alignment/`,
[scripts/evaluate_ig_vad_alignment.py](../scripts/evaluate_ig_vad_alignment.py).

### 3.4 Hypothesis: wrong evaluation granularity

A local per-transition test ([scripts/evaluate_local_transition.py](../scripts/evaluate_local_transition.py)): one
attribution per VAD transition, compared within ±W frames vs. a matched control window. The primary test
(`TCN.TCN.23.conv1d`, W=10) was **pre-registered before the full run** to avoid multiple-comparisons fishing (the
project had already caught two post-hoc-selection problems: the max-diff metric and the entropy-direction bug).

- Pilot (5 pairs, n=13): promising trend, r=+0.582, p=0.068.
- Full scale (35 pairs, n=107): the primary test **reversed sign** — r=−0.230, p=0.040 (significant in the *opposite*
  direction).

The pilot's trend was an underpowered false signal; pre-registration caught it. Attribution does not concentrate
near true VAD transitions. Artifacts: `results/local_transition/`.

### 3.5 The validated positive finding (and its limits)

A final orientation-invariant check ([scripts/evaluate_auc_check.py](../scripts/evaluate_auc_check.py)) measured the
**continuous** CAM's discriminative signal vs. Silero with no threshold or orientation decision baked in (AUC-ROC
maps to 1−AUC under sign inversion, so it's orientation-invariant):

- **Block 23: AUC-ROC = 0.723 ± 0.149** (p≈3e-11, n=70, r=+0.914) — a real, robust ranking-level signal.
- Block 9: AUC = 0.417 (p=0.0023, r=−0.420) — significantly *below* 0.5, no positive signal.

This doesn't contradict the F1 nulls: AUC measures ranking quality independent of threshold/base rate, while
F1 is a thresholded decision metric against a high-base-rate reference. The block-23 AUC finding is retained as the
single validated positive result in the README.

---

## Phase 4 — MAE null controls (2026-10-09)

The original "speaker discriminability" finding (real MAE 0.233 vs. uniform-noise 0.416, p=9e-17, N=100) was found
to be **insufficiently controlled**. The uniform-noise baseline does not preserve the maps' value distributions
(sparsity), so the gap could be explained by sparsity alone rather than any speaker-dependent temporal alignment.

Pre-registered script: `scripts/evaluate_mae_null_controls.py` (committed before execution, commit `59b7c40`).

### 2A — MAE null controls (block 9, N=100, PRIMARY)

| Null control | Null MAE | Wilcoxon p | Rank-biserial r |
|---|---|---|---|
| N1 circular-shift (preserves value dist. + temporal smoothness) | 0.253 ± 0.087 | 0.487 | −0.080 |
| N2 full temporal permutation (secondary) | 0.252 ± 0.087 | 0.462 | −0.085 |
| N3 constant-map baseline (secondary) | 0.208 ± 0.108 | 5.5e-09 | +0.672 |

The real MAE is **not** significantly below N1 or N2. The apparent discriminability is an artifact of the maps'
sparsity, not speaker-dependent temporal alignment. The claim is removed from the README's positive findings.

### 2B — Speaker-specificity (block 23, n=50 eligible, PRIMARY)

Exclusive-activity AUC = 0.519 ± 0.310 vs. 0.5: p=0.630. No evidence the CAM is speaker-specific in the intended
sense (frames where exactly one speaker is active are not ranked above the reverse). Own-vs-other ΔAUC = +0.024
(secondary, p=0.089) — borderline, not significant at α=0.05.

The term "speaker-specific" has been removed from the README; the CAMs carry non-random structure (vs. uniform
noise), but that structure does not discriminate between speakers in the intended sense.

---

## Phase 5 — Closing the gaps (2026-10-12)

Follow-up on the pre-registered 2A/2B report. Script: `scripts/evaluate_union_auc_check.py` (exploratory).
Artifacts: `results/mae_null_controls/union_auc_check.{json,csv}`.

### 5.1 Baseline reproduction (Step 1)

The original full-scale statistic reproduces **exactly** from the stored CSV (N=100, real MAE **0.2333 ± 0.0909**,
uniform-noise 0.4157 ± 0.0513, p = 9.0e-17). The 0.243 in the null run is a **sample-composition difference**: the
original run used each speaker's exact saved utterance; the null script resolved each speaker to the
alphabetically-first utterance (only 84/100 pairs identical). Same 200 speakers, different utterance for some pairs.
Conclusion unchanged. **0.233 is authoritative.**

### 5.2 2B sample size and eligible frames (Step 2)

- Eligible instances: **50/200** in the pre-registered run (40/200 on exact pairs — same cause as above).
- Median |E_s| = |E_o| = **12** across all instances (23 among eligible); mean frame agreement between the two
  reference VADs = **0.782**. Exclusive-activity frames are scarce.
- 2B secondary ΔAUC: block 23 = **+0.024** (p=0.089, r=+0.136, n=200); block 9 = **+0.004** (p=0.903).
- 2A secondary (block 23): N1 p=0.672 (r=−0.049); N2 p=0.818 (r=−0.027); N3 p=6.0e-09 (r=+0.670) — same as block 9.

### 5.3 Exploratory union-AUC check (Step 3)

Does the validated block-23 ranking signal track the target speaker or the overall mixture activity?
- `AUC_own = 0.695 ± 0.179` (prevalence 0.792); `AUC_union = 0.709 ± 0.174` (prevalence 0.897).
- Δ(union−own) = +0.013; frac(union ≥ own) = 0.680; paired Wilcoxon p = 0.124 (r=+0.087, n=200).

Own-speaker and union references give **statistically indistinguishable** AUCs (0.695 vs. 0.709, p = 0.124), so the
ranking signal is **not shown to be specific to the target speaker**; it is *consistent with* tracking overall speech
activity. (Corrected wording — the earlier draft stated mixture-tracking as fact, which the n.s. result does not support.)

### 5.4 Power of 2B (Step 4)

n_eligible = 40–50 (vs. 70 for the validated VAD result). 80%-power detectable standardized effect dz ≈ 0.40–0.44 ⇒
detectable exclusive-AUC shift ≈ **0.124–0.137** (AUC ≳ 0.62–0.64). The observed 0.519 is far below that, so 2B is
**underpowered**. It supports "speaker specificity **not shown**", **not** "not speaker-specific". README updated.

---

## Phase 6 — Exact-utterance re-run, bootstrap CI, and held-out replication (2026-10-12)

Scripts: `scripts/evaluate_mae_null_controls.py` (fixed to exact saved utterances) and
`scripts/evaluate_vad_replication.py` (pre-registered before execution, commit `09ac91a`).
Artifacts: `results/mae_null_controls/`, `results/replication/replication_summary.json`.

### 6.1 Utterance-resolution audit (Step 2)

Only `evaluate_mae_null_controls.py` used alphabetical resolution. All headline scripts
(`evaluate_vad_alignment`, `evaluate_auc_check`, `validate_vad_auc_f1`, `evaluate_separation_alignment`) use a
**seeded random** `SpeakerSampler` (seed=123, reproducible); `validate_vad_auc` reads existing CSVs (no audio loading).
So only the null-control script was affected, by ~0.01 in the real-MAE mean.

### 6.2 2A on exact saved utterances (Step 2)

Real MAE now reproduces at **0.2333** (the authoritative value). Corrected block-9 numbers:

| Null | Null MAE | p | r |
|---|---|---|---|
| N1 circular-shift (PRIMARY) | 0.246 ± 0.081 | 0.202 | −0.147 |
| N2 full permutation (secondary) | 0.246 ± 0.080 | 0.207 | −0.145 |
| N3 constant-map (secondary) | 0.201 ± 0.101 | 7.7e-08 | +0.619 |

Block 23 (secondary): N1 p=0.103, N2 p=0.136, N3 p=1.5e-06. Conclusions unchanged: not below value-preserving nulls.

### 6.3 Bootstrap CI for 2B (Step 3)

Exclusive-activity AUC (block 23, exact pairs, n=40 eligible): mean **0.482**, 95% bootstrap CI **[0.389, 0.576]**
(10,000 resamples). The CI **excludes a speaker-specific effect as large as the overall ranking effect** (~0.2 AUC
shift ⇒ ~0.7). Wording: **large speaker-specific effects excluded; small effects cannot be excluded.**

### 6.4 Held-out replication of the block-23 VAD result (Step 4)

The layer choice and VAD-AUC hypothesis were developed on the original 35 pairs; the N=100 pool contains 65 pairs
never used in any layer/method investigation. Pre-registered primary: block 23, 65 new pairs (130 instances),
per-instance AUC(CAM_s, v_s) vs. own reference VAD, one-sample Wilcoxon vs. 0.5.

| Set | AUC | 95% CI | Wilcoxon p | r |
|---|---|---|---|---|
| Development (35 pairs, n=70) | 0.723 ± 0.149 | [0.687, 0.757] | 3.0e-11 | +0.914 |
| **Held-out (65 pairs, n=130)** | **0.681 ± 0.192** | **[0.647, 0.712]** | **8.1e-15** | **+0.786** |

**The effect replicates**: AUC clearly above 0.5, CI excludes 0.5, p ≈ 8e-15. Block 9 on the same held-out pairs is
below 0.5 (0.443 ± 0.174, CI [0.412, 0.473], p=1.2e-03) — consistent with the development set. This is now the
strongest positive claim in the repo: replicated on data never touched during method development.

---

## Phase 7 — Pair-level clustered re-analysis (2026-10-12)

The per-instance tests treat the two speakers of a pair as independent, but they share a mixture and are correlated.
This re-analysis clusters at the **pair** level: average the two per-instance AUCs within each pair, run the
one-sample Wilcoxon vs. 0.5 across **pairs**, and compute the 95% CI with a bootstrap that resamples **pairs**, not
instances. Script: `scripts/evaluate_pair_level.py`. Artifacts: `results/pair_level/pair_level_summary.json`.

### Old (instance-level) vs. new (pair-level)

| Metric | Instance-level | Pair-level |
|---|---|---|
| Block 23, dev 35 pairs | 0.723 ± 0.149 (n=70), p=3.0e-11, r=+0.914 | 0.723 ± 0.093 (n=35), CI [0.692, 0.753], p=5.8e-11, r=+1.000 |
| Block 23, held-out 65 pairs | 0.681 ± 0.192 (n=130), p=8.1e-15, r=+0.786 | 0.681 ± 0.130 (n=65), CI [0.649, 0.713], p=7.4e-11, r=+0.929 |
| Block 9, dev 35 pairs | 0.417 ± 0.192 (n=70), p=0.0023, r=−0.420 | 0.417 ± 0.105 (n=35), CI [0.384, 0.452], p=8.4e-05, r=−0.721 |
| Block 9, held-out 65 pairs | 0.443 ± 0.174 (n=130), p=1.2e-03, r=−0.327 | 0.443 ± 0.129 (n=65), CI [0.412, 0.474], p=1.5e-03, r=−0.454 |
| 2B exclusive-activity (block 23) | 0.482 ± 0.301 (n=40), p=0.856, r=−0.026 | 0.482 ± 0.169 (n=20 both-eligible pairs), CI [0.406, 0.554], p=0.748, r=−0.081 |

**Conclusion unchanged under pair-level clustering.** The block-23 VAD-timing signal stays clearly above 0.5 (held-out
p=7.4e-11 across pairs); block 9 stays below 0.5; the 2B exclusive-activity test stays non-significant and
underpowered (pair-level CI [0.406, 0.554] still excludes a ~0.2 shift). Clustering inflates neither the point
estimates nor the significance — the within-pair correlation is weak enough that instance-level and pair-level agree.

---

## Phase 8 — Robustness of the validated VAD signal (AUC-based) (2026-10-12)

The earlier robustness analysis measured speaker-vs-speaker MAE vs. a uniform-noise control — a metric later shown to
be uninformative (sparse maps beat uniform noise trivially), so those numbers are superseded. This re-tests robustness
on the one validated, replicated statistic: the block-23 VAD-logit CAM ranking speech-active frames above inactive
ones (pair-level AUC-ROC vs. the speaker's own clean-source reference VAD).

Pre-registered script: `scripts/evaluate_robustness_auc.py` (commit `ad0d44c`, before execution).
Artifacts: `results/robustness_auc/robustness_auc_heldout.{csv,json}` and `..._dev.{csv,json}`.

### Degradation validation (Step 3)

- Realized SNR matches nominal exactly (10/5/0 dB -> 10.00/5.00/0.00 dB).
- Measured T60 (Schroeder) close to nominal: 0.2 -> 0.164 s; 0.4 -> 0.390 s; 0.6 -> 0.637 s.
- Direct-path delay = 68 samples = 4.25 ms < 16 ms (one STFT frame) — no reference shift needed.
- Babble built from 8 leftover LibriSpeech utterances from speakers **not** in the mixture.

### Reference VAD choice (Step 4)

Silero VAD on the clean, dry sources (ground truth for when each speaker talks, independent of the degradation).
Consequence: in reverberation, reverberant tails extend acoustic energy beyond dry-source activity, so the reference
under-marks tail frames — the reverb AUC numbers are conservative.

### Results (held-out 65 pairs, block 23, pair-level)

| Condition | CAM AUC [95% CI] | Persists? (Holm) | Δ vs clean [95% CI] | Robust? (−0.06) | Network-VAD AUC |
|---|---|---|---|---|---|
| clean | 0.681 [0.649, 0.713] | — | — | — | 0.827 |
| reverb T60=0.2 | 0.687 [0.652, 0.720] | yes (9e-10) | +0.006 [−0.026, +0.037] | **yes** | 0.845 |
| reverb T60=0.4 | 0.665 [0.630, 0.699] | yes (1e-09) | −0.016 [−0.049, +0.017] | **yes** | 0.830 |
| reverb T60=0.6 | 0.651 [0.623, 0.679] | yes (3e-10) | −0.030 [−0.064, +0.003] | borderline | 0.843 |
| noise SNR=10 | 0.644 [0.620, 0.667] | yes (3e-10) | −0.037 [−0.069, −0.004] | borderline | 0.855 |
| noise SNR=5 | 0.614 [0.591, 0.635] | yes (1e-09) | −0.067 [−0.103, −0.031] | no | 0.844 |
| noise SNR=0 | 0.577 [0.552, 0.604] | yes (1e-06) | −0.103 [−0.139, −0.067] | no | 0.815 |
| combo T60=0.4+SNR=5 | 0.625 [0.601, 0.649] | yes (7e-10) | −0.056 [−0.090, −0.021] | no | 0.836 |

Two results are **borderline** per the pre-registered rule (CI lower bound within 0.01 of the −0.06 margin):
reverb T60=0.6 (−0.064) and noise SNR=10 (−0.069). Per the discipline, these are reported as borderline, not called
robust.

### Three-way classification and multiplicity sensitivity (2026-10-13)

Script: `scripts/evaluate_robustness_sensitivity.py` (no new simulations; recomputes per-pair AUCs with the same
seeded pipeline, cached in `results/robustness_auc/per_pair_auc_*.json`). Artifacts:
`results/robustness_auc/robustness_sensitivity_{heldout,dev}.{csv,json}`.

**Three-way non-inferiority classification** (margin −0.06, 95% pair-bootstrap CI of AUC_condition − AUC_clean):
robust (lower > −0.06), inferior (upper < −0.06), inconclusive (otherwise).

| Condition | Δ [95% CI] | Verdict (95%) | Bonferroni Δ lower/upper (99.3%) | Verdict (Bonf) | Changed? |
|---|---|---|---|---|---|
| reverb T60=0.2 | +0.006 [−0.026, +0.037] | robust | [−0.038, +0.049] | robust | no |
| reverb T60=0.4 | −0.016 [−0.049, +0.017] | robust | [−0.062, +0.029] | inconclusive | **yes** |
| reverb T60=0.6 | −0.030 [−0.064, +0.003] | inconclusive | [−0.077, +0.015] | inconclusive | no |
| noise SNR=10 | −0.037 [−0.069, −0.004] | inconclusive | [−0.080, +0.007] | inconclusive | no |
| noise SNR=5 | −0.067 [−0.103, −0.031] | inconclusive | [−0.114, −0.016] | inconclusive | no |
| noise SNR=0 | −0.103 [−0.139, −0.067] | **inferior** | [−0.152, −0.053] | inconclusive | **yes** |
| combo | −0.056 [−0.090, −0.021] | inconclusive | [−0.102, −0.008] | inconclusive | no |

**Correction to earlier wording**: the prior log/README said the signal "degrades beyond the margin at SNR ≤ 5 dB and
the combined condition". From the saved CIs, only **SNR = 0 dB** is entirely beyond the margin (upper bound −0.067 <
−0.06). SNR = 5 dB (upper −0.031) and the combined condition (upper −0.021) have upper bounds *inside* the margin — so
the correct statement is: *degradation is significant (CI excludes 0) at SNR ≤ 5 dB and the combined condition, but
degradation beyond the margin is established only at 0 dB; robustness is not demonstrated at 5 dB or in the combined
condition.*

**Multiplicity sensitivity (Bonferroni 99.3% CIs; primary verdicts unchanged)**: two verdicts are sensitive.
T60=0.4 s moves robust → inconclusive (adjusted lower bound −0.062 crosses the margin), and SNR=0 moves inferior →
inconclusive (adjusted upper bound −0.053 crosses back inside). T60=0.2 s stays robust at both coverages. The
pre-registered primary verdicts are the 95% ones; the Bonferroni column is a sensitivity check only.

**Net (held-out, block 23), condition by condition:** the signal **persists** (above chance, Holm-significant) in
every condition; **robust within the margin** at T60=0.2 s (stable) and T60=0.4 s (with the multiplicity caveat);
**inconclusive** at T60=0.6, SNR=10, SNR=5, and the combined condition; **inferior** at SNR=0 (borderline; flips to
inconclusive under Bonferroni). The network's own VAD stays high everywhere (0.81–0.86), so the CAM-AUC drop under
strong noise reflects the *explanation* degrading faster than the network. Development set (35 pairs, secondary):
same direction (robust at all T60 and SNR=10; inconclusive at SNR=5 and combo; inferior at SNR=0), consistent with
held-out. Reverberation references are dry-source (under-mark reverb tails), so reverb numbers are conservative.




F1-vs-chance is a hard-threshold decision against a strong majority-class baseline. Re-calibrating F1 at the
Youden's-J-optimal threshold still gives 0.679 < 0.815 chance. AUC-PR baselined against the 0.814 prevalence gives
0.908 (block 23) vs. 0.809 (block 9), consistent with the AUC-ROC result.

**Net:** block 23 carries genuine ranking-level VAD-timing information, but not enough to beat a majority-class
threshold decision. Investigated and closed.

## Phase 4 — Separation (IBM) ground-truth alignment

Question: does Grad-CAM align with the Ideal Binary Mask (which T-F region belongs to which speaker)? Ground truth:
IBM from the clean sources (the literal signal the mixture was built from).

- **Permutation bug caught and fixed**: the network is PIT-trained, so output slot 0/1 doesn't map consistently to
  true speaker 0/1. Scoring against the wrong assignment made the ceiling look broken (0.088 < 0.697 chance).
  Best-permutation assignment fixed it: corrected ceiling F1 ≈ 0.63–0.67 — the network separates well.
  [scripts/evaluate_separation_alignment.py](../scripts/evaluate_separation_alignment.py).
- **Full-scale result**: mask-value-targeted CAM was significantly **below** the chance baseline at both layers
  (F1 ≈ 0.19–0.21 vs. 0.502; p<10⁻¹⁰, r≈−0.93..−0.99, n=70).
- **Inversion ruled out**: a flip test initially "fixed" it (F1 ≈ 0.62), but the orientation-invariant AUC-ROC
  ([scripts/evaluate_auc_check.py](../scripts/evaluate_auc_check.py)) showed ≈0.497–0.498 at both layers — no real
  signal either way. The flip-test improvement was a sparse-CAM/base-rate artifact, not a genuine inversion (the
  earlier "ReLU sign" explanation was itself wrong and was discarded).

**Net:** the network separates well, but single-point Grad-CAM carries no measurable IBM alignment. Closed.

## What's genuinely still open

- Accent diversity (LibriSpeech has no accent labels; needs Common Voice / VCTK).
- The noise proxy in the robustness test is a babble-sum of leftover LibriSpeech utterances, not a real ambient-noise
  corpus like WHAM!.
- The VAD reference is Silero VAD output, not hand-labeled ground truth.
- CPU-only verification (GPU supported but not separately validated).
