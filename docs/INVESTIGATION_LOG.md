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
subgroup breakdowns). Robustness to noise and reverberation tested (the MAE discriminability holds; VAD-F1 degrades
mildly under noise, not reverb). This is the still-valid positive finding, kept separate from the VAD-alignment
investigation below.

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
