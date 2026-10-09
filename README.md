# Sep-TFAnet-VAD Grad-CAM

Grad-CAM interpretability for [Sep-TFAnet-VAD](https://github.com/MordehayM/Sep-TFAnet-VAD), a PyTorch
speaker-separation/VAD network — adapted from image-based Grad-CAM and validated on real LibriSpeech audio.

**Full investigation narrative** (every hypothesis, bug fix, and pre-registration decision, in order) is preserved in
[docs/INVESTIGATION_LOG.md](docs/INVESTIGATION_LOG.md). This README is the scannable, result-first summary.

## What Grad-CAM successfully explains

**VAD timing, at the ranking level (validated).** At the late TCN layer `TCN.TCN.23.conv1d`, the continuous VAD-logit
CAM carries real discriminative signal about voice-activity timing: AUC-ROC = **0.723 ± 0.149** (p ≈ 3×10⁻¹¹, n=70,
rank-biserial r = +0.914). The ranking signal is real — but it does **not** beat a majority-class-matched threshold
decision, because ~82% of frames are genuinely speech-active (re-calibrated F1 = 0.679 vs. a chance baseline of
0.815).


- **Speaker discriminability (MAE, pre-registered null controls)**: the speaker-vs-speaker MAE is *not*
  significantly different from value-distribution-preserving nulls (circular-shift null N1: p=0.487; full temporal
  permutation null N2: p=0.462) at the pre-registered layer `TCN.TCN.9.conv1d` on N=100 pairs. The earlier
  "discriminability" result (real MAE 0.233 vs. uniform-noise 0.416, p=9e-17) is **not sufficient** to claim the
  CAMs carry speaker-discriminative structure — that difference is explained by the maps' value distributions
  (sparsity), not by any genuine speaker-dependent temporal alignment. The claim is removed from the positive
  findings above and placed here as a null result. (Pre-registered script: `scripts/evaluate_mae_null_controls.py`.)
- **Speaker-specificity (exclusive-activity AUC, pre-registered)**: at `TCN.TCN.23.conv1d`, restricting to frames
  where exactly one speaker is active, the CAM does not rank "speaker s active, other not" frames above "other
  active, speaker s not" frames above chance (exclusive-activity AUC = 0.519 vs. 0.5, p=0.630, n=50 eligible).
  The CAMs are not speaker-specific in the intended sense.
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
pre-registered null controls that preserve the maps' value distributions and temporal smoothness.

| Comparison | Real MAE | Null MAE | Wilcoxon p | Rank-biserial r |
|---|---|---|---|---|
| Real vs. uniform-noise control (original, insufficient) | 0.233 ± 0.091 | 0.416 ± 0.051 | 9.0e-17 | −0.917 |
| Real vs. **N1 circular-shift null** (PRIMARY) | 0.243 ± 0.100 | 0.253 ± 0.087 | 0.487 | −0.080 |
| Real vs. **N2 full-permutation null** (secondary) | 0.243 ± 0.100 | 0.252 ± 0.087 | 0.462 | −0.085 |
| Real vs. **N3 constant-map baseline** (secondary) | 0.243 ± 0.100 | 0.208 ± 0.108 | 5.5e-09 | +0.672 |

The MAE is *not* significantly below the value-distribution-preserving nulls (N1, N2). The apparent discriminability
is an artifact of the maps' sparsity (value distribution), not speaker-dependent temporal alignment. The N3 result
(real > constant) is a secondary check that the maps are not simply degenerate.

### Speaker-specificity (exclusive-activity AUC, pre-registered, block 23)

Restricting to frames where exactly one speaker is active (n=50 eligible instances):

| Test | AUC | Null | Wilcoxon p | Rank-biserial r |
|---|---|---|---|---|
| Exclusive-activity AUC vs. 0.5 (PRIMARY) | 0.519 ± 0.310 | 0.5 | 0.630 | +0.086 |
| Own-vs-other ΔAUC (secondary) | +0.024 | 0 | 0.089 | +0.136 |

No evidence that the CAM is speaker-specific in the intended sense.

### VAD-timing ranking signal (continuous-CAM AUC-ROC, n=70)

| Layer | AUC-ROC | AUC-PR (baseline = prevalence 0.814) | Wilcoxon vs. 0.5 |
|---|---|---|---|
| Block 9 | 0.417 ± 0.192 | 0.809 | p=0.0023, r=−0.420 (below 0.5) |
| **Block 23** | **0.723 ± 0.149** | **0.908** | **p=3.03e-11, r=+0.914 (above 0.5)** |

### Separation (IBM) alignment (orientation-invariant AUC-ROC, n=70)

| Layer | AUC-ROC | Interpretation |
|---|---|---|
| Block 9 | 0.497 ± 0.068 | No signal (at chance) |
| Block 23 | 0.498 ± 0.099 | No signal (at chance) |

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
