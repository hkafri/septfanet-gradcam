"""Per-frame decision-map figures (Step 3). Visual, mostly descriptive.

Pair selection: the 8 held-out pairs in results/per_frame_maps/selection.json,
chosen by the rule declared and committed in scripts/select_per_frame_pairs.py
BEFORE any figure was generated (6 b23-AUC percentile pairs 5/20/40/60/80/95 +
2 max-exclusive-activity pairs; thesis pair = percentile pair closest to the
held-out median AUC).

Per pair (layer TCN.TCN.23.conv1d, variant "gap", SI-SDR matched), one figure
with two stacked speaker blocks, each block:
  A  log-power spectrogram of the mixture (model.spectrum), Silero reference
     v_c bar, the network's sigma(z_c) curve, frames with sigma > 0.5 marked;
  B  the matrix M_c[tau, t] (x = map position t, y = decision frame tau;
     active-decision rows first, then inactive rows, separated and labelled;
     dashed marks at t = tau; side strip = v_c(tau) per row; perceptually
     uniform sequential colormap + colorbar; per-row min-max -- the same figure
     with raw rows and one global scale is saved with suffix _raw);
  C  (i) the single thesis map (target t*_c = argmax|z|, sign +1 as in the
     thesis) and (ii) the mean of all active rows, with v_c shaded.
A second figure per pair shows M_c and M_c' side by side on the same colour
scale. Random-weights-control versions (Step 5 sanity check, seed 0, no
checkpoint) are saved with suffix _random. figures/per_frame_maps_thesis.pdf
holds Panels A+B of the median-AUC pair, 6.3 in wide.

Every matrix is saved to results/per_frame_maps/matrices_heldout<idx>.npz
(raw M for both variants/slots/networks, tau lists, signs, z, v).

New files only; nothing existing is modified.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from silero_vad import load_silero_vad

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

gradcam_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(gradcam_root))

from scripts.evaluate_baselines import resolve_all_pairs, load_and_pad, normalize_audio, minmax, NUM_FRAMES
from scripts.evaluate_per_frame_maps import build_model, compute_pair_cache, LAYER
from scripts.per_frame_cam import per_frame_cam_matrix

OUT_DIR = gradcam_root / "results" / "per_frame_maps"
FIG_W = 6.3  # inches, thesis text width
CMAP = "viridis"       # perceptually uniform sequential (magnitude job)
CMAP_SPEC = "magma"    # spectrogram, distinct from the CAM colormap
C_NET = "#2a7de1"      # network sigma(z) curve
C_MARK = "#e1762a"     # sigma > 0.5 marks
C_REF = "0.45"         # Silero reference

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def row_minmax(M):
    out = np.zeros_like(M)
    for i, r in enumerate(M):
        out[i] = minmax(r)
    return out


def speaker_arrays(data, slot):
    """Matched reference + matrix + logits for one slot of a cached pair."""
    s2s = data["slot_to_source"]
    v = [data["v0"].astype(bool), data["v1"].astype(bool)]
    return {
        "v_own": v[s2s[slot]], "v_other": v[1 - s2s[slot]],
        "z": data["z"][slot].astype(np.float64),
        "gap": data[f"gap{slot}"].astype(np.float64),
        "gradact": data[f"gradact{slot}"].astype(np.float64),
    }


def grouped_rows(M, z):
    """Active rows (tau ascending) first, then inactive rows; returns
    (rows, taus, n_active)."""
    act = np.nonzero(z > 0)[0]
    inact = np.nonzero(z <= 0)[0]
    order = np.concatenate([act, inact])
    return M[order], order, len(act)


def panel_A(ax_spec, ax_vad, data, sp, label):
    spec = data["spectrum"].astype(np.float64)
    ax_spec.imshow(spec, aspect="auto", origin="lower", cmap=CMAP_SPEC,
                   extent=[0, NUM_FRAMES, 0, spec.shape[0]])
    ax_spec.set_ylabel("freq bin")
    ax_spec.set_xticks([])
    ax_spec.set_title(label, loc="left")
    sig = 1.0 / (1.0 + np.exp(-sp["z"]))
    t = np.arange(NUM_FRAMES)
    ax_vad.fill_between(t, 0, sp["v_own"].astype(float), step="mid",
                        color=C_REF, alpha=0.35, lw=0, label="Silero ref. $v_c$")
    ax_vad.plot(t, sig, color=C_NET, lw=1.0, label=r"network $\sigma(z_c)$")
    on = sig > 0.5
    ax_vad.plot(t[on], np.full(on.sum(), 1.06), ls="none", marker="|", ms=3,
                color=C_MARK, label=r"$\sigma>0.5$")
    ax_vad.axhline(0.5, color="0.75", lw=0.6, ls=":")
    ax_vad.set_ylim(-0.04, 1.14)
    ax_vad.set_xlim(0, NUM_FRAMES)
    ax_vad.set_yticks([0, 0.5, 1])
    ax_vad.set_ylabel("VAD")
    ax_vad.set_xticks([])


def panel_B(ax_strip, ax_mat, M_disp, taus, n_active, v_own, cax=None,
            vmin=0.0, vmax=1.0, ylabel=True):
    im = ax_mat.imshow(M_disp, aspect="auto", origin="upper", cmap=CMAP,
                       vmin=vmin, vmax=vmax,
                       extent=[0, NUM_FRAMES, len(taus), 0], interpolation="nearest")
    # dashed t = tau marks, one per row (a diagonal within each block)
    ax_mat.plot(taus[:n_active] + 0.5, np.arange(n_active) + 0.5,
                ls="--", lw=0.7, color="w", dashes=(2, 2))
    if len(taus) > n_active:
        ax_mat.plot(taus[n_active:] + 0.5, np.arange(n_active, len(taus)) + 0.5,
                    ls="--", lw=0.7, color="w", dashes=(2, 2))
        ax_mat.axhline(n_active, color="w", lw=1.2)
        ax_mat.text(2, max(n_active - 2, 1), "active decisions ($z>0$)",
                    color="w", fontsize=6.5, va="bottom")
        ax_mat.text(2, min(n_active + 2, len(taus) - 1), "inactive decisions ($z<0$)",
                    color="w", fontsize=6.5, va="top")
    ax_mat.set_xlabel("map position $t$ (frame)")
    ax_mat.set_yticks([])
    # side strip: v_own at each row's decision frame
    strip = v_own[taus].astype(float)[:, None]
    ax_strip.imshow(strip, aspect="auto", origin="upper", cmap="Greys",
                    vmin=0, vmax=1.4, interpolation="nearest")
    ax_strip.set_xticks([]); ax_strip.set_yticks([])
    if ylabel:
        ax_strip.set_ylabel(r"decision frame $\tau$ (row)")
    ax_strip.set_title("$v_c$", fontsize=6, pad=2)
    if cax is not None:
        plt.colorbar(im, cax=cax)
    return im


def panel_C(ax, sp, thesis_map, tstar):
    t = np.arange(NUM_FRAMES)
    ax.fill_between(t, 0, sp["v_own"].astype(float), step="mid", color=C_REF,
                    alpha=0.25, lw=0, label="$v_c$")
    ax.plot(t, minmax(thesis_map), color="#7c3aed", lw=1.0,
            label=rf"thesis single map ($t^*={tstar}$)")
    act = np.nonzero(sp["z"] > 0)[0]
    rows = sp["gap"][act]
    keep = rows.sum(axis=1) > 1e-12
    mean_map = rows[keep].mean(axis=0) if keep.any() else np.zeros(NUM_FRAMES)
    ax.plot(t, minmax(mean_map), color="#0f766e", lw=1.0, label="mean of active rows")
    ax.axvline(tstar, color="#7c3aed", lw=0.6, ls=":")
    ax.set_xlim(0, NUM_FRAMES)
    ax.set_ylim(-0.04, 1.28)
    ax.set_yticks([0, 1])
    ax.set_ylabel("map (min-max)")
    ax.set_xlabel("map position $t$ (frame)")
    ax.legend(ncol=3, loc="upper center", fontsize=6.5, frameon=True,
              framealpha=0.85, edgecolor="none", borderpad=0.2)


def thesis_map_for(model, pair, slot, z, device="cpu"):
    """The single map of the thesis: target t* = argmax|z|, sign +1 (plain
    backward of the logit, as in evaluate_baselines.compute_cam)."""
    sig0, sig1 = load_and_pad(pair[0]), load_and_pad(pair[1])
    audio = torch.from_numpy(normalize_audio(sig0 + sig1)).unsqueeze(0)
    tstar = int(np.argmax(np.abs(z)))
    row = per_frame_cam_matrix(model, audio, LAYER, slot, [tstar], +1, "gap", device)[0]
    return row, tstar


def pair_figure(data, pair, model, out_stem, title, raw=False, device="cpu"):
    fig = plt.figure(figsize=(FIG_W, 9.2))
    gs = fig.add_gridspec(2, 1, hspace=0.24)
    for slot in range(2):
        sp = speaker_arrays(data, slot)
        sub = gs[slot].subgridspec(4, 2, height_ratios=[1.0, 0.55, 2.6, 0.9],
                                   width_ratios=[0.035, 1.0], hspace=0.14, wspace=0.03)
        ax_spec = fig.add_subplot(sub[0, 1])
        ax_vad = fig.add_subplot(sub[1, 1])
        ax_strip = fig.add_subplot(sub[2, 0])
        ax_mat = fig.add_subplot(sub[2, 1])
        ax_c = fig.add_subplot(sub[3, 1])
        src = data["slot_to_source"][slot]
        panel_A(ax_spec, ax_vad, data, sp, f"speaker $c$ = source {src} (slot {slot})")
        M, taus, n_act = grouped_rows(sp["gap"], sp["z"])
        if raw:
            vmax = np.percentile(M, 99.5) or 1.0
            disp, vmin = M, 0.0
        else:
            disp, vmin, vmax = row_minmax(M), 0.0, 1.0
        pos = ax_mat.get_position()
        cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.013, pos.height])
        panel_B(ax_strip, ax_mat, disp, taus, n_act, sp["v_own"], cax=cax,
                vmin=vmin, vmax=float(vmax))
        row_t, tstar = thesis_map_for(model, pair, slot, sp["z"], device)
        panel_C(ax_c, sp, row_t, tstar)
    norm_txt = ("raw rows, one global scale (99.5th-percentile clip)"
                if raw else "per-row min-max normalised rows")
    fig.suptitle(f"{title}\nPanel B: {norm_txt}; variant gap; layer {LAYER}", y=0.995,
                 fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def speakers_side_by_side(data, out_stem, title):
    fig = plt.figure(figsize=(FIG_W, 3.4))
    gs = fig.add_gridspec(1, 2, wspace=0.12)
    mats = []
    for slot in range(2):
        sp = speaker_arrays(data, slot)
        M, taus, n_act = grouped_rows(sp["gap"], sp["z"])
        mats.append((row_minmax(M), taus, n_act, sp))
    for slot, (disp, taus, n_act, sp) in enumerate(mats):
        sub = gs[slot].subgridspec(1, 2, width_ratios=[0.04, 1.0], wspace=0.03)
        ax_strip = fig.add_subplot(sub[0, 0])
        ax_mat = fig.add_subplot(sub[0, 1])
        cax = None
        if slot == 1:
            pos = ax_mat.get_position()
            cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.015, pos.height])
        panel_B(ax_strip, ax_mat, disp, taus, n_act, sp["v_own"], cax=cax,
                ylabel=(slot == 0))
        ax_mat.set_title(f"$M_c$, source {data['slot_to_source'][slot]}")
    fig.suptitle(f"{title} -- both speakers, same colour scale (per-row min-max)",
                 y=1.00, fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def thesis_figure(data, out_path):
    """Panels A+B for both speakers of the median-AUC pair, 6.3 in wide."""
    fig = plt.figure(figsize=(FIG_W, 4.6))
    gs = fig.add_gridspec(1, 2, wspace=0.14)
    for slot in range(2):
        sp = speaker_arrays(data, slot)
        sub = gs[slot].subgridspec(3, 2, height_ratios=[0.85, 0.45, 2.6],
                                   width_ratios=[0.045, 1.0], hspace=0.16, wspace=0.03)
        ax_spec = fig.add_subplot(sub[0, 1])
        ax_vad = fig.add_subplot(sub[1, 1])
        ax_strip = fig.add_subplot(sub[2, 0])
        ax_mat = fig.add_subplot(sub[2, 1])
        panel_A(ax_spec, ax_vad, data, sp,
                f"speaker $c$ = source {data['slot_to_source'][slot]}")
        M, taus, n_act = grouped_rows(sp["gap"], sp["z"])
        cax = None
        if slot == 1:
            pos = ax_mat.get_position()
            cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.015, pos.height])
        panel_B(ax_strip, ax_mat, row_minmax(M), taus, n_act, sp["v_own"],
                cax=cax, ylabel=(slot == 0))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_matrices(idx, caches_real, caches_random):
    out = {}
    for net, data in (("real", caches_real), ("random", caches_random)):
        for slot in range(2):
            out[f"{net}_gap{slot}"] = data[f"gap{slot}"]
            out[f"{net}_gradact{slot}"] = data[f"gradact{slot}"]
            out[f"{net}_signs{slot}"] = np.where(data["z"][slot] > 0, 1, -1).astype(np.int8)
        out[f"{net}_z"] = data["z"]
        out[f"{net}_slot_to_source"] = data["slot_to_source"]
    out["taus"] = np.arange(NUM_FRAMES)
    out["v0"] = caches_real["v0"]
    out["v1"] = caches_real["v1"]
    np.savez_compressed(OUT_DIR / f"matrices_heldout{idx:02d}.npz", **out)


def main():
    device = "cpu"
    sel = json.loads((OUT_DIR / "selection.json").read_text())
    held = [p for p in resolve_all_pairs() if p[2] == "heldout"]
    vad_model = load_silero_vad()
    models = {net: build_model(net, device) for net in ("real", "random")}

    thesis_idx = None
    for p in sel["pairs"]:
        idx = p["heldout_index"]
        pair = held[idx]
        data = {net: compute_pair_cache(models[net], vad_model, pair, net, idx, device)
                for net in ("real", "random")}
        save_matrices(idx, data["real"], data["random"])
        roles = ",".join(p["roles"])
        base = OUT_DIR / f"per_frame_heldout{idx:02d}"
        title = (f"held-out pair {idx} ({p['utt_0']} + {p['utt_1']}), roles: {roles}, "
                 f"b23 AUC {p['b23_auc']:.3f}")
        pair_figure(data["real"], pair, models["real"], str(base), title, raw=False, device=device)
        pair_figure(data["real"], pair, models["real"], str(base) + "_raw", title, raw=True, device=device)
        pair_figure(data["random"], pair, models["random"], str(base) + "_random",
                    title + " -- RANDOM-WEIGHTS CONTROL (seed 0, no checkpoint)",
                    raw=False, device=device)
        speakers_side_by_side(data["real"], str(base) + "_speakers", title)
        if "thesis_figure" in p["roles"]:
            thesis_idx = idx
            (gradcam_root / "figures").mkdir(exist_ok=True)
            thesis_figure(data["real"], gradcam_root / "figures" / "per_frame_maps_thesis.pdf")
        print(f"  [+] figures for held-out pair {idx} ({roles})")
    print(f"[+] thesis figure from held-out pair {thesis_idx} -> figures/per_frame_maps_thesis.pdf")


if __name__ == "__main__":
    main()
