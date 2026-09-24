"""
For the concluded random sweep (extract_TR_from_model_random_sweep.*), compare
the "on-side" threshold voltage each method reports against the model's true
on-threshold, VTON_model = Vt + Vtr:

  1. extract_VTR            -> VTON                 (already in the sweep CSV)
  2. extract_VTR_derivative -> Vg_peak              (on-side of VTR_deriv)
  3. extract_VTR_gm_gmid    -> VT_rising             (on-side of VTR_gm_gmid)

Unlike Vt (fixed, see vtoff_vs_model_Vt.py), VTON_model varies across the
sweep through Vtr_true, so R^2 is meaningful here. Reuses the Id-Vg curves
already saved in extract_TR_from_model_random_sweep_idvg.npz -- the
drain-current model is NOT re-run, only the (cheap) extraction functions are
re-applied to recover Vg_peak/VT_rising, which were not saved the first time.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from new_TR_extraction_funs import extract_VTR_derivative, extract_VTR_gm_gmid, style_axes
from TR_related_equations import TRParams

script_dir = Path(__file__).parent.resolve()

sweep_df = pd.read_csv(script_dir / "extract_TR_from_model_random_sweep.csv")
npz = np.load(script_dir / "extract_TR_from_model_random_sweep_idvg.npz")
VGS = npz["VGS"]
ID_matrix = npz["ID"]

assert len(sweep_df) == len(ID_matrix), "CSV and Id-Vg npz row counts don't match"

ID_LIMIT = 1e-15
Vt_true = TRParams().Vt  # 0.3 V, fixed for every point in this sweep
sweep_df["VTON_model"] = Vt_true + sweep_df["Vtr_true"]

Vg_peak_list = np.full(len(sweep_df), np.nan)
VT_rising_list = np.full(len(sweep_df), np.nan)

for i in range(len(sweep_df)):
    ID = ID_matrix[i]
    if not np.all(np.isfinite(ID)):
        continue
    try:
        _, det_deriv = extract_VTR_derivative(VGS, ID, ID_limit=ID_LIMIT, window_length=5)
        Vg_peak_list[i] = det_deriv["Vg_peak"]
    except Exception:
        pass
    try:
        _, det_gm_gmid = extract_VTR_gm_gmid(VGS, ID, ID_limit=ID_LIMIT, window_length=5)
        VT_rising_list[i] = det_gm_gmid["VT_rising"]
    except Exception:
        pass

sweep_df["Vg_peak"] = Vg_peak_list
sweep_df["VT_rising"] = VT_rising_list

out_csv = script_dir / "extract_TR_from_model_random_sweep_vton.csv"
sweep_df.to_csv(out_csv, index=False)
print(f"Saved augmented sweep (with on-side thresholds) to {out_csv}")

METHODS = [
    ("VTON", "extract_VTR\n($V_{T,on}$, raw $I_D$ extrapolation)", "#2a78d6"),
    ("Vg_peak", "extract_VTR_derivative\n($g_m$ rising-edge centroid)", "#eb6834"),
    ("VT_rising", "extract_VTR_gm_gmid\n($g_m$ 50% crossing)", "#1baf7a"),
]

x_true = sweep_df["VTON_model"].to_numpy()

print(f"\nModel VTON = Vt + Vtr  (Vt = {Vt_true:.4f} V, Vtr swept)")
print("\nOn-side threshold vs. model VTON:")
for col, _, _ in METHODS:
    y = sweep_df[col].to_numpy()
    finite = np.isfinite(x_true) & np.isfinite(y)
    err = y[finite] - x_true[finite]
    r2 = np.corrcoef(x_true[finite], y[finite])[0, 1] ** 2 if finite.sum() > 1 else np.nan
    print(f"  {col:12s}: n={int(finite.sum()):4d}  R^2={r2:.4f}  "
          f"bias={err.mean():+.4f} V  std={err.std():.4f} V")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)

    lims = [min(0.0, x_true.min()) - 0.05, x_true.max() + 0.1]

    for ax, (col, label, color) in zip(axes, METHODS):
        y = sweep_df[col].to_numpy()
        finite = np.isfinite(x_true) & np.isfinite(y)
        x_f, y_f = x_true[finite], y[finite]

        r2 = np.corrcoef(x_f, y_f)[0, 1] ** 2 if len(x_f) > 1 else np.nan

        ax.scatter(x_f, y_f, s=8, facecolor=color, edgecolor="none", alpha=0.35, zorder=3)
        ax.plot(lims, lims, color="gray", lw=1, ls=":", zorder=1)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel("model $V_{TON}$ = $V_T$ + $V_{tr}$ (V)", fontsize=12)
        ax.set_title(f"{label}\n$R^2$ = {r2:.3f}  (n={len(x_f)})", fontsize=11)
        style_axes(ax)

    axes[0].set_ylabel("on-side threshold, extracted (V)", fontsize=12)
    fig.suptitle("On-side threshold vs. model $V_{TON}$ (random sweep, reusing saved $I_D$-$V_G$ curves)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    fpath = script_dir / "model_VTON_comparison.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"\nSaved comparison plot to {fpath}")
except ImportError:
    pass
