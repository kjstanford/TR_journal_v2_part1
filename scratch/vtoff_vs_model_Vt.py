"""
For the concluded random sweep (extract_TR_from_model_random_sweep.*), compare
the "off-side" threshold voltage each method reports against the model's
actual threshold parameter Vt:

  1. extract_VTR            -> VTOFF                (already in the sweep CSV)
  2. extract_VTR_derivative -> Vg_trough            (off-side of VTR_deriv)
  3. extract_VTR_gm_gmid    -> VT_gm_over_id         (off-side of VTR_gm_gmid)

Reuses the Id-Vg curves already saved in extract_TR_from_model_random_sweep_idvg.npz
-- the drain-current model is NOT re-run, only the (cheap) extraction functions
are re-applied to get Vg_trough/VT_gm_over_id, which were not saved to the
summary CSV the first time around.
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
idx = npz["idx"]

assert len(sweep_df) == len(ID_matrix), "CSV and Id-Vg npz row counts don't match"

ID_LIMIT = 1e-15
Vt_true = TRParams().Vt  # 0.3 V, fixed for every point in this sweep (Vt was not swept)

Vg_trough_list = np.full(len(sweep_df), np.nan)
VT_gm_over_id_list = np.full(len(sweep_df), np.nan)

for i in range(len(sweep_df)):
    ID = ID_matrix[i]
    if not np.all(np.isfinite(ID)):
        continue
    try:
        _, det_deriv = extract_VTR_derivative(VGS, ID, ID_limit=ID_LIMIT, window_length=5)
        Vg_trough_list[i] = det_deriv["Vg_trough"]
    except Exception:
        pass
    try:
        _, det_gm_gmid = extract_VTR_gm_gmid(VGS, ID, ID_limit=ID_LIMIT, window_length=5)
        VT_gm_over_id_list[i] = det_gm_gmid["VT_gm_over_id"]
    except Exception:
        pass

sweep_df["Vg_trough"] = Vg_trough_list
sweep_df["VT_gm_over_id"] = VT_gm_over_id_list

out_csv = script_dir / "extract_TR_from_model_random_sweep_vtoff.csv"
sweep_df.to_csv(out_csv, index=False)
print(f"Saved augmented sweep (with off-side thresholds) to {out_csv}")

METHODS = [
    ("VTOFF", "extract_VTR\n($V_{T,off}$, raw $I_D$ extrapolation)", "#2a78d6"),
    ("Vg_trough", "extract_VTR_derivative\n($g_m/I_D$ trough)", "#eb6834"),
    ("VT_gm_over_id", "extract_VTR_gm_gmid\n($g_m/I_D$ 50% crossing)", "#1baf7a"),
]

print(f"\nModel Vt (constant, not swept) = {Vt_true:.4f} V")
print("\nOff-side threshold vs. model Vt:")
for col, _, _ in METHODS:
    y = sweep_df[col].to_numpy()
    finite = np.isfinite(y)
    err = y[finite] - Vt_true
    r2 = np.corrcoef(np.full(finite.sum(), Vt_true), y[finite])[0, 1] ** 2 if finite.sum() > 1 else np.nan
    print(f"  {col:15s}: n={int(finite.sum()):4d}  mean={y[finite].mean():.4f} V  "
          f"bias={err.mean():+.4f} V  std={err.std():.4f} V  R^2(vs Vt)={r2}")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharex=True, sharey=True)

    all_y = np.concatenate([sweep_df[col].to_numpy() for col, _, _ in METHODS])
    all_y = all_y[np.isfinite(all_y)]
    pad = 0.05 * (all_y.max() - all_y.min() + 1e-6)
    ylims = [all_y.min() - pad, all_y.max() + pad]
    xlims = [Vt_true - 0.05, Vt_true + 0.05]

    for ax, (col, label, color) in zip(axes, METHODS):
        y = sweep_df[col].to_numpy()
        finite = np.isfinite(y)
        y_f = y[finite]
        x_f = np.full(finite.sum(), Vt_true)

        r2 = np.corrcoef(x_f, y_f)[0, 1] ** 2 if finite.sum() > 1 and y_f.std() > 0 else np.nan

        # jitter the constant model-Vt x-position slightly so the vertical
        # scatter is visible instead of a single line of overlapping points
        rng = np.random.default_rng(0)
        x_jitter = x_f + rng.uniform(-0.01, 0.01, size=x_f.shape)

        ax.scatter(x_jitter, y_f, s=8, facecolor=color, edgecolor="none", alpha=0.35, zorder=3)
        ax.axhline(Vt_true, color="gray", lw=1, ls=":", zorder=1)
        ax.axvline(Vt_true, color="gray", lw=1, ls=":", zorder=1)

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xlabel("model $V_T$ (V, fixed)", fontsize=12)
        r2_str = f"{r2:.3f}" if np.isfinite(r2) else "n/a (model $V_T$ is constant)"
        ax.set_title(f"{label}\n$R^2$ = {r2_str}  (n={int(finite.sum())})", fontsize=11)
        style_axes(ax)

    axes[0].set_ylabel("off-side threshold, extracted (V)", fontsize=12)
    fig.suptitle("Off-side threshold vs. model $V_T$ (random sweep, reusing saved $I_D$-$V_G$ curves)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    fpath = script_dir / "model_VTOFF_vs_Vt_comparison.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"\nSaved comparison plot to {fpath}")
except ImportError:
    pass
