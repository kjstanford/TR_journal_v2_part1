"""
Validate the three VTR extraction methods on the TR drain-current model (TR_related_equations.py)
with following parametric sweeps.
Vtr in [0, 1] V, phi_tail in [0.1, 1] eV, Cgch with EOT in [0.5, 5] nm, n in [1, 1.6]
Do an exhaustive sweep of all combinations of these parameters and generate a CSV file with the results.
Finally plot extracted VTR vs. model VTR for each method (combining all sweeps) and report the R^2 for each method.

  1. extract_VTR            -- subthreshold/above-threshold linear extrapolation
  2. extract_VTR_derivative -- gm/Id trough vs gm rising-edge centroid
  3. extract_VTR_gm_gmid    -- gm/Id 50%-of-max crossing vs gm 50%-of-max crossing

Geometry/mobility: L = 2 um, W = 1 um, mu_eff = 20 cm^2/V.s.
No CV/Cgg data is generated here, so extract_VTR_derivative/extract_VTR_gm_gmid
fall back to their gm-only default (see compute_gmid_gmcgg's docstring).
The model's own Vtr parameter is taken as the "true" VTR that each method is
trying to recover.
"""

from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from new_TR_extraction_funs import extract_VTR, extract_VTR_derivative, extract_VTR_gm_gmid, style_axes
from TR_related_equations import TRParams, drain_current

script_dir = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Geometry / bias
# ---------------------------------------------------------------------------
L_um, W_um = 2.0, 1.0
MU_EFF = 20.0  # cm^2/V.s
W_OVER_L = W_um / L_um

VDS = 0.05
VS = 0.0
T = 300.0
ID_LIMIT = 1e-15  # model has no measurement noise, see extract_TR_from_model.py history

VG_MIN, VG_MAX, N_VG = -1.0, 5.0, 151
VGS = np.linspace(VG_MIN, VG_MAX, N_VG)

# SiO2 relative permittivity, used to turn an EOT sweep into Cgch = eps_ox/EOT.
EPS_0 = 8.8541878128e-14  # F/cm
EPS_OX_REL = 3.9


def cgch_from_eot(eot_nm):
    """Cgch [F/cm^2] for an EOT given in nm (Cgch = eps_ox/EOT)."""
    return EPS_OX_REL * EPS_0 / (eot_nm * 1e-7)


# ---------------------------------------------------------------------------
# Exhaustive parameter grid
# ---------------------------------------------------------------------------
VTR_VALS = np.linspace(0.0, 1.0, 5)         # V
PHI_TAIL_VALS = np.linspace(0.01, 0.1, 4)   # V (per model's phi_tail field)
EOT_VALS = np.linspace(0.5, 5.0, 4)         # nm
N_VALS = np.linspace(1.0, 1.6, 4)           # body factor

p_base = TRParams()

common_kwargs = dict(off_frac=1e-3, on_frac=0.2, window_length=5, npts_fit=5,
                      VOV_limit=4.0, ID_limit=ID_LIMIT)

all_combos = list(product(VTR_VALS, PHI_TAIL_VALS, EOT_VALS, N_VALS))

# Physical validity: the tail charge scale Cgch*Vtr should not exceed the
# free-charge scale C2D*phi_tail, else Q_tail would dominate Q_free entirely.
combos = [(Vtr, phi_tail, eot_nm, n_val) for Vtr, phi_tail, eot_nm, n_val in all_combos
          if cgch_from_eot(eot_nm) * Vtr <= p_base.C2D * phi_tail]
n_skipped = len(all_combos) - len(combos)

print(f"Sweeping {len(combos)} combinations of (Vtr, phi_tail, EOT, n) "
      f"({n_skipped} of {len(all_combos)} skipped for Cgch*Vtr > C2D*phi_tail)...")

rows = []
n_ok, n_fail = 0, 0
for Vtr, phi_tail, eot_nm, n_val in combos:
    pc = replace(p_base, Vtr=Vtr, phi_tail=phi_tail, Cgch=cgch_from_eot(eot_nm), n=n_val)

    try:
        ID = np.array([drain_current(VDS, VS, vg, pc, mu_eff=MU_EFF, W_over_L=W_OVER_L)
                       for vg in VGS])

        VTON, VTOFF, VTR, _det = extract_VTR(VGS, ID, VDS, T=T, return_details=True,
                                              **common_kwargs, correct_series_R=False)
        VTR_deriv, _det_deriv = extract_VTR_derivative(VGS, ID, ID_limit=ID_LIMIT, window_length=5)
        VTR_gm_gmid, det_gm_gmid = extract_VTR_gm_gmid(VGS, ID, ID_limit=ID_LIMIT, window_length=5)
        VT_gm_over_id = det_gm_gmid["VT_gm_over_id"]
        VTR_hybrid = VTON - VT_gm_over_id
    except Exception as exc:  # noqa: BLE001 -- degenerate combos are expected at grid extremes
        n_fail += 1
        rows.append({
            "Vtr_true": Vtr, "phi_tail": phi_tail, "EOT_nm": eot_nm, "n": n_val,
            "VTON": np.nan, "VTOFF": np.nan, "VTR": np.nan,
            "VTR_deriv": np.nan, "VTR_gm_gmid": np.nan, "VTR_hybrid": np.nan, "error": str(exc),
        })
        continue

    n_ok += 1
    rows.append({
        "Vtr_true": Vtr, "phi_tail": phi_tail, "EOT_nm": eot_nm, "n": n_val,
        "VTON": VTON, "VTOFF": VTOFF, "VTR": VTR,
        "VTR_deriv": VTR_deriv, "VTR_gm_gmid": VTR_gm_gmid, "VTR_hybrid": VTR_hybrid, "error": "",
    })

print(f"Done: {n_ok} succeeded, {n_fail} failed out of {len(combos)}.")

sweep_df = pd.DataFrame(rows)
csv_path = script_dir / "extract_TR_from_model_sweep.csv"
sweep_df.to_csv(csv_path, index=False)
print(f"Saved sweep results to {csv_path}")

# ---------------------------------------------------------------------------
# Extracted VTR vs. model (true) VTR, combining all sweeps, per method
# ---------------------------------------------------------------------------
METHODS = ["VTR", "VTR_deriv", "VTR_gm_gmid", "VTR_hybrid"]
METHOD_LABELS = {
    "VTR": "extract_VTR\n(raw $I_D$ extrapolation)",
    "VTR_deriv": "extract_VTR_derivative\n($g_m/I_D$ trough - $g_m$ centroid)",
    "VTR_gm_gmid": "extract_VTR_gm_gmid\n($g_m/I_D$ / $g_m$ 50% crossings)",
    "VTR_hybrid": "hybrid: $V_{TON}$ - $V_{T,g_m/I_D}$\n(extract_VTR on-side, extract_VTR_gm_gmid off-side)",
}
METHOD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#a04ec9"]

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(19.5, 5.2), sharex=True, sharey=True)

    x_true = sweep_df["Vtr_true"].to_numpy()
    lims = [min(0.0, x_true.min()) - 0.05, x_true.max() + 0.1]

    for ax, method, color in zip(axes, METHODS, METHOD_COLORS):
        y = sweep_df[method].to_numpy()
        finite = np.isfinite(x_true) & np.isfinite(y)
        x_f, y_f = x_true[finite], y[finite]

        r2 = np.corrcoef(x_f, y_f)[0, 1] ** 2 if len(x_f) > 1 else np.nan

        ax.scatter(x_f, y_f, s=14, facecolor=color, edgecolor="none", alpha=0.5, zorder=3)
        ax.plot(lims, lims, color="gray", lw=1, ls=":", zorder=1)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.set_xlabel("$V_{TR}$, model (V)", fontsize=12)
        ax.set_title(f"{METHOD_LABELS[method]}\n$R^2$ = {r2:.3f}  (n={len(x_f)})", fontsize=11)
        style_axes(ax)

    axes[0].set_ylabel("$V_{TR}$, extracted (V)", fontsize=12)
    fig.suptitle("Model $V_{TR}$ sweep: extracted vs. true, all methods "
                 f"({len(sweep_df)} combinations of $V_{{tr}}$, $\\phi_{{tail}}$, EOT, $n$)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    fpath = script_dir / "model_VTR_sweep_comparison.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"\nSaved sweep comparison scatter plot to {fpath}")
except ImportError:
    pass

print("\nR^2 summary (extracted vs. model VTR, all sweeps combined):")
for method in METHODS:
    y = sweep_df[method].to_numpy()
    finite = np.isfinite(x_true) & np.isfinite(y)
    if finite.sum() > 1:
        r2 = np.corrcoef(x_true[finite], y[finite])[0, 1] ** 2
    else:
        r2 = np.nan
    print(f"  {method:15s}: R^2 = {r2:.4f}  (n={int(finite.sum())})")
