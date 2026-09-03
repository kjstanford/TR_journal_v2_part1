"""
Generate Id-Vg data from the TR drain-current model (TR_related_equations.py)
for the three standard Vtr/phi_tail cases documented in scratch.MD, then run
both VTR extraction methods from new_TR_extraction_funs.py on that data:

  1. extract_VTR         -- subthreshold/above-threshold linear extrapolation
  2. extract_VTR_derivative -- gm/Id trough vs gm rising-edge centroid

Geometry/mobility per request: L = 2 um, W = 1 um, mu_eff = 20 cm^2/V.s.
No CV/Cgg data is generated here, so extract_VTR_derivative falls back to its
gm-only default (see compute_gmid_gmcgg's docstring).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from new_TR_extraction_funs import *
from TR_related_equations import TRParams, drain_current, make_cases

script_dir = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Geometry / bias
# ---------------------------------------------------------------------------
L_um, W_um = 2.0, 1.0
MU_EFF = 20.0  # cm^2/V.s
W_OVER_L = W_um / L_um

VDS = 0.05   # linear-mode bias (same convention as extract_VTR's docstring
             # and TR_related_equations.py main()'s plot_f_t/derivative_extrema)
VS = 0.0
T = 300.0
ID_LIMIT = 1e-13  # model has no measurement noise; this just keeps the very
                   # deepest (numerically negligible) subthreshold tail out
                   # of the SS/gm-Id search, same role as extract_VTR's ID_limit

VG_MIN, VG_MAX, N_VG = -0.25, 1.5, 201
VGS = np.linspace(VG_MIN, VG_MAX, N_VG)

# ---------------------------------------------------------------------------
# The three standard cases (scratch.MD Sec. 6 / TR_related_equations.py main()):
# phi_tail is irrelevant when Vtr = 0, so that case needs no tail label.
# ---------------------------------------------------------------------------
p = TRParams()
cases = make_cases(p, [
    ("Vtr0", {"Vtr": 0.0}),
    ("Vtr0.3_phitail0.045", {"Vtr": 0.3, "phi_tail": 0.045}),
    ("Vtr0.3_phitail0.09", {"Vtr": 0.3, "phi_tail": 0.09}),
])

common_kwargs = dict(off_frac=1e-3, on_frac=0.2, window_length=5, npts_fit=5,
                      VOV_limit=4.0, ID_limit=ID_LIMIT)

# dataviz-skill validated categorical palette, slots 1-3 (blue/orange/aqua)
CASE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]

extracted_list = []
idvg_curves = []

for label, pc, _style in cases:
    ID = np.array([drain_current(VDS, VS, vg, pc, mu_eff=MU_EFF, W_over_L=W_OVER_L)
                   for vg in VGS])
    idvg_curves.append((label, ID))

    print(f"\n=== {label} (Vt={pc.Vt:.2f} V, Vtr={pc.Vtr:.2f} V, "
          f"phi_tail={pc.phi_tail:.3f} V) ===")
    print(f"Id: {ID.min():.4e} to {ID.max():.4e} A over "
          f"Vg {VGS.min():.3f} to {VGS.max():.3f} V, Vd={VDS} V")

    # -- Method 1: extract_VTR (subthreshold/above-threshold extrapolation) --
    VTON, VTOFF, VTR, det = extract_VTR(VGS, ID, VDS, T=T, return_details=True,
                                         **common_kwargs, correct_series_R=False)
    print(f"extract_VTR           : VTON={VTON:.4f} V  VTOFF={VTOFF:.4f} V  "
          f"VTR={VTR:.4f} V  (method={det['method']})")

    try:
        import matplotlib.pyplot as plt
        fig = plot_extraction(VGS, ID, VTON, VTOFF, det,
                               title=f"{label}: extract_VTR")
        fpath = script_dir / f"model_{label}_{det['method']}.png"
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"    -> saved {fpath}")
    except ImportError:
        pass

    # -- Method 2: extract_VTR_derivative (gm/Id trough vs gm centroid) --
    VTR_deriv, det_deriv = extract_VTR_derivative(VGS, ID, ID_limit=ID_LIMIT,
                                                   window_length=5)
    print(f"extract_VTR_derivative: VTR_deriv={VTR_deriv:.4f} V  "
          f"(trough={det_deriv['Vg_trough']:.4f} V, "
          f"centroid={det_deriv['Vg_peak']:.4f} V, rising={det_deriv['rising_kind']})")

    try:
        plot_derivative_extrema(label, VGS, VTR_deriv, det_deriv,
                                 script_dir / f"model_{label}_VTR_derivative.png",
                                 f"{label}: VTR from derivatives")
        plot_gm_and_gmid(label, VGS, det_deriv,
                          script_dir / f"model_{label}_gm_and_gmid.png",
                          f"{label}: gm and gm/Id")
    except ImportError:
        pass

    extracted_list.append({
        "case": label, "Vtr": pc.Vtr, "phi_tail": pc.phi_tail,
        "VTON": VTON, "VTOFF": VTOFF, "VTR": VTR, "VTR_deriv": VTR_deriv,
    })

# ---------------------------------------------------------------------------
# Combined Id-Vg overview (semilog), one curve per case
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for (label, ID), color in zip(idvg_curves, CASE_COLORS):
        ax.semilogy(VGS, np.abs(ID), lw=2, color=color, label=label)
    ax.set_xlabel("$V_G$ (V)")
    ax.set_ylabel("$I_D$ (A)")
    ax.set_title(f"Model $I_D$ vs $V_G$  ($V_D$={VDS} V, L={L_um} $\\mu$m, "
                 f"W={W_um} $\\mu$m, $\\mu_{{eff}}$={MU_EFF:g} cm$^2$/V$\\cdot$s)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fpath = script_dir / "model_IdVg_3case.png"
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    print(f"\nSaved combined Id-Vg plot to {fpath}")
except ImportError:
    pass

extracted_df = pd.DataFrame(extracted_list)
extracted_df.to_csv(script_dir / "extract_TR_from_model_extracted.csv", index=False)

print("\nExtracted VTR summary (both methods):")
print(extracted_df.to_string(index=False))
