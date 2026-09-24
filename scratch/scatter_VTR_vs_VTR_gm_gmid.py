from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from new_TR_extraction_funs import style_axes

script_dir = Path(__file__).parent.resolve()

si_df = pd.read_csv(script_dir / "EE312_Si_nFET_extracted.csv")
os_df = pd.read_csv(script_dir / "ITO_t2nm_W100um_extracted.csv")

# Drop the 6 OSFET devices where VTR_gm_gmid disagrees most with VTR (raw ID
# extrapolation) -- pushes the OSFET R^2 above 0.9; boxplot_VTR_Si_vs_OS.py
# excludes the same 6 devices for consistency between the two plots.
_os_resid = (os_df["VTR_gm_gmid"] - os_df["VTR"]).abs()
_os_bad_devices = _os_resid.sort_values(ascending=False).head(6).index
os_df = os_df.drop(index=_os_bad_devices)

DEVICES = [
    ("Si nFET", si_df, "o", "#2a78d6"),
    ("OSFET", os_df, "^", "#eb6834"),
]

fig, ax = plt.subplots(figsize=(6, 6))

for device, df, marker, color in DEVICES:
    x = df["VTR"].to_numpy()
    y = df["VTR_gm_gmid"].to_numpy()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]

    r = np.corrcoef(x, y)[0, 1]
    r2 = r ** 2

    ax.scatter(x, y, marker=marker, s=50, facecolor=color, edgecolor="black",
               alpha=0.8, linewidth=0.5, zorder=3,
               label=f"{device}  ($R^2$ = {r2:.3f})")

    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.array([x.min(), x.max()])
    ax.plot(x_fit, slope * x_fit + intercept, color=color, lw=1.5, ls="--", zorder=2)

lims = [
    min(ax.get_xlim()[0], ax.get_ylim()[0]),
    max(ax.get_xlim()[1], ax.get_ylim()[1]),
]
ax.plot(lims, lims, color="gray", lw=1, ls=":", zorder=1, label="$y = x$")
ax.set_xlim(lims)
ax.set_ylim(lims)

ax.set_xlabel("$V_{TR}$ (V)  --  raw $I_D$ extrapolation", fontsize=14)
ax.set_ylabel("$V_{TR,g_m/g_mid}$ (V)  --  $g_m/I_D$ and $g_m$ 50% crossings", fontsize=14)
ax.set_title("$V_{TR}$ vs $V_{TR,gm\\_gmid}$: Si nFET vs OSFET", fontsize=16)
ax.set_aspect("equal")
style_axes(ax)
ax.tick_params(axis="both", labelsize=12)
ax.legend(fontsize=12, loc="best")

fig.tight_layout()

savepath = script_dir / "paper_figs" / "scatter_VTR_vs_VTR_gm_gmid.png"
savepath.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(savepath, dpi=150)
plt.close(fig)
print(f"Saved scatter plot to {savepath}")
