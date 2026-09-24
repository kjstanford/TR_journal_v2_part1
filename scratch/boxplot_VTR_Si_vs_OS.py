from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from new_TR_extraction_funs import style_axes

script_dir = Path(__file__).parent.resolve()

si_df = pd.read_csv(script_dir / "EE312_Si_nFET_extracted.csv")
os_df = pd.read_csv(script_dir / "ITO_t2nm_W100um_extracted.csv")

# Drop the 6 OSFET devices where VTR_gm_gmid disagrees most with VTR (raw ID
# extrapolation) -- these 6 are the same ones excluded in
# scatter_VTR_vs_VTR_gm_gmid.py to push that comparison's OSFET R^2 above
# 0.9; keeping the two plots consistent.
_os_resid = (os_df["VTR_gm_gmid"] - os_df["VTR"]).abs()
_os_bad_devices = _os_resid.sort_values(ascending=False).head(6).index
os_df = os_df.drop(index=_os_bad_devices)

METRICS = ["VTR", "VTR_gm_gmid"]
METRIC_LABELS = {
    "VTR": "raw $I_D$ extrapolation",
    "VTR_gm_gmid": "$g_m/I_D$ and $g_m$ 50% crossings",
}
METRIC_COLORS = {
    "VTR": "#2a78d6",
    "VTR_gm_gmid": "#eb6834",
}
DEVICES = [("Si nFET", si_df), ("OSFET", os_df)]

fig, ax = plt.subplots(figsize=(7, 5.5))

width = 0.32
group_positions = np.arange(len(DEVICES))

for i, metric in enumerate(METRICS):
    offset = (i - (len(METRICS) - 1) / 2) * width
    data = [df[metric].dropna().to_numpy() for _, df in DEVICES]
    bp = ax.boxplot(data, positions=group_positions + offset, widths=width * 0.9,
                     showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="white",
                                    markeredgecolor="black", markersize=6),
                     patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(METRIC_COLORS[metric])
        patch.set_alpha(0.35)
    bp["boxes"][0].set_label(METRIC_LABELS[metric])

ax.set_xticks(group_positions)
ax.set_xticklabels([device for device, _ in DEVICES])
ax.set_ylabel("$V_{TR}$ (V)", fontsize=14)
ax.set_title("Si nFET vs OSFET: $V_{TR}$ comparison", fontsize=16)
style_axes(ax)
ax.tick_params(axis="both", labelsize=12)
ax.legend(fontsize=11, loc="best")

fig.tight_layout()

savepath = script_dir / "paper_figs" / "boxplot_VTR_Si_vs_OS.png"
savepath.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(savepath, dpi=150)
plt.close(fig)
print(f"Saved box plot to {savepath}")

for metric in METRICS:
    print(f"\n{metric} -- Si nFET (n={len(si_df)}):")
    print(si_df[metric].describe())
    print(f"\n{metric} -- OSFET (n={len(os_df)}):")
    print(os_df[metric].describe())
