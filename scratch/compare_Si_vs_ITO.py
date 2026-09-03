from pathlib import Path
import numpy as np
import pandas as pd
from new_TR_extraction_funs import *

script_dir = Path(__file__).parent.resolve()
data_dir = script_dir / "paper_data" / "compare_Si_vs_ITO"

T = 300  # Temperature in Kelvin
common_kwargs = dict(off_frac=1e-3, on_frac=0.2, window_length=5, npts_fit=5, VOV_limit=4.0, ID_limit=2e-12)

extracted_list = []


def extract_and_report(device, file_name, VGS_full, ID_full, VDS_full, label_suffix=""):
    VDS = VDS_full[0]  # Assuming VDS is constant for the sweep
    VGS, ID = get_sweep(VGS_full, ID_full, direction='forward')

    print(f"Loaded {len(VGS)} forward-sweep points from {file_name}{label_suffix}")
    print(f"VGS: {VGS.min():.3f} to {VGS.max():.3f} V   VDS = {VDS:.4f} V\n")

    label = f"{device}{label_suffix} (No R correction, raw ID)"
    VTON, VTOFF, VTR, det = extract_VTR(VGS, ID, VDS, T=T,
                                         return_details=True, **common_kwargs, correct_series_R=False)
    print(f"{label:45s}: VTON={VTON:.3f} V  VTOFF={VTOFF:.3f} V  "
          f"VTR={VTR:.3f} V  (method={det['method']}, "
          f"R_tot_est={det['Rtot_used']})")

    try:
        import matplotlib.pyplot as plt
        fig = plot_extraction(VGS, ID, VTON, VTOFF, det, title=label)
        safe_suffix = label_suffix.replace(" ", "_")
        fname = f"{Path(file_name).stem}{safe_suffix}_{det['method']}.png"
        fpath = data_dir / fname
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"    -> saved plot to {fpath}")
    except ImportError:
        pass

    extracted_list.append(
        {
            "device": device,
            "file_name": f"{file_name}{label_suffix}",
            "VDS": VDS,
            "VTON": VTON,
            "VTOFF": VTOFF,
            "VTR": VTR,
        }
    )


# ---------------------------------------------------------------------------
# Si nFET
# ---------------------------------------------------------------------------
si_file = data_dir / "nFET_IdVg_Vdpar [dev5_100_100_pre(1) ; 3_30_2025 3_57_31 PM].csv"
df_list, N1 = agilent_csv_cleaner(si_file)
print(f"Processed {si_file.name} into {len(df_list)} dataset{('s' if len(df_list) != 1 else '')}.")

df = df_list[0]
VGS_full = df[' Vg'].to_numpy()
ID_full = df[' absId'].to_numpy()
VDS_full = df[' Vd'].to_numpy()
extract_and_report("Si nFET", si_file.name, VGS_full, ID_full, VDS_full)

# ---------------------------------------------------------------------------
# OSFET
# ---------------------------------------------------------------------------
os_file = data_dir / "! FET3ttest [post_D_6_0(2) ; 6_25_2026 4_40_40 PM].csv"
df_list, N1 = agilent_csv_cleaner(os_file)
print(f"Processed {os_file.name} into {len(df_list)} dataset{('s' if len(df_list) != 1 else '')}.")

# Each measured block sweeps Vg at two Vd levels (0.05 V, 1.5 V); VTR
# extraction requires the linear-mode (low VDS) sweep, so only the
# low-VDS dataset from each block is used (df_list[0::2]).
for block_idx, df in enumerate(df_list[0::2]):
    VGS_full = df[' Vg'].to_numpy()
    ID_full = df[' absId'].to_numpy()
    VDS_full = df[' Vd'].to_numpy()
    extract_and_report("OSFET", os_file.name, VGS_full, ID_full, VDS_full,
                        label_suffix=f" block{block_idx}")

extracted_df = pd.DataFrame(extracted_list)
extracted_df.to_csv(script_dir / "compare_Si_vs_ITO_extracted.csv", index=False)

print("\nExtracted VTR summary:")
print(extracted_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Cgg vs Vg (CV data)
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

# style_axes, freq_label, SEQ_BLUE_4, compute_gmid_gmcgg, plot_gmid_gmcgg,
# extract_VTR_derivative, plot_derivative_extrema, COLOR_GM_ID, COLOR_GM_CGG
# all come from new_TR_extraction_funs (imported via `import *` above) --
# kept there so other scripts can reuse them without duplicating this file.

# --- Si nFET: Agilent-format CV sweep, 4 freqs x (fwd+bwd) x 121 pts, all in
#     a single Dimension2=1 block -> slice it manually. ---
cv_si_file = data_dir / "CVSweep_nFreq_Kasidit_2 [dev8_10_5_CgsdVg(1) ; 3_31_2025 2_49_02 PM].csv"
cv_df_list, _ = agilent_csv_cleaner(cv_si_file)
cv_si_df = cv_df_list[0]
Vg_si_all = cv_si_df[' Vsweep'].to_numpy()
Cgg_si_all = np.abs(cv_si_df[' Cdata'].to_numpy())

si_freqs = [1_000_000, 500_000, 100_000, 50_000]  # Hz; Freq1..Freq4 (TestParameter.Value)
npts_per_dir = 121

fig, ax = plt.subplots(figsize=(6, 4.5))
for i, (freq, color) in enumerate(zip(si_freqs, SEQ_BLUE_4)):
    fwd = slice(i * 2 * npts_per_dir, i * 2 * npts_per_dir + npts_per_dir)
    ax.plot(Vg_si_all[fwd], Cgg_si_all[fwd], lw=2, color=color, label=freq_label(freq))
ax.set_xlabel("$V_G$ (V)")
ax.set_ylabel("$C_{GG}$ (F)")
ax.set_title("Si nFET: $|C_{GG}|$ vs $V_G$ (forward sweep)")
style_axes(ax)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
si_cv_path = data_dir / "Si_nFET_Cgg_vs_Vg.png"
fig.savefig(si_cv_path, dpi=150)
plt.close(fig)
print(f"\nSaved Si nFET Cgg-Vg plot to {si_cv_path}")

# --- OSFET: plain CSV, 6 log-spaced freqs, explicit fwd/bwd column. Plot a
#     representative subset of 4 (lowest, two mid, highest) to keep the
#     sequential ramp legible -- see SEQ_BLUE_4. ---
cv_os_file = data_dir / "row_0_col_0_20260625_163657_cvf_data.csv"
cv_os_df = pd.read_csv(cv_os_file)
os_freqs_all = sorted(cv_os_df["frequency"].unique())
os_freqs = [os_freqs_all[i] for i in (0, 2, 4, 5)]

fig, ax = plt.subplots(figsize=(6, 4.5))
for freq, color in zip(os_freqs, SEQ_BLUE_4):
    sub = cv_os_df[(cv_os_df["frequency"] == freq) & (cv_os_df["direction"] == "fwd")].sort_values("voltage")
    ax.plot(sub["voltage"], sub["capacitance"].abs(), lw=2, color=color, label=freq_label(freq))
ax.set_xlabel("$V_G$ (V)")
ax.set_ylabel("$C_{GG}$ (F)")
ax.set_title("OSFET: $|C_{GG}|$ vs $V_G$ (forward sweep)")
style_axes(ax)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
os_cv_path = data_dir / "OSFET_Cgg_vs_Vg.png"
fig.savefig(os_cv_path, dpi=150)
plt.close(fig)
print(f"Saved OSFET Cgg-Vg plot to {os_cv_path}")

# ---------------------------------------------------------------------------
# gm/Id and gm/Cgg vs Vg, twin axis (one figure per device)
# ---------------------------------------------------------------------------
# compute_gmid_gmcgg / plot_gmid_gmcgg live in new_TR_extraction_funs.py.

# --- Si nFET: Id-Vg at Vd=0.1 V; Cgg at 1 MHz (forward, first CV segment). ---
si_df_list, _ = agilent_csv_cleaner(si_file)
si_id_df = si_df_list[0]
si_VGS, si_ID = get_sweep(si_id_df[' Vg'].to_numpy(), si_id_df[' absId'].to_numpy(), direction='forward')
si_Vg_cv = Vg_si_all[0:npts_per_dir]     # 1 MHz forward segment
si_Cgg_cv = Cgg_si_all[0:npts_per_dir]

plot_gmid_gmcgg("Si nFET", si_VGS, si_ID,
                 data_dir / "Si_nFET_gmId_gmCgg_vs_Vg.png",
                 "Si nFET: $g_m/I_D$ and $g_m/C_{GG}$ vs $V_G$  ($V_D$ = 0.1 V, $C_{GG}$ @ 1 MHz)",
                 Vg_cv=si_Vg_cv, Cgg_cv=si_Cgg_cv)

# --- OSFET: Id-Vg at Vd=0.05 V (block0); Cgg at 1 MHz (forward). ---
os_df_list, _ = agilent_csv_cleaner(os_file)
os_id_df = os_df_list[0]  # block0, Vd = 0.05 V
os_VGS, os_ID = get_sweep(os_id_df[' Vg'].to_numpy(), os_id_df[' absId'].to_numpy(), direction='forward')
os_cv_sub = cv_os_df[(cv_os_df["frequency"] == os_freqs_all[-1]) & (cv_os_df["direction"] == "fwd")].sort_values("voltage")
os_Vg_cv = os_cv_sub["voltage"].to_numpy()
os_Cgg_cv = os_cv_sub["capacitance"].abs().to_numpy()

plot_gmid_gmcgg("OSFET", os_VGS, os_ID,
                 data_dir / "OSFET_gmId_gmCgg_vs_Vg.png",
                 "OSFET: $g_m/I_D$ and $g_m/C_{GG}$ vs $V_G$  ($V_D$ = 0.05 V, $C_{GG}$ @ 1 MHz)",
                 Vg_cv=os_Vg_cv, Cgg_cv=os_Cgg_cv)

# ---------------------------------------------------------------------------
# VTR from derivative extrema of gm/Id and gm/Cgg
# ---------------------------------------------------------------------------
# extract_VTR_derivative / plot_derivative_extrema live in
# new_TR_extraction_funs.py (see their docstrings/comments there for the
# gm/Id-trough vs gm/Cgg-rising-edge-centroid method and why).
VTR_deriv_list = []

si_VTR_deriv, si_det = extract_VTR_derivative(si_VGS, si_ID, si_Vg_cv, si_Cgg_cv)
plot_derivative_extrema("Si nFET", si_VGS, si_VTR_deriv, si_det,
                          data_dir / "Si_nFET_VTR_derivative.png",
                          "Si nFET: $d(g_m/I_D)/dV_G$ and $d(g_m/C_{GG})/dV_G$")
plot_gm_and_gmid("Si nFET", si_VGS, si_det,
                  data_dir / "Si_nFET_gm_and_gmid.png",
                  "Si nFET: $g_m$ and $g_m/I_D$ vs $V_G$")
VTR_deriv_list.append({"device": "Si nFET", "VTR_deriv": si_VTR_deriv,
                        "Vg_trough_gmid": si_det["Vg_trough"], "Vg_centroid_gmcgg": si_det["Vg_peak"]})

os_VTR_deriv, os_det = extract_VTR_derivative(os_VGS, os_ID, os_Vg_cv, os_Cgg_cv)
plot_derivative_extrema("OSFET", os_VGS, os_VTR_deriv, os_det,
                          data_dir / "OSFET_VTR_derivative.png",
                          "OSFET: $d(g_m/I_D)/dV_G$ and $d(g_m/C_{GG})/dV_G$")
plot_gm_and_gmid("OSFET", os_VGS, os_det,
                  data_dir / "OSFET_gm_and_gmid.png",
                  "OSFET: $g_m$ and $g_m/I_D$ vs $V_G$")
VTR_deriv_list.append({"device": "OSFET", "VTR_deriv": os_VTR_deriv,
                        "Vg_trough_gmid": os_det["Vg_trough"], "Vg_centroid_gmcgg": os_det["Vg_peak"]})

VTR_deriv_df = pd.DataFrame(VTR_deriv_list)
VTR_deriv_df.to_csv(script_dir / "compare_Si_vs_ITO_VTR_derivative.csv", index=False)

print("\nVTR from derivative extrema (gm/Cgg rising-edge centroid - gm/Id trough):")
print(VTR_deriv_df.to_string(index=False))
print("\nFor comparison, VTR from extract_VTR (raw ID, subthreshold/above-threshold extrapolation):")
print(extracted_df[["device", "file_name", "VTR"]].to_string(index=False))
