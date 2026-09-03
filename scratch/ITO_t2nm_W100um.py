from pathlib import Path
import numpy as np
import pandas as pd
import os
import re
from new_TR_extraction_funs import *

script_dir = Path(__file__).parent.resolve()

compiled_csv_path = script_dir / "ITO_t2nm_L2um_W100um_compiled.csv"

if not compiled_csv_path.exists():
    print(f"Compiled CSV file {compiled_csv_path} does not exist. Creating it now.")

    baseline_dir = script_dir / "paper_data" / "ITO_t2nm_L25um_W100um"

    baseline_main_csvs = [f for f in os.listdir(baseline_dir) if f.endswith(".csv") and f.startswith("IdVg_main_")]
    print(f"Found {len(baseline_main_csvs)} baseline main CSV files.") 

    data_dict = {}
    for csv_f in baseline_main_csvs:
        m = re.search(r"DieR_(?P<DieR>[^_]+)_DieC_(?P<DieC>[^_]+).*_ID_(?P<ID>[^_]+)\.csv$", csv_f)

        if not m:
            print(f"Filename {csv_f} does not match the expected pattern.")
            continue
        DieR = m.group("DieR")
        DieC = m.group("DieC")
        ID = ord(m.group("ID")) - ord('A')

        df = pd.read_csv(baseline_dir / csv_f)

        col_prefix = f"{DieC}_{DieR}_{ID}"

        vg_mask = ( df["Drain_Voltage"] == 0.05 ) & ( df["Cycle"] == 1 )
        vd_list = df["Drain_Voltage"].unique()
        cycle_list = df["Cycle"].unique()
        data_dict[f"{col_prefix}_Vg"] =  df["Gate_Voltage"][vg_mask].to_numpy()
        
        for vd in vd_list:
            for cycle in cycle_list:
                mask = ( df["Drain_Voltage"] == vd ) & ( df["Cycle"] == cycle )
                data_dict[f"{col_prefix}_Id_{vd}_{cycle}"] = df["Drain_Current"][mask].to_numpy()

    compiled_df = pd.DataFrame({key: pd.Series(value) for key, value in data_dict.items()})

    compiled_csv_path.parent.mkdir(parents=True, exist_ok=True)
    compiled_df.to_csv(compiled_csv_path, index=False)


compiled_df = pd.read_csv(compiled_csv_path)

VDS = 0.05
T = 300.0
VTR_list = []
VTR_deriv_list = []

for col_idx, col_name in enumerate(compiled_df.columns):
    # ["1_1_1", "1_1_2", "1_0_2"] for L = 2um
    # ["2_1_2"] for L = 5um
    # [] for L = 25um
    if not col_name.endswith("_Vg") or col_name.strip('_Vg') in ["2_1_2"]:
        continue

    vg_values = compiled_df[col_name].dropna().to_numpy()
    print(f"Column {col_name} (index {col_idx}) has {len(vg_values)} Vg values")
    id_values = compiled_df.iloc[:, col_idx + 2].dropna().to_numpy()
    print(f"Column {compiled_df.columns[col_idx + 2]} (index {col_idx + 2}) has {len(id_values)} Id values")

    VGS, ID = get_sweep(vg_values, id_values, direction="forward")

    print(f"Loaded {len(VGS)} VGS values and {len(ID)} ID values for device ID: {col_name.strip('_Vg')}")
    print(f"VGS: {VGS.min():.3f} to {VGS.max():.3f} V\n")

    common_kwargs = dict(off_frac=1e-3, on_frac=0.2, window_length=5, npts_fit=5, VOV_limit=4.0, ID_limit=2e-12)

    label, kwargs = "No R correction (raw ID)", dict(correct_series_R=False)
    VTON, VTOFF, VTR, det = extract_VTR(VGS, ID, VDS, T=T, return_details=True, **common_kwargs, **kwargs)
    VTR_list.append(VTR)
    print(f"{label:35s}: VTON={VTON:.3f} V  VTOFF={VTOFF:.3f} V  "
            f"VTR={VTR:.3f} V  (method={det['method']}, "
            f"R_tot_est={det['Rtot_used']})")

    try:
        import matplotlib.pyplot as plt
        fig = plot_extraction(VGS, ID, VTON, VTOFF, det, title=label)
        fname = f"{col_name.strip('_Vg')}_{det['method']}.png"
        fpath = script_dir / "paper_data" / "dmp_plots" / fname
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"    -> saved plot to {fpath}\n")
    except ImportError:
        pass

    # Alternative VTR: Vg-separation between the trough of d(gm/Id)/dVg and
    # the rising-edge centroid of d(gm/Cgg)/dVg -- here gm/Cgg falls back to
    # plain gm (no CV/Cgg data for this device set), see
    # extract_VTR_derivative's docstring in new_TR_extraction_funs.py.
    VTR_deriv, det_deriv = extract_VTR_derivative(VGS, ID, ID_limit=2e-12, window_length=5)
    VTR_deriv_list.append(VTR_deriv)
    print(f"{'VTR from gm/Id, gm derivatives':35s}: VTR_deriv={VTR_deriv:.3f} V  "
          f"(trough={det_deriv['Vg_trough']:.3f} V, "
          f"centroid={det_deriv['Vg_peak']:.3f} V, rising={det_deriv['rising_kind']})")

    try:
        deriv_fname = f"{col_name.strip('_Vg')}_VTR_derivative.png"
        deriv_fpath = script_dir / "paper_data" / "dmp_plots" / deriv_fname
        deriv_fpath.parent.mkdir(parents=True, exist_ok=True)
        plot_derivative_extrema(col_name.strip('_Vg'), VGS, VTR_deriv, det_deriv,
                                 deriv_fpath, f"{col_name.strip('_Vg')}: VTR from derivatives")
        print(f"    -> saved plot to {deriv_fpath}\n")

        gmid_fname = f"{col_name.strip('_Vg')}_gm_and_gmid.png"
        gmid_fpath = script_dir / "paper_data" / "dmp_plots" / gmid_fname
        plot_gm_and_gmid(col_name.strip('_Vg'), VGS, det_deriv,
                          gmid_fpath, f"{col_name.strip('_Vg')}: gm and gm/Id")
        print(f"    -> saved plot to {gmid_fpath}\n")
    except ImportError:
        pass

print(f"\nExtracted VTR values for {len(VTR_list)} devices: {[float(f'{VTR:.4f}') for VTR in VTR_list]}")
# for idx, VTR in enumerate(VTR_list):
#     print(f"Device {idx + 1}: VTR = {VTR:.3f} V")
stats = pd.Series(VTR_list).describe()
print(stats)

print(f"\nExtracted VTR_deriv values for {len(VTR_deriv_list)} devices: "
      f"{[float(f'{v:.4f}') for v in VTR_deriv_list]}")
stats_deriv = pd.Series(VTR_deriv_list).describe()
print(stats_deriv)

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(VTR_list, bins=20, edgecolor="black")
    ax.set_xlabel("VTR (V)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of VTR values")
    hist_path = fpath.parent / "VTR_histogram.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"Saved VTR histogram to {hist_path}")
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(VTR_deriv_list, bins=20, edgecolor="black")
    ax.set_xlabel("VTR_deriv (V)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of VTR_deriv values")
    hist_path_deriv = fpath.parent / "VTR_deriv_histogram.png"
    fig.savefig(hist_path_deriv, dpi=150)
    plt.close(fig)
    print(f"Saved VTR_deriv histogram to {hist_path_deriv}")
except ImportError:
    pass
