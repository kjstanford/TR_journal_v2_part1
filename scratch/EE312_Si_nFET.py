from pathlib import Path
import numpy as np
import pandas as pd
import os
import re
from new_TR_extraction_funs import *

script_dir = Path(__file__).parent.resolve()
data_dir = script_dir / "paper_data" / "EE312_Si_nFET"
# compiled_csv_path = script_dir / "EE312_Si_nFET_compiled.csv"

extracted_list = []
T = 300  # Temperature in Kelvin
data_dict = {}
for file in data_dir.iterdir():
    if file.is_file() and file.suffix == ".csv" and file.name.__contains__("nFET_IdVg_Vdpar"):
        df_list, N1 = agilent_csv_cleaner(file)
        print(f"Processed {file.name} into {len(df_list)} dataset{('s' if len(df_list) != 1 else '')}.")
        df = df_list[0]
        VGS_full = df[' Vg'].to_numpy()
        ID_full = df[' absId'].to_numpy()
        VDS_full = df[' Vd'].to_numpy()
        # data_dict[f"{file.name}_VGS"] = VGS_full
        # data_dict[f"{file.name}_ID"] = ID_full
        VDS = VDS_full[0]  # Assuming VDS is constant for the sweep
        VGS, ID = get_sweep(VGS_full, ID_full, direction='forward')

        print(f"Loaded {len(VGS)} forward-sweep points from {file.name}")
        print(f"VGS: {VGS.min():.3f} to {VGS.max():.3f} V   VDS = {VDS:.4f} V\n")
    
        common_kwargs = dict(off_frac=1e-3, on_frac=0.2, window_length=5, npts_fit=5, VOV_limit=4.0, ID_limit=2e-12)

        label, kwargs = "No R correction (raw ID)", dict(correct_series_R=False)
        VTON, VTOFF, VTR, det = extract_VTR(VGS, ID, VDS, T=T,
                                                        return_details=True, **common_kwargs, **kwargs)
        print(f"{label:35s}: VTON={VTON:.3f} V  VTOFF={VTOFF:.3f} V  "
                f"VTR={VTR:.3f} V  (method={det['method']}, "
                f"R_tot_est={det['Rtot_used']})")

        try:
            import matplotlib.pyplot as plt
            fig = plot_extraction(VGS, ID, VTON, VTOFF, det, title=label)
            fname = f"{file.stem}_{det['method']}.png"
            fpath = data_dir / fname
            fig.savefig(fpath, dpi=150)
            plt.close(fig)
            print(f"    -> saved plot to {fpath}")
        except ImportError:
            pass

        # Alternative VTR: Vg-separation between the trough of d(gm/Id)/dVg
        # and the rising-edge centroid of d(gm/Cgg)/dVg -- here gm/Cgg falls
        # back to plain gm (no CV/Cgg data for this device set), see
        # extract_VTR_derivative's docstring in new_TR_extraction_funs.py.
        VTR_deriv, det_deriv = extract_VTR_derivative(VGS, ID, ID_limit=2e-12,
                                                       window_length=5)
        print(f"{'VTR from gm/Id, gm derivatives':35s}: VTR_deriv={VTR_deriv:.3f} V  "
              f"(trough={det_deriv['Vg_trough']:.3f} V, "
              f"centroid={det_deriv['Vg_peak']:.3f} V, rising={det_deriv['rising_kind']})")

        try:
            plot_derivative_extrema(file.stem, VGS, VTR_deriv, det_deriv,
                                     data_dir / f"{file.stem}_VTR_derivative.png",
                                     f"{file.stem}: VTR from derivatives")
            plot_gm_and_gmid(file.stem, VGS, det_deriv,
                              data_dir / f"{file.stem}_gm_and_gmid.png",
                              f"{file.stem}: gm and gm/Id")
        except ImportError:
            pass

        extracted_list.append(
            {
                "file_name": file.name,
                "VDS": VDS,
                "VTON": VTON,
                "VTOFF": VTOFF,
                "VTR": VTR,
                "VTR_deriv": VTR_deriv,
            }
        )

# compiled_df = pd.DataFrame({key: pd.Series(value) for key, value in data_dict.items()})
# compiled_df.to_csv(compiled_csv_path, index=False)

extracted_df = pd.DataFrame(extracted_list)
extracted_df.to_csv(script_dir / "EE312_Si_nFET_extracted.csv", index=False)

print("Statistics of extracted VTR values:")
vtr_stats = extracted_df['VTR'].describe()
print(vtr_stats)

print("\nStatistics of extracted VTR_deriv values (gm/Id trough - gm centroid):")
vtr_deriv_stats = extracted_df['VTR_deriv'].describe()
print(vtr_deriv_stats)

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(extracted_df['VTR'], bins=20, edgecolor="black")
    ax.set_xlabel("VTR (V)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of VTR values")
    hist_path = data_dir / "VTR_histogram.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"Saved VTR histogram to {hist_path}")
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(extracted_df['VTR_deriv'], bins=20, edgecolor="black")
    ax.set_xlabel("VTR_deriv (V)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of VTR_deriv values")
    hist_path_deriv = data_dir / "VTR_deriv_histogram.png"
    fig.savefig(hist_path_deriv, dpi=150)
    plt.close(fig)
    print(f"Saved VTR_deriv histogram to {hist_path_deriv}")
except ImportError:
    pass