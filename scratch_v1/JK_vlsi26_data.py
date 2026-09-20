from pathlib import Path
import numpy as np
import pandas as pd
import os
import re
from new_TR_extraction_funs import *

script_dir = Path(__file__).parent.resolve()
main_data_dir = script_dir / "paper_data" / "JK_vlsi26_data"

extracted_list = []
T = 300  # Temperature in Kelvin
data_dict = {}

def get_L_Lc_from_filename(filename):
    # Use regex to extract L and Lc from the filename (order-independent)
    L_match = re.search(r'(?<!L)L(\d+)nm', filename)
    Lc_match = re.search(r'Lc(\d+)nm', filename)
    if L_match and Lc_match:
        L = float(L_match.group(1))
        Lc = float(Lc_match.group(1))
        return L, Lc
    else:
        raise ValueError(f"Filename {filename} does not contain L and Lc information.")

sample_names = ["Ni_O_400C_10mins_run_Jan13_2026", "Ni_run_Jan12_2026"]
for sample in sample_names:
    data_dir = main_data_dir / sample
    for file in data_dir.iterdir():
        if file.is_file() and file.suffix == ".csv" and file.name.__contains__("IdVg_main_"):
            df = pd.read_csv(file)
            # print("Processing file:", file.name, "in", sample, "with shape:", df.shape, "and columns:", df.columns.tolist())
            VDS_list = df['Drain_Voltage'].unique().tolist()
            cycle_list = df['Cycle'].unique().tolist()
            L, Lc = get_L_Lc_from_filename(file.name)
            for VDS in VDS_list:
                for cycle in cycle_list:
                    if VDS > 0.1 or L < 700 or Lc < 500:
                        # print(f"Skipping VDS = {VDS} V, Cycle = {cycle}, L = {L} nm, Lc = {Lc} nm")
                        continue
                    print(f"Processing Drain Voltage: {VDS} V and Cycle: {cycle}")
                    mask = (df['Drain_Voltage'] == VDS) & (df['Cycle'] == cycle)
                    VGS_full = df.loc[mask, 'Gate_Voltage'].to_numpy()
                    ID_full = df.loc[mask, 'Drain_Current'].to_numpy()
                    # print(f"Loaded {len(VGS_full)} points for VDS = {VDS} V and Cycle = {cycle}")
                    VGS, ID = get_sweep(VGS_full, ID_full, direction='forward')
                    print(f"Loaded {len(VGS)} forward-sweep points for VDS = {VDS} V and Cycle = {cycle}")
                    print(f"VGS: {VGS.min():.3f} to {VGS.max():.3f} V   VDS = {VDS:.4f} V\n")

                    common_kwargs = dict(off_frac=1e-3, on_frac=0.2, window_length=5, npts_fit=5, VOV_limit=4.0, ID_limit=2e-12)
                    
                    label, kwargs = "No R correction (raw ID)", dict(correct_series_R=False)
                    VTON, VTOFF, VTR, det = extract_VTR(VGS, ID, VDS, T=T, return_details=True, **common_kwargs, **kwargs)
                    print(f"{label:35s}: VTON={VTON:.3f} V  VTOFF={VTOFF:.3f} V  "
                            f"VTR={VTR:.3f} V  (method={det['method']}, "
                            f"R_tot_est={det['Rtot_used']})")

                    extracted_list.append(
                        {
                            "file_name": file.name,
                            "L": L,
                            "Lc": Lc,
                            "sample": sample,
                            "Cycle": cycle,
                            "VDS": VDS,
                            "VTON": VTON,
                            "VTOFF": VTOFF,
                            "VTR": VTR,
                        }
                    )


extracted_df = pd.DataFrame(extracted_list)
extracted_df.to_csv(main_data_dir / "JK_vlsi26_extracted.csv", index=False)

