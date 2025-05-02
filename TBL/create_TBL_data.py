'''
For this specific script, need to run from the WM_data_creation directory

python TBL/create_TBL_data.py

'''
import os
from scipy.io import loadmat
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import pandas as pd
from utils import *

# --- Set up paths and constants ---
sys.path.append('../')
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path(load_bfm_path=False)  # Ensure the BFM_PATH and subdirectories are in the system path
savedatapath = os.path.join(WM_DATA_PATH, 'data')
UP_FRAC = 0.20
DOWN_FRAC = 0.01
UP_FRAC_SEP = 0.05
DOWN_FRAC_SEP = 0.005

TBL_PATH = "/home/yuenongling/Codes/BL/TBLS"
fpath_format = TBL_PATH + "/data/postnpz_20250321/TBL_Retheta_670_theta_{angle}deg_medium_avg_slice.npz"
stats_BL = pkl.load(open(TBL_PATH + '/stats/20250321/stats_Re670_C_filtered_gauss.pkl', 'rb'))

# --- Retheta and some other setups ---
Retheta = 670
nu = nu_dict[Retheta]
angle_list = [-4, -3, -2, -1, 5, 10, 15, 20]
n_diff = 10

for i, angle in enumerate(angle_list):
    # Lists to collect data
    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []

    # Read in the boundary layer thickness
    delta = stats_BL[angle].Delta_edge
    Cf    = stats_BL[angle].Cf
    up    = stats_BL[angle].up
    dPedx    = stats_BL[angle].dPedx
    utau_x = np.sqrt(abs(Cf)/2)

    x, y, z, xm, ym, zm, Umean, V, Pmean, tau_x, up, xg, yg = read_npz_stats(fpath_format.format(angle=angle), nu, filter=True)

    Umean = (Umean[1:,1:-1] + Umean[:-1,1:-1]) / 2

    for idx, x_ in enumerate(xm):
        # NOTE: Remove inflow and outflow
        if angle < 0:
            if x_ < 1.4 or x_ > 5.5:
                continue
        else:
            if x_ < 1.4 or x_ > 7:
                continue
            if angle == 20:
                if (x_ > 5 and x_ < 6):
                    continue

        # Plot every n_diff point (donsampling)
        if not (idx % n_diff == 0 or abs(Cf[idx]) < 0.0005):
            # NOTE: To keep more points near separation
            continue

        if Cf[idx] < 0:
            idy_low = np.where(ym >= DOWN_FRAC_SEP*delta[idx])[0][0]
            idy_high = np.where(ym <= UP_FRAC_SEP*delta[idx])[0][-1]

            # Find the lowest U velo
            idx_min = np.argmin(Umean[idx, :])

            idy_high = np.min([idy_high, idx_min])
            idy_low = np.min([idy_low, idx_min//2])

            if idy_low == 0 or idy_high == 0:
                continue
            if idy_low >= idy_high:
                continue
            
        else:
            idy_low = np.where(y >= DOWN_FRAC*delta[idx])[0][0]
            idy_high = np.where(y <= UP_FRAC*delta[idx])[0][-1]

        utau = utau_x[idx]

        # NOTE: Local Reynolds number
        u_1 = Umean[idx, idy_low:idy_high]
        u_2 = find_k_y_values(ym[idy_low:idy_high], Umean[idx, :], ym, k=1)
        u_3 = find_k_y_values(ym[idy_low:idy_high], Umean[idx, :], ym, k=2)
        u_4 = find_k_y_values(ym[idy_low:idy_high], Umean[idx, :], ym, k=3)

        # NOTE: Find dU/dy at corresponding locations
        dUdy = np.gradient(Umean[idx,:], ym)
        dudy_1 = dUdy[idy_low:idy_high]
        dudy_2 = find_k_y_values(ym[idy_low:idy_high], dUdy, ym, k=1)
        dudy_3 = find_k_y_values(ym[idy_low:idy_high], dUdy, ym, k=2)
        
        pi_1 = Umean[idx, idy_low:idy_high] * ym[idy_low:idy_high] / nu
        pi_3 = u_2 * ym[idy_low:idy_high] / nu
        pi_4 = u_3 * ym[idy_low:idy_high] / nu
        pi_5 = u_4 * ym[idy_low:idy_high] / nu

        pi_6 = dudy_1 * ym[idy_low:idy_high]**2 / nu
        pi_7 = dudy_2 * ym[idy_low:idy_high]**2 / nu
        pi_8 = dudy_3 * ym[idy_low:idy_high]**2 / nu

        # Normal pressure "gradient"
        delta_p = (Pmean[idx+1, idy_low:idy_high] - Pmean[idx+1, 0]) /  ym[idy_low:idy_high]
        upn = np.sign(delta_p) * (abs(delta_p) * nu )**(1/3)
        pi_9 = upn * ym[idy_low:idy_high] / nu

        # NOTE: Local pressure-gradient based Reynolds number
        u_p = up[idx]
        pi_2 = u_p * ym[idy_low:idy_high] / nu

        # NOTE: Local friction velocity based Reynolds number
        pi_out = utau * ym[idy_low:idy_high] / nu

        inputs_dict = {
            'u1_y_over_nu': pi_1,  # U_i[bot_index] * y_i[bot_index] / nu_i,
            'up_y_over_nu': pi_2,
            'u2_y_over_nu': pi_3,
            'u3_y_over_nu': pi_4,
            'u4_y_over_nu': pi_5,
            'dudy1_y_pow2_over_nu': pi_6,
            'dudy2_y_pow2_over_nu': pi_7,
            'dudy3_y_pow2_over_nu': pi_8,
            'upn_y_over_nu': pi_9,
        }
        all_inputs_data.append(pd.DataFrame(inputs_dict))

        # --- Calculate Output Feature ---
        # Output is y+ (utau * y / nu)
        output_dict = {
            'utau_y_over_nu': pi_out
        }
        all_output_data.append(pd.DataFrame(output_dict))

        # --- Collect Unnormalized Inputs ---
        unnorm_dict = {
            'y': ym[idy_low:idy_high],
            'u1': u_1,
            'nu': nu * np.ones_like(ym[idy_low:idy_high]),
            'utau': np.full_like(ym[idy_low:idy_high], utau),
            'up': np.full_like(ym[idy_low:idy_high], u_p),
            'u2': u_2,
            'u3': u_3,
            'u4': u_4,
            'dudy1': dudy_1,
            'dudy2': dudy_2,
            'dudy3': dudy_3,
            'upn': upn,
        }
        all_unnormalized_inputs_data.append(pd.DataFrame(unnorm_dict))

        # --- Collect Flow Type Information ---
        # Using format: [case_name, reference_nu, x_coord, delta, edge_velocity]
        # For channel flow: x=0, delta=1 (half-channel height), Ue=0 (or U_bulk if needed)
        flow_type_dict = {
            'case_name': ['TBL'] * len(ym[idy_low:idy_high]),
            'nu': [nu] * len(ym[idy_low:idy_high]),
            'x': x_ * np.ones_like(ym[idy_low:idy_high]),
            'delta': delta[idx] * np.ones_like(ym[idy_low:idy_high]),
        }
        all_flow_type_data.append(pd.DataFrame(flow_type_dict))

# Concatenate data from all Re_num cases into single DataFrames
    inputs_df = pd.concat(all_inputs_data, ignore_index=True)
    output_df = pd.concat(all_output_data, ignore_index=True)
    flow_type_df = pd.concat(all_flow_type_data, ignore_index=True)
    unnormalized_inputs_df = pd.concat(all_unnormalized_inputs_data, ignore_index=True)

# Save DataFrames to HDF5 file
    output_filename = os.path.join(savedatapath, f'TBL_{angle}_data.h5')
    print(f"\nSaving data to HDF5 file: {output_filename}")
        # Use fixed format for better performance with numerical data
    inputs_df.to_hdf(output_filename, key='inputs', mode='w', format='fixed')
    output_df.to_hdf(output_filename, key='output', mode='a', format='fixed')
    unnormalized_inputs_df.to_hdf(output_filename, key='unnormalized_inputs', mode='a', format='fixed')
# Use table format for flow_type if it contains strings, to keep them
    flow_type_df.to_hdf(output_filename, key='flow_type', mode='a', format='table')
    print("Data successfully saved.")

# Print summary shapes
    print(f"Final Shapes:")
    print(f"  Inputs: {inputs_df.shape}")
    print(f"  Output: {output_df.shape}")
    print(f"  Flow Type: {flow_type_df.shape}")
    print(f"  Unnormalized Inputs: {unnormalized_inputs_df.shape}")

# --- Sanity Check ---
#     print("\n--- Sanity Check: Comparing HDF5 with Original Pickle ---")
#     with open(f'/home/yuenongling/Codes/BFM/WM_Opt/data/TBL_{angle}_data.pkl', 'rb') as f:
#         original_data = pkl.load(f)
#
# # Load corresponding data from HDF5
#     inputs_hdf = inputs_df[inputs_df.index.isin(np.arange(len(original_data['inputs'])))].values
#     output_hdf = output_df[output_df.index.isin(np.arange(len(original_data['output'])))].values.flatten()
#     flow_type_hdf = flow_type_df[flow_type_df.index.isin(np.arange(len(original_data['flow_type'])))].values
#     unnormalized_inputs_hdf = unnormalized_inputs_df[
#         unnormalized_inputs_df.index.isin(np.arange(len(original_data['unnormalized_inputs'])))].values
#
#     print(f"  Inputs match: {np.allclose(original_data['inputs'], inputs_hdf)}")
#     print(f"  Output match: {np.allclose(original_data['output'], output_hdf)}")
#     print(f"  Flow type match: {np.array_equal(original_data['flow_type'].astype(str), flow_type_hdf.astype(str))}")
#     print(
#         f"  Unnormalized inputs match: {np.allclose(original_data['unnormalized_inputs'].flatten(), unnormalized_inputs_hdf.flatten(), rtol=1e-5, atol=1e-4)}")
