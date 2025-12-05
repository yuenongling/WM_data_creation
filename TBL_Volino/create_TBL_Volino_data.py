import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import os
from scipy import interpolate
import scipy.io
import re # Using re for a slightly more robust extraction

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
casename = 'TBL_Volino'
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
savedatapath = os.path.join(WM_DATA_PATH, 'data')
currentpath = os.path.join(WM_DATA_PATH, casename)
statspath = os.path.join(currentpath, 'stats')

# --- Hardcoded some data ---
rho = 1
UP_FRAC = 0.25
DOWN_FRAC = 0.000
NUM_STATIONS = 12  # Number of stations per case
cases = np.arange(1, 9)

# --- Hardcoded beta from the paper ---
beta = {}
beta[1] = [-0.91, -0.82, -0.79, -0.72, -0.74, -0.72, 0, 0, 0, 2.47, 3.84, 5.97]
beta[2] = [-1.15, -1.15, -0.98, -0.89, -0.84, -0.69, 0, 0, 0, 2.65, 4.09, 6.60]
beta[3] = [-0.57, -0.61, -0.56, -0.56, -0.56, -0.54, 0, 0, 0, 0.76, 0.95, 1.09]
beta[4] = [-0.58, -0.65, -0.65, -0.61, -0.60, -0.58, 0, 0, 0, 0.79, 0.92, 1.02]
beta[5] = [-0.65, -0.63, -0.57, -0.54, -0.54, -0.54, 0, 0, 0, 0.77, 0.92, 1.11]
beta[6] = [-0.24, -0.24, -0.27, -0.28, -0.29, -0.32, 0, 0, 0, 0.38, 0.42, 0.46]
beta[7] = [-0.28, -0.32, -0.32, -0.34, -0.34, -0.36, 0, 0, 0, 0.38, 0.43, 0.48]
beta[8] = [-0.30, -0.31, -0.31, -0.32, -0.33, -0.35, 0, 0, 0, 0.36, 0.39, 0.42]


for case in cases:

    print(f"\nProcessing case: {case}")

    data = scipy.io.loadmat(os.path.join(statspath, f'SmCase{case}All.mat'))

    utau   = data['utaug']
    Uinfty = data['ui']
    x      = data['x'][0]  # x-coordinates in m
    y      = data['y'] * 0.001 # in m
    u      = data['u']
    nu     = data['visc']
    delta_star = data['del1'] * 0.001  # in m
    delta99 = data['del99'] * 0.001  # in m

    # Get beta for the current case
    beta_  = beta[case]
    dpdx   = beta_ * utau**2 / delta_star

    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []

    for i in range(NUM_STATIONS):

        # --- Load data ---
        nu_i = nu[:, i][0]

        # Velocity data
        y_i = y[:, i]
        x_ = x[i]
        U_i = u[:, i]
        delta99_i = delta99[:, i][0]  # in m
        up_i = np.sign(dpdx[:,i]) * (np.abs(dpdx[:,i]) * nu_i / rho)**(1/3)  # in m/s
        up_i = up_i[0]  # in m/s

        # Local utau
        utau_i = utau[:,i][0]  # in m/s

        # Find points within the boundary layer region of interest
        bot_index = np.where((y_i >= DOWN_FRAC * delta99_i) & (y_i <= UP_FRAC * delta99_i))[0]

        U2 = find_k_y_values(y_i[bot_index], U_i, y_i, k=1)
        U3 = find_k_y_values(y_i[bot_index], U_i, y_i, k=2)
        U4 = find_k_y_values(y_i[bot_index], U_i, y_i, k=3)

        pi_1 = y_i * U_i / nu_i
        pi_1 = pi_1[bot_index]
        pi_2 = up_i * y_i[bot_index] / nu_i if up_i != 0 else np.zeros_like(y_i[bot_index])
        pi_3 = U2 * y_i[bot_index] / nu_i
        pi_4 = U3 * y_i[bot_index] / nu_i
        pi_5 = U4 * y_i[bot_index] / nu_i
        # Calculate velocity gradients
        dUdy = np.gradient(U_i, y_i)
        dudy_1 = dUdy[bot_index]
        dudy_2 = find_k_y_values(y_i[bot_index], dUdy, y_i, k=1)
        dudy_3 = find_k_y_values(y_i[bot_index], dUdy, y_i, k=2)

        pi_6 = dudy_1 * y_i[bot_index]**2 / nu_i
        pi_7 = dudy_2 * y_i[bot_index]**2 / nu_i
        pi_8 = dudy_3 * y_i[bot_index]**2 / nu_i

        # Calculate output (y+)
        pi_out = utau_i * y_i / nu_i
        pi_out = pi_out[bot_index]

        # --- Calculate Input Features (Pi Groups) ---
        # Note:  dPdx is NOT zero here
        # Calculate dimensionless inputs using safe names
        inputs_dict = {
            'u1_y_over_nu': pi_1,  # U_i[bot_index] * y_i[bot_index] / nu_i,
            'up_y_over_nu': pi_2,
            'u2_y_over_nu': pi_3,
            'u3_y_over_nu': pi_4,
            'u4_y_over_nu': pi_5,
            'dudy1_y_pow2_over_nu': pi_6,
            'dudy2_y_pow2_over_nu': pi_7,
            'dudy3_y_pow2_over_nu': pi_8,
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
            'y': y_i[bot_index],
            'u1': U_i[bot_index],
            'nu': np.full_like(y_i[bot_index], nu_i), 
            'utau': np.full_like(y_i[bot_index], utau_i),
            'up': np.full_like(y_i[bot_index], up_i),
            'u2': U2,
            'u3': U3,
            'u4': U4,
            'dudy1': dudy_1,
            'dudy2': dudy_2,
            'dudy3': dudy_3,
        }

        all_unnormalized_inputs_data.append(pd.DataFrame(unnorm_dict))

        # --- Collect Flow Type Information ---
        # Using format: [case_name, reference_nu, x_coord, delta, edge_velocity]
        # For channel flow: x=0, delta=1 (half-channel height), Ue=0 (or U_bulk if needed)
        len_y = len(y_i[bot_index])
        flow_type_dict = {
            'case_name': [casename] * len_y,
            'nu': [nu_i] * len_y,
            'x': [x_] * len_y,
            'delta': [delta99_i] * len_y,
        }
        # Add Retau for reference if needed, maybe as an extra column or replacing 'edge_velocity'
        # flow_type_dict['Retau'] = [Re_num] * len(y_sel)
        all_flow_type_data.append(pd.DataFrame(flow_type_dict))


# Concatenate data from all Re_num cases into single DataFrames
    inputs_df = pd.concat(all_inputs_data, ignore_index=True)
    output_df = pd.concat(all_output_data, ignore_index=True)
    flow_type_df = pd.concat(all_flow_type_data, ignore_index=True)
    unnormalized_inputs_df = pd.concat(all_unnormalized_inputs_data, ignore_index=True)

# Save DataFrames to HDF5 file
    output_filename = os.path.join(savedatapath, f'{casename}_{case}_data.h5')
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
