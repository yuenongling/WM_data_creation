import numpy as np
import pandas as pd
import os
import pickle as pkl  # Import pickle
import re

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
savedatapath = os.path.join(WM_DATA_PATH, 'data')
currentpath = os.path.join(WM_DATA_PATH, 'duct_Roma', 'stats')

# --- Load the APG data ---
UP_FRAC = 0.25
DOWN_FRAC = 0.002

Re_all = [150, 220, 500, 1000]
Re_all_real = [150, 227, 519, 1055]
Re_b   = [4410, 7000, 17800, 40000]

all_inputs_data = []
all_output_data = []
all_flow_type_data = []
all_unnormalized_inputs_data = []

# Set a random nu for dimensional calculations
Ub = 17.5
nu_all = 2 * Ub / np.array(Re_b)

for i, Re in enumerate(Re_all):

    nu = nu_all[i]

    # --- Load data from files ---
    pkl_filepath = os.path.join(currentpath , f'plotyz_Retau{Re}.pkl') # Replace with your file path
    result = pkl.load(open(pkl_filepath, 'rb'))

    # --- Load friction velocity data ---
    retau = Re_all_real[i]
    reb   = Re_b[i] / 2 # Note that the definition of Re_b is 2*h*ub/nu

    for z_ in result.keys():
        
        if z_ < -0.99:
            continue
            print(f"Skipping z = {z_} as it is less than -0.99")

        result_ = result[z_]
        y = result_[:, 0] + 1 # convert y to positive values
        z = result_[:, 1]
        assert np.abs(z - z_).max() < 1e-6, "z coordinate mismatch"
        delta = 1 + z_ # local "boundary layer thickness"

        u = result_[:, 2]
        w = result_[:, 3]
        umag = np.sqrt(u**2 + w**2) # magnitude of velocity

        tauw_ratio = result[z_][:, 4]
        assert np.unique(tauw_ratio).size == 1, "tauw_ratio should be constant for each z_"
        retau_local = np.unique(tauw_ratio)[0] * retau  # Friction velocity

        # Calculate dimensionless wall distance and filter data
        bot_index = np.where((((y >= DOWN_FRAC*delta)) & (y <= UP_FRAC*delta)))[0]

        # No equivalent to 'x' or 'dPdx' in channel flow, so using 0 as placeholder
        x = 0
        up = - (retau**2 / 16)**(1/3) / reb # up / ub

        # Calculate interpolated U values
        U2 = find_k_y_values(y[bot_index], umag, y, k=1)
        U3 = find_k_y_values(y[bot_index], umag, y, k=2)
        U4 = find_k_y_values(y[bot_index], umag, y, k=3)

        # Calculate velocity gradient
        dUdy = np.gradient(umag, y) # dU/dy * R**2 / nu
        dudy_1 = dUdy[bot_index]
        dudy_2 = find_k_y_values(y[bot_index], dUdy, y, k=1)
        dudy_3 = find_k_y_values(y[bot_index], dUdy, y, k=2)

        # NOTE: Inputs
        pi_1 = umag * y * reb # U * y / nu
        pi_1 = pi_1[bot_index]
        pi_2 = y * up * reb
        pi_2 = pi_2[bot_index]
        pi_3 = U2 * y[bot_index] * reb
        pi_5 = U3 * y[bot_index] * reb
        pi_4 = U4 * y[bot_index] * reb

        # WARNING: These might be incorrect but I do not bother to fix them as they are useless
        pi_6 = dudy_1 * y[bot_index]**2 * reb
        pi_7 = dudy_2 * y[bot_index]**2 * reb
        pi_8 = dudy_3 * y[bot_index]**2 * reb

        # NOTE: Outputs (mimicking KTH script output feature)
        pi_out = retau_local * y
        pi_out = pi_out[bot_index]

        # --- Calculate Input Features (Pi Groups) ---
        # Calculate dimensionless inputs using safe names
        inputs_dict = {
            'u1_y_over_nu': pi_1,  # U_i[bot_index] * y_i[bot_index] / nu_i,
            'up_y_over_nu': pi_2,
            'upn_y_over_nu': pi_2, # pi_2 == 0
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
            'y': y[bot_index],
            'u1': umag[bot_index]*Ub,
            'nu': np.full_like(y[bot_index], nu),
            'utau': np.full_like(y[bot_index], retau_local * nu),
            'up': np.full_like(y[bot_index], up),
            'upn': np.full_like(y[bot_index], up),
            'u2': U2*Ub,
            'u3': U3*Ub,
            'u4': U4*Ub,
            'dudy1': dudy_1,
            'dudy2': dudy_2,
            'dudy3': dudy_3,
        }
        all_unnormalized_inputs_data.append(pd.DataFrame(unnorm_dict))

        # --- Collect Flow Type Information ---
        # Using format: [case_name, reference_nu, x_coord, delta, edge_velocity]
        # For channel flow: x=0, delta=1 (half-channel height), Ue=0 (or U_bulk if needed)
        flow_type_dict = {
            'case_name': ['duct'] * len(y[bot_index]),
            'nu': [1/retau] * len(y[bot_index]),
            'x': [z_] * len(y[bot_index]),
            'delta': [delta] * len(y[bot_index]),
            'Retau': [retau_local] * len(y[bot_index]),
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
output_filename = os.path.join(savedatapath, 'DUCT_data.h5')
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
