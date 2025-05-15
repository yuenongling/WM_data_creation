import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import os
from scipy import interpolate

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
savedatapath = os.path.join(WM_DATA_PATH, 'data')
currentpath = os.path.join(WM_DATA_PATH, 'TBL_APG_Gasser')
statspath = os.path.join(currentpath, 'stats')


# --- Hardcoded some data ---
x = [-100, 0, 50, 100, 200, 300]
tauw = [3.282, 2.213, 0.664, 0.235, 0.434, 0.646]
delta99 = np.array([15, 30, 18, 25, 30, 30]) * 1e-3
rho = 1.11
nu  = 1.65E-05

# Fractions for defining the boundary layer region of interest
UP_FRAC = 0.2    # Upper fraction of boundary layer to consider
DOWN_FRAC = 0.002 # Lower fraction of boundary layer to consider

all_inputs_data = []
all_output_data = []
all_flow_type_data = []
all_unnormalized_inputs_data = []

for i, x_ in enumerate(x):
    # Each x_ (xA) represents one case (with different pressure gradient) with data measured at one xm downstream location

    # Calculate the friction velocity
    tauw_i = tauw[i]
    utau_i = np.sqrt(tauw_i / rho)
    # A: Distance from the wall y [mm]
    # B: Dimensionless velocity U/UD
    # C: Yplus
    # D: Uplus
    vel_file = os.path.join(statspath, f'station_{i}.dat')
    vel = np.loadtxt(vel_file, usecols=(0, 1, 2, 3))
    y_i = vel[:, 0] * 1E-3  # Convert to meters
    U_over_UD = vel[:, 1]
    Uplus = vel[:, 3]
    U_i     = Uplus * utau_i
    UD    = np.mean(U_i / U_over_UD)
    
    cp_file = os.path.join(statspath, f'station_p_{i}.dat')
    cp = np.loadtxt(cp_file, usecols=(0, 1))
    dcpdx = np.gradient(cp[:, 1], cp[:, 0])
    dcpdx_i = dcpdx[5] # The middle one is the point of interest
    dpdx_i = dcpdx_i * 0.5 * rho * UD**2 # dpdx / (uref**2)
    up_i = np.sign(dpdx_i) * (nu * abs(dpdx_i) / rho)**(1/3)

    # Find points within the boundary layer region of interest
    delta99_i = delta99[i]
    bot_index = np.where((y_i >= DOWN_FRAC*delta99_i) & (y_i <= UP_FRAC*delta99_i))[0]

    # Calculate velocity at different y positions
    U2 = find_k_y_values(y_i[bot_index], U_i, y_i, k=1)
    U3 = find_k_y_values(y_i[bot_index], U_i, y_i, k=2)
    U4 = find_k_y_values(y_i[bot_index], U_i, y_i, k=3)
    
    # Calculate non-dimensional inputs
    pi_1 = y_i * U_i / nu
    pi_1 = pi_1[bot_index]
    pi_2 = up_i * y_i / nu if up_i != 0 else np.zeros_like(y_i[bot_index])
    pi_2 = pi_2[bot_index]
    pi_3 = U2 * y_i[bot_index] / nu
    pi_4 = U3 * y_i[bot_index] / nu
    pi_5 = U4 * y_i[bot_index] / nu
    
    # Calculate velocity gradients
    dUdy = np.gradient(U_i, y_i)
    dudy_1 = dUdy[bot_index]
    dudy_2 = find_k_y_values(y_i[bot_index], dUdy, y_i, k=1)
    dudy_3 = find_k_y_values(y_i[bot_index], dUdy, y_i, k=2)
    
    pi_6 = dudy_1 * y_i[bot_index]**2 / nu
    pi_7 = dudy_2 * y_i[bot_index]**2 / nu
    pi_8 = dudy_3 * y_i[bot_index]**2 / nu
    
    # Calculate output (y+)
    pi_out = utau_i * y_i / nu
    pi_out = pi_out[bot_index]
    # Flow type information
    flow_type_tmp = np.array([
        ['APG_Gasser', nu, x, delta99_i, 0] for _ in range(len(bot_index))
    ], dtype=object)
    
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
        'nu': np.full_like(y_i[bot_index], nu), 
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
        'case_name': ['APG_Gasser'] * len_y,
        'nu': [nu] * len_y,
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
output_filename = os.path.join(savedatapath, 'APG_Gasser_data.h5')
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
