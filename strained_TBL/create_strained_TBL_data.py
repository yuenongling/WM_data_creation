import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import os
from scipy import interpolate

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
casename = 'strained_TBL'
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
savedatapath = os.path.join(WM_DATA_PATH, 'data')
currentpath = os.path.join(WM_DATA_PATH, casename)
statspath = os.path.join(currentpath, 'stats')

# --- Hardcoded some data ---
rho = 1.1182
nu  = 0.16620E-04 
Q   = 1000.18
delta  = 13 * 1e-3 # in m

# load p and cf data
with open(os.path.join(statspath, 'cp_conv.pkl'), 'rb') as f:
    p_data = pkl.load(f)
with open(os.path.join(statspath, 'cf_conv.pkl'), 'rb') as f:
    cf_data = pkl.load(f)

# Fractions for defining the boundary layer region of interest
UP_FRAC = 0.2    # Upper fraction of boundary layer to consider
DOWN_FRAC = 0.002 # Lower fraction of boundary layer to consider

all_inputs_data = []
all_output_data = []
all_flow_type_data = []
all_unnormalized_inputs_data = []

z_list =[-150, -100, 0]  # -200  does not have Cf data
for z in z_list:
    with open(os.path.join(statspath, f'vel_z{z}_conv.pkl'), 'rb') as f:
        vel_data = pkl.load(f)

    p_i_data = p_data['data_by_z_station'][z]
    xp   = p_i_data['x [mm]'] * 1e-3 # in m
    p_i = p_i_data['Pstat[Pa]']

    dpdx = np.gradient(p_i, xp) # in Pa/m
    up = np.sign(dpdx) * (np.abs(dpdx) * nu / rho)**1/3 # in m/s

    cf_data_i = cf_data['data_by_z_station'][z]
    x_to_investigate = cf_data_i['x [mm]'] * 1e-3 # in m
    cf = cf_data_i['Cf']
    utau = np.sqrt(cf * Q) # in m/s

    for i, x_ in enumerate(x_to_investigate):
        # Each x_ (xA) represents one case (with different pressure gradient) with data measured at one xm downstream location
        
        utau_i = utau[i] # in m/s
        delta99_i = delta # in m
        nu_i = nu # in m^2/s

        # A: Distance from the wall y [mm]
        # B: Dimensionless velocity U/UD
        # C: Yplus
        # D: Uplus
        if x_*1e3 not in vel_data['data_by_x_station']:
            print(f"Warning: x={x_*1e3} mm not found in velocity data. Skipping this x.")
            continue

        vel_file = vel_data['data_by_x_station'][x_*1e3]['velocity_profile']

        U_i = vel_file['Q [m/s]'].values
        y_i = vel_file['y [mm]'].values * 1e-3 # in m

        # Find points within the boundary layer region of interest
        bot_index = np.where((y_i >= DOWN_FRAC * delta99_i) & (y_i <= UP_FRAC * delta99_i))[0]

        # Calculate velocity at different y positions
        U2 = find_k_y_values(y_i[bot_index], U_i, y_i, k=1)
        U3 = find_k_y_values(y_i[bot_index], U_i, y_i, k=2)
        U4 = find_k_y_values(y_i[bot_index], U_i, y_i, k=3)

        # WARNING: pressure gradient along the test wall is minimized by the geometry of the opposite wall; assume zero here
        up_i = np.interp(x_, xp, up) # in m/s
        
        # Calculate non-dimensional inputs
        pi_1 = y_i * U_i / nu_i
        pi_1 = pi_1[bot_index]
        pi_2 = up_i * y_i / nu_i if up_i != 0 else np.zeros_like(y_i[bot_index])
        pi_2 = pi_2[bot_index]
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
    output_filename = os.path.join(savedatapath, f'{casename}_z{z}_data.h5')
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
