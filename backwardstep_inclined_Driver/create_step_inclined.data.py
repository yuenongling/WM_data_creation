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
currentpath = os.path.join(WM_DATA_PATH, 'backwardstep_inclined_Driver')
statspath = os.path.join(currentpath, 'stats')

cp_file = os.path.join(statspath, 'cp_deg6.dat')
cf_file = os.path.join(statspath, 'bkst-tauw.dat')
profile_file = os.path.join(statspath, 'deg6_station_data.pkl')

# Define constants from the image
Re_H = 36000
Uref = 1.0  # Reference velocity (normalized)

# Fractions for defining the boundary layer region of interest
UP_FRAC = 0.25    # Upper fraction of boundary layer to consider
DOWN_FRAC = 0.000 # Lower fraction of boundary layer to consider

"""
Process the extracted CSV files to calculate non-dimensional parameters
"""
# Load the extracted data
cf_data = pd.read_csv(cf_file)
cp_data = np.loadtxt(cp_file, comments='#')
x_cp = cp_data[:, 0]
cp   = cp_data[:, 1]
cf_data = np.loadtxt(cf_file, comments='#')
x_cf = cf_data[:, 0]
cf   = cf_data[:, 1]

profile_data = pkl.load(open(profile_file, 'rb'))

# Get unique stations
stations = list(profile_data.keys())

# Reference values
nu = 1.0 / Re_H  # Non-dimensional kinematic viscosity

# Calculate the pressure gradient
dPdx = np.gradient(cp, x_cp) * 0.5 * Uref**2
up = np.sign(dPdx) * (nu * abs(dPdx))**(1/3)

all_inputs_data = []
all_output_data = []
all_flow_type_data = []
all_unnormalized_inputs_data = []

# Hardcoded by looking at the data in the profile file
delta99 = [2.8, 2.8, 2.8, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.6, 3.6, 3.6, 3.6, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]

for i, station in enumerate(stations):
    print(f"Processing station {station}")
    
    # Extract profile data for this station
    station_data = profile_data[station]

    # Get x position from the station name
    x = float(station)
    
    # Get the nearest Cf value to this x position
    Cf = np.interp(x, x_cf, cf)
    
    # Calculate utau from Cf
    utau_i = np.sqrt(abs(Cf)/2) * Uref
    
    # Extract y and U data
    y_i = station_data['data']['Y/H'].values
    if x < 0: # if the x position is negative, it is before the step 
        y_i -= 1

    U_i = station_data['data']['U/Ur'].values
    
    # Boundary layer thickness (estimate from max y value)
    delta99_i = delta99[i]
    
    # Skip the first point if it is zero
    if U_i[0] == 0 or y_i[0] == 0:
        y_i = y_i[1:]
        U_i = U_i[1:]
    
    # Calculate pressure gradient velocity scale
    up_i = np.interp(x, x_cp, up)
    
    # Find points within the boundary layer region of interest
    bot_index = np.where((y_i >= DOWN_FRAC*delta99_i) & (y_i <= UP_FRAC*delta99_i))[0]

    if len(bot_index) < 2:
        print(f"  Skipping station {station}: not enough points in boundary layer")
        continue
    
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
        ['backstep', nu, x, delta99_i, 0] for _ in range(len(bot_index))
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
        'case_name': ['backstep'] * len_y,
        'nu': [nu] * len_y,
        'x': [x] * len_y,
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
output_filename = os.path.join(savedatapath, 'backstep_inclined_data.h5')
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
