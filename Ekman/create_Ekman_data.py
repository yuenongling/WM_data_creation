import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import os
from scipy import interpolate

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
casename = 'Ekman'
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
savedatapath = os.path.join(WM_DATA_PATH, 'data')
currentpath = os.path.join(WM_DATA_PATH, casename)
statspath = os.path.join(currentpath, 'stats')

# --- Hardcoded some data ---
utau = 0.05580 
Retau = 914.74451

# --- Load data ---
data = np.loadtxt(os.path.join(statspath, 'stats.dat'), comments='#')
y_i = data[:, 0]  # y/delta
yplus        = data[:, 3]  # y+
Uplus        = data[:, 4]  # U+
Wplus        = data[:, 5]  # W+
U_i = np.sqrt(Uplus**2 + Wplus**2)  # Magnitude of velocity

# Fractions for defining the boundary layer region of interest
UP_FRAC = 0.2   
DOWN_FRAC = 0.05

all_inputs_data = []
all_output_data = []
all_flow_type_data = []
all_unnormalized_inputs_data = []

# Find points within the boundary layer region of interest
bot_index = np.where((y_i >= DOWN_FRAC) & (y_i <= UP_FRAC))[0]

# Calculate velocity at different y positions
U2 = find_k_y_values(y_i[bot_index], U_i, y_i, k=1)
U3 = find_k_y_values(y_i[bot_index], U_i, y_i, k=2)
U4 = find_k_y_values(y_i[bot_index], U_i, y_i, k=3)

# WARNING: pressure gradient along the test wall is minimized by the geometry of the opposite wall; assume zero here
up_i = - (Retau)**(2/3) # up * delta / nu

# Calculate non-dimensional inputs
pi_1 = U_i[bot_index] * y_i[bot_index]
pi_2 = up_i * y_i[bot_index]
pi_3 = U2 * y_i[bot_index]
pi_4 = U3 * y_i[bot_index]
pi_5 = U4 * y_i[bot_index]

# Calculate velocity gradients
dUdy = np.gradient(U_i, y_i)
dudy_1 = dUdy[bot_index]
dudy_2 = find_k_y_values(y_i[bot_index], dUdy, y_i, k=1)
dudy_3 = find_k_y_values(y_i[bot_index], dUdy, y_i, k=2)

pi_6 = dudy_1 * y_i[bot_index]**2
pi_7 = dudy_2 * y_i[bot_index]**2
pi_8 = dudy_3 * y_i[bot_index]**2

# Calculate output (y+)
pi_out = Retau * y_i
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
    'nu': np.full_like(y_i[bot_index], 1/Retau), 
    'utau': np.full_like(y_i[bot_index], utau),
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
    'nu': [1/Retau] * len_y,
    'x': [0] * len_y,
    'delta': [1] * len_y,
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
output_filename = os.path.join(savedatapath, f'{casename}_data.h5')
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
