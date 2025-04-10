import pandas as pd
import numpy as np
import sys
import pickle as pkl
import os
# import scipy.signal # Keep if needed later, otherwise remove

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
savedatapath = os.path.join(WM_DATA_PATH, 'data')
currentpath = os.path.join(WM_DATA_PATH, 'sep_bubble_Coleman')
statspath = os.path.join(currentpath, 'data')

def find_separation_region(Cf):
    """
    Find the start and end indices of the first continuous separation region (Cf < 0).
    Returns None if no separation is found.
    """
    neg_cf_indices = np.where(Cf < 0)[0]
    if len(neg_cf_indices) == 0:
        print("Info: No separation region found (Cf never below 0).")
        return None, None

    # Find the first index
    idx_first = neg_cf_indices[0]

    # Find the last index - more robustly find the end of the *first* block
    diff = np.diff(neg_cf_indices)
    discontinuity = np.where(diff > 1)[0]
    if len(discontinuity) == 0:
        # Only one continuous block
        idx_last = neg_cf_indices[-1]
    else:
        # End index of the first block
        idx_last = neg_cf_indices[discontinuity[0]]

    return idx_first, idx_last


# --- Configuration ---

interval = int(sys.argv[1]) if len(sys.argv) > 1 else 10

for case_name in ['A', 'B', 'C']:
    print("--- Starting Comprehensive Post-Processing ---")

# Define input and output file paths
    input_pickle_path = os.path.join(statspath, f'tecplot_data_Case{case_name}.pkl')
# output_raw_profiles_path = f'./data_processed/data_Case{case_name}_every_{interval}_rawprofiles.pkl' # Renamed for clarity
    output_derived_data_path = os.path.join(savedatapath, f'bub_{case_name}_data.h5')

# Check if input pickle file exists
    if not os.path.exists(input_pickle_path):
        print(f"Error: Input data file not found: {input_pickle_path}")
        print(f"Please run 'python extract_data_comprehensive.py {case_name}' first.")
        sys.exit(1)

    print(f"Processing Case: {case_name}")
    print(f"Loading pre-processed data from: {input_pickle_path}")
# print(f"Save raw profiles: {save_raw_profiles}, Sampling Interval: {interval}")

# --- Load Pre-Processed Data ---
    try:
        with open(input_pickle_path, 'rb') as f:
            loaded_data = pkl.load(f)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data from Pickle file: {e}")
        sys.exit(1)

# Extract data components
    case_name_loaded = loaded_data['case_name']
    Re = loaded_data['Re']
    surface_data = loaded_data['surface_data']
    volume_data = loaded_data['volume_data']
    dimensions = loaded_data['dimensions']

    x = surface_data['x']
# Cp = surface_data['Cp'] # Not directly used in this part of the script
    Cf = surface_data['Cf']
    delta = surface_data['delta']
# dPdx_smooth = surface_data['dPdx_smooth'] # Not directly used here
    up = surface_data['up'] # Use the pre-calculated 'up'

    y_vol = volume_data['y'] # Shape (max_J, max_I)
    U_vol = volume_data['U'] # Shape (max_J, max_I)

    max_I = dimensions['I']
    max_J = dimensions['J']

# Basic verification
    if case_name_loaded != case_name:
        print(f"Warning: Loaded data is for case {case_name_loaded}, expected {case_name}")
    if y_vol.shape != (max_J, max_I) or U_vol.shape != (max_J, max_I):
        print("Warning: Loaded volume data shapes do not match saved dimensions!")
    if len(x) != max_J or len(Cf) != max_J or len(delta) != max_J or len(up) != max_J:
        print("Warning: Length of loaded surface arrays does not match volume dimension J!")


# --- Analysis and Processing ---

# Define y-range parameters (relative to delta)
    UP_DELTA = 0.2
    DOWN_DELTA = 0.008
    UP_DELTA_SEP = 0.05 # Use different limits in separation
    DOWN_DELTA_SEP = 0.005

# Find separation region indices using the loaded Cf
    idx_first_sep, idx_last_sep = find_separation_region(Cf)

# Determine which x indices to process
    investigation_limit = 10.0 # Based on original script using x_to_investigate=10
    base_indices = np.where((x > -investigation_limit) & (x < investigation_limit))[0]
# Sample these base indices
    sampled_indices = base_indices[::interval]

# Add more points near separation boundaries if separation exists
    if idx_first_sep is not None:
        print(f"Separation identified: x_indices [{idx_first_sep}, {idx_last_sep}], x_values [{x[idx_first_sep]:.3f}, {x[idx_last_sep]:.3f}]")
        # Define how many points to add around separation (e.g., 100 points before/after)
        points_around_sep = 100
        indices_near_sep_start = np.arange(max(0, idx_first_sep - points_around_sep), idx_first_sep + points_around_sep)
        indices_near_sep_end = np.arange(max(0, idx_last_sep - points_around_sep), min(max_J, idx_last_sep + points_around_sep))
        # Combine base sampled indices and near-separation indices
        x_idx_in_range = np.concatenate((sampled_indices, indices_near_sep_start, indices_near_sep_end))
    else:
        x_idx_in_range = sampled_indices # No separation, just use sampled indices

# Remove duplicates and sort
    x_idx_in_range = np.sort(np.unique(x_idx_in_range))
# Ensure indices are within bounds
    x_idx_in_range = x_idx_in_range[(x_idx_in_range >= 0) & (x_idx_in_range < max_J)]

    print(f"Total number of x-locations to process: {len(x_idx_in_range)}")

# Initialize lists to store results efficiently
    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []

# --- Main Processing Loop ---
    for i, x_idx in enumerate(x_idx_in_range):

        # Extract profile and surface data for this x_idx
        current_x = x[x_idx]
        current_delta = delta[x_idx]
        current_Cf = Cf[x_idx]
        current_up = up[x_idx] # Use pre-calculated 'up'

        # Extract the velocity profile corresponding to this x_idx (J index)
        # y_data is the y-coordinates for this profile (I varies)
        # U_data is the U-velocity for this profile (I varies)
        y_data = y_vol[x_idx, :]
        U_data = U_vol[x_idx, :]

        # Progress indicator
        if (i + 1) % 50 == 0 or i == len(x_idx_in_range) - 1:
            print(f'Processing x = {current_x:.4f} (Index {x_idx}, {i+1}/{len(x_idx_in_range)})')

        # Determine the relevant range of y indices based on delta and separation status
        is_separated = (idx_first_sep is not None) and (idx_first_sep <= x_idx <= idx_last_sep)

        if is_separated:
            y_limit_lower = DOWN_DELTA_SEP * current_delta
            y_limit_upper = UP_DELTA_SEP * current_delta
        else:
            y_limit_lower = DOWN_DELTA * current_delta
            y_limit_upper = UP_DELTA * current_delta

        # Find indices within the y-range [y_limit_lower, y_limit_upper]
        valid_y_indices = np.where((y_data >= y_limit_lower) & (y_data <= y_limit_upper))[0]

        # Special handling for separation region to ensure we don't go past U=0 if UP_DELTA_SEP is large
        if is_separated:
            # Find first index where U > 0 (after the first point, assuming U near wall can be < 0)
            positive_U_indices = np.where(U_data[1:] > 0)[0]
            if len(positive_U_indices) > 0:
                idx_U_becomes_positive = positive_U_indices[0] + 1 # Add 1 because we sliced from index 1
                # Filter valid_y_indices to be below this point
                valid_y_indices = valid_y_indices[valid_y_indices < idx_U_becomes_positive]
            # else: U is never positive (highly unlikely/problematic profile) -> valid_y_indices might become empty

        # Skip if no valid y-points are found in the specified range
        if len(valid_y_indices) == 0:
            print(f'---> Skipping x = {current_x:.4f}: No data points found in y range [{y_limit_lower:.4e}, {y_limit_upper:.4e}] after checks.')
            continue

        # Extract U and y data only within the valid range
        y_in_range_vals = y_data[valid_y_indices]
        U_in_range_vals = U_data[valid_y_indices]

        # Calculate U at k*y using interpolation on the *full* profile
        # Pass the y values *within the range* to find U at k*y_in_range
        U_2 = find_k_y_values(y_in_range_vals, U_data, y_data, k=1)
        U_3 = find_k_y_values(y_in_range_vals, U_data, y_data, k=2)
        U_4 = find_k_y_values(y_in_range_vals, U_data, y_data, k=3)

        # Handle potential NaNs from interpolation (if k*y was outside profile range)
        # Decide on strategy: skip point, set to 0, etc. Let's skip for now if any U_k is NaN.
        # if np.isnan(U_2).any() or np.isnan(U_3).any() or np.isnan(U_4).any():
        #     print(f'---> Skipping x = {current_x:.4f}: NaN encountered in U_k calculation (likely k*y outside profile range).')
        #     continue

        # Calculate dimensionless inputs (pi groups)
        pi_1 = U_in_range_vals * y_in_range_vals * Re
        pi_2 = current_up * y_in_range_vals * Re
        pi_3 = U_2 * y_in_range_vals * Re
        pi_4 = U_3 * y_in_range_vals * Re
        pi_5 = U_4 * y_in_range_vals * Re

        # Calculate dimensionless output (friction velocity based)
        utau = np.sqrt(np.abs(current_Cf * 0.5)) # utau / U_inf
        pi_out = utau * y_in_range_vals * Re

        # Calculate gradient for consistency with other dataset
        dudy = np.gradient(U_data, y_data)
        dudy_1 = dudy[valid_y_indices]
        dudy_2 = find_k_y_values(y_in_range_vals, dudy, y_data, k=1)
        dudy_3 = find_k_y_values(y_in_range_vals, dudy, y_data, k=2)

        pi_6 = dudy_1 * y_in_range_vals**2 * Re
        pi_7 = dudy_2 * y_in_range_vals**2 * Re
        pi_8 = dudy_3 * y_in_range_vals**2 * Re

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
            'y': y_in_range_vals,
            'u1': U_in_range_vals,
            'nu': np.full_like(y_in_range_vals, 1/Re),
            'utau': np.full_like(y_in_range_vals, utau),
            'up': np.full_like(y_in_range_vals, current_up),
            'u2': U_2,
            'u3': U_3,
            'u4': U_4,
            'dudy1': dudy_1,
            'dudy2': dudy_2,
            'dudy3': dudy_3,
        }
        all_unnormalized_inputs_data.append(pd.DataFrame(unnorm_dict))


        # --- Collect Flow Type Information ---
        # Using format: [case_name, reference_nu, x_coord, delta, edge_velocity]
        # For channel flow: x=0, delta=1 (half-channel height), Ue=0 (or U_bulk if needed)
        len_y = len(y_in_range_vals)
        flow_type_dict = {
            'case_name': [f'bub_{case_name}'] * len_y,
            'nu': [1/Re] * len_y,
            'x': [current_x] * len_y,
            'delta': [current_delta] * len_y,
        }
        # Add Retau for reference if needed, maybe as an extra column or replacing 'edge_velocity'
        # flow_type_dict['Retau'] = [Re_num] * len(y_sel)
        all_flow_type_data.append(pd.DataFrame(flow_type_dict))


# --- Finalize and Save ---

# Concatenate data from all Re_num cases into single DataFrames
    inputs_df = pd.concat(all_inputs_data, ignore_index=True)
    output_df = pd.concat(all_output_data, ignore_index=True)
    flow_type_df = pd.concat(all_flow_type_data, ignore_index=True)
    unnormalized_inputs_df = pd.concat(all_unnormalized_inputs_data, ignore_index=True)

# Save DataFrames to HDF5 file
    output_filename = output_derived_data_path
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
    print("--- Comprehensive Post-Processing Finished ---")
