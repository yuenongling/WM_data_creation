import numpy as np
import pandas as pd
import os
import pickle as pkl  # Import pickle
import re

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
savedatapath = os.path.join(WM_DATA_PATH, 'data')
currentpath = os.path.join(WM_DATA_PATH, 'pipe_Roma', 'stats')

# --- Load the APG data ---
UP_FRAC = 0.2
DOWN_FRAC = 0.025

def read_pipe_data(filepath):
    """
    Reads the pipe flow data from a text file.

    Args:
        filepath (str): The path to the data file.

    Returns:
        numpy.ndarray: A NumPy array containing the numerical data, or None if an error occurs.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Extract Friction Reynolds number
        re_tau = None
        for line in lines:
            match = re.search(r"%\s*Friction Reynolds number\s*=\s*([\d.]+)", line)
            if match:
                re_tau = float(match.group(1))
                break

        # Find the line where the numerical data starts
        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('%') and line.strip():  # Find the first non-comment, non-empty line
                data_start = i
                break

        # Extract and parse the numerical data
        data_lines = lines[data_start:]
        data = []
        for line in data_lines:
            values = line.split()
            if values: #prevent empty lines from causing issues.
                data.append([float(val) for val in values])

        return re_tau, np.array(data)

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except ValueError:
        print("Error: Invalid numerical data in the file.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

Re_all = [500, 1140, 2000, 3000, 6000]

all_inputs_data = []
all_output_data = []
all_flow_type_data = []
all_unnormalized_inputs_data = []

for Re in Re_all:

# Example usage:
    filepath = os.path.join(currentpath , f'Pipe_profiles_Retau{Re}.dat') # Replace with your file path
    retau, data = read_pipe_data(filepath)

    if data is not None:

        print("Data loaded successfully for Re_tau =", retau)
        # Access individual columns, for example:
        y_plus = data[:, 0]
        Uz_plus = data[:, 1]

        y = y_plus / retau # y / R
        uu = Uz_plus * retau # U * R / nu

        # Calculate dimensionless wall distance and filter data
        bot_index = np.where((((y_plus > 50) & (y >= DOWN_FRAC)) & (y <= UP_FRAC)))[0]

        # No equivalent to 'x' or 'dPdx' in channel flow, so using 0 as placeholder
        x = 0
        # up = utau * (1/retau) **1/3
        # up*y/nu = yplus * (1/retau) ** 1/3
        dPdx = - (retau) #  This is placeholder for dPdx  
        up = - (1/retau)**(1/3)

        # Calculate interpolated U values
        U2 = find_k_y_values(y[bot_index], uu, y, k=1)
        U3 = find_k_y_values(y[bot_index], uu, y, k=2)
        U4 = find_k_y_values(y[bot_index], uu, y, k=3)

        # Calculate velocity gradient
        dUdy = np.gradient(uu, y) # dU/dy * R**2 / nu
        dudy_1 = dUdy[bot_index]
        dudy_2 = find_k_y_values(y[bot_index], dUdy, y, k=1)
        dudy_3 = find_k_y_values(y[bot_index], dUdy, y, k=2)

        # NOTE: Inputs (mimicking KTH script input features)
        pi_1 = uu * y # U * y / nu
        pi_1 = pi_1[bot_index]
        pi_2 = y_plus * up
        pi_2 = pi_2[bot_index]
        pi_3 = U2 * y[bot_index]
        pi_4 = U3 * y[bot_index]
        pi_5 = U4 * y[bot_index]
        pi_6 = dudy_1 * y[bot_index]**2 
        pi_7 = dudy_2 * y[bot_index]**2 
        pi_8 = dudy_3 * y[bot_index]**2 

        # NOTE: Outputs (mimicking KTH script output feature)
        pi_out = retau * y
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
            'u1': uu[bot_index],
            'nu': np.full_like(y[bot_index], 1),
            'utau': np.full_like(y[bot_index], retau),
            'up': np.full_like(y[bot_index], up),
            'upn': np.full_like(y[bot_index], up),
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
        flow_type_dict = {
            'case_name': ['pipe'] * len(y[bot_index]),
            'nu': [1/retau] * len(y[bot_index]),
            'x': [0] * len(y[bot_index]),
            'delta': [1] * len(y[bot_index]),
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
output_filename = os.path.join(savedatapath, 'PIPE_data.h5')
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
# print("\n--- Sanity Check: Comparing HDF5 with Original Pickle ---")
# with open('/home/yuenongling/Codes/BFM/WM_Opt/data/PIPE_data.pkl', 'rb') as f:
#     original_data = pkl.load(f)
#
# # Load corresponding data from HDF5
# inputs_hdf = inputs_df[inputs_df.index.isin(np.arange(len(original_data['inputs'])))].values
# output_hdf = output_df[output_df.index.isin(np.arange(len(original_data['output'])))].values.flatten()
# flow_type_hdf = flow_type_df[flow_type_df.index.isin(np.arange(len(original_data['flow_type'])))].values
# unnormalized_inputs_hdf = unnormalized_inputs_df[
#     unnormalized_inputs_df.index.isin(np.arange(len(original_data['unnormalized_inputs'])))].values
#
# print(f"  Inputs match: {np.allclose(original_data['inputs'], inputs_hdf)}")
# print(f"  Output match: {np.allclose(original_data['output'], output_hdf)}")
# print(f"  Flow type match: {np.array_equal(original_data['flow_type'].astype(str), flow_type_hdf.astype(str))}")
# print(
#     f"  Unnormalized inputs match: {np.allclose(original_data['unnormalized_inputs'].flatten(), unnormalized_inputs_hdf.flatten(), rtol=1e-5, atol=1e-4)}")
