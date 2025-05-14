import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
datapath = os.path.join(WM_DATA_PATH, 'data')
STATS_FILE = os.path.join(WM_DATA_PATH, 'swept_wing', 'stats')

# --- Load the APG data ---
UP_FRAC = 0.2
DOWN_FRAC = 0.01

# --- read cp ---
cp_data = np.loadtxt(os.path.join(STATS_FILE, 'f0251a-pt3.dat'), comments='%', usecols=(2,4))
x_cp = cp_data[:, 0]
cp = cp_data[:, 1]
dcpdx = np.gradient(cp, x_cp) * 2 # dpdx / (rho * Ue^2)

# --- Flow parameters ---
Retheta = 3400
theta   = 1.342607515480526 * 1e-3 # (m) from Fig. 9 of Van Den Berg et al. 1975; used for define Retheta

# --- read station data ---
import pprint # For pretty printing the dictionary

def parse_data_file(filepath):
    """
    Args:
        filepath (str): The path to the data file.

    Returns:
        dict: A dictionary where keys are station numbers (int).
              Each value is a dictionary with two keys:
              'station_info': A dict of station parameters.
              'velocities': A dict where keys are column names and
                            values are lists of floats/ints for that column.
    """
    all_stations_data = {}
    current_station_data = None
    reading_velocities = False

    # Define the keys based on the header lines
    # %Station   x(m)   Ue/Uref    Alpha     Cf     Betaw
    station_info_keys = ['Station', 'x(m)', 'Ue/Uref', 'Alpha', 'Cf', 'Betaw']
    
    # %Norm.y U/Ue  W/Ue  True y, m      U/Ue        W/Ue
    # To avoid duplicate keys like 'U/Ue', we'll qualify them.
    velocity_data_keys = ['Norm.y', 'U/Ue_norm', 'W/Ue_norm', 'True_y_m', 'U/Ue_true', 'W/Ue_true']

    try:
        with open(filepath, 'r') as f:
            for line_number, raw_line in enumerate(f, 1):
                line = raw_line.strip()

                if not line:  # Skip empty lines
                    continue

                if line.startswith("%Station"):
                    # This line is the header for station info.
                    # The actual data is expected on the next non-comment line.
                    # For this specific format, it's immediately the next line.
                    reading_velocities = False # Stop reading velocities if we were
                    current_station_data = {} # Prepare for new station data
                    
                    # The station_info_keys are defined above.
                    # The next line should contain the values.
                    continue # The actual data is on the *next* line

                if line.startswith("%Norm.y"):
                    # This is the header for velocity data.
                    # Actual data starts from the next line.
                    if current_station_data and 'station_info' in current_station_data:
                        reading_velocities = True
                        current_station_data['velocities'] = {key: [] for key in velocity_data_keys}
                    else:
                        # Found velocity header without preceding station info context
                        print(f"Warning: Line {line_number}: Found velocity header without active station context.")
                        reading_velocities = False
                    continue

                # If we are expecting station data (after %Station header)
                if current_station_data is not None and 'station_info' not in current_station_data and not line.startswith('%'):
                    parts = line.split()
                    if len(parts) == len(station_info_keys):
                        try:
                            info = {}
                            info[station_info_keys[0]] = int(parts[0]) # Station number
                            for i in range(1, len(station_info_keys)):
                                info[station_info_keys[i]] = float(parts[i])
                            
                            station_num = info['Station']
                            current_station_data['station_info'] = info
                            all_stations_data[station_num] = current_station_data
                        except ValueError:
                            print(f"Warning: Line {line_number}: Could not parse station data: {line}")
                            current_station_data = None # Invalidate
                    else:
                        # This line was not station data as expected, might be another comment
                        if not line.startswith('%'): # if it's not a comment, it's unexpected
                             print(f"Warning: Line {line_number}: Expected station data, found: {line}")
                             current_station_data = None # Invalidate

                # If we are reading velocity data
                elif reading_velocities and current_station_data and not line.startswith('%'):
                    parts = line.split()
                    if len(parts) == len(velocity_data_keys):
                        try:
                            for i, key in enumerate(velocity_data_keys):
                                current_station_data['velocities'][key].append(float(parts[i]))
                        except ValueError:
                            print(f"Warning: Line {line_number}: Could not parse velocity data row: {line}")
                            # Optionally, stop reading velocities for this station
                            # reading_velocities = False 
                    else:
                        # This might be the end of the velocity block (e.g. a stray line)
                        # or a malformed line.
                        # If it's not a comment, it signals end of velocity data.
                        print(f"Warning: Line {line_number}: Malformed velocity data or end of block: {line}")
                        reading_velocities = False 
                
                elif line.startswith('%') and reading_velocities:
                    # A comment line signifies the end of the current velocity data block
                    reading_velocities = False

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
        
    return all_stations_data

# --- Load the data ---
all_stations_data = parse_data_file(os.path.join(STATS_FILE, 'f0251a-pt1.dat'))

# --- Initialize variables ---
all_inputs_data = []
all_output_data = []
all_flow_type_data = []
all_unnormalized_inputs_data = []

# --- Go over all stations and extract data ---
#
# NOTE: Read from f0251a-pt1.dat
delta99 = [1.3000E-02, 1.6000E-02, 2.0001E-02, 2.4001E-02, 2.5991E-02, 2.9990E-02, 4.0000E-02, 4.9999E-02, 6.0010E-02,6.7501E-02]
delta99 = np.array(delta99) / theta
delta99_index = 0

# station_info_keys = ['Station', 'x(m)', 'Ue/Uref', 'Alpha', 'Cf', 'Betaw']
# velocity_data_keys = ['Norm.y', 'U/Ue_norm', 'W/Ue_norm', 'True_y_m', 'U/Ue_true', 'W/Ue_true']
for key, station_data in all_stations_data.items():

    y_i = np.array(station_data['velocities']['True_y_m']) / theta  # y / theta
    U = np.array(station_data['velocities']['U/Ue_true'])  # Normalized velocity
    W = np.array(station_data['velocities']['W/Ue_true'])  # Normalized velocity
    U_i = np.sqrt(U**2 + W**2) * station_data['station_info']['Ue/Uref'] # Magnitude of velocity; no separation, so safe to use

    # get dpdx
    cp_ind = np.argmin(abs(x_cp - station_data['station_info']['x(m)']))
    dpdx_i = dcpdx[cp_ind] # dpdx from cp data (dpdx / uref**2)
    up_i = np.sign(dpdx_i) * (abs(dpdx_i) * theta / Retheta ) ** (1 / 3)  # up / uref

    # tau_w
    cf_i = station_data['station_info']['Cf']  # Skin friction coefficient
    utau_i = np.sqrt(cf_i / 2)  # Friction velocity (utau / uref) 

    # 
    delta99_i = delta99[delta99_index]  # delta99 from the list
    delta99_index += 1
    bot_index = np.where((y_i >= DOWN_FRAC * delta99_i) & (y_i <= UP_FRAC * delta99_i))[0]

    U2 = find_k_y_values(y_i[bot_index], U_i, y_i, k=1)
    U3 = find_k_y_values(y_i[bot_index], U_i, y_i, k=2)
    U4 = find_k_y_values(y_i[bot_index], U_i, y_i, k=3)

    # --- Calculate Input Features (Pi Groups) ---
    # Calculate dimensionless inputs using safe names
    inputs_dict = {
        'u1_y_over_nu': U_i[bot_index] * y_i[bot_index] * Retheta,
        'up_y_over_nu': up_i * y_i[bot_index] * Retheta,
        'upn_y_over_nu': 0 * y_i[bot_index],
        'u2_y_over_nu': U2 * y_i[bot_index] * Retheta,
        'u3_y_over_nu': U3 * y_i[bot_index] * Retheta,
        'u4_y_over_nu': U4 * y_i[bot_index] * Retheta,
        'dudy1_y_pow2_over_nu':  0 * y_i[bot_index],
        'dudy2_y_pow2_over_nu':  0 * y_i[bot_index],
        'dudy3_y_pow2_over_nu':  0 * y_i[bot_index],
    }
    all_inputs_data.append(pd.DataFrame(inputs_dict))

    # --- Calculate Output Feature ---
    # Output is y+ (utau * y / nu)
    output_dict = {
        'utau_y_over_nu': utau_i * y_i[bot_index] * Retheta
    }
    all_output_data.append(pd.DataFrame(output_dict))

    # --- Collect Unnormalized Inputs ---
    unnorm_dict = {
        'y': y_i[bot_index],
        'u1': U_i[bot_index],
        'nu': np.full_like(y_i[bot_index], 1/Retheta),
        'utau': np.full_like(y_i[bot_index], utau_i),
        'up': np.full_like(y_i[bot_index], up_i),
        'upn': np.full_like(y_i[bot_index], 0),
        'u2': U2,
        'u3': U3,
        'u4': U4,
        'dudy1': np.full_like(y_i[bot_index], 0),
        'dudy2': np.full_like(y_i[bot_index], 0),
        'dudy3': np.full_like(y_i[bot_index], 0),
    }
    all_unnormalized_inputs_data.append(pd.DataFrame(unnorm_dict))

    # --- Collect Flow Type Information ---
    # Using format: [case_name, reference_nu, x_coord, delta, edge_velocity]
    # For channel flow: x=0, delta=1 (half-channel height), Ue=0 (or U_bulk if needed)
    flow_type_dict = {
        'case_name': ['swept_wing'] * len(y_i[bot_index]),
        'nu': [1/Retheta] * len(y_i[bot_index]),
        'x': [key] * len(y_i[bot_index]),
        'delta': [delta99_i] * len(y_i[bot_index]),
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
output_filename = os.path.join(datapath, 'swept_wing_data.h5')
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
