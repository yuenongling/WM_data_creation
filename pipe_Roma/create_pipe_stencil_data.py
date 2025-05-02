'''
For this specific script, need to run from the directory containing data_processing_utils.py
and the 'pipe_Roma/stats' subdirectory with the data files.
Assuming the structure is something like:
/your_project/
├── data_processing_utils.py
├── pipe_Roma/
│   └── stats/
│       └── Pipe_profiles_Retau*.dat
└── create_pipe_flow_data.py (this script)

python create_pipe_flow_data.py

'''
import numpy as np
import pandas as pd
import os
import pickle as pkl
import re # Import re for regular expressions

# --- Set up paths and constants ---
# data_processing_utils is expected to contain find_k_y_values and import_path
from data_processing_utils import find_k_y_values, import_path

# WM_DATA_PATH should be the base path where your simulation data is stored
# import_path() is assumed to return this base path
WM_DATA_PATH = import_path()

# Define the output directory for the HDF5 file
savedatapath = os.path.join(WM_DATA_PATH, 'data')

# Define the path to the raw pipe data files
currentpath = os.path.join(WM_DATA_PATH, 'pipe_Roma', 'stats')

# Y-fraction limits for selecting points (in units of pipe radius R)
UP_FRAC = 0.2   # Upper limit of y/R
DOWN_FRAC = 0.025 # Lower limit of y/R

# --- Function to Read Pipe Data ---
def read_pipe_data(filepath):
    """
    Reads the pipe flow data from a text file.

    Args:
        filepath (str): The path to the data file.

    Returns:
        tuple: A tuple containing (re_tau, data_array) or (None, None) if an error occurs.
               data_array is a NumPy array with columns: y+, Uz+, uu+, vv+, ww+, uv+
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Extract Friction Reynolds number (Retau) from comments
        re_tau = None
        for line in lines:
            match = re.search(r"%\s*Friction Reynolds number\s*=\s*([\d.]+)", line)
            if match:
                re_tau = float(match.group(1))
                break

        # Find the line where the numerical data starts
        data_start = 0
        for i, line in enumerate(lines):
            # Find the first non-comment line that is not just whitespace
            if not line.strip().startswith('%') and line.strip():
                data_start = i
                break

        # Extract and parse the numerical data
        data_lines = lines[data_start:]
        data = []
        for line in data_lines:
            values = line.split()
            if values: # prevent empty lines from causing issues.
                try:
                    # Attempt to convert values to float
                    data.append([float(val) for val in values])
                except ValueError:
                    print(f"Warning: Skipping line with non-float values: {line.strip()}")
                    continue # Skip this line if conversion fails

        if not data:
             print(f"Error: No valid numerical data found in {filepath}")
             return None, None

        return re_tau, np.array(data)

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred while reading {filepath}: {e}")
        return None, None

# --- List of Reynolds numbers (Retau) to process ---
Re_all = [500, 1140, 2000, 3000, 6000]

# --- Lists to collect data from all Re_num cases ---
all_inputs_data = []
all_output_data = []
all_flow_type_data = []
all_unnormalized_inputs_data = []

print("Starting pipe flow data generation...")

# --- Loop through each Reynolds number ---
for Re in Re_all:
    print(f"\nProcessing Re_tau = {Re}...")

    # Construct the file path for the current Reynolds number
    filepath = os.path.join(currentpath , f'Pipe_profiles_Retau{Re}.dat')

    # Read the data for the current Reynolds number
    retau, data = read_pipe_data(filepath)

    if data is not None:
        if retau is None:
            print(f"Warning: Could not find Friction Reynolds number in {filepath}. Skipping.")
            continue

        print(f"Data loaded successfully for Re_tau = {retau}")

        # Extract columns based on typical pipe profile file format: y+, Uz+, uu+, vv+, ww+, uv+
        if data.shape[1] < 6:
             print(f"Error: Expected at least 6 columns in {filepath}, but found {data.shape[1]}. Skipping.")
             continue

        y_plus = data[:, 0] # Dimensionless wall distance y+
        Uz_plus = data[:, 1] # Mean axial velocity in wall units Uz/utau
        # Other columns (uu+, vv+, ww+, uv+) are available but not used in the current Pi formulation

        # Convert to physical units scaled by pipe radius R (assuming R=1 for normalization)
        # In this scaling, nu = 1/Retau and utau = 1
        # y = y+ * nu / utau = y+ * (1/Retau) / 1 = y+ / Retau
        # Uz = Uz+ * utau = Uz+ * 1 = Uz+
        # So, y in this script corresponds to y/R, and uu corresponds to U_axial * R / nu
        # Let's re-align the variable names to be less confusing and match the Pi definitions better.
        # The Pi definitions seem to use physical units scaled such that nu is explicit.
        # Let's assume the input data Uz_plus is U/utau and y_plus is y utau / nu.
        # We need U, y, nu, utau in consistent physical units.
        # Let's assume the data is normalized by utau and nu.
        # U_phys = Uz_plus * utau
        # y_phys = y_plus * nu / utau

        # However, the original script calculated pi_1 = uu * y, where uu = Uz_plus * retau and y = y_plus / retau.
        # This means pi_1 = (Uz_plus * retau) * (y_plus / retau) = Uz_plus * y_plus = (U/utau) * (y utau / nu) = U y / nu.
        # This matches the definition of pi_1.
        # The original script's variable names were a bit misleading. Let's stick to the Pi definitions.
        # Let's use y_phys and U_phys where 'phys' means consistent physical units where nu is explicit.
        # We can choose a scaling where utau = 1. Then nu = 1/Retau.
        utau_phys = 1.0 # Choose utau = 1
        nu_phys = 1.0 / retau # Then nu = 1/Retau
        y_phys = y_plus * nu_phys / utau_phys # y = y+ * nu / utau
        U_phys = Uz_plus * utau_phys # U = U+ * utau

        # Calculate dimensionless wall distance (y+) using the consistent physical units
        yplus_check = y_phys * utau_phys / nu_phys # Should recover y_plus

        # Filter data based on physical y range (scaled by R) and y+ > 50
        # The original script used y >= DOWN_FRAC and y <= UP_FRAC, where y = y_plus / retau.
        # Let's use y_phys >= DOWN_FRAC and y_phys <= UP_FRAC
        bot_index = np.where(((yplus_check > 50) & (y_phys >= DOWN_FRAC) & (y_phys <= UP_FRAC)))[0]

        if len(bot_index) == 0:
            print(f"No data points found in the specified range (y/R=[{DOWN_FRAC}, {UP_FRAC}] and y+ > 50) for Re_tau = {retau}")
            continue

        print(f"Selected {len(bot_index)} points for Re_tau = {retau}")

        # --- Loop through selected y-indices to create data points ---
        # Each selected index 'i' from bot_index corresponds to one row in the output DataFrames
        for i in bot_index:
            y_val_phys = y_phys[i] # The physical y-coordinate for this data point

            # Calculate velocity gradient profile in physical units
            # dU/dy in physical units: d(U+ * utau) / d(y+ * nu / utau) = (utau / (nu/utau)) * dU+/dy+ = (utau^2 / nu) * dU+/dy+
            # In our chosen scaling (utau=1, nu=1/Retau), this is (1 / (1/Retau)) * dU+/dy+ = Retau * dU+/dy+
            # The gradient of Uz_plus w.r.t y_plus is dU+/dy+
            dUplus_dyplus = np.gradient(Uz_plus, y_plus)
            dUdy_phys_profile = dUplus_dyplus * (utau_phys**2 / nu_phys) # Convert dU+/dy+ to dU/dy in physical units

            # Get velocity and gradient values at the required k*y locations using find_k_y_values
            # find_k_y_values interpolates the profile (U_phys or dUdy_phys_profile) w.r.t its grid (y_phys)
            # at the target y-values (y_val_phys * 2^k).
            U1_phys = U_phys[i] # U at y_val_phys (k=0)
            U2_phys = find_k_y_values(np.array([y_val_phys]), U_phys, y_phys, k=1)[0] # U at 2*y_val_phys (k=1)
            U3_phys = find_k_y_values(np.array([y_val_phys]), U_phys, y_phys, k=2)[0] # U at 4*y_val_phys (k=2)
            U4_phys = find_k_y_values(np.array([y_val_phys]), U_phys, y_phys, k=3)[0] # U at 8*y_val_phys (k=3)

            dudy1_phys = dUdy_phys_profile[i] # dU/dy at y_val_phys (k=0)
            dudy2_phys = find_k_y_values(np.array([y_val_phys]), dUdy_phys_profile, y_phys, k=1)[0] # dU/dy at 2*y_val_phys (k=1)
            dudy3_phys = find_k_y_values(np.array([y_val_phys]), dUdy_phys_profile, y_phys, k=2)[0] # dU/dy at 4*y_val_phys (k=2)

            # For pipe flow, there is a mean pressure gradient, but the Pi_2 and Pi_9 terms
            # relate to the *local* pressure gradient or normal pressure gradient.
            # For fully developed pipe flow, the pressure gradient is only in the axial direction (dP/dz).
            # The terms 'up' and 'upn' in the Pi groups are typically related to dP/dx and dP/dy.
            # In standard pipe flow analysis, these terms are zero or handled differently.
            # Based on the original script setting pi_2 and pi_9 to 0, we'll continue this.
            up_val_phys = 0.0
            upn_val_phys = 0.0
            # The axial pressure gradient dP/dz is related to tauw: dP/dz = -2*tauw/R.
            # In our scaling (utau=1, R=1), tauw = utau^2 = 1, so dP/dz = -2.
            # This axial pressure gradient is a global parameter, not a local 'up' or 'upn'.
            # We will keep up_val_phys and upn_val_phys as 0 to match the original script's Pi definitions.


            # --- Calculate Dimensionless Input Features (Pi Groups) for this y_val ---
            # Note: nu_phys and y_val_phys are the scaling factors for the Pi groups for this row.
            # Since pipe flow is homogeneous in x (axial direction) and theta (azimuthal),
            # the flow properties at x, x-2y, and x+2y for a given y are the same.
            # We use the calculated physical values and the physical nu for non-dimensionalization.
            pi_1 = U1_phys * y_val_phys / nu_phys
            pi_2 = up_val_phys * y_val_phys / nu_phys # Always 0 for pipe flow (based on original script)
            pi_3 = U2_phys * y_val_phys / nu_phys
            pi_4 = U3_phys * y_val_phys / nu_phys
            pi_5 = U4_phys * y_val_phys / nu_phys
            pi_6 = dudy1_phys * y_val_phys**2 / nu_phys
            pi_7 = dudy2_phys * y_val_phys**2 / nu_phys
            pi_8 = dudy3_phys * y_val_phys**2 / nu_phys
            pi_9 = upn_val_phys * y_val_phys / nu_phys # Always 0 for pipe flow (based on original script)

            # --- Structure Input Features with Stencil Locations ---
            # Create a dictionary for the inputs of this single row (corresponding to y_val)
            combined_inputs_dict = {}
            input_bases = [
                'u1_y_over_nu', 'up_y_over_nu', 'u2_y_over_nu', 'u3_y_over_nu',
                'u4_y_over_nu', 'dudy1_y_pow2_over_nu', 'dudy2_y_pow2_over_nu',
                'dudy3_y_pow2_over_nu', 'upn_y_over_nu',
            ]
            # Use specific suffixes or no suffix for 'current'
            locations = {
                'current': '', # No suffix for current
                'upstream': '_upstream',
                'downstream': '_downstream'
            }

            # Populate the combined inputs dictionary by replicating the values
            for base in input_bases:
                # Get the calculated value for this base input at the current y_val
                # Mapping base names to calculated Pi values
                if base == 'u1_y_over_nu': value = pi_1
                elif base == 'up_y_over_nu': value = pi_2
                elif base == 'u2_y_over_nu': value = pi_3
                elif base == 'u3_y_over_nu': value = pi_4
                elif base == 'u4_y_over_nu': value = pi_5
                elif base == 'dudy1_y_pow2_over_nu': value = pi_6
                elif base == 'dudy2_y_pow2_over_nu': value = pi_7
                elif base == 'dudy3_y_pow2_over_nu': value = pi_8
                elif base == 'upn_y_over_nu': value = pi_9
                else: value = np.nan # Should not happen

                # Assign the same value to all three stencil locations with appropriate suffix
                for loc_name, suffix in locations.items():
                    combined_inputs_dict[f'{base}{suffix}'] = value

            all_inputs_data.append(combined_inputs_dict) # Append dictionary for this row

            # --- Structure Unnormalized Inputs with Stencil Locations ---
            combined_unnorm_dict = {}
            unnorm_bases = [
                 'y', 'u1', 'nu', 'utau', 'up', 'upn', 'u2', 'u3', 'u4', 'dudy1', 'dudy2', 'dudy3', 'x'
            ]

            # Populate the combined unnormalized dictionary by replicating the values
            for base in unnorm_bases:
                 # Get the calculated value for this base unnormalized input at the current y_val
                 if base == 'y': value = y_val_phys
                 elif base == 'u1': value = U1_phys
                 elif base == 'nu': value = nu_phys
                 elif base == 'utau': value = utau_phys # utau is 1.0 in this scaling
                 elif base == 'up': value = up_val_phys # 0.0
                 elif base == 'upn': value = upn_val_phys # 0.0
                 elif base == 'u2': value = U2_phys
                 elif base == 'u3': value = U3_phys
                 elif base == 'u4': value = U4_phys
                 elif base == 'dudy1': value = dudy1_phys
                 elif base == 'dudy2': value = dudy2_phys
                 elif base == 'dudy3': value = dudy3_phys
                 elif base == 'x': value = 0.0 # x is 0 for pipe flow (no streamwise variation)
                 else: value = np.nan # Should not happen

                 # Assign the same value to all three stencil locations with appropriate suffix
                 for loc_name, suffix in locations.items():
                      combined_unnorm_dict[f'{base}{suffix}'] = value

            all_unnormalized_inputs_data.append(combined_unnorm_dict) # Append dictionary for this row


            # --- Calculate Output Feature for this y_val ---
            # Output is y+ (utau * y / nu)
            # Using the physical units and definitions: y+ = y_phys * utau_phys / nu_phys
            output_dict = {
                'utau_y_over_nu': y_val_phys * utau_phys / nu_phys # This is y+
            }
            all_output_data.append(output_dict) # Append dictionary for this row

            # --- Collect Flow Type Information for this y_val ---
            # Using format: [case_name, reference_nu, x_coord, delta, edge_velocity]
            # For pipe flow: x=0, delta=1 (pipe radius R), Ue=0 (or U_bulk if needed)
            flow_type_dict = {
                'case_name': 'pipe',
                'nu': nu_phys, # Using nu_phys as reference value
                'x': 0.0, # No streamwise variation for pipe flow
                'delta': 1.0, # Using pipe radius R = 1 in this scaling
                'Retau': retau # Include Retau for reference
            }
            all_flow_type_data.append(flow_type_dict) # Append dictionary for this row


    else:
        # read_pipe_data returned None, indicating an error
        print(f"Skipping Re_tau = {Re} due to data loading error.")


if not all_inputs_data:
    print("No data was generated for any Reynolds number. Check input Reynolds numbers and data loading.")
else:
    # Concatenate data from all Re_num cases into single DataFrames
    print(f"\nConcatenating data from {len(Re_all)} Reynolds numbers...")
    inputs_df = pd.DataFrame(all_inputs_data)
    output_df = pd.DataFrame(all_output_data)
    flow_type_df = pd.DataFrame(all_flow_type_data)
    unnormalized_inputs_df = pd.DataFrame(all_unnormalized_inputs_data)

    # --- Filter rows with NaN in inputs_df ---
    # Although we don't expect NaNs with this logic (unless find_k_y_values returns NaN),
    # we keep the filtering step as a safety measure.
    print(f"Original shapes before NaN filter: Inputs={inputs_df.shape}, Output={output_df.shape}, Flow Type={flow_type_df.shape}, Unnormalized Inputs={unnormalized_inputs_df.shape}")
    rows_before_filter = inputs_df.shape[0]

    # Find rows where NO element is NaN in the inputs DataFrame
    valid_rows_mask = inputs_df.notna().all(axis=1)

    # Use the boolean mask to select only the valid rows from all DataFrames
    inputs_df_filtered = inputs_df[valid_rows_mask].copy()
    output_df_filtered = output_df[valid_rows_mask].copy()
    flow_type_df_filtered = flow_type_df[valid_rows_mask].copy()
    unnormalized_inputs_df_filtered = unnormalized_inputs_df[valid_rows_mask].copy()

    rows_after_filter = inputs_df_filtered.shape[0]
    print(f"Filtered out {rows_before_filter - rows_after_filter} rows containing NaN in inputs.")
    print(f"Filtered shapes: Inputs={inputs_df_filtered.shape}, Output={output_df_filtered.shape}, Flow Type={flow_type_df_filtered.shape}, Unnormalized Inputs={unnormalized_inputs_df_filtered.shape}")

    # Reorder unnorm columns before output
    ordered_unnorm_bases = unnorm_bases.copy()
    for i in ['_upstream', '_downstream']:
        ordered_unnorm_bases += [f"{base}{i}" for base in unnorm_bases]
    unnormalized_inputs_df_filtered = unnormalized_inputs_df_filtered[ordered_unnorm_bases]


    # Ensure output directory exists
    os.makedirs(savedatapath, exist_ok=True)

    # Save DataFrames to HDF5 file
    # Use a filename indicating pipe flow and the stencil structure
    output_filename = os.path.join(savedatapath, "PIPE_data_stencils.h5")
    print(f"\nSaving filtered data to HDF5 file: {output_filename}")
    try:
        # Use fixed format for better performance with numerical data
        inputs_df_filtered.to_hdf(output_filename, key='inputs', mode='w', format='fixed')
        output_df_filtered.to_hdf(output_filename, key='output', mode='a', format='fixed')
        unnormalized_inputs_df_filtered.to_hdf(output_filename, key='unnormalized_inputs', mode='a', format='fixed')
        # Use table format for flow_type if it contains strings, to keep them
        flow_type_df_filtered.to_hdf(output_filename, key='flow_type', mode='a', format='table')
        print("Data successfully saved.")
    except Exception as e:
        print(f"Error saving HDF5 file: {e}")
        # Consider cleaning up partially written files if save fails
        pass # Or handle error appropriately

    # Print final summary shapes
    print(f"Final Filtered Shapes:")
    print(f"  Inputs: {inputs_df_filtered.shape}")
    print(f"  Output: {output_df_filtered.shape}")
    print(f"  Flow Type: {flow_type_df_filtered.shape}")
    print(f"  Unnormalized Inputs: {unnormalized_inputs_df_filtered.shape}")

# --- Execution Example ---
if __name__ == "__main__":
    # The main execution is handled by the script flow above.
    # You can add specific calls here if you want to run with different parameters
    # or perform post-processing on the saved file.
    print("\nScript finished.")

    # Example of how to load the saved data:
    # try:
    #     output_filename = os.path.join(savedatapath, "PIPE_data_stencils.h5")
    #     print(f"\nLoading saved data from {output_filename} for verification...")
    #     inputs_loaded = pd.read_hdf(output_filename, key='inputs')
    #     output_loaded = pd.read_hdf(output_filename, key='output')
    #     flow_type_loaded = pd.read_hdf(output_filename, key='flow_type')
    #     unnormalized_inputs_loaded = pd.read_hdf(output_filename, key='unnormalized_inputs')
    #     print("Data loaded successfully.")
    #     print(f"Loaded Inputs shape: {inputs_loaded.shape}")
    #     # print first few rows
    #     # print("\nFirst 5 rows of loaded Inputs:")
    #     # print(inputs_loaded.head())
    # except FileNotFoundError:
    #     print(f"Error: Saved file not found at {output_filename}")
    # except Exception as e:
    #     print(f"Error loading saved HDF5 file: {e}")

