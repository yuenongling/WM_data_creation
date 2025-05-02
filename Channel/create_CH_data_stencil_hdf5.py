'''
Stencil version of channel data creation
'''
import os
import numpy as np
import pandas as pd

# Assuming these imports work based on your project structure
# data_processing_utils is expected to contain find_k_y_values and import_path
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
import common

Retau_v = [180,    550,   950,  2000,   4200,  5200, 10000] # list of Re_tau
inu_v   = [3250, 11180, 20580, 48500, 112500, 125000, 290000] # inverse of nu
utau_v  = [0.057231059, 0.048904658, 0.045390026, 0.04130203, 0.0371456, 4.14872e-02, 0.034637]

Re_dict = {}
for i, Re in enumerate(Retau_v):
    Re_dict[Re] = {
        'inu': inu_v[i],
        'utau': utau_v[i]
    }

# --- Set up paths ---
# WM_DATA_PATH should be the base path where your simulation data is stored
# import_path() is assumed to return this base path
WM_DATA_PATH = import_path()
# Define the output directory for the HDF5 file
output_dir = os.path.join(WM_DATA_PATH, "data")

# --- Helper function to sample value from a 1D profile along y ---
def sample_y_profile(y_profile, y_grid, target_y):
    """
    Samples a value from a 1D profile (defined on y_grid) at a target y-value.
    Uses linear interpolation.
    """
    if y_grid.size == 0 or y_profile.size == 0:
        return np.nan
    # np.interp requires y_grid to be increasing. Assume ym is increasing.
    return np.interp(target_y, y_grid, y_profile)

# --- Helper function to check if an x-index passes the original filtering ---
def passes_original_filter_by_index(check_idx, angle, xm, Cf, n_diff):
    """
    Checks if a given x-index corresponds to a location that would have
    been selected by the original filtering logic.
    """
    # This function is not strictly needed for channel flow data generation
    # because channel flow is homogeneous in x and we don't filter based on x-location
    # for the 'current' point. However, keeping it for consistency if this function
    # were to be adapted for other flow types or more complex filtering.
    # For channel flow, this will always return True if check_idx is valid.
    if not (0 <= check_idx < len(xm)):
        return False # Invalid index

    # For channel flow, x range and Cf filters are not applicable in the same way
    # as TBL. We'll simplify this for CH flow: if the index is valid, assume it passes.
    # If this function were used for TBL, the full logic would be needed.
    return True


# --- Main Data Creation Function ---
def create_channel_flow_data_hdf5(RE_NUMS, output_dir=output_dir, Verbose=False):
    """
    Generates input and output data for channel flow with replicated stencil
    locations (current, upstream, downstream) due to streamwise homogeneity,
    and saves it into an HDF5 file using Pandas DataFrames.

    Args:
        RE_NUMS (list or array): List of Reynolds numbers (Retau) to process.
        output_dir (str): Directory to save the HDF5 file.
        Verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        str: Path to the saved HDF5 file.
    """
    # Constants for data selection (in physical units, scaled by half-channel height)
    # These limits define the y-range from which points are selected for each Re_tau case.
    y_lower_limit = 0.025
    y_up_limit = 0.2

    # Lists to collect data from all Re_num cases
    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []

    print("Starting channel flow data generation...")
    for Re_num in RE_NUMS:
        if Verbose:
            print(f"\nProcessing Re_tau = {Re_num}...")

        # Load raw data using provided utility functions
        utau = Re_dict[Re_num]['utau']
        nu   = 1/Re_dict[Re_num]['inu'] # Inverse of nu
        y, Udns_wall_units = common.read_ch_prof(Re_num)

        # Remove the first element if y is zero or U is zero, as it can cause issues with log scale or division
        if Udns_wall_units[0] == 0 or y[0] == 0:
            y = y[1:]
            Udns_wall_units = Udns_wall_units[1:]

        # Calculate friction velocity and convert mean velocity to physical units
        Udns_phys = Udns_wall_units * utau # Convert from wall units to physical velocity

        # Calculate dimensionless wall distance (y+)
        yplus = y * utau / nu

        # Select indices within the desired physical y range AND where y+ > 50
        # The y+ > 50 condition is often used to select points assumed to be in the log-law region
        bot_index = np.where(((y >= y_lower_limit) & (y <= y_up_limit) & (yplus > 50)))[0]

        if len(bot_index) == 0:
            if Verbose:
                print(f"No data points found in the specified range (y=[{y_lower_limit}, {y_up_limit}] and y+ > 50) for Re_tau = {Re_num}")
            continue

        if Verbose:
            print(f"Selected {len(bot_index)} points for Re_tau = {Re_num}")

        # --- Loop through selected y-indices to create data points ---
        # Each selected y-index corresponds to one row in the output DataFrames
        for i in bot_index:
            y_val = y[i] # The physical y-coordinate for this data point

            # Calculate velocity gradient profile in physical units
            dUdy_phys = np.gradient(Udns_phys, y)

            # Get velocity and gradient values at the required k*y locations using find_k_y_values
            # find_k_y_values is assumed to interpolate Udns_phys and dUdy_phys profiles at y_val * 2^k
            U1 = Udns_phys[i] # U at y_val (k=0)
            U2 = find_k_y_values(np.array([y_val]), Udns_phys, y, k=1)[0] # U at 2*y_val (k=1)
            U3 = find_k_y_values(np.array([y_val]), Udns_phys, y, k=2)[0] # U at 4*y_val (k=2)
            U4 = find_k_y_values(np.array([y_val]), Udns_phys, y, k=3)[0] # U at 8*y_val (k=3)

            dudy1 = dUdy_phys[i] # dU/dy at y_val (k=0)
            dudy2 = find_k_y_values(np.array([y_val]), dUdy_phys, y, k=1)[0] # dU/dy at 2*y_val (k=1)
            dudy3 = find_k_y_values(np.array([y_val]), dUdy_phys, y, k=2)[0] # dU/dy at 4*y_val (k=2)

            # For channel flow, dPdx = 0, so up and upn are 0
            dPdx = 0
            up_val = 0.0
            upn_val = 0.0

            # --- Calculate Dimensionless Input Features (Pi Groups) for this y_val ---
            # These are the base values calculated at the current y-location.
            # Since channel flow is homogeneous in x, these values are the same
            # for upstream and downstream locations at the same y.
            pi_1 = U1 * y_val / nu
            pi_2 = up_val * y_val / nu # Always 0 for channel flow
            pi_3 = U2 * y_val / nu
            pi_4 = U3 * y_val / nu
            pi_5 = U4 * y_val / nu
            pi_6 = dudy1 * y_val**2 / nu
            pi_7 = dudy2 * y_val**2 / nu
            pi_8 = dudy3 * y_val**2 / nu
            pi_9 = upn_val * y_val / nu # Always 0 for channel flow

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
                if base == 'u1_y_over_nu': value = pi_1
                elif base == 'up_y_over_nu': value = pi_2
                elif base == 'u2_y_over_nu': value = pi_3
                elif base == 'u3_y_over_nu': value = pi_4
                elif base == 'u4_y_over_nu': value = pi_5
                elif base == 'dudy1_y_pow2_over_nu': value = pi_6
                elif base == 'dudy2_y_pow2_over_nu': value = pi_7
                elif base == 'dudy3_y_pow2_over_nu': value = pi_8
                elif base == 'upn_y_over_nu': value = pi_9
                else: value = np.nan # Should not happen with defined bases

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
                 if base == 'y': value = y_val
                 elif base == 'u1': value = U1
                 elif base == 'nu': value = nu
                 elif base == 'utau': value = utau
                 elif base == 'up': value = up_val
                 elif base == 'upn': value = upn_val
                 elif base == 'u2': value = U2
                 elif base == 'u3': value = U3
                 elif base == 'u4': value = U4
                 elif base == 'dudy1': value = dudy1
                 elif base == 'dudy2': value = dudy2
                 elif base == 'dudy3': value = dudy3
                 elif base == 'x': value = 0.0 # x is 0 for channel flow
                 else: value = np.nan # Should not happen

                 # Assign the same value to all three stencil locations with appropriate suffix
                 for loc_name, suffix in locations.items():
                      combined_unnorm_dict[f'{base}{suffix}'] = value

            all_unnormalized_inputs_data.append(combined_unnorm_dict) # Append dictionary for this row


            # --- Calculate Output Feature for this y_val ---
            # Output is y+ (utau * y / nu)
            output_dict = {
                'utau_y_over_nu': utau * y_val / nu
            }
            all_output_data.append(output_dict) # Append dictionary for this row

            # --- Collect Flow Type Information for this y_val ---
            flow_type_dict = {
                'case_name': 'Channel',
                'nu': nu, # Using nu as reference value
                'x': 0.0, # No streamwise variation for channel flow
                'delta': 1.0, # Using half-channel height = 1 as reference length
                'Retau': Re_num # Include Retau for reference
            }
            all_flow_type_data.append(flow_type_dict) # Append dictionary for this row


    if not all_inputs_data:
        print("No data was generated. Check input Reynolds numbers and data loading.")
        return None

    # Concatenate data from all Re_num cases into single DataFrames
    print(f"\nConcatenating data from {len(RE_NUMS)} Reynolds numbers...")
    inputs_df = pd.DataFrame(all_inputs_data)
    output_df = pd.DataFrame(all_output_data)
    flow_type_df = pd.DataFrame(all_flow_type_data)
    unnormalized_inputs_df = pd.DataFrame(all_unnormalized_inputs_data)

    # --- Filter rows with NaN in inputs_df ---
    # Although we don't expect NaNs with this logic (unless find_k_y_values returns NaN),
    # we keep the filtering step as a safety measure and for consistency with the TBL script.
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
    os.makedirs(output_dir, exist_ok=True)

    # Save DataFrames to HDF5 file
    # Use a filename indicating channel flow and the stencil structure
    output_filename = os.path.join(output_dir, "CH_data_stencils.h5")
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
        return None

    # Print final summary shapes
    print(f"Final Filtered Shapes:")
    print(f"  Inputs: {inputs_df_filtered.shape}")
    print(f"  Output: {output_df_filtered.shape}")
    print(f"  Flow Type: {flow_type_df_filtered.shape}")
    print(f"  Unnormalized Inputs: {unnormalized_inputs_df_filtered.shape}")

    return output_filename

# --- Execution Example ---
if __name__ == "__main__":
    # Example Reynolds numbers (Retau) for channel flow
    RE_NUMS_CHANNEL = [550, 950, 2000, 4200, 5200, 10000]
    saved_file_path = create_channel_flow_data_hdf5(RE_NUMS_CHANNEL, Verbose=True)

    if saved_file_path:
        print(f"\nTo load the data later:")
        print(f"import pandas as pd")
        print(f"inputs_loaded = pd.read_hdf('{saved_file_path}', key='inputs')")
        print(f"output_loaded = pd.read_hdf('{saved_file_path}', key='output')")
        print(f"# etc.")

