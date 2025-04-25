import os
import numpy as np
import pandas as pd

from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
import utils  # Assumes 'utils' module is accessible from '../NNOpt/'
import common # Assumes 'common' module is accessible from '../NNOpt/post_proc/'

# --- Main Data Creation Function ---
def create_channel_flow_data_hdf5(RE_NUMS, output_dir=os.path.join(WM_DATA_PATH, "data"), Verbose=False):
    """
    Generates input and output data for channel flow, mimicking the KTH data
    creation process, and saves it into an HDF5 file using Pandas DataFrames
    with explicit, safe variable names.

    Args:
        RE_NUMS (list or array): List of Reynolds numbers (Retau) to process.
        output_dir (str): Directory to save the HDF5 file.
        Verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        str: Path to the saved HDF5 file.
    """
    # Constants for data selection
    y_lower_limit = 0.025 # Lower boundary layer limit (in physical units)
    y_up_limit = 0.2    # Upper boundary layer limit (in physical units)

    # Lists to collect data from all Re_num cases
    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []

    print("Starting data generation...")
    for Re_num in RE_NUMS:
        if Verbose:
            print(f"\nProcessing Re_tau = {Re_num}...")

        # Load raw data using provided utility functions
        try:
            nu, [tauw, _], _ = utils.get_data(Re=Re_num) # Assuming utils.get_data provides nu and tauw
            y, Udns = common.read_ch_prof(Re_num) # Assuming common.read_ch_prof provides y and U (in wall units)
        except Exception as e:
            print(f"Error loading data for Re_tau={Re_num}: {e}")
            continue

        # Remove the first element if it's zero (common issue in profile data)
        if Udns[0] == 0 or y[0] == 0:
            y = y[1:]
            Udns = Udns[1:]

        # Calculate friction velocity and scale velocity to physical units
        utau = np.sqrt(abs(tauw)).squeeze()
        Udns_phys = Udns * utau # Convert from wall units to physical velocity

        # Calculate dimensionless wall distance and filter data based on physical y range
        yplus = y * utau / nu
        # Select indices within the desired physical y range and where y+ > 50 (log-law region assumption)
        bot_index = np.where(((y >= y_lower_limit) & (y <= y_up_limit) & (yplus > 50)))[0]

        if len(bot_index) == 0:
            if Verbose:
                print(f"No data points found in the specified range for Re_tau = {Re_num}")
            continue

        if Verbose:
            print(f"Selected {len(bot_index)} points for Re_tau = {Re_num}")

        # Extract data for the selected indices
        y_sel = y[bot_index]
        U_sel = Udns_phys[bot_index]
        nu_sel = np.full_like(y_sel, nu) # nu is constant for the case

        # --- Calculate Input Features (Pi Groups) ---
        # Note: Channel flow has dPdx = 0
        dPdx = 0

        # Interpolate velocities at required k*y locations
        U2 = find_k_y_values(y_sel, Udns_phys, y, k=1)
        U3 = find_k_y_values(y_sel, Udns_phys, y, k=2)
        U4 = find_k_y_values(y_sel, Udns_phys, y, k=3)

        # Calculate velocity gradient and interpolate
        dUdy_phys = np.gradient(Udns_phys, y)
        dudy1 = dUdy_phys[bot_index]
        dudy2 = find_k_y_values(y_sel, dUdy_phys, y, k=1)
        dudy3 = find_k_y_values(y_sel, dUdy_phys, y, k=2)

        # Calculate dimensionless inputs using safe names
        inputs_dict = {
            'u1_y_over_nu': U_sel * y_sel / nu_sel,          # pi_1
            'up_y_over_nu': np.full_like(y_sel, dPdx),     # pi_2 (is zero for CH)
            'upn_y_over_nu': np.full_like(y_sel, dPdx),     # pi_2 (is zero for CH)
            'u2_y_over_nu': U2 * y_sel / nu_sel,             # pi_3
            'u3_y_over_nu': U3 * y_sel / nu_sel,             # pi_4
            'u4_y_over_nu': U4 * y_sel / nu_sel,             # pi_5
            'dudy1_y_pow2_over_nu': dudy1 * y_sel**2 / nu_sel,   # pi_6
            'dudy2_y_pow2_over_nu': dudy2 * y_sel**2 / nu_sel,   # pi_7
            'dudy3_y_pow2_over_nu': dudy3 * y_sel**2 / nu_sel    # pi_8
        }
        all_inputs_data.append(pd.DataFrame(inputs_dict))

        # --- Calculate Output Feature ---
        # Output is y+ (utau * y / nu)
        output_dict = {
            'utau_y_over_nu': utau * y_sel / nu_sel
        }
        all_output_data.append(pd.DataFrame(output_dict))

        # --- Collect Unnormalized Inputs ---
        unnorm_dict = {
            'y': y_sel,
            'u1': U_sel,
            'nu': nu_sel,
            'utau': np.full_like(y_sel, utau),
            'up': np.full_like(y_sel, dPdx), # dPdx is 0 for channel flow
            'upn': np.full_like(y_sel, dPdx), # dPdx is 0 for channel flow
            'u2': U2,
            'u3': U3,
            'u4': U4,
            'dudy1': dudy1,
            'dudy2': dudy2,
            'dudy3': dudy3
        }
        all_unnormalized_inputs_data.append(pd.DataFrame(unnorm_dict))

        # --- Collect Flow Type Information ---
        # Using format: [case_name, reference_nu, x_coord, delta, edge_velocity]
        # For channel flow: x=0, delta=1 (half-channel height), Ue=0 (or U_bulk if needed)
        flow_type_dict = {
            'case_name': ['Channel'] * len(y_sel),
            'nu': [nu] * len(y_sel), # Using nu as reference value
            'x': [0] * len(y_sel),       # No streamwise variation for channel flow
            'delta': [1.0] * len(y_sel),       # Using half-channel height = 1
        }
        # Add Retau for reference if needed, maybe as an extra column or replacing 'edge_velocity'
        flow_type_dict['Retau'] = [Re_num] * len(y_sel)
        all_flow_type_data.append(pd.DataFrame(flow_type_dict))

    if not all_inputs_data:
        print("No data was generated. Check input Reynolds numbers and data loading.")
        return None

    # Concatenate data from all Re_num cases into single DataFrames
    inputs_df = pd.concat(all_inputs_data, ignore_index=True)
    output_df = pd.concat(all_output_data, ignore_index=True)
    flow_type_df = pd.concat(all_flow_type_data, ignore_index=True)
    unnormalized_inputs_df = pd.concat(all_unnormalized_inputs_data, ignore_index=True)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save DataFrames to HDF5 file
    output_filename = os.path.join(output_dir, "CH_data.h5")
    print(f"\nSaving data to HDF5 file: {output_filename}")
    try:
        # Use fixed format for better performance with numerical data
        inputs_df.to_hdf(output_filename, key='inputs', mode='w', format='fixed')
        output_df.to_hdf(output_filename, key='output', mode='a', format='fixed')
        unnormalized_inputs_df.to_hdf(output_filename, key='unnormalized_inputs', mode='a', format='fixed')
        # Use table format for flow_type if it contains strings, to keep them
        flow_type_df.to_hdf(output_filename, key='flow_type', mode='a', format='table')
        print("Data successfully saved.")
    except Exception as e:
        print(f"Error saving HDF5 file: {e}")
        return None

    # Print summary shapes
    print(f"Final Shapes:")
    print(f"  Inputs: {inputs_df.shape}")
    print(f"  Output: {output_df.shape}")
    print(f"  Flow Type: {flow_type_df.shape}")
    print(f"  Unnormalized Inputs: {unnormalized_inputs_df.shape}")

    return output_filename

# --- Execution Example ---
if __name__ == "__main__":
    RE_NUMS_CHANNEL = [550, 950, 2000, 4200] # Example Reynolds numbers (Retau)
    saved_file_path = create_channel_flow_data_hdf5(RE_NUMS_CHANNEL, Verbose=True)

    if saved_file_path:
        print(f"\nTo load the data later:")
        print(f"import pandas as pd")
        print(f"inputs_loaded = pd.read_hdf('{saved_file_path}', key='inputs')")
        print(f"output_loaded = pd.read_hdf('{saved_file_path}', key='output')")
        print(f"# etc.")
