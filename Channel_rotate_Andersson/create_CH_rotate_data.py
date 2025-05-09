import os
import numpy as np
import pandas as pd

from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
import common

# Parameters used by Andersson et al. 1993
Retau = 194
utau = 1 # The unrotated case has utau = dp/dx = 1
nu = utau / Retau  # KTH data uses nu = 1/Re_tau

# Ro = [0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.50]
Ro = ["00", "01", "05", "10", "15", "20", "50"]

utau_dict = {}

utau_dict["00"] = [1.002, 1.004]
utau_dict["01"] = [1.023, 0.970]
utau_dict["05"] = [1.140, 0.805]
utau_dict["10"] = [1.185, 0.760]
utau_dict["15"] = [1.213, 0.754]
utau_dict["20"] = [1.217, 0.707]
utau_dict["50"] = [1.207, 0.679]


# --- Main Data Creation Function ---
def create_channel_flow_data_hdf5(RO_NUMS, output_dir=os.path.join(WM_DATA_PATH, "data"), Verbose=False):
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
    for Ro_num in RO_NUMS:
        if Verbose:
            print(f"\nProcessing Ro = {Ro_num}...")

        data = np.loadtxt(os.path.join(WM_DATA_PATH, "Channel_rotate_Andersson", "stats", f"f2_r_{Ro_num}.dat"), unpack=True)
        y = data[0, :]
        Udns = data[1, :]

        # Remove indices where Udns == 0
        y = y[Udns != 0]
        Udns = Udns[Udns != 0]

        y_top = 1 - y[y > 0]
        Udns_top = Udns[y > 0]
        Retau_top = utau_dict[Ro_num][0] * Retau  # Retau for the top half
        utau_top = Retau_top * nu  # Calculate utau for the top half

        y_bottom = y[y < 0] + 1
        Udns_bottom = Udns[y < 0]
        Retau_bottom = utau_dict[Ro_num][1] * Retau  # Retau for the top half
        utau_bottom = Retau_bottom * nu  # Calculate utau for the bottom half

        # NOTE: Do separate calculations for top and bottom halves
        # Get lazy so just copy and paste...
        #
        # NOTE: 1. Bottom half
        # Calculate dimensionless wall distance and filter data based on physical y range
        yplus = y_bottom * Retau_bottom
        # Select indices within the desired physical y range and where y+ > 50 (log-law region assumption)
        bot_index = np.where(((y_bottom >= y_lower_limit) & (y_bottom <= y_up_limit) ))[0]
        if Verbose:
            print(f"Selected {len(bot_index)} points for Ro = {Ro} (Bottom half)")

        # Extract data for the selected indices
        y_sel = y_bottom[bot_index]
        U_sel = Udns_bottom[bot_index]
        nu_sel = np.full_like(y_sel, nu) # nu is constant for the case

        # --- Calculate Input Features (Pi Groups) ---
        # Note: Channel flow has dPdx = - utau ** 2
        dPdx = - utau_bottom  # Here is (utau * 1 / nu) ** 2; nu will be cancel out for the pi groups
        up = np.sign(dPdx) * np.sqrt(np.abs(dPdx) * nu_sel)**1/3  # Up is the velocity gradient at the wall

        # Interpolate velocities at required k*y locations
        U2 = find_k_y_values(y_sel, Udns_bottom, y_bottom, k=1)
        U3 = find_k_y_values(y_sel, Udns_bottom, y_bottom, k=2)
        U4 = find_k_y_values(y_sel, Udns_bottom, y_bottom, k=3)

        # Calculate velocity gradient and interpolate
        dUdy_bottom = np.gradient(Udns_bottom, y_bottom)
        dudy1 = dUdy_bottom[bot_index]
        dudy2 = find_k_y_values(y_sel, dUdy_bottom, y_bottom, k=1)
        dudy3 = find_k_y_values(y_sel, dUdy_bottom, y_bottom, k=2)

        # Calculate dimensionless inputs using safe names
        inputs_dict = {
            'u1_y_over_nu': U_sel * y_sel / nu_sel,          # pi_1
            'up_y_over_nu': up * y_sel / nu_sel,     # pi_2 (is zero for CH)
            'upn_y_over_nu': np.full_like(y_sel, 0),     # pi_2 (is zero for CH)
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
            'utau_y_over_nu': utau_bottom * y_sel / nu_sel
        }
        all_output_data.append(pd.DataFrame(output_dict))

        # --- Collect Unnormalized Inputs ---
        unnorm_dict = {
            'y': y_sel,
            'u1': U_sel,
            'nu': nu_sel,
            'utau': np.full_like(y_sel, utau),
            'up': np.full_like(y_sel, up), # dPdx is 0 for channel flow
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
            'case_name': [f'Rot_Ro{Ro_num}_bot'] * len(y_sel),
            'nu': [nu] * len(y_sel), # Using nu as reference value
            'x': [0] * len(y_sel),       # No streamwise variation for channel flow
            'delta': [1.0] * len(y_sel),       # Using half-channel height = 1
        }
        # Add Retau for reference if needed, maybe as an extra column or replacing 'edge_velocity'
        flow_type_dict['Retau'] = [Retau_bottom] * len(y_sel)
        all_flow_type_data.append(pd.DataFrame(flow_type_dict))

        # NOTE: 2. Top half
        # Calculate dimensionless wall distance and filter data based on physical y range
        yplus = y_top * Retau_top
        # Select indices within the desired physical y range and where y+ > 50 (log-law region assumption)
        bot_index = np.where(((y_top >= y_lower_limit) & (y_top <= y_up_limit) ))[0]
        if Verbose:
            print(f"Selected {len(bot_index)} points for Ro = {Ro} (top half)")

        # Extract data for the selected indices
        y_sel = y_top[bot_index]
        U_sel = Udns_top[bot_index]
        nu_sel = np.full_like(y_sel, nu) # nu is constant for the case

        # --- Calculate Input Features (Pi Groups) ---
        # Note: Channel flow has dPdx = - utau ** 2
        dPdx = - utau_top  # Here is (utau * 1 / nu) ** 2; nu will be cancel out for the pi groups
        up = np.sign(dPdx) * np.sqrt(np.abs(dPdx) * nu_sel)**1/3  # Up is the velocity gradient at the wall

        # Interpolate velocities at required k*y locations
        U2 = find_k_y_values(y_sel, Udns_top, y_top, k=1)
        U3 = find_k_y_values(y_sel, Udns_top, y_top, k=2)
        U4 = find_k_y_values(y_sel, Udns_top, y_top, k=3)

        # Calculate velocity gradient and interpolate
        dUdy_top = np.gradient(Udns_top, y_top)
        dudy1 = dUdy_top[bot_index]
        dudy2 = find_k_y_values(y_sel, dUdy_top, y_top, k=1)
        dudy3 = find_k_y_values(y_sel, dUdy_top, y_top, k=2)

        # Calculate dimensionless inputs using safe names
        inputs_dict = {
            'u1_y_over_nu': U_sel * y_sel / nu_sel,          # pi_1
            'up_y_over_nu': up * y_sel / nu_sel,     # pi_2 (is zero for CH)
            'upn_y_over_nu': np.full_like(y_sel, 0),     # pi_2 (is zero for CH)
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
            'utau_y_over_nu': utau_top * y_sel / nu_sel
        }
        all_output_data.append(pd.DataFrame(output_dict))

        # --- Collect Unnormalized Inputs ---
        unnorm_dict = {
            'y': y_sel,
            'u1': U_sel,
            'nu': nu_sel,
            'utau': np.full_like(y_sel, utau),
            'up': np.full_like(y_sel, up), # dPdx is 0 for channel flow
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
            'case_name': [f'Rot_Ro{Ro_num}_bot'] * len(y_sel),
            'nu': [nu] * len(y_sel), # Using nu as reference value
            'x': [0] * len(y_sel),       # No streamwise variation for channel flow
            'delta': [1.0] * len(y_sel),       # Using half-channel height = 1
        }
        # Add Retau for reference if needed, maybe as an extra column or replacing 'edge_velocity'
        flow_type_dict['Retau'] = [Retau_top] * len(y_sel)
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
    output_filename = os.path.join(output_dir, "CH_rot_data.h5")
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

saved_file_path = create_channel_flow_data_hdf5(Ro, Verbose=True)

if saved_file_path:
    print(f"\nTo load the data later:")
    print(f"import pandas as pd")
    print(f"inputs_loaded = pd.read_hdf('{saved_file_path}', key='inputs')")
    print(f"output_loaded = pd.read_hdf('{saved_file_path}', key='output')")
    print(f"# etc.")
