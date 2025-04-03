import numpy as np
import pandas as pd
import os

from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path

# --- Main Data Creation Function ---
def create_synthetic_data_hdf5(RE_NUMS, output_dir=os.path.join(WM_DATA_PATH, "data"), n_points=100, Verbose=False):
    """
    Generates synthetic input and output data based on the log-law,
    mimicking the structure of create_channel_flow_data_hdf5, and saves
    it into an HDF5 file using Pandas DataFrames with explicit, safe variable names.

    Args:
        RE_NUMS (list or array): List of Reynolds numbers (Retau) to process.
        output_dir (str): Directory to save the HDF5 file.
        n_points (int): Number of points to generate per Re_num.
        Verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        str: Path to the saved HDF5 file or None if failed.
    """
    # Constants for log-law and data selection
    kappa = 0.41
    B = 5.2
    y_lower_limit = 0.05 # Lower boundary layer limit (physical units) - adjust if needed
    y_up_limit = 0.15    # Upper boundary layer limit (physical units) - adjust if needed
    yplus_min_limit = 50 # Minimum y+ to consider (log-law region assumption)

    # Lists to collect data from all Re_num cases
    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []

    print("Starting synthetic data generation...")
    for Re_num in RE_NUMS:
        if Verbose:
            print(f"\nProcessing Re_tau = {Re_num}...")

        # --- Generate Synthetic Profile ---
        # Define nu based on Re_tau (assuming delta=1 for simplicity like channel flow)
        # Re_tau = utau * delta / nu => utau = Re_tau * nu / delta
        # We need to fix nu or utau. Let's fix nu and calculate utau.
        # Using a fixed nu consistent with the CH example script.
        # Alternatively, derive nu from Re_num assuming utau=1, etc.
        nu = 8e-6 # Fix kinematic viscosity (adjust if needed)
        utau = Re_num * nu # Calculate friction velocity (assuming delta=1)

        # Generate y points logarithmically within the physical limits
        y = np.logspace(np.log10(y_lower_limit), np.log10(y_up_limit), n_points)

        # Calculate y+
        yplus = y * utau / nu

        # Generate velocity using log law (in physical units)
        Ulog_plus = (1 / kappa) * np.log(yplus) + B
        Udns_phys = Ulog_plus * utau

        # Filter points based on y+ limit
        bot_index = np.where(yplus >= yplus_min_limit)[0]

        if len(bot_index) == 0:
            if Verbose:
                print(f"No data points found with y+ >= {yplus_min_limit} for Re_tau = {Re_num}")
            continue

        if Verbose:
            print(f"Selected {len(bot_index)} points for Re_tau = {Re_num}")

        # Extract data for the selected indices
        y_sel = y[bot_index]
        U_sel = Udns_phys[bot_index]
        nu_sel = np.full_like(y_sel, nu) # nu is constant for the case

        # --- Calculate Input Features (Pi Groups) ---
        # Note: Synthetic log-law profile assumes dPdx = 0
        dPdx = 0

        # Interpolate velocities at required k*y locations
        # Need the full profile (y, Udns_phys) for interpolation
        U2 = find_k_y_values(y_sel, Udns_phys, y, k=1)
        U3 = find_k_y_values(y_sel, Udns_phys, y, k=2)
        U4 = find_k_y_values(y_sel, Udns_phys, y, k=3)

        # Calculate velocity gradient and interpolate
        # Need the full profile (y, Udns_phys) for gradient calculation
        dUdy_phys = np.gradient(Udns_phys, y)
        dudy1 = dUdy_phys[bot_index]
        dudy2 = find_k_y_values(y_sel, dUdy_phys, y, k=1)
        dudy3 = find_k_y_values(y_sel, dUdy_phys, y, k=2)

        # Calculate dimensionless inputs using safe names
        inputs_dict = {
            'u1_y_over_nu': U_sel * y_sel / nu_sel,
            'up_y_over_nu': np.full_like(y_sel, dPdx), # pi_2 (zero for SYN)
            'u2_y_over_nu': U2 * y_sel / nu_sel,
            'u3_y_over_nu': U3 * y_sel / nu_sel,
            'u4_y_over_nu': U4 * y_sel / nu_sel,
            'dudy1_y_pow2_over_nu': dudy1 * y_sel**2 / nu_sel, # pi_6
            'dudy2_y_pow2_over_nu': dudy2 * y_sel**2 / nu_sel, # pi_7
            'dudy3_y_pow2_over_nu': dudy3 * y_sel**2 / nu_sel  # pi_8
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
            'dpds': np.full_like(y_sel, dPdx), # dPdx is 0
            'u2': U2,
            'u3': U3,
            'u4': U4,
            'dudy1': dudy1,
            'dudy2': dudy2,
            'dudy3': dudy3
        }
        all_unnormalized_inputs_data.append(pd.DataFrame(unnorm_dict))

        # --- Collect Flow Type Information ---
        flow_type_dict = {
            'case_name': ['Synthetic'] * len(y_sel),
            'nu': [nu] * len(y_sel),
            'x': [0] * len(y_sel), # No streamwise variation
            'delta': [1.0] * len(y_sel), # Assumed delta=1 for Re_tau calc
            'Retau': [Re_num] * len(y_sel) # Store the target Re_tau
        }
        all_flow_type_data.append(pd.DataFrame(flow_type_dict))

    if not all_inputs_data:
        print("No data was generated. Check input Reynolds numbers and filtering.")
        return None

    # Concatenate data from all Re_num cases into single DataFrames
    inputs_df = pd.concat(all_inputs_data, ignore_index=True)
    output_df = pd.concat(all_output_data, ignore_index=True)
    flow_type_df = pd.concat(all_flow_type_data, ignore_index=True)
    unnormalized_inputs_df = pd.concat(all_unnormalized_inputs_data, ignore_index=True)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save DataFrames to HDF5 file
    output_filename = os.path.join(output_dir, "SYN_data.h5")
    print(f"\nSaving synthetic data to HDF5 file: {output_filename}")
    try:
        # Use fixed format for better performance with numerical data
        inputs_df.to_hdf(output_filename, key='inputs', mode='w', format='fixed')
        output_df.to_hdf(output_filename, key='output', mode='a', format='fixed')
        unnormalized_inputs_df.to_hdf(output_filename, key='unnormalized_inputs', mode='a', format='fixed')
        # Use table format for flow_type if it contains strings
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
    # Define target Re_tau values for synthetic data generation
    RE_NUMS_SYNTHETIC = [10000, 50000, 100000, 500000] # Example Re_tau values
    saved_file_path = create_synthetic_data_hdf5(RE_NUMS_SYNTHETIC, Verbose=True)

    if saved_file_path:
        print(f"\nTo load the data later:")
        print(f"import pandas as pd")
        print(f"inputs_loaded = pd.read_hdf('{saved_file_path}', key='inputs')")
        print(f"output_loaded = pd.read_hdf('{saved_file_path}', key='output')")
        print(f"# etc.")
