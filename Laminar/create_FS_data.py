# ./FS_folder/create_FS_data.py
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from falkner_skan import falkner_skan

# Assuming data_processing_utils.py is in a parent directory or can be imported
# Make sure the directory containing data_processing_utils.py is in sys.path
try:
    from data_processing_utils import find_k_y_values, import_path
    WM_DATA_PATH = import_path() # Ensure the BFM_PATH and subdirectories are in the system path
except ImportError:
    print("Error: Could not import data_processing_utils. Ensure it's in sys.path.")
    print("Attempting to set a default path for WM_DATA_PATH. Please verify.")
    # Fallback if import_path is not available or fails
    WM_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if not os.path.exists(os.path.join(WM_DATA_PATH, 'data_processing_utils.py')):
         print("Warning: data_processing_utils.py not found at expected fallback location.")
         print("Please manually add the directory containing data_processing_utils.py to sys.path.")
         sys.exit(1)


# --- Path Adjustments ---
parent_dir = WM_DATA_PATH # This should be the base directory for outputs


# --- Physical and Simulation Constants ---
# Vinf = 7.0 # This seems unused in the current script logic
nu = 1.45e-6
c = 0.08
rho = 1.225
UP_FRAC = 0.25
DOWN_FRAC = 0.005
a_values = [2.0, 1.0, 0.6, 0.3, 0.1, 0.0, -0.05, -0.08, -0.09043]
# Modified: Reynolds numbers from 100 to 10000
reynolds_numbers = np.logspace(2, 4, 20) # 20 points logarithmically spaced between 10^2 (100) and 10^4 (10000)
x_over_c_locations = np.linspace(0.1, 1.0, 20)
YPLUS_THRESHOLD = 5.0 # Minimum y+ to include


# --- Profile Calculation Function ---
def calculate_falkner_skan_profile(a_value, Re, x_over_c_val):
    """
    Calculates detailed boundary layer profile data for a given Falkner-Skan case.
    (Same implementation as before)
    """
    try:
        # eta_edge increased slightly for potentially thicker boundary layers at lower Re
        eta, f0, f1, f2 = falkner_skan(m=a_value, eta_edge=15, n_points=300, max_iter=200)
        # Added check for negative skin friction, indicating separation (f2[0] <= 0)
        # For negative a_values, f2[0] can be very small positive near separation
        # A more robust check for separation might be needed depending on 'falkner_skan' implementation
        # For now, keep the original check, but be aware negative f2[0] means separation
        if f2[0] <= 1e-6 and a_value < 0:
             # print(f"Skipping separated profile: a={a_value}, Re={Re}, x/c={x_over_c_val}")
             return None
    except Exception as e:
        # print(f"Error calculating FS profile (a={a_value}, Re={Re}, x/c={x_over_c_val}): {e}")
        return None

    x_val = x_over_c_val * c
    # Calculate U_inf based on Re and characteristic length c
    U_inf = Re * nu / c

    # Calculate edge velocity ue
    if abs(a_value) < 1e-9: # ZPG case (a=0)
        ue_val = U_inf
    elif x_val == 0: # Leading edge
        # ue is zero at leading edge for non-zero 'a', but x/c=0.1 is the minimum
        # This case might not be strictly necessary given x_over_c_locations start at 0.1
         ue_val = 0 # Limit case, practically not used due to x_over_c > 0
    else:
        # Falkner-Skan wedge flow ue = C * x^m, C = U_inf / c^m = Re * nu / (c * c^a)
        # ue = (Re * nu / c^(a+1)) * x^a = (Re * nu / c) * (x/c)^a = U_inf * (x/c)^a
        ue_val = U_inf * (x_val / c) ** a_value

    # Filter out cases with very low edge velocity (e.g., near leading edge for non-zero 'a' or just numerically small)
    if ue_val <= 1e-6:
        # print(f"Skipping profile due to low edge velocity (ue={ue_val}): a={a_value}, Re={Re}, x/c={x_over_c_val}")
        return None

    # Falkner-Skan length scale dFS = sqrt(nu * x / ue)
    # Need to handle x_val = 0 carefully, but our x_over_c starts at 0.1
    if x_val <= 1e-9: # Should not happen with x_over_c_locations[0] = 0.1
         # print(f"Skipping profile due to low x_val ({x_val}): a={a_value}, Re={Re}, x/c={x_over_c_val}")
         return None
    dFS_val = np.sqrt(nu * x_val / ue_val)

    # Pressure gradient parameter beta = 2*m / (m+1). dp/dx = -rho * ue * due/dx
    # due/dx = U_inf * a * x^(a-1) / c^a = (ue/x) * a
    # dp/dx = -rho * ue * (ue/x) * a = -rho * a * ue^2 / x
    # Handle x_val = 0 separately, though it's not used
    dp_dx = -rho * a_value * ue_val**2 / x_val if x_val > 0 else 0

    y_vals = eta * dFS_val
    u_vals = f1 * ue_val
    dudy_vals = f2 * ue_val / dFS_val # d(f1)/d(eta) * d(eta)/dy = f2 * (1/dFS)

    # Find delta_99 (y where u/ue >= 0.99)
    idx_99 = np.where(f1 >= 0.99)[0]
    delta_99 = y_vals[idx_99[0]] if len(idx_99) > 0 else y_vals[-1] # Use last y if 0.99 isn't reached

    # Calculate wall shear stress and friction velocity
    # tau_w = mu * (du/dy)_wall = rho * nu * f2[0] * ue / dFS
    # cf = tau_w / (0.5 * rho * ue^2) = (rho * nu * f2[0] * ue / dFS) / (0.5 * rho * ue^2)
    # cf = 2 * nu * f2[0] / (dFS * ue) = 2 * nu * f2[0] / (sqrt(nu * x / ue) * ue)
    # cf = 2 * nu * f2[0] / sqrt(nu * x * ue) = 2 * f2[0] * sqrt(nu / (x * ue))
    # cf = 2 * f2[0] / sqrt(Re_x) where Re_x = ue * x / nu
    # Falkner-Skan specific formula: cf = sqrt(2*(a+1)) * f2[0] / sqrt(Re_x)
    # The factor sqrt(2*(a+1)) accounts for the velocity profile shape parameter beta = 2*a/(a+1)
    # beta = 2m/(m+1), for FS m=a, so beta = 2a/(a+1). The sqrt term comes from the transformation.
    # Check source for sqrt(2*(a+1)). White's Viscous Fluid Flow Eq 4-152 gives C_f * sqrt(Re_x) = sqrt(2(m+1)) * f''(0)
    # So C_f = sqrt(2(m+1)) * f''(0) / sqrt(Re_x). With m=a, and f''(0)=f2[0]
    Re_x = ue_val * x_val / nu
    cf_val = 0
    tau_w = 0
    # Wall shear stress and cf are only well-defined for attached flow (tau_w > 0 or close to zero)
    # Also requires Re_x > 0
    # For converging flow (a > 0), f2[0] > 0. For parallel flow (a=0), f2[0] > 0.
    # For diverging flow (a < 0), f2[0] decreases and becomes 0 at separation (a approx -0.09046)
    if Re_x > 1e-9 and f2[0] > 0: # Check for non-zero Re_x and positive wall shear (f2[0] proportional to wall shear)
         cf_val = np.sqrt(2 * (a_value + 1)) * f2[0] / np.sqrt(Re_x) if (a_value + 1) > 0 else 0
         tau_w = 0.5 * cf_val * ue_val**2 * rho
    tau_w = max(tau_w, 0.0) # Ensure non-negative wall shear, though f2[0] > 0 check should handle this

    u_tau = np.sqrt(tau_w / rho) if tau_w > 0 else 0.0 # Avoid sqrt of zero/negative

    return {
        'a_value': a_value, 'Re': Re, 'x_over_c': x_over_c_val, 'y': y_vals,
        'u': u_vals, 'dudy': dudy_vals, 'delta_99': delta_99, 'u_tau': u_tau,
        'dp_dx': dp_dx, 'ue': ue_val, 'nu': nu, 'Re_x': Re_x, 'cf': cf_val, 'tau_w': tau_w
    }


# --- Helper Function for Processing and Saving ---
# Modified to process a list of profiles and save to a single HDF5 file
def process_and_save_all_profiles(profiles_list, output_filename, output_data_dir):
    """
    Processes a list of profile dictionaries, creates DataFrames, and saves them
    to a single HDF5 file.

    Args:
        profiles_list (list): List of profile dictionaries for all cases.
        output_filename (str): The full path and filename for the output HDF5 file.
        output_data_dir (str): Directory to save the HDF5 file (will be created if needed).
    """
    if not profiles_list:
        print("No profiles to process. Exiting saving process.")
        return None

    print(f"\nProcessing {len(profiles_list)} collected profiles...")

    all_inputs_rows = []
    all_output_rows = []
    all_flow_type_rows = []
    all_unnormalized_inputs_rows = []
    processed_count = 0

    for profile in tqdm(profiles_list, desc="Processing profiles for saving"):
        y_vals, u_vals, dudy_vals = profile['y'], profile['u'], profile['dudy']
        delta_99, u_tau, dp_dx, profile_nu = profile['delta_99'], profile['u_tau'], profile['dp_dx'], profile['nu']

        # Skip profiles where u_tau is zero or near zero (e.g., separated or potential issues)
        if u_tau <= 1e-9:
            # print(f"Skipping profile with zero u_tau: a={profile['a_value']}, Re={profile['Re']}, x/c={profile['x_over_c']}")
            continue

        # Filter points within the specified boundary layer fractions
        idx_low_bl_arr = np.where(y_vals >= DOWN_FRAC * delta_99)[0]
        idx_up_bl_arr = np.where(y_vals <= UP_FRAC * delta_99)[0]

        if len(idx_low_bl_arr) == 0 or len(idx_up_bl_arr) == 0:
             print(f"Skipping profile due to empty BL fraction range: a={profile['a_value']}, Re={profile['Re']}, x/c={profile['x_over_c']}")
             continue
        # Find the indices that are within both ranges and form a continuous block
        # The intersection of indices would be idx_low_bl_arr intersect idx_up_bl_arr
        # A simpler way is to find the first index >= DOWN_FRAC*delta_99 and the last index <= UP_FRAC*delta_99
        idx_start_bl = idx_low_bl_arr[0]
        idx_end_bl = idx_up_bl_arr[-1]

        if idx_end_bl < idx_start_bl:
            # print(f"Skipping profile where BL fraction range is inverted: a={profile['a_value']}, Re={profile['Re']}, x/c={profile['x_over_c']}")
            continue

        # Get the original indices in y_vals corresponding to the valid y+ points in the selected BL range
        final_indices_start = idx_start_bl
        final_indices_end = idx_start_bl

        if final_indices_end < final_indices_start:
            # print(f"Skipping profile (final) due to inverted index range after y+ filter: a={profile['a_value']}, Re={profile['Re']}, x/c={profile['x_over_c']}")
            continue

        # Final selected data points
        y_sel = y_vals[final_indices_start:final_indices_end+1]
        U_sel = u_vals[final_indices_start:final_indices_end+1]
        nu_sel = np.full_like(y_sel, profile_nu) # Should be constant anyway, but for clarity
        dudy1 = dudy_vals[final_indices_start:final_indices_end+1]

        # Calculate pressure gradient term up_val
        up_val = np.sign(dp_dx) * (abs(dp_dx / rho) * profile_nu)**(1/3)
        # Ensure up_val is an array or compatible for calculations
        up_val_sel = np.full_like(y_sel, up_val)


        # Calculate U2, U3, U4, dudy2, dudy3 using the original full profile data
        # find_k_y_values interpolates, it needs the y locations to interpolate AT (y_sel)
        # and the original full profile y, u, dudy arrays to interpolate FROM (y_vals, u_vals, dudy_vals)
        try:
            U2 = find_k_y_values(y_sel, u_vals, y_vals, k=1)
            U3 = find_k_y_values(y_sel, u_vals, y_vals, k=2)
            U4 = find_k_y_values(y_sel, u_vals, y_vals, k=3)
            dudy2 = find_k_y_values(y_sel, dudy_vals, y_vals, k=1)
            dudy3 = find_k_y_values(y_sel, dudy_vals, y_vals, k=2)
        except Exception as e:
             print(f"Error in find_k_y_values for profile (a={profile['a_value']}, Re={profile['Re']}, x/c={profile['x_over_c']}): {e}")
             continue # Skip this profile if interpolation fails


        processed_count += 1 # Count profiles that passed filtering

        # Append data for each point in the selected range
        for i in range(len(y_sel)):
            # Inputs (Normalized)
            all_inputs_rows.append({
                'u1_y_over_nu': U_sel[i] * y_sel[i] / nu_sel[i],
                'up_y_over_nu': up_val_sel[i] * y_sel[i] / nu_sel[i],
                'upn_y_over_nu': 0.0, # upn is for non-equilibrium pressure gradient, not in FS
                'u2_y_over_nu': U2[i] * y_sel[i] / nu_sel[i],
                'u3_y_over_nu': U3[i] * y_sel[i] / nu_sel[i],
                'u4_y_over_nu': U4[i] * y_sel[i] / nu_sel[i],
                'dudy1_y_pow2_over_nu': dudy1[i] * y_sel[i]**2 / nu_sel[i],
                'dudy2_y_pow2_over_nu': dudy2[i] * y_sel[i]**2 / nu_sel[i],
                'dudy3_y_pow2_over_nu': dudy3[i] * y_sel[i]**2 / nu_sel[i]
            })
            # Output (Normalized)
            all_output_rows.append({'utau_y_over_nu': u_tau * y_sel[i] / nu_sel[i]})

            # Unnormalized Inputs (for reference)
            all_unnormalized_inputs_rows.append({
                'y': y_sel[i], 'u1': U_sel[i], 'nu': nu_sel[i], 'utau': u_tau,
                'up': up_val_sel[i], 'upn': 0.0, 'u2': U2[i], 'u3': U3[i], 'u4': U4[i],
                'dudy1': dudy1[i], 'dudy2': dudy2[i], 'dudy3': dudy3[i]
            })

            # Flow Type and Case Info (Repeated for each point, but this is how it was structured)
            # Determine region for flow_type
            if dp_dx < -1e-6: current_region = 'FPG'
            elif dp_dx > 1e-6: current_region = 'APG'
            else: current_region = 'ZPG' # abs(dp_dx) <= 1e-6
            all_flow_type_rows.append({
                'case_name': f"FS_{current_region}", # Case name based on type
                'a_value': profile['a_value'],
                'Re_c': profile['Re'],
                'x_over_c': profile['x_over_c'],
                'delta99': delta_99,
                'ue': profile['ue'],
                'dp_dx': dp_dx
            })

    print(f"\nFinished processing. Generated data points from {processed_count} profiles.")

    if not all_inputs_rows:
        print("No data points collected after processing and filtering profiles. No file will be saved.")
        return None

    print(f"Creating DataFrames...")
    inputs_df = pd.DataFrame(all_inputs_rows)
    output_df = pd.DataFrame(all_output_rows)
    flow_type_df = pd.DataFrame(all_flow_type_rows)
    unnormalized_inputs_df = pd.DataFrame(all_unnormalized_inputs_rows)

    print(f"Ensuring output directory exists: {output_data_dir}")
    os.makedirs(output_data_dir, exist_ok=True)

    print(f"\nSaving combined data to HDF5 file: {output_filename}")
    try:
        # Use 'w' mode for the first key to create/overwrite the file
        inputs_df.to_hdf(output_filename, key='inputs', mode='w', format='fixed')
        # Use 'a' mode for subsequent keys to append to the same file
        output_df.to_hdf(output_filename, key='output', mode='a', format='fixed')
        unnormalized_inputs_df.to_hdf(output_filename, key='unnormalized_inputs', mode='a', format='fixed')
        # Use format='table' for flow_type as it might be useful for querying, though 'fixed' is also fine
        flow_type_df.to_hdf(output_filename, key='flow_type', mode='a', format='table')
        print("Combined data successfully saved.")
        saved_successfully = True
    except Exception as e:
        print(f"Error saving combined HDF5 file: {e}")
        saved_successfully = False
        # Clean up potentially incomplete file
        if os.path.exists(output_filename):
            try:
                os.remove(output_filename)
                print(f"Removed incomplete file: {output_filename}")
            except Exception as clean_e:
                print(f"Error cleaning up incomplete file: {clean_e}")
        return None

    if saved_successfully:
        print(f"\nFinal Combined Data Shapes:")
        print(f"  Inputs: {inputs_df.shape}")
        print(f"  Output: {output_df.shape}")
        print(f"  Flow Type: {flow_type_df.shape}")
        print(f"  Unnormalized Inputs: {unnormalized_inputs_df.shape}")
        return output_filename
    else:
        return None


# --- Main Execution ---
if __name__ == "__main__":
    # --- Calculate all profiles first ---
    all_profiles = []
    print("Calculating Falkner-Skan profiles for all parameter combinations...")
    total_combinations = len(a_values) * len(reynolds_numbers) * len(x_over_c_locations)
    with tqdm(total=total_combinations, desc="Calculating Profiles") as pbar:
        for a in a_values:
            for Re in reynolds_numbers:
                for x_loc in x_over_c_locations:
                    profile = calculate_falkner_skan_profile(a, Re, x_loc)
                    if profile is not None:
                        all_profiles.append(profile)
                    pbar.update(1)

    if not all_profiles:
        print("No valid profiles were generated overall. Exiting.")
        sys.exit(1)
    print(f"\nGenerated {len(all_profiles)} valid profiles in total.")

    # --- Define output directory and single filename ---
    output_save_dir = os.path.join(parent_dir, "data") # e.g., BFM_PATH/data/
    # Modified: Define a single output filename for all data
    combined_output_filename = os.path.join(output_save_dir, "FS_ALL_combined_data.h5")

    # --- Process and Save ALL profiles into a single file ---
    print("\n--- Processing and Saving All Data ---")
    saved_file = process_and_save_all_profiles(all_profiles, combined_output_filename, output_save_dir)

    print("\n--- Summary ---")
    if saved_file:
        print(f"Successfully saved combined data to: {saved_file}")
    else:
        print("No combined data file was saved.")
