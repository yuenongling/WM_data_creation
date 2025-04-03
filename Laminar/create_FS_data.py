# ./FS_folder/create_FS_data.py
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from falkner_skan import falkner_skan

from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path

# --- Path Adjustments ---
parent_dir = WM_DATA_PATH


# --- Physical and Simulation Constants ---
Vinf = 7.0
nu = 1.45e-6
c = 0.08
rho = 1.225
UP_FRAC = 0.25
DOWN_FRAC = 0.05
a_values = [2.0, 1.0, 0.6, 0.3, 0.1, 0.0, -0.05, -0.08, -0.09043]
reynolds_numbers = np.logspace(3, 6, 5)
x_over_c_locations = np.linspace(0.1, 1.0, 20)
YPLUS_THRESHOLD = 5.0 # Minimum y+ to include


# --- Profile Calculation Function ---
def calculate_falkner_skan_profile(a_value, Re, x_over_c_val):
    """
    Calculates detailed boundary layer profile data for a given Falkner-Skan case.
    (Same implementation as before)
    """
    try:
        eta, f0, f1, f2 = falkner_skan(m=a_value, eta_edge=10, n_points=200, max_iter=150)
        if f2[0] <= 1e-6 and a_value < 0: return None
    except Exception as e:
        return None
    x_val = x_over_c_val * c
    U_inf = Re * nu / c
    if abs(a_value) < 1e-9: ue_val = U_inf
    elif x_val == 0: ue_val = 0
    else: ue_val = U_inf * (x_val / c) ** a_value
    if ue_val <= 1e-6: return None
    dFS_val = np.sqrt(nu * x_val / ue_val)
    dp_dx = -rho * a_value * ue_val**2 / x_val if x_val > 0 else 0
    y_vals = eta * dFS_val
    u_vals = f1 * ue_val
    dudy_vals = f2 * ue_val / dFS_val
    idx_99 = np.where(f1 >= 0.99)[0]
    delta_99 = y_vals[idx_99[0]] if len(idx_99) > 0 else y_vals[-1]
    Re_x = ue_val * x_val / nu
    cf_val = 0; tau_w = 0
    if Re_x > 0 and (a_value + 1) > 0:
         cf_val = 2 * np.sqrt((a_value + 1) / 2) * Re_x**(-0.5) * f2[0]
         tau_w = 0.5 * cf_val * ue_val**2 * rho
    tau_w = max(tau_w, 0.0)
    u_tau = np.sqrt(tau_w / rho)
    return {
        'a_value': a_value, 'Re': Re, 'x_over_c': x_over_c_val, 'y': y_vals,
        'u': u_vals, 'dudy': dudy_vals, 'delta_99': delta_99, 'u_tau': u_tau,
        'dp_dx': dp_dx, 'ue': ue_val, 'nu': nu
    }


# --- Helper Function for Processing and Saving ---
def process_and_save_profiles(profiles_list, region_name, output_data_dir):
    """
    Processes a list of profile dictionaries, creates DataFrames, and saves them to HDF5.

    Args:
        profiles_list (list): List of profile dictionaries for a specific region (or all).
        region_name (str): Name of the region ('FPG', 'ZPG', 'APG', 'ALL').
        output_data_dir (str): Directory to save the HDF5 file.
    """
    if not profiles_list:
        print(f"No profiles to process for region: {region_name}")
        return None

    print(f"\nProcessing {len(profiles_list)} profiles for region: {region_name}...")

    all_inputs_rows = []
    all_output_rows = []
    all_flow_type_rows = []
    all_unnormalized_inputs_rows = []

    for profile in tqdm(profiles_list, desc=f"Processing {region_name}"):
        y_vals, u_vals, dudy_vals = profile['y'], profile['u'], profile['dudy']
        delta_99, u_tau, dp_dx, profile_nu = profile['delta_99'], profile['u_tau'], profile['dp_dx'], profile['nu']

        idx_low_bl_arr = np.where(y_vals >= DOWN_FRAC * delta_99)[0]
        idx_up_bl_arr = np.where(y_vals <= UP_FRAC * delta_99)[0]

        if len(idx_low_bl_arr) == 0 or len(idx_up_bl_arr) == 0: continue
        idx_low_bl, idx_up_bl = idx_low_bl_arr[0], idx_up_bl_arr[-1]
        if idx_up_bl <= idx_low_bl: continue

        yplus_check = y_vals[idx_low_bl:idx_up_bl+1] * u_tau / profile_nu
        valid_yplus_indices = np.where(yplus_check > YPLUS_THRESHOLD)[0]
        if len(valid_yplus_indices) == 0: continue
        final_indices = (idx_low_bl + valid_yplus_indices[0], idx_low_bl + valid_yplus_indices[-1])
        if final_indices[1] <= final_indices[0]: continue

        y_sel = y_vals[final_indices[0]:final_indices[1]+1]
        U_sel = u_vals[final_indices[0]:final_indices[1]+1]
        nu_sel = np.full_like(y_sel, profile_nu)
        dudy1 = dudy_vals[final_indices[0]:final_indices[1]+1]
        up_val = np.sign(dp_dx) * (abs(dp_dx / rho) * profile_nu)**(1/3)

        U2 = find_k_y_values(y_sel, u_vals, y_vals, k=1)
        U3 = find_k_y_values(y_sel, u_vals, y_vals, k=2)
        U4 = find_k_y_values(y_sel, u_vals, y_vals, k=3)
        dudy2 = find_k_y_values(y_sel, dudy_vals, y_vals, k=1)
        dudy3 = find_k_y_values(y_sel, dudy_vals, y_vals, k=2)

        for i in range(len(y_sel)):
            all_inputs_rows.append({
                'u1_y_over_nu': U_sel[i] * y_sel[i] / nu_sel[i],
                'up_y_over_nu': up_val * y_sel[i] / nu_sel[i],
                'u2_y_over_nu': U2[i] * y_sel[i] / nu_sel[i],
                'u3_y_over_nu': U3[i] * y_sel[i] / nu_sel[i],
                'u4_y_over_nu': U4[i] * y_sel[i] / nu_sel[i],
                'dudy1_y_pow2_over_nu': dudy1[i] * y_sel[i]**2 / nu_sel[i],
                'dudy2_y_pow2_over_nu': dudy2[i] * y_sel[i]**2 / nu_sel[i],
                'dudy3_y_pow2_over_nu': dudy3[i] * y_sel[i]**2 / nu_sel[i]
            })
            all_output_rows.append({'utau_y_over_nu': u_tau * y_sel[i] / nu_sel[i]})
            all_unnormalized_inputs_rows.append({
                'y': y_sel[i], 'u1': U_sel[i], 'nu': nu_sel[i], 'utau': u_tau,
                'up': up_val, 'u2': U2[i], 'u3': U3[i], 'u4': U4[i],
                'dudy1': dudy1[i], 'dudy2': dudy2[i], 'dudy3': dudy3[i]
            })

            # Determine region for flow_type (even if saving 'ALL')
            if dp_dx < -1e-6: current_region = 'FPG'
            elif dp_dx > 1e-6: current_region = 'APG'
            else: current_region = 'ZPG'
            all_flow_type_rows.append({
                'case_name': f"FS_{current_region}", 'a_value': profile['a_value'],
                'Re_c': profile['Re'], 'x_over_c': profile['x_over_c'],
                'delta99': delta_99, 'ue': profile['ue'], 'dp_dx': dp_dx
            })

    if not all_inputs_rows:
        print(f"No data points collected after processing profiles for region: {region_name}.")
        return None

    inputs_df = pd.DataFrame(all_inputs_rows)
    output_df = pd.DataFrame(all_output_rows)
    flow_type_df = pd.DataFrame(all_flow_type_rows)
    unnormalized_inputs_df = pd.DataFrame(all_unnormalized_inputs_rows)

    os.makedirs(output_data_dir, exist_ok=True)
    output_filename = os.path.join(output_data_dir, f"FS_{region_name}_data.h5")
    print(f"\nSaving {region_name} data to HDF5 file: {output_filename}")
    try:
        inputs_df.to_hdf(output_filename, key='inputs', mode='w', format='fixed')
        output_df.to_hdf(output_filename, key='output', mode='a', format='fixed')
        unnormalized_inputs_df.to_hdf(output_filename, key='unnormalized_inputs', mode='a', format='fixed')
        flow_type_df.to_hdf(output_filename, key='flow_type', mode='a', format='table')
        print(f"{region_name} data successfully saved.")
    except Exception as e:
        print(f"Error saving {region_name} HDF5 file: {e}")
        return None

    print(f"Final {region_name} Shapes:\n  Inputs: {inputs_df.shape}\n  Output: {output_df.shape}\n  Flow Type: {flow_type_df.shape}\n  Unnormalized Inputs: {unnormalized_inputs_df.shape}")
    return output_filename


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

    # --- Categorize profiles ---
    fpg_profiles = [p for p in all_profiles if p['dp_dx'] < -1e-6]
    zpg_profiles = [p for p in all_profiles if abs(p['dp_dx']) <= 1e-6]
    apg_profiles = [p for p in all_profiles if p['dp_dx'] > 1e-6]
    print(f"Categorized profiles: FPG={len(fpg_profiles)}, ZPG={len(zpg_profiles)}, APG={len(apg_profiles)}")

    # --- Define output directory ---
    output_save_dir = os.path.join(parent_dir, "data") # e.g., all_data_creation/data/

    # --- Process and Save each category + ALL ---
    saved_files = []
    # Save FPG
    fpg_file = process_and_save_profiles(fpg_profiles, "FPG", output_save_dir)
    if fpg_file: saved_files.append(fpg_file)
    # Save ZPG
    zpg_file = process_and_save_profiles(zpg_profiles, "ZPG", output_save_dir)
    if zpg_file: saved_files.append(zpg_file)
    # Save APG
    apg_file = process_and_save_profiles(apg_profiles, "APG", output_save_dir)
    if apg_file: saved_files.append(apg_file)
    # Save ALL
    all_file = process_and_save_profiles(all_profiles, "ALL", output_save_dir)
    if all_file: saved_files.append(all_file)

    print("\n--- Summary ---")
    if saved_files:
        print("Successfully saved the following files:")
        for f in saved_files:
            print(f"- {f}")
    else:
        print("No data files were saved.")
