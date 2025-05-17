import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import os
from scipy import interpolate
import re # Using re for a slightly more robust extraction

# Extract Ue and X station 
def extract_header_data(filepath):
    """
    Extracts X (in mm) and Ue/Uref values from the header of the given file.

    Args:
        filepath (str): The path to the data file.

    Returns:
        tuple: (x_value, ue_uref_value) or (None, None) if not found.
    """
    x_value = None
    ue_uref_value = None

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip() # Remove leading/trailing whitespace

                # Look for X value
                # Example: #            X=  50.8 mm      Tunnel Run#= 638
                if "X=" in line and "mm" in line:
                    try:
                        # Split by "X=", take the part after it
                        # Then split by "mm", take the part before it
                        val_str = line.split("X=")[1].split("mm")[0].strip()
                        x_value = float(val_str)
                    except (IndexError, ValueError) as e:
                        print(f"Warning: Could not parse X value from line: '{line}'. Error: {e}")
                        pass # Continue, maybe it's in a different format or another line

                # Look for Ue/Uref value
                # Example: Ue/Uref=   0.996      Date= 428.88
                if "Ue/Uref=" in line:
                    try:
                        # Split by "Ue/Uref=", take the part after it
                        # Then split by whitespace and take the first token
                        val_str = line.split("Ue/Uref=")[1].split()[0].strip()
                        ue_uref_value = float(val_str)
                    except (IndexError, ValueError) as e:
                        print(f"Warning: Could not parse Ue/Uref value from line: '{line}'. Error: {e}")
                        pass # Continue

                # If we've found both, we can stop reading the file
                if x_value is not None and ue_uref_value is not None:
                    break
            
            # Alternative using regular expressions (more robust to spacing variations)
            # Reset for regex search if values not found by simple split
            if x_value is None or ue_uref_value is None:
                f.seek(0) # Go back to the beginning of the file for regex search
                file_content = f.read()
                
                if x_value is None:
                    match_x = re.search(r"X=\s*([\d\.]+)\s*mm", file_content)
                    if match_x:
                        try:
                            x_value = float(match_x.group(1))
                        except ValueError:
                             print(f"Warning: Regex could not parse X value: '{match_x.group(1)}'")
                
                if ue_uref_value is None:
                    match_ue_uref = re.search(r"Ue/Uref=\s*([\d\.]+)", file_content)
                    if match_ue_uref:
                        try:
                            ue_uref_value = float(match_ue_uref.group(1))
                        except ValueError:
                            print(f"Warning: Regex could not parse Ue/Uref value: '{match_ue_uref.group(1)}'")


    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None

    return x_value, ue_uref_value

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
casename = 'spinning_Driver'
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
savedatapath = os.path.join(WM_DATA_PATH, 'data')
currentpath = os.path.join(WM_DATA_PATH, casename)
statspath = os.path.join(currentpath, 'stats')

# --- Hardcoded some data ---
Uref = 30
nu = (Uref * 18e-3) / 6000 # Re = Uref * L / nu

rho = 1
UP_FRAC = 0.20
DOWN_FRAC = 0.025

cases = ['as1', 'cs0', 'cs1', 'ds0', 'ds1']

filenumber_case ={'as1': 12, 
                  'cs0': 13, 
                  'cs1': 13, 
                  'ds0': 12,
                  'ds1': 12}

for case in cases:

    print(f"\nProcessing case: {case}")

    # Cf data
    cf_data = np.loadtxt(os.path.join(statspath, f'cf_{case}.dat'), comments='#')
    xcf = cf_data[:, 0]
    cf_x = cf_data[:, 2]
    cf_z = cf_data[:, 3]
    cf = np.sqrt(cf_x**2 + cf_z**2)

    if 'as' not in case:
        cp_data = np.loadtxt(os.path.join(statspath, f'cp_{case}.dat'), comments='#')
        xcp = cp_data[:, 0]
        cp  = cp_data[:, 1]

        dpdx = np.gradient(cp, xcp) * 0.5 * rho * Uref**2 # in Pa/m 
        up = np.sign(dpdx) * (np.abs(dpdx) * nu /  rho) **(1/3) # in m/s

    else:
        cp_data = 0

    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []

    for i in range(filenumber_case[case]):

        fname = f'ldv_{case}_0{i+1}.dat' if i < 9 else f'ldv_{case}_{i+1}.dat'

        # --- Load data ---
        # Load velocity data
        filename = os.path.join(statspath, fname)
        # Extract X and Ue/Uref from the header of the velocity file
        x_, Ue_Uref = extract_header_data(filename)
        nu_i = nu
        up_i = np.interp(x_, xcp, up) if 'as' not in case else 0 # in m/s

        # Velocity data
        vel_data = np.loadtxt(filename, comments='#', usecols=(1,2,4))
        y_i = vel_data[:, 0] * 1e-3 # in m
        U = vel_data[:, 1] * Ue_Uref * Uref # in m/s
        W = vel_data[:, 2] * Ue_Uref * Uref # in m/s
        U_i = np.sqrt(U**2 + W**2) 

        # Estimate delta99
        delta99_i = np.interp(0.99, U_i/U_i.max(), y_i) # in m

        # Screening valid x_
        if x_ is None or Ue_Uref is None:
            print(f"Could not extract X or Ue/Uref from {filename}. Skipping...")
            continue

        # Get local Cf value
        if x_ < xcf[0] or x_ > xcf[-1]:
            print(f"X value {x_} is out of range for cf data. Skipping...")
            continue
        else:
            cf_x_ = np.interp(x_, xcf, cf)
            utau_i = np.sqrt(cf_x_ * 0.5 * rho * Uref**2) # in m/s

        # Find points within the boundary layer region of interest
        bot_index = np.where((y_i >= DOWN_FRAC * delta99_i) & (y_i <= UP_FRAC * delta99_i))[0]

        U2 = find_k_y_values(y_i[bot_index], U_i, y_i, k=1)
        U3 = find_k_y_values(y_i[bot_index], U_i, y_i, k=2)
        U4 = find_k_y_values(y_i[bot_index], U_i, y_i, k=3)

        pi_1 = y_i * U_i / nu_i
        pi_1 = pi_1[bot_index]
        pi_2 = up_i * y_i[bot_index] / nu_i if up_i != 0 else np.zeros_like(y_i[bot_index])
        pi_3 = U2 * y_i[bot_index] / nu_i
        pi_4 = U3 * y_i[bot_index] / nu_i
        pi_5 = U4 * y_i[bot_index] / nu_i
        # Calculate velocity gradients
        dUdy = np.gradient(U_i, y_i)
        dudy_1 = dUdy[bot_index]
        dudy_2 = find_k_y_values(y_i[bot_index], dUdy, y_i, k=1)
        dudy_3 = find_k_y_values(y_i[bot_index], dUdy, y_i, k=2)

        pi_6 = dudy_1 * y_i[bot_index]**2 / nu_i
        pi_7 = dudy_2 * y_i[bot_index]**2 / nu_i
        pi_8 = dudy_3 * y_i[bot_index]**2 / nu_i

        # Calculate output (y+)
        pi_out = utau_i * y_i / nu_i
        pi_out = pi_out[bot_index]

        # --- Calculate Input Features (Pi Groups) ---
        # Note:  dPdx is NOT zero here
        # Calculate dimensionless inputs using safe names
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
            'y': y_i[bot_index],
            'u1': U_i[bot_index],
            'nu': np.full_like(y_i[bot_index], nu_i), 
            'utau': np.full_like(y_i[bot_index], utau_i),
            'up': np.full_like(y_i[bot_index], up_i),
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
        len_y = len(y_i[bot_index])
        flow_type_dict = {
            'case_name': [casename] * len_y,
            'nu': [nu_i] * len_y,
            'x': [x_] * len_y,
            'delta': [delta99_i] * len_y,
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
    output_filename = os.path.join(savedatapath, f'{casename}_{case}_data.h5')
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


# #     y_data = np.loadtxt(os.path.join(statspath, f'{case}y.dat'), comments='#', usecols=(1,2,3,4,7))
# #
# #     x_to_investigate = y_data[:, 0] * 1e-3 # in m
#     Rex = y_data[:, 1]
#     Uo  = y_data[:, 2] # in m/s
#     ##
#     nu  = (Uo * x_to_investigate) / Rex
#     print(f"mean nu: {np.mean(nu)}, std nu: {np.std(nu)}")
#     nu  = np.mean(nu)
#     ##
#     CF  = y_data[:, 3]
#     # utau = np.sqrt(CF * 0.5 * rho * Uo**2) # in m/s
#     delta = y_data[:, 4] * 1e-3 # in m
# #
# #     vel_data = pkl.load(open(os.path.join(statspath, f'{case}.pkl'), 'rb'))
# #
# #     # --- pressure gradient data ---
# #     if 'c' not in case:
# #         up = np.zeros(len(x_to_investigate)) # in m/s
# #     else:
# #         cp_data = np.loadtxt(os.path.join(statspath, f'cp.dat'))
# #         xp = cp_data[:, 0] # in m
# #         cp = cp_data[:, 1]
# #         dpdx = np.gradient(cp, xp) # in Pa/m
# #         up  = np.sign(dpdx) * (np.abs(dpdx) * nu /  rho) **(1/3) # in m/s
# #
# #
# #     for i, x_ in enumerate(x_to_investigate):
# #         # Each x_ (xA) represents one case (with different pressure gradient) with data measured at one xm downstream location
# #
# #         delta99_i = delta[i] # in m
# #         nu_i = nu # in m^2/s
# #
# #         vel_file = vel_data[int(x_*1e3)]['data']
# #
# #         U_i = vel_file['U']
# #         y_i = vel_file['Y/DEL'] * delta99_i
# #
# #         # Use Uplus from the velocity file
# #         Uplus = vel_file['UPLUS']
# #         utau  = U_i / Uplus
# #         utau_i = np.mean(utau)
# #         print(f"mean utau: {np.mean(utau)}, std utau: {np.std(utau)}")
# #
# #
# #         # Calculate velocity at different y positions
# #
# #         # WARNING: pressure gradient along the test wall is minimized by the geometry of the opposite wall; assume zero here
# #         if 'c' in case:
# #             up_i = np.interp(x_, xp, up) # in m/s
# #         else:
# #             up_i = up[i] # in m/s
# #
# #         # Calculate non-dimensional inputs
# #
# #
