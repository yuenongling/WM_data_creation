from scipy.io import loadmat
import numpy as np
import pandas as pd
import os
import pickle as pkl  # Import pickle

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
datapath = os.path.join(WM_DATA_PATH, 'data')
APG_MAT_FILE = os.path.join(WM_DATA_PATH, 'apg_KTH', 'data', 'APG.mat')

# --- Load the APG data ---
results = loadmat(APG_MAT_FILE, squeeze_me=True)
subcases = ['b1n', 'b2n', 'm13n', 'm16n', 'm18n']
UP_FRAC = 0.2
DOWN_FRAC = 0.01


for subcase in subcases:
# Lists to collect data
    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []

    result = results[subcase]

    # Read data
    Cf = result['Cf']  # Cf = 2(utau / Ue)^2

    U = result['U']
    y = result['y']
    Ue = result['Ue']
    utau = np.array([np.sqrt(Cf[i] / 2) * Ue[i] for i, _ in enumerate(Cf)])
    dPdx = result['beta'] * utau**2 / result['deltas']  # NOTE: get back dPdx from beta
    theta = result['theta']
    nu = result['nu']
    up = np.sign(dPdx) * (abs(nu * dPdx)) ** (1 / 3)
    delta99 = result['delta99']
    beta = result['beta']
    P    = result['P']

    breakpoint()
    # Albert parameter: alber = theta / Ue**2 * dPdx
    albert = theta / Ue**2 * dPdx
    xa = result['x']

    for idx, x in enumerate(xa):

        # NOTE: Discard points after 2000
        if x > 2000:
            continue
        # if albert[i] > 0.1:
        #     break
        y_i = y[idx]
        U_i = U[idx]
        P_i = P[idx]

        delta99_i = delta99[idx]
        nu_i = nu[idx]
        x_i = xa[idx]
        utau_i = utau[idx]
        beta_i = beta[idx]
        dPdx_i = dPdx[idx]

        # Skip the first point if it is zero
        if U_i[0] == 0 or y_i[0] == 0:
            y_i = y_i[1:]
            U_i = U_i[1:]

        y_i = np.array(y_i)
        U_i = np.array(U_i)
        up_i = up[idx]

        delta_p = P_i - P_i[0]  # Pressure difference from the first point
        up_n_i = np.sign(delta_p) * (abs(nu_i * delta_p)) ** (1 / 3)

        bot_index = np.where((y_i >= DOWN_FRAC * delta99_i) & (y_i <= UP_FRAC * delta99_i))[0]

        U2 = find_k_y_values(y_i[bot_index], U_i, y_i, k=1)
        U3 = find_k_y_values(y_i[bot_index], U_i, y_i, k=2)
        U4 = find_k_y_values(y_i[bot_index], U_i, y_i, k=3)

        # --- Calculate Input Features (Pi Groups) ---
        # Note:  dPdx is NOT zero here
        # Calculate dimensionless inputs using safe names
        inputs_dict = {
            'u1_y_over_nu': U_i[bot_index] * y_i[bot_index] / nu_i,
            'up_y_over_nu': up_i * y_i[bot_index] / nu_i,  # pi_2 (is NOT zero for APG)
            'upn_y_over_nu': up_n_i[bot_index] * y_i[bot_index] / nu_i,  # pi_2 (is NOT zero for APG)
            'u2_y_over_nu': U2 * y_i[bot_index] / nu_i,
            'u3_y_over_nu': U3 * y_i[bot_index] / nu_i,
            'u4_y_over_nu': U4 * y_i[bot_index] / nu_i,
            'dudy1_y_pow2_over_nu': np.gradient(U_i, y_i)[bot_index] * y_i[bot_index] ** 2 / nu_i,
            'dudy2_y_pow2_over_nu': find_k_y_values(y_i[bot_index], np.gradient(U_i, y_i), y_i, k=1) * y_i[
                bot_index] ** 2 / nu_i,
            'dudy3_y_pow2_over_nu': find_k_y_values(y_i[bot_index], np.gradient(U_i, y_i), y_i, k=2) * y_i[
                bot_index] ** 2 / nu_i
        }
        all_inputs_data.append(pd.DataFrame(inputs_dict))

        # --- Calculate Output Feature ---
        # Output is y+ (utau * y / nu)
        output_dict = {
            'utau_y_over_nu': utau_i * y_i[bot_index] / nu_i
        }
        all_output_data.append(pd.DataFrame(output_dict))

        # --- Collect Unnormalized Inputs ---
        unnorm_dict = {
            'y': y_i[bot_index],
            'u1': U_i[bot_index],
            'nu': np.full_like(y_i[bot_index], nu_i),
            'utau': np.full_like(y_i[bot_index], utau_i),
            'up': np.full_like(y_i[bot_index], up_i),
            'upn': up_n_i[bot_index],
            'u2': U2,
            'u3': U3,
            'u4': U4,
            'dudy1': np.gradient(U_i, y_i)[bot_index],
            'dudy2': find_k_y_values(y_i[bot_index], np.gradient(U_i, y_i), y_i, k=1),
            'dudy3': find_k_y_values(y_i[bot_index], np.gradient(U_i, y_i), y_i, k=2)
        }
        all_unnormalized_inputs_data.append(pd.DataFrame(unnorm_dict))

        # --- Collect Flow Type Information ---
        # Using format: [case_name, reference_nu, x_coord, delta, edge_velocity]
        # For channel flow: x=0, delta=1 (half-channel height), Ue=0 (or U_bulk if needed)
        flow_type_dict = {
            'case_name': ['apg_kth'] * len(y_i[bot_index]),
            'nu': [nu_i] * len(y_i[bot_index]),
            'x': [x_i] * len(y_i[bot_index]),
            'delta': [delta99_i] * len(y_i[bot_index]),
            'albert': [albert[idx]] * len(y_i[bot_index])
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
    output_filename = os.path.join(datapath, 'apg_' + subcase + '_data.h5')
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

    # NOTE: This is outdated code for sanity check, uncomment if needed
    # Only to check with original pickle data
    
    # --- Sanity Check ---
    # print("\n--- Sanity Check: Comparing HDF5 with Original Pickle ---")
    # with open('/home/yuenongling/Codes/BFM/WM_Opt/data/apg_' + subcase + '_data.pkl', 'rb') as f:
    #     original_data = pkl.load(f)
    #
    # # Load corresponding data from HDF5
    # inputs_hdf = inputs_df[inputs_df.index.isin(np.arange(len(original_data['inputs'])))].values
    # output_hdf = output_df[output_df.index.isin(np.arange(len(original_data['output'])))].values.flatten()
    # flow_type_hdf = flow_type_df[flow_type_df.index.isin(np.arange(len(original_data['flow_type'])))].values
    # unnormalized_inputs_hdf = unnormalized_inputs_df[
    #     unnormalized_inputs_df.index.isin(np.arange(len(original_data['unnormalized_inputs'])))].values
    #
    # print(f"\nSubcase: {subcase}")
    # print(f"  Inputs match: {np.allclose(original_data['inputs'], inputs_hdf)}")
    # print(f"  Output match: {np.allclose(original_data['output'], output_hdf)}")
    # print(f"  Flow type match: {np.array_equal(original_data['flow_type'].astype(str), flow_type_hdf.astype(str))}")
    # print(
    #     f"  Unnormalized inputs match: {np.allclose(original_data['unnormalized_inputs'].flatten(), unnormalized_inputs_hdf.flatten(), rtol=1e-5, atol=1e-4)}")
