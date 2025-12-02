from scipy.io import loadmat
import numpy as np
import pandas as pd
import os
import pickle as pkl  # Import pickle

def save_to_hdf5(idx_all, xall, y, U, P, nu, utau, dPdx, up, delta99, reg_name='laminar'):

    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []

# Albert parameter: alber = theta / Ue**2 * dPdx
    for idx in idx_all:
        x = xall[idx]

        # Downsample the data
        if idx % 10 != 0:
            continue

        # if albert[i] > 0.1:
        #     break
        y_i = y
        U_i = U[:, idx]
        P_i = P[:, idx]

        delta99_i = delta99[idx]
        nu_i = nu
        x_i = xall[idx]
        utau_i = utau[idx]
        dPdx_i = dPdx[idx]
        up_i = up[idx]

        delta_p = P_i - P_i[0]  # Pressure difference from the first point
        up_n_i = np.sign(delta_p) * (abs(nu_i * delta_p)) ** (1 / 3)

        if reg_name == 'laminar':
            bot_index = np.where((y_i >= 0.05 * delta99_i) & (y_i <= UP_FRAC * delta99_i))[0]
        else:
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
            'case_name': [f'transition_JHU_{reg_name}'] * len(y_i[bot_index]),
            'nu': [nu_i] * len(y_i[bot_index]),
            'x': [x_i] * len(y_i[bot_index]),
            'delta': [delta99_i] * len(y_i[bot_index]),
            'albert': [0] * len(y_i[bot_index])
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
    output_filename = os.path.join(datapath, f'transition_JHU_{reg_name}_data.h5')
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

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
datapath = os.path.join(WM_DATA_PATH, 'data')
DATA_FILE = os.path.join(WM_DATA_PATH, 'transition_JHU', 'stats', 'stats.pkl')

# --- Load the APG data ---
with open(DATA_FILE, 'rb') as f:
    result = pkl.load(f)

UP_FRAC = 0.2
DOWN_FRAC = 0.01



# Read data
tauw = result['tauw']  # Wall shear stress
utau = np.sqrt(tauw)  # Wall shear velocity

U = result['um']
y = result['y']
x = result['x']
xall = result['x']
nu = 1.25e-3
P    = result['p']
delta99 = result['delta99']

dPdx = np.gradient(P[0, :], xall)  # Gradient of pressure with respect to x
# Smooth the pressure gradient
dPdx = np.convolve(dPdx, np.ones(20)/20, mode='same')
up = np.sign(dPdx) * (abs(nu * dPdx)) ** (1 / 3)

# Depending on xall, we save to different regions
x_laminar = xall[(xall > 100) & (xall < 200)]
x_laminar_idx = np.where((xall > 100) & (xall < 200))[0]
x_transition_idx = np.where((xall > 200) & (xall < 460))[0]
x_turbulent_idx = np.where((xall > 460) & (xall < 800))[0]

save_to_hdf5(x_laminar_idx   , x, y, U, P, nu, utau, dPdx, up, delta99, reg_name='laminar')
save_to_hdf5(x_transition_idx, x, y, U, P, nu, utau, dPdx, up, delta99, reg_name='transition')
save_to_hdf5(x_turbulent_idx , x, y, U, P, nu, utau, dPdx, up, delta99, reg_name='turbulent')



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
