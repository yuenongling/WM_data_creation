from scipy.io import loadmat
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from airfoil_util import *
import pandas as pd
import os

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
savedatapath = os.path.join(WM_DATA_PATH, 'data')
datapath = os.path.join(WM_DATA_PATH, 'airfoil_KTH', 'data')

# --- Set up cases ---
subcases = ['0012', '4412']
Res = {}
Res['0012'] = [400_000]
Res['4412'] = [100_000, 200_000, 400_000, 1_000_000]

UP_FRAC = 0.20
DOWN_FRAC = 0.025

for case in subcases:
    cos_theta = np.loadtxt(datapath + f'/cosine_angle_naca{case}.txt', skiprows=1)

    # result = results[subcase]
    results = loadmat(datapath + f'/naca{case}.mat', squeeze_me = True)

    for Re in Res[case]:

        all_inputs_data = []
        all_output_data = []
        all_flow_type_data = []
        all_unnormalized_inputs_data = []

        if case == '0012':
            case_string = f'top4n12'
        else:
            case_string = f'top{int(Re // 100_000)}n'
        result = results[case_string]
        print('Investigating case:', case_string, 'for Re =', Re)

        # Read data
        Cf = result['Cf'] # Cf = 2(utau / Ue)^2

        U = result['U']
        Ue = result['Ue']
        utau = result['ut']
        # Retau = [utau[i] * delta[i] / nu[i] for i, _ in enumerate(utau)]

        delta = result['delta99']
        x    = result['xa'].astype(float)
        radius_curve = naca_airfoil_curvature(x, case)
        delta_over_r = delta / radius_curve['radius']
        print('delta_over_r', delta_over_r)
        plt.plot(x, delta_over_r, label=f'naca_{case}_{case_string}')


        deltas = result['deltas']
        beta = result['beta']
        theta = result['theta']
        nu = result['nu']

        ya    = result['ya'].astype(float)
        cos  = np.interp(x, cos_theta[:,0], cos_theta[:,1])
        # NOTE: Local wall-normal coordinate
        y = result['yn']
        U = result['U']

        # WARNING: Here dPdx is edge pressure gradient
        # We do correction here where we 
        #   1. Compare the pressure difference of near-wall and edge
        #   2. Multiply the edge pressure gradient with the ratio of near-wall and edge pressure difference
        dPdx = beta * utau**2 / deltas # NOTE: get back dPdx from beta

        up   = np.sign(dPdx) * (abs(nu * dPdx)) ** (1/3)
        Ue   = result['Ue']

        # Albert parameter: alber = theta / Ue**2 * dPdx
        albert = theta / Ue**2 * dPdx

        for idx, x_ in enumerate(x):

            if x_ > 0.952:
                continue

            y_i = y[idx]
            U_i = U[idx]

            delta99_i   = delta[idx]
            nu_i = nu[idx]
            x_i  = x_
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

            # print('up', up)

            bot_index = np.where((y_i >= DOWN_FRAC*delta99_i) & (y_i <= UP_FRAC*delta99_i))[0]


            U2 = find_k_y_values(y_i[bot_index], U_i, y_i, k=1)
            U3 = find_k_y_values(y_i[bot_index], U_i, y_i, k=2)
            U4 = find_k_y_values(y_i[bot_index], U_i, y_i, k=3)

            pi_1 = y_i * U_i / nu_i
            pi_1 = pi_1[bot_index]
            pi_2 = up_i * y_i / nu_i
            pi_2 = pi_2[bot_index]
            pi_3 = U2 * y_i[bot_index] / nu_i
            pi_4 = U3 * y_i[bot_index] / nu_i
            pi_5 = U4 * y_i[bot_index] / nu_i

            # NOTE: Velocity gradient
            dUdy = np.gradient(U_i, y_i)
            dudy_1 = dUdy[bot_index]
            dudy_2 = find_k_y_values(y_i[bot_index], dUdy, y_i, k=1)
            dudy_3 = find_k_y_values(y_i[bot_index], dUdy, y_i, k=2)

            pi_6 = dudy_1 * y_i[bot_index]**2 / nu_i
            pi_7 = dudy_2 * y_i[bot_index]**2 / nu_i
            pi_8 = dudy_3 * y_i[bot_index]**2 / nu_i

            pi_out = utau_i * y_i / nu_i
            pi_out = pi_out[bot_index]

            # --- Calculate Input Features (Pi Groups) ---
            # Note:  dPdx is NOT zero here
            # Calculate dimensionless inputs using safe names
            inputs_dict = {
                'u1_y_over_nu': U_i[bot_index] * y_i[bot_index] / nu_i,
                'up_y_over_nu': up_i * y_i[bot_index] / nu_i,  # pi_2 (is NOT zero for APG)
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
                'case_name': ['naca_'+case+'_'+case_string] * len(y_i[bot_index]),
                'nu': [nu_i] * len(y_i[bot_index]),
                'x': [x_i] * len(y_i[bot_index]),
                'delta': [delta99_i] * len(y_i[bot_index]),
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
        savename = 'naca_' + case + '_' + case_string + '_data.h5'
        output_filename = os.path.join(savedatapath, savename)
        print(f"\nSaving data to HDF5 file: {output_filename}")
        inputs_df.to_hdf(output_filename, key='inputs', mode='w', format='fixed')
        output_df.to_hdf(output_filename, key='output', mode='a', format='fixed')
        unnormalized_inputs_df.to_hdf(output_filename, key='unnormalized_inputs', mode='a', format='fixed')
        # Use table format for flow_type if it contains strings, to keep them
        flow_type_df.to_hdf(output_filename, key='flow_type', mode='a', format='table')
        print("Data successfully saved.")

        # --- Sanity Check ---
        print("\n--- Sanity Check: Comparing HDF5 with Original Pickle ---")
        with open('/home/yuenongling/Codes/BFM/WM_Opt/data/' + 'naca_' + case + '_' + case_string + '_data.pkl', 'rb') as f:
            original_data = pkl.load(f)

        # Load corresponding data from HDF5
        inputs_hdf = inputs_df[inputs_df.index.isin(np.arange(len(original_data['inputs'])))].values
        output_hdf = output_df[output_df.index.isin(np.arange(len(original_data['output'])))].values.flatten()
        flow_type_hdf = flow_type_df[flow_type_df.index.isin(np.arange(len(original_data['flow_type'])))].values
        unnormalized_inputs_hdf = unnormalized_inputs_df[
            unnormalized_inputs_df.index.isin(np.arange(len(original_data['unnormalized_inputs'])))].values

        print(f"\nSubcase: naca_{case}_{case_string}")
        print(f"  Inputs match: {np.allclose(original_data['inputs'], inputs_hdf)}")
        print(f"  Output match: {np.allclose(original_data['output'], output_hdf)}")
        print(f"  Flow type match: {np.array_equal(original_data['flow_type'].astype(str), flow_type_hdf.astype(str))}")
        print(
            f"  Unnormalized inputs match: {np.allclose(original_data['unnormalized_inputs'].flatten(), unnormalized_inputs_hdf.flatten(), rtol=1e-5, atol=1e-4)}")
