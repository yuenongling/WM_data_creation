import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import h5py
import os
import pandas as pd

cases = [ {'fn':'ztmd_zpgc.h5',
            'lbl':r'ZPG-C',
            'c':'#37b24d', ## green
            },
            {'fn':'ztmd_pgc.h5',
            'lbl':r'PG-C',
            'c':'#228be6', ## blue
            },
        ]

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
datapath = os.path.join(WM_DATA_PATH, 'data')
STATS_FILE = os.path.join(WM_DATA_PATH, 'curved_TBL_Appelbaum', 'stats')

UP_FRAC = 0.2
DOWN_FRAC = 0.01

def find_k_y_values(y, U_all, y_all, k=2):
    '''
    Given U and y array with the same length, find the U value at k*y values.
    '''
    U_at_ky = np.interp((2*k+1)*y, y_all, U_all)
    return U_at_ky

fig_curve, ax_curve = plt.subplots(1, 1, figsize=(6, 6))

for subcase in cases:

    filename = subcase['fn'] ## filename

    # WARNING: Only extract pg case
    if filename != 'ztmd_pgc.h5':
        continue

    # inputs = np.empty((1,8))
    # output = np.empty((1,))
    # flow_type = np.empty((1,5))
    # unnormalized_inputs = np.empty((1,11))

    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []

    with h5py.File(os.path.join(STATS_FILE,filename), 'r') as h5file:
        lchar   = h5file.attrs['lchar']
        p_inf   = h5file.attrs['p_inf']
        rho_inf = h5file.attrs['rho_inf']
        U_inf   = h5file.attrs['U_inf']

        Retau   = np.copy(h5file['data_1Dx/Re_tau'][()])
        Medge   = np.copy(h5file['data_1Dx/M_edge'][()])
        delta99 = np.copy(h5file['data_1Dx/d99'][()])
        tauw    = np.copy(h5file['data_1Dx/tau_wall'][()])
        nu_w    = np.copy(h5file['data_1Dx/nu_wall'][()])
        rho_w   = np.copy(h5file['data_1Dx/rho_wall'][()])

        x       = np.copy(h5file['dims/stang']) ## streamwise path length at wall
        y       = np.copy(h5file['dims/snorm']) ## wall-normal coordinate
        x_phys  = np.copy(h5file['dims/x']) 
        y_phys  = np.copy(h5file['dims/y']) 

        p_wall  = np.copy(h5file['data/p'][()].T)[:,0]

        R_local = np.copy(h5file['dims/crv_R'])

        rho = np.copy(h5file['data/rho'][()].T)
        U   = np.copy(h5file['data/u'][()].T)
        nu  = np.copy(h5file['data/nu'][()].T)

        # NOTE: Derived quantities
        dPdx    = np.gradient(p_wall, x)
        up      = np.sign(dPdx) * (abs(nu_w * dPdx / rho_w)) ** (1/3)
        utau    = np.sqrt(tauw / rho_w)

        # NOTE: Check utau by comparing with Retau
        # Already checked, they are the same
        # Retau_mycalc = delta99 * utau / nu_w
        #
        delta_over_R = delta99 / R_local
        

    for idx, x_ in enumerate(x):

        if idx % 100 == 0:
            print(f'processing {x_} [{idx}] ...')
        else:
            continue

        U_i = U[idx]
        y_i = y
        delta99_i   = delta99[idx]
        utau_i      = utau[idx]
        up_i        = up[idx]

        nu_i        = nu[idx]

        # Skip the first point if it is zero
        if U_i[0] == 0 or y_i[0] == 0:
            y_i = y_i[1:]
            U_i = U_i[1:]
            nu_i = nu_i[1:]

        bot_index = np.where((y_i >= DOWN_FRAC*delta99_i) & (y_i <= UP_FRAC*delta99_i))[0]

        U2 = find_k_y_values(y_i[bot_index], U_i, y_i, k=1)
        U3 = find_k_y_values(y_i[bot_index], U_i, y_i, k=2)
        U4 = find_k_y_values(y_i[bot_index], U_i, y_i, k=3)

        # NOTE: Inputs
        pi_1 = y_i * U_i / nu_i
        pi_1 = pi_1[bot_index]
        pi_2 = up_i * y_i / nu_i
        pi_2 = pi_2[bot_index]
        pi_3 = U2 * y_i[bot_index] / nu_i[bot_index]
        pi_4 = U3 * y_i[bot_index] / nu_i[bot_index]
        pi_5 = U4 * y_i[bot_index] / nu_i[bot_index]

        # NOTE: Velocity gradient
        dUdy = np.gradient(U_i, y_i)
        dudy_1 = dUdy[bot_index]
        dudy_2 = find_k_y_values(y_i[bot_index], dUdy, y_i, k=1)
        dudy_3 = find_k_y_values(y_i[bot_index], dUdy, y_i, k=2)

        pi_6 = dudy_1 * y_i[bot_index]**2 / nu_i[bot_index] 
        pi_7 = dudy_2 * y_i[bot_index]**2 / nu_i[bot_index] 
        pi_8 = dudy_3 * y_i[bot_index]**2 / nu_i[bot_index] 

        # --- Calculate Input Features (Pi Groups) ---
        # Calculate dimensionless inputs using safe names
        inputs_dict = {
            'u1_y_over_nu': pi_1,
            'up_y_over_nu': pi_2,
            'upn_y_over_nu': 0 * pi_2,
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
            'utau_y_over_nu': utau_i * y_i[bot_index] / nu_i[bot_index]
        }
        all_output_data.append(pd.DataFrame(output_dict))

        # NOTE: Also collect unnormalized inputs
        # 1. y coordinate 
        # 2. U velocity 
        # 3. kinematic viscosity
        # 4. friction velocity
        # 5. pressure gradient velocity
        # 6. U2 velocity
        # 7. U3 velocity
        # 8. U4 velocity
        # 6. dudy1 velocity
        # 7. dudy2 velocity
        # 8. dudy3 velocity

        # --- Collect Unnormalized Inputs ---
        unnorm_dict = {
            'y': y_i[bot_index],
            'u1': U_i[bot_index],
            'nu': nu_i[bot_index],
            'utau': np.full_like(y_i[bot_index], utau_i),
            'up': np.full_like(y_i[bot_index], up_i),
            'upn': np.full_like(y_i[bot_index], 0),
            'u2': U2,
            'u3': U3,
            'u4': U4,
            'dudy1': np.gradient(U_i, y_i)[bot_index],
            'dudy2': find_k_y_values(y_i[bot_index], np.gradient(U_i, y_i), y_i, k=1),
            'dudy3': find_k_y_values(y_i[bot_index], np.gradient(U_i, y_i), y_i, k=2)
        }
        all_unnormalized_inputs_data.append(pd.DataFrame(unnorm_dict))

        # WARNING: Flow type is an array of five elements
        # 1. Flow type
        # 2. Reynolds number (can be used to normalize data)
        # 3. x coordinate (unnormalized)
        # 4. delta (boundary layer thickness)
        # 5. ...
        # flow_type_tmp = np.tile(['curve', nu_i[bot_index], x[idx], delta99_i, 0], (len(bot_index), 1))
        # --- Collect Flow Type Information ---
        # Using format: [case_name, reference_nu, x_coord, delta, edge_velocity]
        # For channel flow: x=0, delta=1 (half-channel height), Ue=0 (or U_bulk if needed)
        flow_type_dict = {
            'case_name': ['curved_BL'] * len(y_i[bot_index]),
            'nu': nu_i[bot_index],
            'x': [x_] * len(y_i[bot_index]),
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
    output_filename = os.path.join(datapath, 'curved_TBL_FPG_data.h5')
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
