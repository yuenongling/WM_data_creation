import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
datapath = os.path.join(WM_DATA_PATH, 'data')
statspath = os.path.join(WM_DATA_PATH, 'diffuer_Ohlsson', 'stats')
UP_FRAC = 0.2
DOWN_FRAC = 0.01
Re = 10000 # Bulk Reynolds number
nu = 1.0 / Re  # Kinematic viscosity

FILE_FORMAT = lambda x: statspath + f'/profiles/C13.2_z0500_x{x}_KTH_DNS.txt'

L = 15
x_all = np.array([-2, 0, 2, 4, 6, 8, 10, 12, 14])

Cf = np.loadtxt(statspath + "/cf.dat", delimiter=',')
dpdx = np.loadtxt(statspath + "/dpdx.dat", delimiter=',')
# dpdx is 100 times the real value
dpdx[:,1] = dpdx[:,1] / 100

dpdx_interpolated = np.interp(x_all/L, dpdx[:, 0], dpdx[:, 1])
Cf_interpolated = np.interp(x_all/L, Cf[:, 0], Cf[:, 1])

# Test interpolation
plt.plot(dpdx[:, 0], dpdx[:, 1], label='Original dpdx')
plt.plot(x_all/L, dpdx_interpolated, 'x', label='Interpolated dpdx')
plt.show()
plt.plot(Cf[:, 0], Cf[:, 1], label='Original Cf')
plt.plot(x_all/L, Cf_interpolated, 'x', label='Interpolated Cf')
plt.show()

# Initialize lists to collect data
all_inputs_data = []
all_output_data = []
all_flow_type_data = []
all_unnormalized_inputs_data = []

for i, x in enumerate(x_all):

    profile = np.loadtxt(FILE_FORMAT(x), comments='#', usecols=(1, 3, 5))

    y = profile[:,0]
    U = np.sqrt(profile[:, 1]**2 + profile[:, 2]**2)
    tauw_i = Cf_interpolated[i] / 2
    utau_i = np.sqrt(tauw_i)
    dpdx_i = dpdx_interpolated[i]
    up_i = np.sign(dpdx_i) * (abs(dpdx_i * nu)) ** (1 / 3)

    # We only care about the lower half of the domain
    # Take max U and find the y values
    max_index = np.argmax(U)
    delta = y[max_index] # boundary layer thickness
    # Skip top half of the domain
    y_i = y[:max_index]
    U_i = U[:max_index]

    # Skip the first point if it is zero
    if U_i[0] == 0 or y_i[0] == 0:
        y_i = y_i[1:]
        U_i = U_i[1:]

    bot_index = np.where((y_i >= DOWN_FRAC * delta) & (y_i <= UP_FRAC * delta))[0]

    U2 = find_k_y_values(y_i[bot_index], U_i, y_i, k=1)
    U3 = find_k_y_values(y_i[bot_index], U_i, y_i, k=2)
    U4 = find_k_y_values(y_i[bot_index], U_i, y_i, k=3)

    # --- Calculate Input Features (Pi Groups) ---
    # Note:  dPdx is NOT zero here
    # Calculate dimensionless inputs using safe names
    inputs_dict = {
        'u1_y_over_nu': U_i[bot_index] * y_i[bot_index] / nu,
        'up_y_over_nu': up_i * y_i[bot_index] / nu,  # pi_2 (is NOT zero for APG)
        'upn_y_over_nu': 0 * y_i[bot_index] / nu,  # pi_2 (is NOT zero for APG)
        'u2_y_over_nu': U2 * y_i[bot_index] / nu,
        'u3_y_over_nu': U3 * y_i[bot_index] / nu,
        'u4_y_over_nu': U4 * y_i[bot_index] / nu,
        'dudy1_y_pow2_over_nu': np.gradient(U_i, y_i)[bot_index] * y_i[bot_index] ** 2 / nu,
        'dudy2_y_pow2_over_nu': find_k_y_values(y_i[bot_index], np.gradient(U_i, y_i), y_i, k=1) * y_i[
            bot_index] ** 2 / nu,
        'dudy3_y_pow2_over_nu': find_k_y_values(y_i[bot_index], np.gradient(U_i, y_i), y_i, k=2) * y_i[
            bot_index] ** 2 / nu
    }
    all_inputs_data.append(pd.DataFrame(inputs_dict))

    # --- Calculate Output Feature ---
    # Output is y+ (utau * y / nu)
    output_dict = {
        'utau_y_over_nu': utau_i * y_i[bot_index] / nu
    }
    all_output_data.append(pd.DataFrame(output_dict))

    # --- Collect Unnormalized Inputs ---
    unnorm_dict = {
        'y': y_i[bot_index],
        'u1': U_i[bot_index],
        'nu': np.full_like(y_i[bot_index], nu),
        'utau': np.full_like(y_i[bot_index], utau_i),
        'up': np.full_like(y_i[bot_index], up_i),
        'upn': 0, 
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
        'case_name': ['diffuser'] * len(y_i[bot_index]),
        'nu': [nu] * len(y_i[bot_index]),
        'x': [x] * len(y_i[bot_index]),
        'delta': [delta] * len(y_i[bot_index]),
    }
    # Add Retau for reference if needed, maybe as an extra column or replacing 'edge_velocity'
    # flow_type_dict['Retau'] = [Re_num] * len(y_sel)
    all_flow_type_data.append(pd.DataFrame(flow_type_dict))

    # Plot the data for visual inspection
    # plt.plot(y, U, label='C13.2_z0500_x0_KTH_DNS')
    # plt.axvline(y[max_index], color='r', linestyle='--', label='y_max')
    # plt.title(f'x = {float(x)/L:.3f}')
    # plt.show()



# Concatenate data from all Re_num cases into single DataFrames
inputs_df = pd.concat(all_inputs_data, ignore_index=True)
output_df = pd.concat(all_output_data, ignore_index=True)
flow_type_df = pd.concat(all_flow_type_data, ignore_index=True)
unnormalized_inputs_df = pd.concat(all_unnormalized_inputs_data, ignore_index=True)

# Save DataFrames to HDF5 file
output_filename = os.path.join(datapath, 'diffuser_data.h5')
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
