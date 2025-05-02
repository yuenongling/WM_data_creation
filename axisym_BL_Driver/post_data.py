import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import os
from scipy import interpolate

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
savedatapath = os.path.join(WM_DATA_PATH, 'data')
currentpath = os.path.join(WM_DATA_PATH, 'axisym_BL_Driver')
statspath = os.path.join(currentpath, 'stats')

cp_file = os.path.join(statspath, 'extracted_data_cp.csv')
cf_file = os.path.join(statspath, 'extracted_data_cf.csv')
profile_file = os.path.join(statspath, 'extracted_data_profile.csv')

# Define constants from the image
M_ref = 0.08812
Re_H = 2_000_000
T_ref = 527
H = 1.0  # Step height (normalized to 1.0)
Uref = 1.0  # Reference velocity (normalized)

# Fractions for defining the boundary layer region of interest
UP_FRAC = 0.2    # Upper fraction of boundary layer to consider
DOWN_FRAC = 0.005 # Lower fraction of boundary layer to consider

def process_data(cf_file, cp_file, profile_file, output_path="./"):
    """
    Process the extracted CSV files to calculate non-dimensional parameters
    """
    # Load the extracted data
    cf_data = pd.read_csv(cf_file)
    cp_data = pd.read_csv(cp_file)
    profile_data = pd.read_csv(profile_file)

    # Get unique stations
    stations = profile_data['station'].unique()
    
    # Reference values
    nu = 1.0 / Re_H  # Non-dimensional kinematic viscosity

    # Calculate the pressure gradient
    dPdx = np.gradient(cp_data['cp'].values, cp_data['x'].values) * 0.5 * Uref**2
    up = np.sign(dPdx) * (nu * abs(dPdx))**(1/3)
    
    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []
    
    # WARNING: Hardcoded by looking at the data in the profile file
    delta99 = [1.294999942E-02, 1.294999942E-02, 3.316999972E-02, 6.633999944E-02, 2.565000020E-02, 3.581000119E-02, 
               4.089000076E-02, 4.089000076E-02, 4.597000033E-02, 4.597000033E-02, 5.612999946E-02, 5.612999946E-02, 6.120999902E-02]

    for i, station in enumerate(stations):
        print(f"Processing station {station}")
        
        # Extract profile data for this station
        station_data = profile_data[profile_data['station'] == station]
        
        # Get x position from the station name
        x = float(station)
        
        # Get the nearest Cf value to this x position
        cf_idx = np.argmin(abs(cf_data['x'].values - x))
        Cf = np.interp(x, cf_data['x'].values, cf_data['cf'].values)
        
        # Calculate utau from Cf
        utau_i = np.sqrt(abs(Cf)/2) * Uref
        
        # Extract y and U data
        y_i = station_data['y'].values

        U_i = station_data['u'].values
        
        # Boundary layer thickness (estimate from max y value)
        delta99_i = delta99[i]
        
        # Skip the first point if it is zero
        if U_i[0] == 0 or y_i[0] == 0:
            y_i = y_i[1:]
            U_i = U_i[1:]
        
        # Calculate pressure gradient velocity scale
        up_i = np.interp(x, cp_data['x'].values, up)
        
        # Calculate Albert parameter
        albert_i = 0
        
        # Find points within the boundary layer region of interest
        bot_index = np.where((y_i >= DOWN_FRAC*delta99_i) & (y_i <= UP_FRAC*delta99_i))[0]
        print(f"  Found {len(bot_index)} points in boundary layer region")

        if len(bot_index) < 2:
            print(f"  Skipping station {station}: not enough points in boundary layer")
            continue
        
        # Calculate velocity at different y positions
        U2 = find_k_y_values(y_i[bot_index], U_i, y_i, k=1)
        U3 = find_k_y_values(y_i[bot_index], U_i, y_i, k=2)
        U4 = find_k_y_values(y_i[bot_index], U_i, y_i, k=3)
        
        # Calculate non-dimensional inputs
        pi_1 = y_i * U_i / nu
        pi_1 = pi_1[bot_index]
        pi_2 = up_i * y_i / nu if up_i != 0 else np.zeros_like(y_i[bot_index])
        pi_2 = pi_2[bot_index]
        pi_3 = U2 * y_i[bot_index] / nu
        pi_4 = U3 * y_i[bot_index] / nu
        pi_5 = U4 * y_i[bot_index] / nu
        
        # Calculate velocity gradients
        dUdy = np.gradient(U_i, y_i)
        dudy_1 = dUdy[bot_index]
        dudy_2 = find_k_y_values(y_i[bot_index], dUdy, y_i, k=1)
        dudy_3 = find_k_y_values(y_i[bot_index], dUdy, y_i, k=2)
        
        pi_6 = dudy_1 * y_i[bot_index]**2 / nu
        pi_7 = dudy_2 * y_i[bot_index]**2 / nu
        pi_8 = dudy_3 * y_i[bot_index]**2 / nu
        
        # Calculate output (y+)
        pi_out = utau_i * y_i / nu
        pi_out = pi_out[bot_index]
        # Flow type information
        flow_type_tmp = np.array([
            ['axisym_BL', nu, x, delta99_i, albert_i] for _ in range(len(bot_index))
        ], dtype=object)
        
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
            'nu': np.full_like(y_i[bot_index], nu), 
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
        'axissym_BL', nu, x, delta99_i, albert_i
        flow_type_dict = {
            'case_name': ['axissym_BL'] * len_y,
            'nu': [nu] * len_y,
            'x': [x] * len_y,
            'delta': [delta99_i] * len_y,
            'temp': [albert_i] * len_y,
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
    output_filename = os.path.join(savedatapath, 'axisym_BL_data.h5')
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

# # --- Sanity Check ---
#     print("\n--- Sanity Check: Comparing HDF5 with Original Pickle ---")
#     with open('/home/yuenongling/Codes/BFM/WM_Opt/data/backstep_data.pkl', 'rb') as f:
#         original_data = pkl.load(f)
#
# # Load corresponding data from HDF5
#     inputs_hdf = inputs_df[inputs_df.index.isin(np.arange(len(original_data['inputs'])))].values
#     output_hdf = output_df[output_df.index.isin(np.arange(len(original_data['output'])))].values.flatten()
#     flow_type_hdf = flow_type_df[flow_type_df.index.isin(np.arange(len(original_data['flow_type'])))].values
#     unnormalized_inputs_hdf = unnormalized_inputs_df[
#         unnormalized_inputs_df.index.isin(np.arange(len(original_data['unnormalized_inputs'])))].values
#
#     print(f"  Inputs match: {np.allclose(original_data['inputs'], inputs_hdf)}")
#     print(f"  Output match: {np.allclose(original_data['output'], output_hdf)}")
#     print(f"  Flow type match: {np.array_equal(original_data['flow_type'].astype(str), flow_type_hdf.astype(str))}")
#     print(
#         f"  Unnormalized inputs match: {np.allclose(original_data['unnormalized_inputs'].flatten(), unnormalized_inputs_hdf.flatten(), rtol=1e-5, atol=1e-4)}")
 
# Process the data
process_data(cf_file, cp_file, profile_file, savedatapath)
