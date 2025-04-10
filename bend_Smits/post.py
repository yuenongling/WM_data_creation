import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.interpolate import interp1d

def smooth_curve(xcp, cp, num_points=100):
    """
    Smooths a curve by adding more points using interpolation.

    Args:
        xcp (numpy.ndarray): The x-coordinates of the curve.
        cp (numpy.ndarray): The y-coordinates of the curve.
        num_points (int): The desired number of points in the smoothed curve.

    Returns:
        tuple: A tuple containing the smoothed x-coordinates and y-coordinates.
    """

    # Ensure xcp is monotonically increasing, if it isn't, sort the xcp and cp based on xcp.
    sorted_indices = np.argsort(xcp)
    xcp_sorted = xcp[sorted_indices]
    cp_sorted = cp[sorted_indices]

    # Create an interpolation function
    f = interp1d(xcp_sorted, cp_sorted, kind='quadratic')  # Use cubic interpolation for smoothness

    # Generate new x-coordinates for the smoothed curve
    xcp_smooth = np.linspace(xcp_sorted.min(), xcp_sorted.max(), num_points)

    # Generate the corresponding y-coordinates
    cp_smooth = f(xcp_smooth)

    return xcp_smooth, cp_smooth


# Define constants similar to the second file
UP_FRAC = 0.25
DOWN_FRAC = 0.01

# --- Set up paths and constants ---
import os
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
savedatapath = os.path.join(WM_DATA_PATH, 'data')
currentpath = os.path.join(WM_DATA_PATH, 'bend_Smits')
statspath = os.path.join(WM_DATA_PATH, 'bend_Smits', 'stats')

# Load your saved CSV files
flow_data = pd.read_csv(statspath + '/Smits_flow_raw_data.csv')
xcp_data = pd.read_csv(statspath + '/Smits_xcp_raw_data.csv')

# Do some processing of xcp_data to get dPdx and other parameters
xcp = xcp_data['X'].values
cp  = xcp_data['CP'].values
dCPdx = np.gradient(cp, xcp)  # Calculate dP/dx

# WARNING: Smooth the curve if needed
xcp, cp = smooth_curve(xcp, cp, num_points=1000)
dCPdx = np.gradient(cp, xcp)

# import matplotlib.pyplot as plt
# plt.plot(xcp, dCPdx, '-o')
# plt.plot(xcp_sm, dCPdx_sm, '--')
# plt.show()


all_inputs_data = []
all_output_data = []
all_flow_type_data = []
all_unnormalized_inputs_data = []

# Get velocity profiles from saved CSV files
station_data = []
for i in range(1, 8):  # Stations 1-7
    try:
        station_df = pd.read_csv(statspath + f'/Station_{i}_velocity_profile.csv')
        station_data.append(station_df)
    except FileNotFoundError:
        print(f"Warning: Station_{i}_velocity_profile.csv not found")

# Combine flow data with pressure gradient information from xcp_data
# We'll use X values to match the data points
combined_data = []

# We need to calculate some additional parameters similar to the second file
# Let's set up the processing for each station
for i, station_df in enumerate(station_data):
    
    x = flow_data['X'].iloc[i]
    nu = flow_data['NU'].iloc[i]
    # Get boundary layer parameters from flow data
    UE = flow_data['UE'].iloc[i]
    DEL995 = flow_data['DEL995'].iloc[i]
    DELS = flow_data['DELS'].iloc[i]
    CF   = flow_data['CFE'].iloc[i]

    # up
    dPdx = interpolate.interp1d(xcp, dCPdx, fill_value="extrapolate")(x) * 0.5*UE**2  # Interpolating dP/dx at this X position
    up = np.sign(dPdx) * (abs(nu * dPdx)) ** (1/3)

    # utau
    # Calculate utau from Cf
    utau = np.sqrt(CF/2) * UE
    
    # Extract data for this position
    y_values = station_df['Y/DEL995'].values
    U_values = station_df['U/UE'].values
    
    # Calculate y in physical units (not normalized)
    y_physical = y_values * DEL995
    
    # Calculate U in physical units (not normalized)
    U_physical = U_values * UE
    
    # Filter points for processing (similar to the second file)
    bot_index = np.where((y_physical >= DOWN_FRAC*DEL995) & (y_physical <= UP_FRAC*DEL995))[0]
    
    if len(bot_index) > 0:
        # Calculate quantities at k*y values
        U2 = find_k_y_values(y_physical[bot_index], U_physical, y_physical, k=1)
        U3 = find_k_y_values(y_physical[bot_index], U_physical, y_physical, k=2)
        U4 = find_k_y_values(y_physical[bot_index], U_physical, y_physical, k=3)
        
        # Calculate dimensionless parameters
        pi_1 = y_physical[bot_index] * U_physical[bot_index] / nu
        pi_2 = up * y_physical[bot_index] / nu
        pi_3 = U2 * y_physical[bot_index] / nu
        pi_4 = U3 * y_physical[bot_index] / nu
        pi_5 = U4 * y_physical[bot_index] / nu
        
        # Calculate velocity gradients
        dUdy = np.gradient(U_physical, y_physical)
        dudy_1 = dUdy[bot_index]
        dudy_2 = find_k_y_values(y_physical[bot_index], dUdy, y_physical, k=1)
        dudy_3 = find_k_y_values(y_physical[bot_index], dUdy, y_physical, k=2)
        
        pi_6 = dudy_1 * y_physical[bot_index]**2 / nu
        pi_7 = dudy_2 * y_physical[bot_index]**2 / nu
        pi_8 = dudy_3 * y_physical[bot_index]**2 / nu
        
        # Output parameter
        pi_out = utau * y_physical[bot_index] / nu
        
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
            'y': y_physical[bot_index],
            'u1': U_physical[bot_index],
            'nu': np.full_like(y_physical[bot_index], nu),
            'utau': np.full_like(y_physical[bot_index], utau),
            'up': np.full_like(y_physical[bot_index], up),
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
        len_y = len(y_physical[bot_index])
        flow_type_dict = {
            'case_name': ['bend_Smits'] * len_y,
            'nu': [nu] * len_y,
            'x': [x] * len_y,
            'delta': [DEL995] * len_y,
            'temp': [0] * len_y,
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
output_filename = os.path.join(savedatapath, 'bend_data.h5')
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
