import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle as pkl
import pandas as pd
import os
import re

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
savedatapath = os.path.join(WM_DATA_PATH, 'data')
dir = os.path.join(WM_DATA_PATH, 'airfoil_Tamaki', '2M')
# Investigate x that are available in the tamaki dataset
x_to_investigate = np.array([0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.825, 0.87, 0.93, 0.99])
# x_to_investigate = np.array([0.2, 0.7, 0.825, 0.87])
Re = 2_100_000

UP_FRAC = 0.20
DOWN_FRAC = 0.01

# NOTE: Function used for calculating directional derivative of pressure 
def calculate_directional_derivative(f_values, curve_points):
    """
    Calculate the directional derivative of function values along a curve.
    
    Parameters:
    f_values (np.ndarray): Function values at points along the curve
    curve_points (np.ndarray): Points along the curve where function is evaluated
                              Shape should be (n, dim) where n is number of points
                              and dim is the dimension of the space
    
    Returns:
    np.ndarray: Directional derivative values along the curve
    """
    # Calculate arc length between consecutive points
    arc_lengths = np.sqrt(np.sum(np.diff(curve_points, axis=0)**2, axis=1))
    
    # Calculate cumulative arc length
    cum_arc_length = np.concatenate(([0], np.cumsum(arc_lengths)))
    
    # Calculate gradient of function values with respect to arc length
    directional_derivative = np.gradient(f_values, cum_arc_length)
    
    return directional_derivative


# First read surface data
surface_data = np.loadtxt(dir+f'/surface_data.txt', skiprows=1)
# NOTE: Delete data from the pressure side
# Integral boundary layer parameters are blank for the pressure side
surface_data = surface_data[np.where(surface_data[:,2] > 0)]
x_surf = surface_data[:,0]
Cp = surface_data[:,1]
def gaussian_kernel(x, xi, bandwidth):
    """
    Gaussian kernel function.
    
    Parameters:
    x (float): The point at which the kernel is evaluated.
    xi (float): The center of the kernel.
    bandwidth (float): The bandwidth of the kernel (controls the spread).
    
    Returns:
    float: The value of the Gaussian kernel at x.
    """
    return np.exp(-0.5 * ((x - xi) / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
def local_gaussian_smooth(x, y, bandwidth=0.1):
    """
    Perform local Gaussian smoothing on the data.
    
    Parameters:
    x (array-like): The independent variable values.
    y (array-like): The dependent variable values.
    bandwidth (float): The bandwidth of the Gaussian kernel (default is 1.0).
    
    Returns:
    y_smooth (array): The smoothed dependent variable values.
    """
    x = np.array(x)
    y = np.array(y)
    y_smooth = np.zeros_like(y)
    
    for i, xi in enumerate(x):
        # Compute the Gaussian weights for all points
        weights = gaussian_kernel(x, xi, bandwidth)
        
        # Normalize the weights
        weights /= np.sum(weights)
        
        # Compute the smoothed value as a weighted average
        y_smooth[i] = np.sum(weights * y)
    
    return y_smooth

# Read delta_99 data
delta_99 = np.loadtxt(dir+f'/Delta99.csv', delimiter=',', skiprows=0)

# Define a data dictionary
data_tamaki = {}

delta_99_x = np.interp(x_to_investigate, delta_99[:,0], delta_99[:,1])

all_inputs_data = []
all_output_data = []
all_flow_type_data = []
all_unnormalized_inputs_data = []

for idx, x in enumerate(x_to_investigate):

    if x < 0.10:
        fname = dir+f'/profile_00{x*100:.0f}.0.txt'
    elif x == 0.825:
        fname = dir+f'/profile_082.5.txt'
    else:
        fname = dir+f'/profile_0{x*100:.0f}.0.txt'
    print(f'Investigating {fname}...')

    data = np.loadtxt(fname, skiprows=1)

    if idx == 0:
        yc = np.zeros((data.shape[0],x_to_investigate.shape[0]))
        U = np.zeros((data.shape[0],x_to_investigate.shape[0]))
        yplus = np.zeros((data.shape[0],x_to_investigate.shape[0]))
        uplus = np.zeros((data.shape[0],x_to_investigate.shape[0]))
        dpds  = np.zeros((data.shape[0],x_to_investigate.shape[0]))

        # Inputs and outputs for the wall model
        pi_1  = np.zeros((data.shape[0],x_to_investigate.shape[0]))
        up  = np.zeros((data.shape[0],x_to_investigate.shape[0]))
        pi_2  = np.zeros((data.shape[0],x_to_investigate.shape[0]))
        pi_3  = np.zeros((data.shape[0],x_to_investigate.shape[0]))
        pi_4  = np.zeros((data.shape[0],x_to_investigate.shape[0]))
        pi_5  = np.zeros((data.shape[0],x_to_investigate.shape[0]))
        output  = np.zeros((data.shape[0],x_to_investigate.shape[0]))

        # Surface data and BL data
        Cp = np.zeros((x_to_investigate.shape[0],))
        Cf = np.zeros((x_to_investigate.shape[0],))
        H  = np.zeros((x_to_investigate.shape[0],))
        Re_theta = np.zeros((x_to_investigate.shape[0],))
        Re_delta_star = np.zeros((x_to_investigate.shape[0],))
        utau = np.zeros((x_to_investigate.shape[0],))

        # An index to record the approximate boundary layer thickness
        idx_bl = np.zeros((x_to_investigate.shape[0],), dtype=int)
        idx_bl_quarter = np.zeros((x_to_investigate.shape[0],), dtype=int)
        idx_bl_low = np.zeros((x_to_investigate.shape[0],), dtype=int)


    yc[:, idx] = data[:,0]
    U[:, idx] = data[:,1]

    yplus[:, idx] = data[:,6]
    uplus[:, idx] = data[:,7]

    # Also record the final index with increasing U (an approximation for boundary layer thickness)
    idx_bl[idx] = int(np.where(yc > delta_99_x[idx])[0][0])

    # Find 0.08 boundary layer thickness indices
    idx_bl_quarter[idx] = int(np.where(yc[:, idx] > UP_FRAC *yc[int(idx_bl[idx]), idx])[0][0])
    # Find 0.02 boundary layer thickness indices
    idx_bl_low[idx] = int(np.where(yc[:, idx] > DOWN_FRAC *yc[int(idx_bl[idx]), idx])[0][0])

    # NOTE: Here is the "old" way of calculating dpds
    # It is at x/delta = 0.05, which might not be the best location
    if x < 0.10:
        fname = dir+f'/u_budget_00{x*100:.0f}.0.txt'
    elif x == 0.825:
        fname = dir+f'/u_budget_082.5.txt'
    else:
        fname = dir+f'/u_budget_0{x*100:.0f}.0.txt'
    # WARNING: There seem to be a mis label of pressure gradient in budget file...
    # Need to recheck
    dpds[:, idx] = np.loadtxt(fname, skiprows=1)[:, -3]

    # Surface data
    surface_idx = np.argmin(abs(surface_data[:, 0] - x))
    Cp[idx] = surface_data[surface_idx, 1]
    Cf[idx] = surface_data[surface_idx, 2]
    H[idx] = surface_data[surface_idx, 3]
    Re_theta[idx] = surface_data[surface_idx, 4]
    Re_delta_star[idx] = surface_data[surface_idx, 5]

    # NOTE: Calculate the utau based on plus values (Only works for 10M data)
    avg_utau  =  yplus[10:50, idx] / yc[10:50, idx] /  Re
    print(np.std(avg_utau))
    if np.std(avg_utau) > 0.1:
        raise ValueError('Standard deviation of utau is too high')
    utau[idx] = avg_utau.mean()


    # Chord-based Reynolds number
    # Calcualte u_1 = U * yc * Re
    pi_1[:, idx] = U[:, idx] * yc[:, idx] * Re
    up[:, idx] = np.sign(dpds[0, idx]) * (abs(dpds[0, idx]) / Re)**(1/3)
    U_2 = find_k_y_values(yc[idx_bl_low[idx]:idx_bl_quarter[idx], idx], U[:, idx], yc[:, idx], k=1)
    U_3 = find_k_y_values(yc[idx_bl_low[idx]:idx_bl_quarter[idx], idx], U[:, idx], yc[:, idx], k=2)
    U_4 = find_k_y_values(yc[idx_bl_low[idx]:idx_bl_quarter[idx], idx], U[:, idx], yc[:, idx], k=3)
    pi_2[:, idx] = up[:, idx] * yc[:, idx] * Re
    pi_3[idx_bl_low[idx]:idx_bl_quarter[idx], idx] = U_2 * yc[idx_bl_low[idx]:idx_bl_quarter[idx], idx] * Re
    pi_4[idx_bl_low[idx]:idx_bl_quarter[idx], idx] = U_3 * yc[idx_bl_low[idx]:idx_bl_quarter[idx], idx] * Re
    pi_5[idx_bl_low[idx]:idx_bl_quarter[idx], idx] = U_4 * yc[idx_bl_low[idx]:idx_bl_quarter[idx], idx] * Re
    output[:, idx] = utau[idx] * yc[:, idx] * Re 

    # NOTE: Velocity gradient
    dUdy = np.gradient(U[:, idx], yc[:, idx])
    dudy_1 = dUdy[idx_bl_low[idx]:idx_bl_quarter[idx]]
    dudy_2 = find_k_y_values(yc[idx_bl_low[idx]:idx_bl_quarter[idx], idx], dUdy, yc[:, idx], k=1)
    dudy_3 = find_k_y_values(yc[idx_bl_low[idx]:idx_bl_quarter[idx], idx], dUdy, yc[:, idx], k=2)

    pi_6 = dudy_1 * yc[idx_bl_low[idx]:idx_bl_quarter[idx], idx]**2 * Re
    pi_7 = dudy_2 * yc[idx_bl_low[idx]:idx_bl_quarter[idx], idx]**2 * Re
    pi_8 = dudy_3 * yc[idx_bl_low[idx]:idx_bl_quarter[idx], idx]**2 * Re

    # --- Calculate Input Features (Pi Groups) ---
    # Note:  dPdx is NOT zero here
    # Calculate dimensionless inputs using safe names
    inputs_dict = {
        'u1_y_over_nu':         pi_1[idx_bl_low[idx]:idx_bl_quarter[idx], idx],  # U_i[bot_index] * y_i[bot_index] / nu_i,
        'up_y_over_nu':         pi_2[idx_bl_low[idx]:idx_bl_quarter[idx], idx],
        'u2_y_over_nu':         pi_3[idx_bl_low[idx]:idx_bl_quarter[idx], idx],
        'u3_y_over_nu':         pi_4[idx_bl_low[idx]:idx_bl_quarter[idx], idx],
        'u4_y_over_nu':         pi_5[idx_bl_low[idx]:idx_bl_quarter[idx], idx],
        'dudy1_y_pow2_over_nu': pi_6,
        'dudy2_y_pow2_over_nu': pi_7,
        'dudy3_y_pow2_over_nu': pi_8,
    }
    all_inputs_data.append(pd.DataFrame(inputs_dict))

    # --- Calculate Output Feature ---
    # Output is y+ (utau * y / nu)
    output_dict = {
        'utau_y_over_nu': output[idx_bl_low[idx]:idx_bl_quarter[idx], idx]
    }
    all_output_data.append(pd.DataFrame(output_dict))

    # --- Collect Unnormalized Inputs ---
    unnorm_dict = {
        'y': yc[idx_bl_low[idx]:idx_bl_quarter[idx], idx],
        'u1': U[idx_bl_low[idx]:idx_bl_quarter[idx], idx],
        'nu': np.full_like(yc[idx_bl_low[idx]:idx_bl_quarter[idx],idx], 1/Re),
        'utau': np.full_like(yc[idx_bl_low[idx]:idx_bl_quarter[idx],idx], utau[idx]),
        'up': np.full_like(yc[idx_bl_low[idx]:idx_bl_quarter[idx],idx], up[0, idx]),
        'u2': U_2,
        'u3': U_3,
        'u4': U_4,
        'dudy1': dudy_1,
        'dudy2': dudy_2,
        'dudy3': dudy_3,
    }
    all_unnormalized_inputs_data.append(pd.DataFrame(unnorm_dict))

    # --- Collect Flow Type Information ---
    # Using format: [case_name, reference_nu, x_coord, delta, edge_velocity]
    # For channel flow: x=0, delta=1 (half-channel height), Ue=0 (or U_bulk if needed)
    flow_type_dict = {
        'case_name': [f'aairfoil_2M'] * len(yc[idx_bl_low[idx]:idx_bl_quarter[idx]]),
        'nu': [1/Re] * len(yc[idx_bl_low[idx]:idx_bl_quarter[idx]]),
        'x': [x] * len(yc[idx_bl_low[idx]:idx_bl_quarter[idx]]),
        'delta': [delta_99_x[idx]] * len(yc[idx_bl_low[idx]:idx_bl_quarter[idx]]),
        'temp': [0] * len(yc[idx_bl_low[idx]:idx_bl_quarter[idx]])
    }
    # Add Retau for reference if needed, maybe as an extra column or replacing 'edge_velocity'
    # flow_type_dict['Retau'] = [Re_num] * len(y_sel)
    all_flow_type_data.append(pd.DataFrame(flow_type_dict))

# data = {'inputs': inputs_export[1:,:], 'output': output_export[1:], 'flow_type': flow_type[1:], 'unnormalized_inputs': unnormalized_inputs[1:]}
# with open(dir+f'AAirfoil_{int(np.floor(Re//(1_000_000)))}M_select_data.pkl', 'wb') as f:
#     pkl.dump(data, f)
# Concatenate data from all Re_num cases into single DataFrames
inputs_df = pd.concat(all_inputs_data, ignore_index=True)
output_df = pd.concat(all_output_data, ignore_index=True)
flow_type_df = pd.concat(all_flow_type_data, ignore_index=True)
unnormalized_inputs_df = pd.concat(all_unnormalized_inputs_data, ignore_index=True)

# Save DataFrames to HDF5 file
output_filename = os.path.join(savedatapath, 'aairfoil_2M_data.h5')
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
