import numpy as np
import pandas as pd
import os
import pickle as pkl  # Import pickle

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
savedatapath = os.path.join(WM_DATA_PATH, 'data')
currentpath = os.path.join(WM_DATA_PATH, 'sep_bubble_Kamogawa')
statspath = os.path.join(WM_DATA_PATH, 'sep_bubble_Kamogawa', 'stats')
# NOTE: that the boundary layer thickness is not given for this case. Use some approx values
UP_FRAC = 0.15
DOWN_FRAC = 0.01

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

def read_grid_2d(filename):
    """
    Read 2D grid coordinates from a Plot3D format binary file.
    
    Args:
        filename (str): Path to the grid file
        
    Returns:
        tuple: (x, y) arrays containing grid coordinates
    """
    with open(filename, 'rb') as f:
        # Read grid dimensions (imax, jmax)
        imax, jmax = np.fromfile(f, dtype=np.int32, count=2)
        
        # Read x and y coordinates
        # Reshape to match the grid dimensions
        x = np.fromfile(f, dtype=np.float32, count=imax*jmax).reshape(jmax, imax)
        y = np.fromfile(f, dtype=np.float32, count=imax*jmax).reshape(jmax, imax)
        
    return x, y

def read_statistics_2d(filename):
    """
    Read 2D flow statistics from a Plot3D format binary file.
    
    Args:
        filename (str): Path to the statistics file
        
    Returns:
        dict: Dictionary containing all flow variables
    """
    var_names = [
        'density',           # Mean density
        'u_velocity',        # Mean x-velocity
        'v_velocity',        # Mean y-velocity
        'pressure',          # Mean pressure
        'reynolds_xx',       # Reynolds normal stress in x direction
        'reynolds_yy',       # Reynolds normal stress in y direction
        'reynolds_zz',       # Reynolds normal stress in z direction
        'reynolds_xy',       # Reynolds shear stress
        'pressure_rms'       # RMS pressure fluctuation
    ]
    
    with open(filename, 'rb') as f:
        # Read dimensions (imax, jmax, nvar)
        imax, jmax, nvar = np.fromfile(f, dtype=np.int32, count=3)
        
        # Initialize dictionary to store all variables
        data = {}
        
        # Read each variable and store in dictionary
        for i, name in enumerate(var_names):
            var_data = np.fromfile(f, dtype=np.float32, count=imax*jmax).reshape(jmax, imax)
            data[name] = var_data
            
    return data

# Example usage
def calculate_pressure_gradient_x(x, pressure):
    """
    Calculate pressure gradient in x-direction using central differencing.
    
    Args:
        x (np.ndarray): Grid x-coordinates
        pressure (np.ndarray): Pressure field
    
    Returns:
        np.ndarray: Pressure gradient dp/dx
    """
    dp_dx = np.zeros_like(pressure)
    # Use central differencing for interior points
    dp_dx[:, 1:-1] = (pressure[:, 2:] - pressure[:, :-2]) / (x[:, 2:] - x[:, :-2])
    # Forward difference for first point
    dp_dx[:, 0] = (pressure[:, 1] - pressure[:, 0]) / (x[:, 1] - x[:, 0])
    # Backward difference for last point
    dp_dx[:, -1] = (pressure[:, -1] - pressure[:, -2]) / (x[:, -1] - x[:, -2])
    return dp_dx

# Reynolds number
Ret = 2000

# Read grid data
x, y = read_grid_2d(os.path.join(statspath, 'grid_2d.xyz'))
print(f"Grid dimensions: {x.shape}")
stats = read_statistics_2d(os.path.join(statspath, 'statistics_2d.fun'))
print("\nAvailable variables:")
for var_name in stats.keys():
    print(f"- {var_name}: {stats[var_name].shape}")
U = stats['u_velocity']
P = stats['pressure']

# Read surface data
data = np.loadtxt(os.path.join(statspath, 'surface_data.txt'), skiprows=1)
Cf = data[:, 2]  # Skin friction coefficient
utau = np.sqrt(abs(Cf / 2))  # Friction velocity
    
# Example: Print some statistics
print("\nSample statistics:")
print(f"Mean density range: {stats['density'].min():.3f} to {stats['density'].max():.3f}")
print(f"Mean x-velocity range: {stats['u_velocity'].min():.3f} to {stats['u_velocity'].max():.3f}")

dp_dx = calculate_pressure_gradient_x(x, stats['pressure'])
dp_dx_w = dp_dx[0, :]  # Pressure gradient at the wall
# Gaussian smoothing dp_dx_w
# dp_dx_w_smooth = local_gaussian_smooth(x[0, :], dp_dx_w, bandwidth=7)
up = np.sign(dp_dx_w) * (abs(dp_dx_w / Ret)) ** (1/3)  # Pressure gradient velocity

sampling_fre = 20

all_inputs_data = []
all_output_data = []
all_flow_type_data = []
all_unnormalized_inputs_data = []

for idx, x_i in enumerate(x[0, :]):

    # WARNING: For this case, only sample separated cases
    # if Cf[idx] > 0:
    #     continue

    if idx % sampling_fre != 0 and np.abs(Cf[idx]) > 0.0001: # Keep more points near separation and reattachment
        continue

    # WARNING: Manually remove points outside the range
    if x_i < 20 or x_i > 252:
        continue
    print('x_i', x_i)

    y_i = y[:, idx]
    U_i = U[:, idx]
    P_i = P[:, idx]

    # FIXME: Do a rough estimate of delta99
    delta99_i = 0.2 * np.max(y)
    utau_i = utau[idx]
    dPdx_i = dp_dx_w[idx]

# Skip the first point if it is zero
    if U_i[0] == 0 or y_i[0] == 0:
        y_i = y_i[1:]
        U_i = U_i[1:]

    y_i = np.array(y_i)
    U_i = np.array(U_i)
    up_i = up[idx]

    # NOTE: Find indices that are within the range of 0.005 to 0.08 of delta99
    # NOTE: Delete points where the flow reverses but U_ > 0
    if Cf[idx] < 0:
        bot_index = np.where((y_i >= DOWN_FRAC*delta99_i) & (y_i <= UP_FRAC*delta99_i) & (U_i < 0))[0]
    else:
        bot_index = np.where((y_i >= DOWN_FRAC*delta99_i) & (y_i <= UP_FRAC*delta99_i))[0]

    if len(bot_index) == 0:
        print('No data points in range at x =', x_i)
        continue

    # NOTE: Unnormalized inputs
    U_ = U_i[bot_index]
    y_ = y_i[bot_index]
    P_ = P_i[bot_index]
    U_2 = find_k_y_values(y_, U_i, y_i, k=1)
    U_3 = find_k_y_values(y_, U_i, y_i, k=2)
    U_4 = find_k_y_values(y_, U_i, y_i, k=3)

    # NOTE: Inputs
    pi_1 = y_ * U_ * Ret
    pi_2 = up_i * y_ * Ret
    pi_3 = U_2 * y_ * Ret
    pi_4 = U_3 * y_ * Ret
    pi_5 = U_4 * y_ * Ret

    # NOTE: Velocity gradient
    dUdy = np.gradient(U_i, y_i)
    dudy_1 = dUdy[bot_index]
    dudy_2 = find_k_y_values(y_, dUdy, y_i, k=1)
    dudy_3 = find_k_y_values(y_, dUdy, y_i, k=2)

    pi_6 = dudy_1 * y_**2 * Ret
    pi_7 = dudy_2 * y_**2 * Ret
    pi_8 = dudy_3 * y_**2 * Ret

    # NOTE: Wall-normal pressure gradient
    delta_p = P_ - P_i[0]  # Pressure difference from the first point
    up_n_i = np.sign(delta_p) * (abs(delta_p / Ret)) ** (1 / 3)
    pi_9 = up_n_i * y_ * Ret

    # NOTE: Outputs
    pi_out = utau_i * y_ * Ret

    # --- Calculate Input Features (Pi Groups) ---
    # Note:  dPdx is NOT zero here
    # Calculate dimensionless inputs using safe names
    inputs_dict = {
        'u1_y_over_nu': pi_1,  # U_i[bot_index] * y_i[bot_index] / nu_i,
        'up_y_over_nu': pi_2,
        'upn_y_over_nu': pi_9,
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
        'y': y_,
        'u1': U_,
        'nu': np.full_like(y_, 1/Ret),
        'utau': np.full_like(y_, utau_i),
        'up': np.full_like(y_, up_i),
        'upn': up_n_i,
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
    len_y = len(y_)
    flow_type_dict = {
        'case_name': ['bub_K'] * len_y,
        'nu': [1/Ret] * len_y,
        'x': [x_i] * len_y,
        'delta': [delta99_i] * len_y,
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
output_filename = os.path.join(savedatapath, 'bub_K_data.h5')
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

# --- Sanity Check ---
# print("\n--- Sanity Check: Comparing HDF5 with Original Pickle ---")
# with open('/home/yuenongling/Codes/BFM/WM_Opt/data/bub_K_data.pkl', 'rb') as f:
#     original_data = pkl.load(f)
#
# # Load corresponding data from HDF5
# inputs_hdf = inputs_df[inputs_df.index.isin(np.arange(len(original_data['inputs'])))].values
# output_hdf = output_df[output_df.index.isin(np.arange(len(original_data['output'])))].values.flatten()
# flow_type_hdf = flow_type_df[flow_type_df.index.isin(np.arange(len(original_data['flow_type'])))].values
# unnormalized_inputs_hdf = unnormalized_inputs_df[
#     unnormalized_inputs_df.index.isin(np.arange(len(original_data['unnormalized_inputs'])))].values
#
# print(f"  Inputs match: {np.allclose(original_data['inputs'], inputs_hdf)}")
# print(f"  Output match: {np.allclose(original_data['output'], output_hdf)}")
# print(f"  Flow type match: {np.array_equal(original_data['flow_type'].astype(str), flow_type_hdf.astype(str))}")
# print(
#     f"  Unnormalized inputs match: {np.allclose(original_data['unnormalized_inputs'].flatten(), unnormalized_inputs_hdf.flatten(), rtol=1e-5, atol=1e-4)}")
