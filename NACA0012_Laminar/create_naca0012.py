'''
Note this dataset has nan values in the wall-normal distance. Just ignore those.
'''
import os
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
# --- Configuration ---
# (Assume these are defined elsewhere in your code)
STATS_PATH = os.path.join(WM_DATA_PATH, "NACA0012_Laminar", "stats")
OUTPUT_PATH = os.path.join(WM_DATA_PATH, "data")

REYNOLDS_NUMBER = 5000
UPPER_FRACTION = 0.15
LOWER_FRACTION = 0.025
UPPER_FRACTION_SEP = 0.1
LOWER_FRACTION_SEP = 0.003

# --- Select whether to save data and which points to inspect ---
import sys
save_data = int(sys.argv[1]) if len(sys.argv) > 1 else 0
# inspect_x = [0.2014175, 0.79862236, 0.90906449, 1.00000186]

def calculate_up(dpds, reynolds_number):
    """Calculates the pressure gradient velocity scale."""
    return np.sign(dpds) * (np.abs(dpds) / reynolds_number) ** (1 / 3)

def interpolate_values(x, x_original, values):
    """Interpolates values at given x locations."""
    return np.interp(x, x_original, values)


def calculate_pi_groups(
    dist_normal, up, upn, kinematic_viscosity, u_interp_1, u_interp_2, u_interp_3, u_interp_4, dudy_1, dudy_2, dudy_3
):
    """Calculates the dimensionless Pi groups."""
    return {
        "u1_y_over_nu": u_interp_1 * dist_normal / kinematic_viscosity,
        "up_y_over_nu": up * dist_normal / kinematic_viscosity,  # up is already scaled by nu
        "upn_y_over_nu": upn * dist_normal / kinematic_viscosity,  # up is already scaled by nu
        "u2_y_over_nu": u_interp_2 * dist_normal / kinematic_viscosity,
        "u3_y_over_nu": u_interp_3 * dist_normal / kinematic_viscosity,
        "u4_y_over_nu": u_interp_4 * dist_normal / kinematic_viscosity,
        "dudy1_y_pow2_over_nu": dudy_1 * dist_normal**2
        / kinematic_viscosity,
        "dudy2_y_pow2_over_nu": dudy_2 * dist_normal**2
        / kinematic_viscosity,
        "dudy3_y_pow2_over_nu": dudy_3 * dist_normal**2
        / kinematic_viscosity,
    }


def calculate_output(utau_interpolated, dist_normal, kinematic_viscosity):
    """Calculates the output feature."""
    utau = np.abs(utau_interpolated)
    return {"utau_y_over_nu": utau * dist_normal / kinematic_viscosity}


def calculate_unnormalized_inputs(
    dist_normal, kinematic_viscosity, utau, up, upn, u_interp_1, u_interp_2, u_interp_3, u_interp_4, dudy_1, dudy_2, dudy_3
):
    """Calculates the unnormalized input features."""
    return {
        "y": dist_normal,
        "u1": u_interp_1,
        "nu": np.full_like(dist_normal, kinematic_viscosity),
        "utau": np.full_like(dist_normal, utau),
        "dpdx": np.full_like(dist_normal, up),
        "upn": upn,
        "u2": u_interp_2,
        "u3": u_interp_3,
        "u4": u_interp_4,
        "dudy1": dudy_1, 
        "dudy2": dudy_2,
        "dudy3": dudy_3,
    }


def calculate_flow_type(x_normal, delta, region, kinematic_viscosity, num_points):
    """Calculates flow type information."""
    return {
        "case_name": [region] * num_points,
        "nu": [kinematic_viscosity] * num_points,
        "x": x_normal,
        "delta": [delta] * num_points,
        "albert": [0] * num_points,  # Placeholder
    }


def save_to_hdf5(data_dict, filename):
    """Saves the data dictionary to an HDF5 file."""

    try:
        inputs_df = data_dict["inputs"]
        output_df = data_dict["output"]
        flow_type_df = data_dict["flow_type"]
        unnormalized_inputs_df = data_dict["unnormalized_inputs"]

        output_filename = os.path.join(OUTPUT_PATH, filename + ".h5")
        inputs_df.to_hdf(output_filename, key="inputs", mode="w", format="fixed")
        output_df.to_hdf(output_filename, key="output", mode="a", format="fixed")
        flow_type_df.to_hdf(output_filename, key="flow_type", mode="a", format="table")
        unnormalized_inputs_df.to_hdf(
            output_filename, key="unnormalized_inputs", mode="a", format="fixed"
        )
        print(f"Data saved to {output_filename}")
    except Exception as e:
        print(f"Error saving data: {e}")


def process_and_save_region_data(
    p,
    u_mag,
    utau_interpolated,
    x_normal, # X coordinates of the wall
    S, # Wall-normal distance
    delta,
    up,
    region,
    ind_start,
    ind_end,
    up_frac=UPPER_FRACTION,
    down_frac=LOWER_FRACTION,
    save_plots=False,
):
    """
    Processes data for a specified region and saves it to an HDF5 file.

    Args:
        ... (same as before)
    """

    # # WARNING: Fix delta to be unit normal distance
    # delta = 0.08

    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []
    kinematic_viscosity = 1 / REYNOLDS_NUMBER

    for i in range(ind_start, ind_end):

        # if region == 'APG' and up[i] < 0:
        #     continue

        # breakpoint()
        delta = np.max(S[i][1:])

        dist_normal_i = S[i]
        idx_low_bl_i = np.where(dist_normal_i > down_frac * delta)[0][0]
        idx_up_bl_i = np.where(dist_normal_i <= up_frac * delta)[0][-1]

        u_interp_1 = interpolate_values(
            dist_normal_i[idx_low_bl_i:idx_up_bl_i], dist_normal_i, u_mag[i]
        )
        u_interp_2 = find_k_y_values(dist_normal_i[idx_low_bl_i:idx_up_bl_i], u_mag[i], dist_normal_i, k=1)
        u_interp_3 = find_k_y_values(dist_normal_i[idx_low_bl_i:idx_up_bl_i], u_mag[i], dist_normal_i, k=2)
        u_interp_4 = find_k_y_values(dist_normal_i[idx_low_bl_i:idx_up_bl_i], u_mag[i], dist_normal_i, k=3)
        # note: velocity gradient
        dudy = np.gradient(u_mag[i], dist_normal_i)
        dudy_1 = dudy[idx_low_bl_i-1:idx_up_bl_i-1]
        dudy_2 = find_k_y_values(dist_normal_i[idx_low_bl_i:idx_up_bl_i], dudy, dist_normal_i, k=1)
        dudy_3 = find_k_y_values(dist_normal_i[idx_low_bl_i:idx_up_bl_i], dudy, dist_normal_i, k=2)

        delta_p = (p[i] - p[i][0]) / (dist_normal_i - dist_normal_i[0])
        upn = calculate_up(delta_p, REYNOLDS_NUMBER)

        pi_groups = calculate_pi_groups(
            dist_normal_i[idx_low_bl_i:idx_up_bl_i],
            up[i],
            upn[idx_low_bl_i:idx_up_bl_i],
            kinematic_viscosity,
            u_interp_1,
            u_interp_2,
            u_interp_3,
            u_interp_4,
            dudy_1,
            dudy_2,
            dudy_3,
        )
        all_inputs_data.append(pd.DataFrame(pi_groups))

        output_data = calculate_output(
            utau_interpolated[i], dist_normal_i[idx_low_bl_i:idx_up_bl_i], kinematic_viscosity
        )
        all_output_data.append(pd.DataFrame(output_data))

        unnormalized_inputs_data = calculate_unnormalized_inputs(
            dist_normal_i[idx_low_bl_i:idx_up_bl_i],
            kinematic_viscosity,
            utau_interpolated[i],
            up[i],
            upn[idx_low_bl_i:idx_up_bl_i],
            u_interp_1,
            u_interp_2,
            u_interp_3,
            u_interp_4,
            dudy_1,
            dudy_2,
            dudy_3,
        )
        all_unnormalized_inputs_data.append(pd.DataFrame(unnormalized_inputs_data))

        flow_type_data = calculate_flow_type(
            x_normal[i],
            delta,
            region,
            kinematic_viscosity,
            len(u_interp_1),
        )
        all_flow_type_data.append(pd.DataFrame(flow_type_data))

    # Concatenate dataframes
    inputs_df = pd.concat(all_inputs_data, ignore_index=True)
    output_df = pd.concat(all_output_data, ignore_index=True)
    flow_type_df = pd.concat(all_flow_type_data, ignore_index=True)
    unnormalized_inputs_df = pd.concat(all_unnormalized_inputs_data, ignore_index=True)

    # # Find rows with NaN values in inputs_df
    nan_rows = inputs_df.isna().any(axis=1)
    #
    # # Get the indices of rows without NaN values
    valid_indices = inputs_df.index[~nan_rows]

    # # Clean all DataFrames by selecting only valid indices
    inputs_df = inputs_df.loc[valid_indices].copy()
    output_df = output_df.loc[valid_indices].copy()
    flow_type_df = flow_type_df.loc[valid_indices].copy()
    unnormalized_inputs_df = unnormalized_inputs_df.loc[valid_indices].copy()

    data_dict = {
        "inputs": inputs_df,
        "output": output_df,
        "flow_type": flow_type_df,
        "unnormalized_inputs": unnormalized_inputs_df,
    }

    save_to_hdf5(data_dict, f"naca0012_laminar_data")

def calculate_tangent_normal_velocity(u, v, unit_tangents, unit_normals):
    """
    Calculates velocity components tangential and normal to a curve.

    Args:
        u (np.ndarray): Velocity component(s) in x-direction.
                        Can be scalar, 1D array, or 2D array (e.g., M profiles x N points).
        v (np.ndarray): Velocity component(s) in y-direction. Must have the same shape as u.
        unit_tangents (np.ndarray): Unit tangent vectors [tx, ty].
                                   Shape should be (2,) for single vector, or (M, 2) for M vectors
                                   corresponding to M profiles/points in u/v.
        unit_normals (np.ndarray): Unit normal vectors [nx, ny].
                                  Shape should be (2,) or (M, 2).

    Returns:
        tuple: (vel_tangent, vel_normal)
               - vel_tangent (np.ndarray): Velocity component(s) parallel to the tangent vector(s).
                                          Same shape as u and v.
               - vel_normal (np.ndarray): Velocity component(s) parallel to the normal vector(s).
                                         Same shape as u and v.
    """
    u = np.asarray(u)
    v = np.asarray(v)
    unit_tangents = np.asarray(unit_tangents)
    unit_normals = np.asarray(unit_normals)

    if u.shape != v.shape:
        raise ValueError("u and v must have the same shape")

    # Determine shapes for broadcasting
    if u.ndim == 0: # Scalar u, v
        if unit_tangents.shape != (2,) or unit_normals.shape != (2,):
             raise ValueError("For scalar u/v, tangents/normals must be single vectors of shape (2,)")
        tx, ty = unit_tangents
        nx, ny = unit_normals
    elif u.ndim == 1: # 1D array u, v (e.g., velocity along one line)
        if unit_tangents.shape == (2,) and unit_normals.shape == (2,):
            # Single tangent/normal applied to all points
            tx, ty = unit_tangents
            nx, ny = unit_normals
        elif unit_tangents.shape == u.shape + (2,) and unit_normals.shape == u.shape + (2,):
             # Different tangent/normal for each point in u/v (less common)
             tx = unit_tangents[:, 0]
             ty = unit_tangents[:, 1]
             nx = unit_normals[:, 0]
             ny = unit_normals[:, 1]
        else:
            raise ValueError(f"Shapes mismatch for 1D u/v ({u.shape}): tangents ({unit_tangents.shape}), normals ({unit_normals.shape})")
    elif u.ndim == 2: # 2D array u, v (e.g., M profiles, N points)
        M, N = u.shape
        if unit_tangents.shape == (M, 2) and unit_normals.shape == (M, 2):
             # Tangent/normal defined per profile (M vectors), broadcast across N points
             tx = unit_tangents[:, 0:1] # Shape (M, 1) for broadcasting
             ty = unit_tangents[:, 1:2] # Shape (M, 1)
             nx = unit_normals[:, 0:1] # Shape (M, 1)
             ny = unit_normals[:, 1:2] # Shape (M, 1)
        elif unit_tangents.shape == (2,) and unit_normals.shape == (2,):
             # Single tangent/normal applied to all profiles/points
             tx, ty = unit_tangents
             nx, ny = unit_normals
        else:
             raise ValueError(f"Shapes mismatch for 2D u/v ({u.shape}): tangents ({unit_tangents.shape}), normals ({unit_normals.shape})")
    else:
        raise ValueError("u and v must be scalar, 1D, or 2D arrays")


    # Calculate projections using dot product (element-wise multiplication and sum)
    vel_tangent = u * tx + v * ty
    vel_normal = u * nx + v * ny

    return vel_tangent, vel_normal


"""Main function to process and save data for different regions of the bump."""

# Load data
with open(os.path.join(STATS_PATH, "NACA0012_Upper_NoLE_dpds.pkl"), "rb") as f:
    dpds_upper = pkl.load(f)
with open(os.path.join(STATS_PATH, "NACA0012_Upper_NoLE_data.pkl"), "rb") as f:
    data = pkl.load(f)

x_normal = []
y_normal = []
u_mag = []
s_dist = []
# p = []  # Placeholder for pressure gradient, not used in this script

x_normal = data['x']
y_normal = data['y']
u_mag  = data['U']
s_dist = data['s_dist']  # Wall-normal distance
p      = data['P']  # Pressure gradient
# p.append(data[key]['p'])  # Pressure gradient, not used in this script
# x_normal = np.array(x_normal)
# y_normal = np.array(y_normal)
# u_mag = np.array(u_mag)
# s_dist = np.array(s_dist)
# p = np.array(p)

up_all = dpds_upper['up']
x_up_all = dpds_upper['x']
up = np.interp(x_normal, x_up_all, up_all)

Uo = 1.00  # Free stream velocity assumed to be 1

cf_data = np.loadtxt(os.path.join(STATS_PATH, "data_ref_cf.csv"), delimiter=',')
cf_interpolated = np.interp(x_normal, cf_data[:, 0], cf_data[:, 1])
utau_interpolated = np.sqrt(np.abs(cf_interpolated) / 2) * Uo

if save_data:
    process_and_save_region_data(
        p,
        u_mag,
        utau_interpolated,
        x_normal, # X coordinates of the wall
        s_dist, # Wall-normal distance
        1, 
        up,
        'naca0012_laminar',
        0, 
        len(u_mag)-1,
        up_frac=UPPER_FRACTION,
        down_frac=LOWER_FRACTION,
        save_plots=False,
    )
#
# if len(inspect_x) > 0:
#     for x in inspect_x:
#         ind = np.argmin(np.abs(x_normal[:, 0] - x))
#         print(f"Inspecting point at x={x_normal[ind, 0]}")
#         plt.figure(figsize=(10, 6))
#         plt.plot(s_dist[ind, 1:], u_mag[ind, :], label='u_mag')
#         # plt.plot(s_dist[ind, 1:], v_mag[ind, :], label='v_mag')
#         plt.xlabel('Wall-normal distance (s)')
#         plt.ylabel('Velocity')
#         plt.title(f'Velocity profiles at x={x_normal[ind, 0]}')
#         plt.legend()
#         plt.grid()
#
#         # Add log law
#         # Plot in plus units
#         fig, ax = plt.subplots(figsize=(10, 6))
#         u_plus = u_mag[ind, :] / utau_interpolated[ind]
#         s_dist_p = s_dist[ind, 1:] * REYNOLDS_NUMBER * utau_interpolated[ind]
#         ax.semilogx(s_dist_p, u_plus, label='u+')
#         y_plus = np.linspace(100, 1000, 100)
#         u_plus_law = 1 / 0.41 * np.log(y_plus) + 5.2
#         ax.semilogx(y_plus, u_plus_law, 'r--', label='Log Law')
#         ax.set_xlabel('Wall-normal distance (s+)')
#         ax.set_ylabel('Velocity (u+)')
#         ax.legend()
#
#         plt.show()
