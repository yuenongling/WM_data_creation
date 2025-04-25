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
STATS_PATH = os.path.join(WM_DATA_PATH, "conv-div_channel_Laval", "stats")
WALL_STATS_PATH = os.path.join(STATS_PATH, "statistics_streamwise.dat")
OUTPUT_PATH = os.path.join(WM_DATA_PATH, "data")
REYNOLDS_NUMBER = 12600
UPPER_FRACTION = 0.20
LOWER_FRACTION = 0.025
UPPER_FRACTION_SEP = 0.005
LOWER_FRACTION_SEP = 0.00003

# --- Select whether to save data and which points to inspect ---
import sys
save_data = int(sys.argv[1]) if len(sys.argv) > 1 else 0
inspect_x = [ 2.0203, 4.0488, 8.05638]
inspect_x = [ ]

# --- Utility Functions ---
def load_bump_data(filename):
    """Loads bump data from a pickle file."""

    file_path = os.path.join(STATS_PATH, filename)
    with open(file_path, "rb") as f:
        data = pkl.load(f)
    return data

def calculate_up(dpds, reynolds_number):
    """Calculates the pressure gradient velocity scale."""
    return np.sign(dpds) * (np.abs(dpds) / reynolds_number) ** (1 / 3)

def load_cf_cp_data():
    """Loads Cf, Cp, and delta data from text files."""

    wall_data = np.loadtxt(WALL_STATS_PATH, comments='#', usecols=(0, 1, 2, 4))

    X = wall_data[:, 0]
    Y = wall_data[:, 1]
    utau_data = wall_data[:, 2]
    cp_data = wall_data[:, 3]

    return X, Y, utau_data, cp_data


def calculate_directional_derivative(f_values, curve_points):
    """Calculates the directional derivative of f along a curve."""

    arc_lengths = np.sqrt(np.sum(np.diff(curve_points, axis=0) ** 2, axis=1))
    cum_arc_length = np.concatenate(([0], np.cumsum(arc_lengths)))
    directional_derivative = np.gradient(f_values, cum_arc_length)
    return directional_derivative


def interpolate_values(x, x_original, values):
    """Interpolates values at given x locations."""
    return np.interp(x, x_original, values)


def calculate_wall_normal_distance(x_normal, y_normal):
    """Calculates the wall-normal distance."""

    dist_normal = np.sqrt(
        (x_normal - x_normal[:, 0][:, np.newaxis]) ** 2
        + (y_normal - y_normal[:, 0][:, np.newaxis]) ** 2
    )
    return dist_normal

import numpy as np

def calculate_tangents_normals(x_coords, y_coords):
    """
    Calculates unit tangent and unit normal vectors for a curve defined by
    x and y coordinates using numpy.gradient.

    Args:
        x_coords (np.ndarray): 1D array of x-coordinates.
        y_coords (np.ndarray): 1D array of y-coordinates (must be same length as x).

    Returns:
        tuple: (unit_tangents, unit_normals)
               - unit_tangents (np.ndarray): Array of shape (N, 2) with unit tangent vectors (tx, ty).
               - unit_normals (np.ndarray): Array of shape (N, 2) with unit normal vectors (nx, ny).
                                            Normal is calculated as a 90-degree counter-clockwise
                                            rotation of the tangent.
    """
    # Calculate derivatives using central differences where possible
    dx = np.gradient(x_coords)
    dy = np.gradient(y_coords)

    # Calculate magnitude of the tangent vector
    magnitude = np.sqrt(dx**2 + dy**2)

    # --- Handle potential zero magnitude cases ---
    # Find indices where magnitude is close to zero
    zero_mag_indices = np.isclose(magnitude, 0)
    if np.any(zero_mag_indices):
        print(f"Warning: Magnitude is close to zero at {np.sum(zero_mag_indices)} points. "
              "Tangent/normal are ill-defined there. Setting them to [NaN, NaN].")
        dx[zero_mag_indices] = np.nan
        dy[zero_mag_indices] = np.nan
        magnitude[zero_mag_indices] = np.nan # Ensure NaNs propagate

    # --- Calculate Unit Tangent ---
    # Suppress division-by-zero warnings temporarily for the zero magnitude case
    with np.errstate(invalid='ignore', divide='ignore'):
        unit_tangent_x = dx / magnitude
        unit_tangent_y = dy / magnitude

    unit_tangents = np.column_stack((unit_tangent_x, unit_tangent_y))

    # --- Calculate Unit Normal (counter-clockwise rotation of tangent) ---
    # Normal vector = (-dy, dx)
    with np.errstate(invalid='ignore', divide='ignore'):
        unit_normal_x = -dy / magnitude
        unit_normal_y = dx / magnitude

    unit_normals = np.column_stack((unit_normal_x, unit_normal_y))

    return unit_tangents, unit_normals


def calculate_bump_parallel_velocity(X, Y, u_velocity, v_velocity):
    """Calculates the velocity component parallel to the bump."""

    # norm_vec = np.array(
    #     [
    #         [x_normal[i, 1] - x_normal[i, 0], y_normal[i, 1] - y_normal[i, 0]]
    #         / calculate_wall_normal_distance(x_normal, y_normal)[i, 1]
    #         for i in range(x_normal.shape[0])
    #     ]
    # )
    # tan_vec = np.array([[norm_vec[i, 1], -norm_vec[i, 0]] for i in range(x_normal.shape[0])])

    u_mag = np.sum(
        np.stack((u_velocity, v_velocity), axis=-1) * tan_vec[:, np.newaxis, :], axis=-1
    )
    return u_mag

def calculate_wall_normal_distance(x_normal, y_normal):
    """Calculates the wall-normal distance."""
    return np.sqrt(
        (x_normal - x_normal[:, 0][:, np.newaxis]) ** 2
        + (y_normal - y_normal[:, 0][:, np.newaxis]) ** 2
    )


def calculate_pi_groups(
    dist_normal, up, kinematic_viscosity, u_interp_1, u_interp_2, u_interp_3, u_interp_4, dudy_1, dudy_2, dudy_3
):
    """Calculates the dimensionless Pi groups."""
    return {
        "u1_y_over_nu": u_interp_1 * dist_normal / kinematic_viscosity,
        "up_y_over_nu": up * dist_normal / kinematic_viscosity,  # up is already scaled by nu
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
    dist_normal, kinematic_viscosity, utau, up, u_interp_1, u_interp_2, u_interp_3, u_interp_4, dudy_1, dudy_2, dudy_3
):
    """Calculates the unnormalized input features."""
    return {
        "y": dist_normal,
        "u1": u_interp_1,
        "nu": np.full_like(dist_normal, kinematic_viscosity),
        "utau": np.full_like(dist_normal, utau),
        "dpdx": np.full_like(dist_normal, up),
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

    # WARNING: Fix delta to be unit normal distance
    delta = 0.6

    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []
    kinematic_viscosity = 1 / REYNOLDS_NUMBER

    for i in range(ind_start, ind_end):

        # if region == 'APG' and up[i] < 0:
        #     continue
        # FIXME: Hardcode to skip some points
        # if (x_normal[i, 0] > 9.0):
        #     print(f"Skipping point at x={x_normal[i,0]}")
        #     continue


        dist_normal_i = S[i]
        idx_low_bl_i = np.where(dist_normal_i > down_frac * delta)[0][0]
        idx_up_bl_i = np.where(dist_normal_i <= up_frac * delta)[0][-1]

        if utau_interpolated[i] < 0:
            idx_up_old = idx_up_bl_i
            idx_up_bl_i = min(
                np.where(u_mag[i, :] > 0)[0][0] - 1, idx_up_bl_i
            )
            if idx_up_bl_i < idx_up_old:
                print(f"WARNING: idx_up_bl < idx_up_old at x={x_normal[i,0]}")

        if idx_up_bl_i <= idx_low_bl_i:
            print(f"WARNING: idx_up_bl == idx_low_bl at x={x_normal[i,0]}")
            continue

        utau_interpolated[i] = np.abs(utau_interpolated[i])

        u_interp_1 = interpolate_values(
            dist_normal_i[idx_low_bl_i:idx_up_bl_i], dist_normal_i[1:], u_mag[i, :]
        )
        u_interp_2 = find_k_y_values(dist_normal_i[idx_low_bl_i:idx_up_bl_i], u_mag[i, :], dist_normal_i[1:], k=1)
        u_interp_3 = find_k_y_values(dist_normal_i[idx_low_bl_i:idx_up_bl_i], u_mag[i, :], dist_normal_i[1:], k=2)
        u_interp_4 = find_k_y_values(dist_normal_i[idx_low_bl_i:idx_up_bl_i], u_mag[i, :], dist_normal_i[1:], k=3)
        # note: velocity gradient
        dudy = np.gradient(u_mag[i,:], dist_normal_i[1:])
        dudy_1 = dudy[idx_low_bl_i-1:idx_up_bl_i-1]
        dudy_2 = find_k_y_values(dist_normal_i[idx_low_bl_i:idx_up_bl_i], dudy, dist_normal_i[1:], k=1)
        dudy_3 = find_k_y_values(dist_normal_i[idx_low_bl_i:idx_up_bl_i], dudy, dist_normal_i[1:], k=2)

        pi_groups = calculate_pi_groups(
            dist_normal_i[idx_low_bl_i:idx_up_bl_i],
            up[i],
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
            x_normal[i, 0],
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

    data_dict = {
        "inputs": inputs_df,
        "output": output_df,
        "flow_type": flow_type_df,
        "unnormalized_inputs": unnormalized_inputs_df,
    }

    save_to_hdf5(data_dict, f"convdiv_data")


def process_separation_zone_data(
    u_mag,
    cf_interpolated,
    x_normal,
    y_normal,
    delta,
    up,
    ind_start,
    ind_end,
    save_plots=False,
):
    """
    Processes and saves data specifically for the separation zone, handling Cf < 0.

    Args:
        ... (same as before)
    """

    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []
    kinematic_viscosity = 1 / REYNOLDS_NUMBER

    pi_1 = (
        u_mag * calculate_wall_normal_distance(x_normal, y_normal)[:, 1:] * REYNOLDS_NUMBER
    )

    for i in range(ind_start, ind_end):

        dist_normal_i = calculate_wall_normal_distance(
            x_normal[i : i + 1], y_normal[i : i + 1]
        )[0]
        idx_low_bl_i = np.where(
            dist_normal_i > LOWER_FRACTION_SEP * delta[i]
        )[0][0]
        idx_up_bl_i = np.where(
            dist_normal_i <= UPPER_FRACTION_SEP * delta[i]
        )[0][-1]

        if cf_interpolated[i] < 0:
            idx_up_old = idx_up_bl_i
            idx_up_bl_i = min(
                np.where(pi_1[i, :] > 0)[0][0] - 1, idx_up_bl_i
            )
            if idx_up_bl_i < idx_up_old:
                print(f"WARNING: idx_up_bl < idx_up_old at x={x_normal[i,0]}")

        if idx_up_bl_i <= idx_low_bl_i:
            print(f"WARNING: idx_up_bl == idx_low_bl at x={x_normal[i,0]}")
            continue

        u_interp_1 = interpolate_values(
            dist_normal_i[idx_low_bl_i:idx_up_bl_i], dist_normal_i[1:], u_mag[i, :]
        )
        u_interp_2 = find_k_y_values(dist_normal_i[idx_low_bl_i:idx_up_bl_i], u_mag[i, :], dist_normal_i[1:], k=1)
        u_interp_3 = find_k_y_values(dist_normal_i[idx_low_bl_i:idx_up_bl_i], u_mag[i, :], dist_normal_i[1:], k=2)
        u_interp_4 = find_k_y_values(dist_normal_i[idx_low_bl_i:idx_up_bl_i], u_mag[i, :], dist_normal_i[1:], k=3)
        # note: velocity gradient
        dudy = np.gradient(u_mag[i,:], dist_normal_i[1:])
        dudy_1 = dudy[idx_low_bl_i-1:idx_up_bl_i-1]
        dudy_2 = find_k_y_values(dist_normal_i[idx_low_bl_i:idx_up_bl_i], dudy, dist_normal_i[1:], k=1)
        dudy_3 = find_k_y_values(dist_normal_i[idx_low_bl_i:idx_up_bl_i], dudy, dist_normal_i[1:], k=2)

        pi_groups = calculate_pi_groups(
            dist_normal_i[idx_low_bl_i:idx_up_bl_i],
            up[i],
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
            cf_interpolated[i], dist_normal_i[idx_low_bl_i:idx_up_bl_i], kinematic_viscosity
        )
        all_output_data.append(pd.DataFrame(output_data))

        unnormalized_inputs_data = calculate_unnormalized_inputs(
            dist_normal_i[idx_low_bl_i:idx_up_bl_i],
            kinematic_viscosity,
            np.sqrt(0.5 * np.abs(cf_interpolated[i])),
            up[i],
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
            x_normal[i, 0],
            delta[i],
            'SEP',
            kinematic_viscosity,
            len(u_interp_1),
        )
        all_flow_type_data.append(pd.DataFrame(flow_type_data))

# Concatenate dataframes
    inputs_df = pd.concat(all_inputs_data, ignore_index=True)
    output_df = pd.concat(all_output_data, ignore_index=True)
    flow_type_df = pd.concat(all_flow_type_data, ignore_index=True)
    unnormalized_inputs_df = pd.concat(all_unnormalized_inputs_data, ignore_index=True)

    data_dict = {
        "inputs": inputs_df,
        "output": output_df,
        "flow_type": flow_type_df,
        "unnormalized_inputs": unnormalized_inputs_df,
    }

    save_to_hdf5(data_dict, "gaussian_2M_data_SEP")

    if save_plots:
        plt.figure()
        sc = plt.scatter(
            inputs_df["u1_y_over_nu"],
            inputs_df["up_y_over_nu"],
            s=3,
            c=output_df["utau_y_over_nu"],
            cmap="rainbow",
            rasterized=True,
        )
        plt.colorbar(sc)
        plt.xlabel(r"$\Pi_1=u_1y/\nu$")
        plt.ylabel(r"$\Pi_2=u_py/\nu$")
        plt.title(f"Gaussian Bump at Re={REYNOLDS_NUMBER} - Separation Zone")
        plt.show()

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
data = load_bump_data("wall_normal_profiles.pkl")
X, Y, utau_data, cp_data = load_cf_cp_data()

# Quick sanity check
#
# PLOT: Contour plot of the velocity field
# plt.scatter(X,Y)
# for key in data.keys():
#     x_norm = data[key]['x_norm']
#     y_norm = data[key]['y_norm']
#     plt.plot(x_norm,y_norm)
# plt.gca().set_aspect('equal')
# plt.show()
#
# PLOT: velocity profiles
# for key in data.keys():
#   s_dist = data[key]['s']
#   u_profile = data[key]['u']
#   plt.plot(u_profile,s_dist)
# plt.show()

x_normal = []
y_normal = []
u_velocity = []
v_velocity = []
s_dist = []

for key in data.keys():
    x_normal.append(data[key]['x_norm'])
    y_normal.append(data[key]['y_norm'])
    u_velocity.append(data[key]['u'])
    v_velocity.append(data[key]['v'])
    s_dist.append(data[key]['s'])
x_normal = np.array(x_normal)
y_normal = np.array(y_normal)
u_velocity = np.array(u_velocity)
v_velocity = np.array(v_velocity)
s_dist = np.array(s_dist)

# Extract relevant fields
utau_interpolated = interpolate_values(x_normal[:, 0], X, utau_data)

# Wall parallel pressure gradient
x_coord = np.vstack([X, Y]).T
Uo = 1.00717
dpds_wall_all = calculate_directional_derivative(
    cp_data, x_coord
) * 0.5*Uo**2
dpds_wall = interpolate_values(x_normal[:, 0], X, dpds_wall_all)
up = calculate_up(dpds_wall, REYNOLDS_NUMBER)

tan, nor = calculate_tangents_normals(X, Y)
tan_local = np.vstack([interpolate_values(x_normal[:, 0], X, tan[:,0]),
                      interpolate_values(x_normal[:, 0], X, tan[:,1])]).T
nor_local = np.vstack([interpolate_values(x_normal[:, 0], X, nor[:,0]),
                      interpolate_values(x_normal[:, 0], X, nor[:,1])]).T

u_mag, v_mag = calculate_tangent_normal_velocity(u_velocity, v_velocity, 
                                                 tan_local, nor_local)

# Crop out zero velocity/distances
u_mag = u_mag[:,1:]
# s_dist = s_dist[:,1:]

if save_data:
    process_and_save_region_data(
        u_mag,
        utau_interpolated,
        x_normal, # X coordinates of the wall
        s_dist, # Wall-normal distance
        1, 
        up,
        'convdiv',
        0, 
        len(u_mag)-1,
        up_frac=UPPER_FRACTION,
        down_frac=LOWER_FRACTION,
        save_plots=False,
    )

if len(inspect_x) > 0:
    for x in inspect_x:
        ind = np.argmin(np.abs(x_normal[:, 0] - x))
        print(f"Inspecting point at x={x_normal[ind, 0]}")
        plt.figure(figsize=(10, 6))
        plt.plot(s_dist[ind, 1:], u_mag[ind, :], label='u_mag')
        # plt.plot(s_dist[ind, 1:], v_mag[ind, :], label='v_mag')
        plt.xlabel('Wall-normal distance (s)')
        plt.ylabel('Velocity')
        plt.title(f'Velocity profiles at x={x_normal[ind, 0]}')
        plt.legend()
        plt.grid()

        # Add log law
        # Plot in plus units
        fig, ax = plt.subplots(figsize=(10, 6))
        u_plus = u_mag[ind, :] / utau_interpolated[ind]
        s_dist_p = s_dist[ind, 1:] * REYNOLDS_NUMBER * utau_interpolated[ind]
        ax.semilogx(s_dist_p, u_plus, label='u+')
        y_plus = np.linspace(100, 1000, 100)
        u_plus_law = 1 / 0.41 * np.log(y_plus) + 5.2
        ax.semilogx(y_plus, u_plus_law, 'r--', label='Log Law')
        ax.set_xlabel('Wall-normal distance (s+)')
        ax.set_ylabel('Velocity (u+)')
        ax.legend()

        plt.show()
