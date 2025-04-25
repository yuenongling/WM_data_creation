import os
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
# --- Configuration ---
# (Assume these are defined elsewhere in your code)
STATS_PATH = os.path.join(WM_DATA_PATH, "smooth_ramp_Uzun", "stats")
WALL_STATS_PATH = os.path.join(STATS_PATH, "dns_wall_quantities_vs_x.txt")
DELTA_STATS_PATH = os.path.join(STATS_PATH, "delta_paper.csv")
OUTPUT_PATH = os.path.join(WM_DATA_PATH, "data")

REYNOLDS_NUMBER = 667_000
UPPER_FRACTION = 0.13
LOWER_FRACTION = 0.025
UPPER_FRACTION_SEP = 0.1
LOWER_FRACTION_SEP = 0.003
H = 0.22

# --- Select whether to save data and which points to inspect ---
import sys
save_data = int(sys.argv[1]) if len(sys.argv) > 1 else 0
inspect_x = [0, 0.5]


# --- Utility Functions ---
def load_bump_data(filename):
    """Loads bump data from a pickle file."""

    file_path = os.path.join(STATS_PATH, filename)
    with open(file_path, "rb") as f:
        data = pkl.load(f)
    return data

def calculate_bump_shape():
    # Before the ramp is flat
    x_before_ramp = np.linspace(-0.62, 0, 10)
    y_before_ramp = H * np.ones_like(x_before_ramp)
    # Ramp shape
    x = np.linspace(0, 1, 100)
    y = H * (1 - 10 * x**3 + 15 * x**4 - 6 * x**5)
    # After the ramp is flat again
    x_after_ramp = np.linspace(1, 4, 20)
    y_after_ramp = np.zeros_like(x_after_ramp)
    x = np.concatenate((x_before_ramp, x, x_after_ramp))
    y = np.concatenate((y_before_ramp, y, y_after_ramp))
    return x, y

def calculate_bump_shape_given_x(x):

    y = np.zeros_like(x)

    for idx in range(len(x)):
        if x[idx] < 0:
            y[idx] = H
        elif 0 <= x[idx] <= 1:
            y[idx] = H * (1 - 10 * x[idx]**3 + 15 * x[idx]**4 - 6 * x[idx]**5)
        else:
            y[idx] = 0

    return y

def analytical_derivative(x, H):
    """
    Calculates the analytical derivative of the bump shape function
    using np.piecewise for vectorized calculation.

    Args:
        x (float or np.array): X-coordinate(s).
        H (float): Height parameter of the bump.

    Returns:
        float or np.array: The derivative dy/dx at the given x-coordinate(s).
    """
    x = np.asarray(x) # Ensure x is an array for np.piecewise

    # Define the conditions and corresponding derivative functions (as lambdas)
    conditions = [
        x < 0,                 # Condition 1: x < 0
        (x >= 0) & (x <= 1),   # Condition 2: 0 <= x <= 1
        x > 1                  # Condition 3: x > 1
    ]

    functions = [
        lambda v: np.zeros_like(v),                             # Result for x < 0
        lambda v: H * (-30*v**2 + 60*v**3 - 30*v**4),            # Result for 0 <= x <= 1
        lambda v: np.zeros_like(v)                              # Result for x > 1
    ]

    # np.piecewise evaluates the functions based on the conditions
    dydx = np.piecewise(x, conditions, functions)
    return dydx

def plot_tangents_normals_analytical(x_curve, y_curve, indices_to_plot, H, line_length=0.1):
    """
    Calculates and plots tangent and normal lines for a given curve
    at specified indices, using the analytical derivative.

    Args:
        x_curve (np.array): X-coordinates of the curve.
        y_curve (np.array): Y-coordinates of the curve.
        indices_to_plot (list): List of indices for plotting.
        H (float): Height parameter (needed for analytical derivative).
        line_length (float): Visual length of the tangent/normal lines.
    """
    if len(x_curve) != len(y_curve):
        raise ValueError("x_curve and y_curve must have the same length")
    if not indices_to_plot:
        print("No indices specified for plotting tangents/normals.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the main bump shape
    ax.plot(x_curve, y_curve, label='Bump Shape', color='blue', linewidth=2)

    print("\nPlotting Tangents and Normals (Analytical Derivative) at selected points:")

    # Plot tangent and normal lines at selected points
    for i, index in enumerate(indices_to_plot):
        if index < 0 or index >= len(x_curve):
            print(f"  Skipping invalid index: {index}")
            continue

        x0, y0 = x_curve[index], y_curve[index]

        # === Use Analytical Derivative ===
        m_tangent = analytical_derivative(x0, H)
        # =================================

        # Plot the point itself
        ax.plot(x0, y0, 'o', color='red', markersize=5)
        print(f"  Index {index}: (x={x0:.3f}, y={y0:.3f}), Analytical Tangent Slope={m_tangent:.3f}")

        # --- Tangent Line Segment --- (Logic remains the same)
        mag_tangent = np.sqrt(1 + m_tangent**2)
        if mag_tangent < 1e-9: mag_tangent = 1e-9
        dx_t = 1 / mag_tangent
        dy_t = m_tangent / mag_tangent
        xt1, yt1 = x0 - dx_t * line_length / 2, y0 - dy_t * line_length / 2
        xt2, yt2 = x0 + dx_t * line_length / 2, y0 + dy_t * line_length / 2
        label_t = 'Tangent' if i == 0 else None
        ax.plot([xt1, xt2], [yt1, yt2], '--', color='green', label=label_t)

        # --- Normal Line Segment --- (Logic remains the same)
        dx_n, dy_n = -dy_t, dx_t # Perpendicular vector
        xn1, yn1 = x0 - dx_n * line_length / 2, y0 - dy_n * line_length / 2
        xn2, yn2 = x0 + dx_n * line_length / 2, y0 + dy_n * line_length / 2
        label_n = 'Normal' if i == 0 else None
        ax.plot([xn1, xn2], [yn1, yn2], ':', color='purple', label=label_n)

    # Plot settings
    ax.set_title("Bump Shape with Tangent and Normal Lines (Analytical Derivative)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.show()

def calculate_tangents_normals_analytical(x_coords, y_coords, H):
    """
    Calculates unit tangent and unit normal vectors for the specific bump shape
    using its analytical derivative. Returns vectors in the same format as the
    user's example function.

    Args:
        x_coords (np.ndarray): 1D array of x-coordinates defining the bump shape.
        y_coords (np.ndarray): 1D array of y-coordinates (must match x_coords).
                               (Note: y_coords are not used in calculation but kept
                                for consistent function signature if needed elsewhere).
        H (float): Height parameter of the bump shape.

    Returns:
        tuple: (unit_tangents, unit_normals)
               - unit_tangents (np.ndarray): Array of shape (N, 2) with unit tangent vectors (tx, ty).
               - unit_normals (np.ndarray): Array of shape (N, 2) with unit normal vectors (nx, ny).
                                             Normal is 90-degree counter-clockwise rotation of tangent.
    """
    if len(x_coords) != len(y_coords):
         raise ValueError("x_coords and y_coords must have the same length")

    # 1. Calculate analytical slope m = dy/dx
    m = analytical_derivative(x_coords, H)

    # 2. Calculate magnitude of the vector (1, m) for normalization
    # magnitude = sqrt(1^2 + m^2)
    magnitude = np.sqrt(1.0 + m**2)

    # --- Handle potential issues ---
    # Check for NaN in slope 'm' or if magnitude is ~0 or NaN.
    # For this specific function, magnitude should always be >= 1.0,
    # but the check is good practice.
    problem_indices = np.isnan(m) | np.isclose(magnitude, 0) | np.isnan(magnitude)

    if np.any(problem_indices):
        num_problems = np.sum(problem_indices)
        print(f"Warning: Slope or magnitude calculation resulted in NaN or zero "
              f"magnitude at {num_problems} points. Tangent/normal are ill-defined "
              f"there. Setting them to [NaN, NaN].")
        # Ensure NaNs propagate by setting magnitude to NaN at problem points
        magnitude[problem_indices] = np.nan
        # Also set slope m to NaN where magnitude is NaN to avoid warnings below
        m[np.isnan(magnitude)] = np.nan


    # --- Calculate Unit Tangent (1/mag, m/mag) ---
    # Suppress division-by-zero/invalid-value warnings for NaN cases
    with np.errstate(invalid='ignore', divide='ignore'):
        unit_tangent_x = 1.0 / magnitude
        unit_tangent_y = m / magnitude

    # Stack into (N, 2) array
    unit_tangents = np.column_stack((unit_tangent_x, unit_tangent_y))

    # --- Calculate Unit Normal (-m/mag, 1/mag) ---
    # Can be calculated directly from unit tangent components: (-ty, tx)
    unit_normal_x = -unit_tangent_y
    unit_normal_y = unit_tangent_x

    # Stack into (N, 2) array
    unit_normals = np.column_stack((unit_normal_x, unit_normal_y))

    return unit_tangents, unit_normals


def calculate_up(dpds, reynolds_number):
    """Calculates the pressure gradient velocity scale."""
    return np.sign(dpds) * (np.abs(dpds) / reynolds_number) ** (1 / 3)

def load_cf_cp_data():
    """Loads Cf, Cp, and delta data from text files."""

    data_wall = np.loadtxt(WALL_STATS_PATH, comments='#', skiprows=1)

    X = data_wall[:, 0]

    cp_data = data_wall[:, 2]
    cf_data = data_wall[:, 1]

    return X, cf_data, cp_data


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
    up_frac=UPPER_FRACTION,
    down_frac=LOWER_FRACTION,
):
    """
    Processes data for a specified region and saves it to an HDF5 file.

    Args:
        ... (same as before)
    """


    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []
    kinematic_viscosity = 1 / REYNOLDS_NUMBER

    for i in range(len(u_mag)):

        delta = delta_local[i]

        dist_normal_i = S[i]
        idx_low_bl_i = np.where(dist_normal_i > down_frac * delta)[0][0]
        idx_up_bl_i = np.where(dist_normal_i <= up_frac * delta)[0][-1]

        # Special tratment for separation zone
        if u_mag[i][0] < 0:
            idx_low_bl_i = np.where(dist_normal_i > LOWER_FRACTION_SEP * delta)[0][0]
            idx_up_bl_i = np.where(dist_normal_i <= UPPER_FRACTION_SEP * delta)[0][-1]
            idx_pos_low = np.where(u_mag[i] > 0)[0][0]
            if idx_pos_low > idx_up_bl_i:
                print(f"WARNING: idx_pos_low > idx_up_bl_i at x={x_normal[i][0]}")
                print(f"idx_pos_low: {idx_pos_low}, idx_up_bl_i: {idx_up_bl_i}")
                print(f"Corresponding dist_normal_i: {dist_normal_i[idx_pos_low]}")
                print(f"Corresponding dist_normal_i: {dist_normal_i[idx_up_bl_i]}")
                continue
            
            if idx_pos_low >= idx_low_bl_i:
                # Replace idx_up_bl_i with idx_pos_low
                idx_up_bl_i = idx_pos_low - 1

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
            x_normal[i][0],
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

    # Find rows with NaN values in inputs_df
    nan_rows = inputs_df.isna().any(axis=1)

    # Get the indices of rows without NaN values
    valid_indices = inputs_df.index[~nan_rows]

    # Clean all DataFrames by selecting only valid indices
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

    save_to_hdf5(data_dict, f"smoothramp_data")


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

######################################
# Plot bump to check
x_bump, y_bump = calculate_bump_shape()

# Choose some indices along the curve to plot tangents/normals
# Points chosen: before ramp, start of ramp, mid-ramp, end of ramp, after ramp
# Note: Indices correspond to the concatenated array 'x_bump'
num_points = len(x_bump)
x_before_ramp = x_bump[x_bump < 0]
x_ramp = x_bump[(x_bump >= 0) & (x_bump <= 1)]
indices = [
    5,                  # Before ramp (index in x_before_ramp)
    len(x_before_ramp), # Approx start of ramp (first point in x_ramp)
    len(x_before_ramp) + 25, # Near steepest part of ramp
    len(x_before_ramp) + 75, # Near end of ramp slope
    len(x_before_ramp) + len(x_ramp) + 5 # After ramp (index in x_after_ramp)
]

# Ensure indices are within bounds
indices = [idx for idx in indices if 0 <= idx < num_points]

plot_tangents_normals_analytical(x_bump, y_bump, indices, H=H, line_length=0.15)
######################################


# Load data
data = load_bump_data("all_stations_data.pkl")
x_cpcf, cf_data, cp_data = load_cf_cp_data()
delta_data = np.loadtxt(DELTA_STATS_PATH, delimiter=',')
delta_x = delta_data[:, 0]
delta   = delta_data[:, 1]

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
s_dist_plus = []
u_plus = []

utau = []

for key in data.keys():
    x_normal.append(data[key][:,0])
    y_normal.append(data[key][:,1])
    s_dist.append(data[key][:,2])
    u_velocity.append(data[key][:,3])
    v_velocity.append(data[key][:,4])
    s_dist_plus.append(data[key][:,5])
    u_plus.append(data[key][:,6])

    utau_all =  data[key][:,3] / data[key][:,6]
    utau_mean, utau_std = np.mean(utau_all), np.std(utau_all)
    print(f"Mean utau: {utau_mean:.3f}, Std: {utau_std:.3f}")
    utau.append(utau_mean)

# x_normal = np.array(x_normal)
# y_normal = np.array(y_normal)
# u_velocity = np.array(u_velocity)
# v_velocity = np.array(v_velocity)
# s_dist = np.array(s_dist)
# s_dist_plus = np.array(s_dist_plus)
# u_plus = np.array(u_plus)

#######################
# Reproduce plots in the paper

# Figure 7
key_of_interest = ['-0.2', '-0.4', '-0.62', '-0.72', '-0.76', '-0.82', '-0.92']
for i, key in enumerate(data.keys()):
    if key in key_of_interest:
        plt.semilogx(s_dist_plus[i], u_plus[i], label=key)
# Log law
x = np.linspace(20, 200, 50)
y_log = 1 / 0.41 * np.log(x) + 5.2
plt.semilogx(x, y_log, 'r--', label='Log Law')
# Viscous sublayer
x = np.linspace(1, 10, 20)
y = x
plt.semilogx(x, y, 'k--', label='Viscous Sublayer')
plt.legend()
plt.xlim(1, 10000)
plt.xlabel('Wall-normal distance (s+)')
plt.ylabel('Velocity (u+)')
plt.show()

# Figure 8a
key_of_interest = ['-0.2', '0.0']
for i, key in enumerate(data.keys()):
    if key in key_of_interest:
        delta_local = interp1d(delta_x, delta, kind='linear', fill_value='extrapolate')(x_normal[i][0])
        plt.semilogx(s_dist[i]/delta_local, u_velocity[i], label=key)
# plt.xlim(0, 1.2)
# Log law
plt.xlabel('Wall-normal distance (s)')
plt.ylabel('Velocity (u)')
plt.show()

#######################
x_stations = np.array([x_normal[i][0] for i in range(len(x_normal))])
delta_local = interp1d(delta_x, delta, kind='linear', fill_value='extrapolate')(x_stations)

Uo = 1.00  # Free stream velocity assumed to be 1

# Wall parallel pressure gradient
x_cp = x_cpcf
y_cp = calculate_bump_shape_given_x(x_cp)
x_cp_coord = np.vstack([x_cp, y_cp]).T

dpds_wall_all = calculate_directional_derivative(
    cp_data, x_cp_coord
) * 0.5*Uo**2
dpds_wall = interpolate_values(x_stations, x_cp, dpds_wall_all)
up = calculate_up(dpds_wall, REYNOLDS_NUMBER)

tan, nor = calculate_tangents_normals_analytical(x_bump, y_bump, H)
tan_local = np.vstack([interpolate_values(x_stations, x_bump, tan[:,0]),
                      interpolate_values(x_stations,  x_bump, tan[:,1])]).T
nor_local = np.vstack([interpolate_values(x_stations, x_bump, nor[:,0]),
                      interpolate_values(x_stations,  x_bump, nor[:,1])]).T

if save_data:
    process_and_save_region_data(
        u_velocity,
        utau,
        x_normal, # X coordinates of the wall
        s_dist, # Wall-normal distance
        delta_local,
        up,
        'smoothramp',
        up_frac=UPPER_FRACTION,
        down_frac=LOWER_FRACTION,
    )

if len(inspect_x) > 0:
    for x in inspect_x:
        ind = np.argmin(np.abs(x_stations - x))
        print(f"Inspecting point at x={x_normal[ind][0]}")

        plt.figure(figsize=(10, 6))
        plt.plot(s_dist[ind], u_velocity[ind], label='u_mag')
        # plt.plot(s_dist[ind, 1:], v_mag[ind, :], label='v_mag')
        plt.xlabel('Wall-normal distance (s)')
        plt.ylabel('Velocity')
        plt.title(f'Velocity profiles at x={x_normal[ind][0]}')
        plt.legend()

        # Plot in plus units
        fig, ax = plt.subplots(figsize=(10, 6))
        u_plus = u_velocity[ind] / utau[ind]
        s_dist_p = s_dist[ind] * REYNOLDS_NUMBER * utau[ind]
        ax.semilogx(s_dist_p, u_plus, label='u+')

        # Add log law
        y_plus = np.linspace(10, 1000, 100)
        u_plus_law = 1 / 0.41 * np.log(y_plus) + 5.2
        ax.semilogx(y_plus, u_plus_law, 'r--', label='Log Law')
        ax.set_xlabel('Wall-normal distance (s+)')
        ax.set_ylabel('Velocity (u+)')
        ax.legend()

        plt.show()
