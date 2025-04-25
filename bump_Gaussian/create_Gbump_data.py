import os
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import sys

from pandas._libs.algos import pad_inplace

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
# --- Configuration ---
# (Assume these are defined elsewhere in your code)
STATS_PATH = os.path.join(WM_DATA_PATH, "bump_Gaussian", "stats")
OUTPUT_PATH = os.path.join(WM_DATA_PATH, "data")
REYNOLDS_NUMBER = int(sys.argv[1])*1_000_000 if len(sys.argv) > 1 else 2_000_000
RE_STRING = f'{REYNOLDS_NUMBER//1_000_000}M'
UPPER_FRACTION = 0.20
LOWER_FRACTION = 0.025
UPPER_FRACTION_SEP = 0.05
LOWER_FRACTION_SEP = 0.003

# --- Select whether to save data and which points to inspect ---
import sys
save_data = int(sys.argv[2]) if len(sys.argv) > 2 else 0

plot_inspection = int(sys.argv[3]) if len(sys.argv) > 3 else 0
inspect_x = [ -0.59999815, -0.04998937, -0.00222268, 0.00998682,  0.02995354, 0.04391049, 0.10057804]

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

def load_cf_cp_delta_data():
    """Loads Cf, Cp, and delta data from text files."""

    cf_data = pd.read_csv(
        os.path.join(STATS_PATH, f"Cf_{RE_STRING}.dat"),
        skiprows=2,
        delim_whitespace=True,
        header=None,
    ).to_numpy()
    cp_data = pd.read_csv(
        os.path.join(STATS_PATH, f"Cp_{RE_STRING}.dat"),
        skiprows=2,
        delim_whitespace=True,
        header=None,
    ).to_numpy()
    delta_data = np.loadtxt(
        os.path.join(STATS_PATH, f"delta_{RE_STRING}.csv"), delimiter=","
    )
    return cf_data, cp_data, delta_data


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


def calculate_bump_parallel_velocity(x_normal, y_normal, u_velocity, v_velocity):
    """Calculates the velocity component parallel to the bump."""

    norm_vec = np.array(
        [
            [x_normal[i, 1] - x_normal[i, 0], y_normal[i, 1] - y_normal[i, 0]]
            / calculate_wall_normal_distance(x_normal, y_normal)[i, 1]
            for i in range(x_normal.shape[0])
        ]
    )
    tan_vec = np.array([[norm_vec[i, 1], -norm_vec[i, 0]] for i in range(x_normal.shape[0])])
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


def calculate_output(cf_interpolated, dist_normal, kinematic_viscosity):
    """Calculates the output feature."""
    utau = np.sqrt(0.5 * np.abs(cf_interpolated))
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
    cf_interpolated,
    x_normal,
    y_normal,
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

    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []
    kinematic_viscosity = 1 / REYNOLDS_NUMBER

    for i in range(ind_start, ind_end):

        if region == 'APG' and up[i] < 0:
            continue

        dist_normal_i = calculate_wall_normal_distance(
            x_normal[i : i + 1], y_normal[i : i + 1]
        )[0]
        idx_low_bl_i = np.where(dist_normal_i > down_frac * delta[i])[0][0]
        idx_up_bl_i = np.where(dist_normal_i <= up_frac * delta[i])[0][-1]

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

        p_i = p[i, :]
        delta_p = (p_i[idx_low_bl_i:idx_up_bl_i] - p_i[0]) / dist_normal_i[idx_low_bl_i:idx_up_bl_i]
        upn = calculate_up(delta_p, REYNOLDS_NUMBER)

        pi_groups = calculate_pi_groups(
            dist_normal_i[idx_low_bl_i:idx_up_bl_i],
            up[i],
            upn,
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
            upn,
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

    save_to_hdf5(data_dict, f"gaussian_{RE_STRING}_data_{region}")

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
        plt.title(f"Gaussian Bump at Re={REYNOLDS_NUMBER} - {region}")
        plt.show()

# # --- Sanity Check ---
#     print("\n--- Sanity Check: Comparing HDF5 with Original Pickle ---")
#     with open(f'/home/yuenongling/Codes/BFM/WM_Opt/data/gaussian_2M_data_{region}.pkl', 'rb') as f:
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


def process_separation_zone_data(
    p,
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

        p_i = p[i, :]
        delta_p = (p_i[idx_low_bl_i:idx_up_bl_i] - p_i[0]) / dist_normal_i[idx_low_bl_i:idx_up_bl_i]
        upn = calculate_up(delta_p, REYNOLDS_NUMBER)

        pi_groups = calculate_pi_groups(
            dist_normal_i[idx_low_bl_i:idx_up_bl_i],
            up[i],
            upn,
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
            upn,
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

    save_to_hdf5(data_dict, f"gaussian_{RE_STRING}_data_SEP")

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


"""Main function to process and save data for different regions of the bump."""

# Load data
data = load_bump_data(f"interpolated_data_Re_{RE_STRING}.pkl")
cf_data, cp_data, delta_data = load_cf_cp_delta_data()

# Extract relevant fields
x_normal = data["x_normal"]
y_normal = data["y_normal"]
cf_interpolated = interpolate_values(x_normal[:, 0], cf_data[:, 0], cf_data[:, 1])
delta = interpolate_values(x_normal[:, 0], delta_data[:, 0], delta_data[:, 1])
u_velocity = data["U/U_inf"]
v_velocity = data["V/U_inf"]

if REYNOLDS_NUMBER == 2_000_000:
    p          = data["p/p_inf"]
else:
    p          = data["P/p_inf"]

# Calculate necessary quantities
dpds_wall = calculate_directional_derivative(
    interpolate_values(x_normal[:, 0], cp_data[:, 0], cp_data[:, 1]), x_normal
) * 2
u_mag = calculate_bump_parallel_velocity(
    x_normal, y_normal, u_velocity, v_velocity
)[:, 1:]  # Remove wall values
up = calculate_up(dpds_wall, REYNOLDS_NUMBER)

# Process and save data for different regions
# For different Reynolds numbers, you can adjust the regions as needed

if save_data:
    if REYNOLDS_NUMBER == 2_000_000:

        # Process and save data for each region
        # Before strong favorable pressure gradient
        x_strong = -0.289
        ind_strong = np.where(x_normal[:, 0] <= x_strong)[0][-1]
        process_and_save_region_data(
            p,
            u_mag,
            cf_interpolated,
            x_normal,
            y_normal,
            delta,
            up,
            "MAPG",
            0,
            ind_strong,
            save_plots=True,
        )

        # Favorable pressure gradient region
        x_apex = -0.024
        ind_apex = np.where(x_normal[:, 0] <= x_apex)[0][-1]
        process_and_save_region_data(
            p,
            u_mag,
            cf_interpolated,
            x_normal,
            y_normal,
            delta,
            up,
            "FPG",
            ind_strong,
            ind_apex,
            save_plots=True,
        )

        # Adverse pressure gradient (separation) region
        x_sep = 0.1
        ind_sep = np.where(x_normal[:, 0] <= x_sep)[0][-1]
        process_and_save_region_data(
            p,
            u_mag,
            cf_interpolated,
            x_normal,
            y_normal,
            delta,
            up,
            "APG",
            ind_apex,
            ind_sep,
            save_plots=True,
        )

        # Additional concave and convex regions (as in the original code)
        # x_concave_low = -0.138
        # ind_concave_low = np.where(x_normal[:, 0] <= x_concave_low)[0][-1]
        # process_and_save_region_data(
        #     u_mag,
        #     cf_interpolated,
        #     x_normal,
        #     y_normal,
        #     delta,
        #     up,
        #     "concave_a",
        #     0,
        #     ind_concave_low,
        #     save_plots=True,
        # )
        #
        # x_concave_up = 0.1  # Only plot up to separation
        # ind_concave_up = np.where(x_normal[:, 0] <= x_concave_up)[0][-1]
        # process_and_save_region_data(
        #     u_mag,
        #     cf_interpolated,
        #     x_normal,
        #     y_normal,
        #     delta,
        #     up,
        #     "convex",
        #     ind_concave_low,
        #     ind_concave_up,
        #     save_plots=True,
        # )

        # process_and_save_region_data(
        #     u_mag,
        #     cf_interpolated,
        #     x_normal,
        #     y_normal,
        #     delta,
        #     up,
        #     "fpg_concave",
        #     ind_strong,
        #     ind_concave_low,
        #     save_plots=True,
        # )
        # process_and_save_region_data(
        #     u_mag,
        #     cf_interpolated,
        #     x_normal,
        #     y_normal,
        #     delta,
        #     up,
        #     "fpg_convex",
        #     ind_concave_low,
        #     ind_apex,
        #     save_plots=True,
        # )

        # Separation zone (using the specialized function)
        x_reatt = 0.42
        ind_reatt = np.where(x_normal[:, 0] <= x_reatt)[0][-1]
        process_separation_zone_data(
            p,
            u_mag,
            cf_interpolated,
            x_normal,
            y_normal,
            delta,
            up,
            ind_sep,
            ind_reatt,
            save_plots=True,
        )

    elif REYNOLDS_NUMBER == 1_000_000:

        # Process and save data for each region
        # Before strong favorable pressure gradient
        x_strong = -0.294
        ind_strong = np.where(x_normal[:, 0] <= x_strong)[0][-1]
        process_and_save_region_data(
            p,
            u_mag,
            cf_interpolated,
            x_normal,
            y_normal,
            delta,
            up,
            "MAPG",
            3,
            ind_strong,
            save_plots=True,
        )

        x_concave = -0.138 
        ind_fpg_concave = np.where(x_normal[:, 0] <= x_concave)[0][-1]
        process_and_save_region_data(
            p,
            u_mag,
            cf_interpolated,
            x_normal,
            y_normal,
            delta,
            up,
            "FPG_concave",
            ind_strong,
            ind_fpg_concave,
            save_plots=True,
        )

        x_fpg = -0.0012
        ind_fpg = np.where(x_normal[:, 0] <= x_fpg)[0][-1]
        process_and_save_region_data(
            p,
            u_mag,
            cf_interpolated,
            x_normal,
            y_normal,
            delta,
            up,
            "FPG_convex",
            ind_fpg_concave,
            ind_fpg,
            save_plots=True,
        )

        # This is the region where apg but laminar (or in process of turbulent transition)
        x_apg_stable = 0.05
        ind_apg_stable  = np.where(x_normal[:, 0] <= x_apg_stable)[0][-1]
        process_and_save_region_data(
            p,
            u_mag,
            cf_interpolated,
            x_normal,
            y_normal,
            delta,
            up,
            "APG_stable",
            ind_fpg,
            ind_apg_stable,
            save_plots=True,
        )

        # This is the region where apg but laminar (or in process of turbulent transition)
        x_sep = 0.195
        ind_sep  = np.where(x_normal[:, 0] <= x_sep)[0][-1]
        process_and_save_region_data(
            p,
            u_mag,
            cf_interpolated,
            x_normal,
            y_normal,
            delta,
            up,
            "APG",
            ind_apg_stable,
            ind_sep,
            save_plots=True,
        )

        # Separation zone (using the specialized function)
        x_reatt = 0.268
        ind_reatt = np.where(x_normal[:, 0] <= x_reatt)[0][-1]
        process_separation_zone_data(
            p,
            u_mag,
            cf_interpolated,
            x_normal,
            y_normal,
            delta,
            up,
            ind_sep,
            ind_reatt,
            save_plots=True,
        )

utau = np.sqrt(0.5 * np.abs(cf_interpolated))
s_dist = calculate_wall_normal_distance(x_normal, y_normal)

if plot_inspection:
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
            u_plus = u_mag[ind, :] / utau[ind]
            s_dist_p = s_dist[ind, 1:] * REYNOLDS_NUMBER * utau[ind]
            ax.semilogx(s_dist_p, u_plus, label='u+')
            y_plus = np.linspace(50, 1000, 100)
            u_plus_law = 1 / 0.41 * np.log(y_plus) + 5.2
            ax.semilogx(y_plus, u_plus_law, 'r--', label='Log Law')
            ax.set_xlabel('Wall-normal distance (s+)')
            ax.set_ylabel('Velocity (u+)')
            ax.legend()

            plt.show()
