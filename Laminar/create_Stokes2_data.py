# ./Stokes2_folder/create_Stokes2_data.py
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import math

# --- Path Adjustments ---
# Assuming data will be saved in a 'data' subdirectory relative to the script location
# You might need to adjust this based on your actual directory structure
script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
output_data_dir = os.path.join(script_dir, "..", "data") # Example: Save in parent_dir/data

# --- Physical and Simulation Constants ---
# Ranges for parameters (adjust as needed)
U0_values = [1.0, 2.0] # Amplitude of wall velocity
omega_values = [1.0, 5.0, 10.0] # Angular frequency (rad/s)
nu_values = [1.0e-6, 1.5e-6] # Kinematic viscosity (m^2/s)
rho = 1.225 # Fluid density (kg/m^3)

# Simulation parameters
time_steps_per_period = 20 # Number of time steps to sample per oscillation period
num_periods_for_steady_state = 2 # Number of periods to pass before sampling (for steady-oscillating state)
num_periods_to_sample = 1 # Number of periods to sample data from the steady state
y_points_per_delta_s = 50 # Number of points per Stokes boundary layer thickness
max_delta_s_multiple = 5 # Maximum y distance to sample (e.g., 5 * delta_s)
YPLUS_THRESHOLD = 5.0 # Minimum y+ to include

# Define the steps for k in the FS-like inputs (velocity and derivative at y + k*delta_s)
# u1: y, u2: y+delta_s, u3: y+2*delta_s, u4: y+3*delta_s
# dudy1: y, dudy2: y+delta_s, dudy3: y+2*delta_s
k_values_u = [0, 1, 2, 3]
k_values_dudy = [0, 1, 2]

# --- Analytical Solution and Derivatives for Stokes' Second Problem (Steady Oscillating State) ---
# Assuming Wall Velocity U_w(t) = U0 * cos(omega * t)
# Velocity: u(y, t) = U0 * exp(-eta) * cos(omega*t - eta) where eta = y * sqrt(omega / (2*nu))
def stokes2_velocity(y, t, U0, omega, nu):
    """Calculates the velocity u(y, t) for Stokes' second problem (steady oscillating state)."""
    if y < 0 or t < 0: return 0.0 # Or handle appropriately for boundary/initial conditions
    eta = y * np.sqrt(omega / (2.0 * nu))
    return U0 * np.exp(-eta) * np.cos(omega * t - eta)

# Derivative of velocity w.r.t. y: du/dy
# du/dy = U0 * sqrt(omega/(2*nu)) * exp(-eta) * [sin(omega*t - eta) - cos(omega*t - eta)]
def stokes2_dudy(y, t, U0, omega, nu):
    """Calculates the velocity derivative du/dy for Stokes' second problem."""
    if y < 0 or t < 0: return 0.0
    eta = y * np.sqrt(omega / (2.0 * nu))
    sqrt_omega_2nu = np.sqrt(omega / (2.0 * nu))
    return U0 * sqrt_omega_2nu * np.exp(-eta) * (np.sin(omega * t - eta) - np.cos(omega * t - eta))

# Derivative of velocity w.r.t. y at y=0: (du/dy)_w
# (du/dy)_w = U0 * sqrt(omega/(2*nu)) * (sin(omega*t) - cos(omega*t))
def stokes2_dudy_wall(t, U0, omega, nu):
    """Calculates the velocity derivative at the wall (y=0)."""
    if t < 0: return 0.0
    sqrt_omega_2nu = np.sqrt(omega / (2.0 * nu))
    return U0 * sqrt_omega_2nu * (np.sin(omega * t) - np.cos(omega * t))

# Second derivative of velocity w.r.t. y: d2u/dy2
# d2u/dy2 = -U0 * (omega/nu) * exp(-eta) * sin(omega*t - eta)
def stokes2_d2udy2(y, t, U0, omega, nu):
    """Calculates the second velocity derivative d2u/dy2."""
    if y < 0 or t < 0: return 0.0
    eta = y * np.sqrt(omega / (2.0 * nu))
    return -U0 * (omega / nu) * np.exp(-eta) * np.sin(omega * t - eta)

# Third derivative of velocity w.r.t. y: d3u/dy3
# d3u/dy3 = U0 * (omega/nu) * sqrt(omega/(2*nu)) * exp(-eta) * [cos(omega*t - eta) + sin(omega*t - eta)]
def stokes2_d3udy3(y, t, U0, omega, nu):
    """Calculates the third velocity derivative d3u/dy3."""
    if y < 0 or t < 0: return 0.0
    eta = y * np.sqrt(omega / (2.0 * nu))
    omega_nu = omega / nu
    sqrt_omega_2nu = np.sqrt(omega / (2.0 * nu))
    return U0 * omega_nu * sqrt_omega_2nu * np.exp(-eta) * (np.cos(omega * t - eta) + np.sin(omega * t - eta))


# --- Helper Functions ---
def stokes_boundary_layer_thickness(omega, nu):
    """Calculates the Stokes boundary layer thickness delta_s = sqrt(2*nu/omega)."""
    if omega <= 0 or nu <= 0: return 0.0
    return np.sqrt(2.0 * nu / omega)

def instantaneous_friction_velocity(t, U0, omega, nu, rho):
    """Calculates the instantaneous friction velocity u_tau(t) = sqrt(|tau_w(t)|/rho)."""
    # tau_w(t) = mu * (du/dy)_w = rho * nu * (du/dy)_w
    # u_tau(t) = sqrt(|rho * nu * (du/dy)_w| / rho) = sqrt(|nu * (du/dy)_w|)
    dudy_w = stokes2_dudy_wall(t, U0, omega, nu)
    return np.sqrt(abs(nu * dudy_w))


# --- Profile Calculation Function ---
def calculate_stokes2_profile(U0, omega, nu, rho, t_val, y_points_per_delta_s, max_delta_s_multiple, yplus_threshold):
    """
    Calculates detailed boundary layer profile data for a given Stokes II case at a specific time.

    Args:
        U0 (float): Amplitude of wall velocity.
        omega (float): Angular frequency.
        nu (float): Kinematic viscosity.
        rho (float): Fluid density.
        t_val (float): Current time.
        y_points_per_delta_s (int): Number of y points to sample per delta_s.
        max_delta_s_multiple (float): Maximum y distance as a multiple of delta_s.
        yplus_threshold (float): Minimum y+ to include.

    Returns:
        list: A list of dictionaries, where each dictionary contains data for a single y point,
              or an empty list if u_tau is zero or invalid parameters.
    """
    delta_s = stokes_boundary_layer_thickness(omega, nu)
    if delta_s <= 1e-12: return [] # Avoid division by zero or invalid cases

    u_tau_val = instantaneous_friction_velocity(t_val, U0, omega, nu, rho)
    if u_tau_val < 1e-9: # u_tau is effectively zero at certain times in the cycle
         return [] # Skip this time step if u_tau is zero

    # Define y points to sample
    max_y = max_delta_s_multiple * delta_s
    num_y_points = int(y_points_per_delta_s * max_delta_s_multiple)
    if num_y_points < 10: num_y_points = 10 # Ensure a minimum number of points
    # Sample y from a small value up to max_y to avoid y=0 issues in normalization
    y_vals = np.linspace(1e-9 * delta_s, max_y, num_y_points) # Start slightly above 0

    # Calculate wall velocity and its acceleration at the current time
    wall_vel = U0 * np.cos(omega * t_val)
    # Calculate wall acceleration (derivative of U0*cos(omega*t))
    dudt_wall = - np.sin(omega * t_val) * omega * U0

    # Calculate the 'up' velocity scale based on the wall's acceleration magnitude and sign
    # This follows the structure provided by the user snippet, substituting wall acceleration
    # for the pressure gradient term seen in the FS script.
    # This IS NOT a pressure gradient term for Stokes II as dp/dx = 0.
    up_val = -np.sign(dudt_wall) * (np.abs(dudt_wall) * nu)**(1/3)
    # Handle the case where dudt_wall is exactly zero - sign(0) is 0, so up_val will be 0, which is correct.


    profile_data = []

    for y in y_vals:
        # Calculate y+
        yplus = y * u_tau_val / nu
        if yplus < yplus_threshold:
            continue # Skip points below the y+ threshold

        # Calculate velocity and derivatives at y and y + k*delta_s
        # Use analytical solutions directly for y + k*delta_s, even if it exceeds max_y
        u_k = [stokes2_velocity(y + k * delta_s, t_val, U0, omega, nu) for k in k_values_u]
        dudy_k = [stokes2_dudy(y + k * delta_s, t_val, U0, omega, nu) for k in k_values_dudy]

        # Calculate second and third derivatives at y
        d2udy2_at_y = stokes2_d2udy2(y, t_val, U0, omega, nu)
        d3udy3_at_y = stokes2_d3udy3(y, t_val, U0, omega, nu)


        # Calculate normalized inputs (matching FS structure where applicable)
        # FS Inputs: u1_y_over_nu, up_y_over_nu, upn_y_over_nu, u2_y_over_nu, u3_y_over_nu, u4_y_over_nu, dudy1_y_pow2_over_nu, dudy2_y_pow2_over_nu, dudy3_y_pow2_over_nu
        inputs_row = {
            'u1_y_over_nu': u_k[0] * y / nu, # u(y) * y / nu
            'up_y_over_nu': up_val * y / nu, # Velocity scale based on wall accel * y / nu
            'upn_y_over_nu': 0.0, # No other pressure gradient analog in standard Stokes II
            'u2_y_over_nu': u_k[1] * y / nu if len(u_k) > 1 else 0.0, # u(y+delta_s) * y / nu
            'u3_y_over_nu': u_k[2] * y / nu if len(u_k) > 2 else 0.0, # u(y+2*delta_s) * y / nu
            'u4_y_over_nu': u_k[3] * y / nu if len(u_k) > 3 else 0.0, # u(y+3*delta_s) * y / nu
            'dudy1_y_pow2_over_nu': dudy_k[0] * y**2 / nu if len(dudy_k) > 0 else 0.0, # du/dy(y) * y^2 / nu
            'dudy2_y_pow2_over_nu': dudy_k[1] * y**2 / nu if len(dudy_k) > 1 else 0.0, # du/dy(y+delta_s) * y^2 / nu
            'dudy3_y_pow2_over_nu': dudy_k[2] * y**2 / nu if len(dudy_k) > 2 else 0.0 # du/dy(y+2*delta_s) * y^2 / nu
        }

        # Calculate normalized output (matching FS structure)
        # FS Output: utau_y_over_nu
        output_row = {
            'utau_y_over_nu': yplus # u_tau * y / nu (i.e., y+)
        }

        # Store unnormalized inputs
        unnormalized_inputs_row = {
            'y': y,
            'u1': u_k[0], 'u2': u_k[1] if len(u_k) > 1 else 0.0,
            'u3': u_k[2] if len(u_k) > 2 else 0.0, 'u4': u_k[3] if len(u_k) > 3 else 0.0,
            'dudy1': dudy_k[0] if len(dudy_k) > 0 else 0.0,
            'dudy2': dudy_k[1] if len(dudy_k) > 1 else 0.0,
            'dudy3': dudy_k[2] if len(dudy_k) > 2 else 0.0,
            'nu': nu,
            'utau': u_tau_val, # Instantaneous u_tau
            'U0': U0,
            'omega': omega,
            't': t_val,
            'delta_s': delta_s,
            'd2udy2': d2udy2_at_y, # Include second derivative at y
            'd3udy3': d3udy3_at_y, # Include third derivative at y
            'wall_vel': wall_vel, # Include instantaneous wall velocity
            'dudt_wall': dudt_wall, # Include instantaneous wall acceleration
            'up': up_val # Include the calculated 'up' velocity scale
        }

        # Store flow type information (case parameters + time)
        flow_type_row = {
            'case_name': f"Stokes2_U{U0}_O{omega}_N{nu}",
            'U0': U0,
            'omega': omega,
            'delta_s': delta_s,
            'nu': nu,
            'rho': rho,
            't': t_val,
            'utau_inst': u_tau_val # Store instantaneous u_tau for reference
        }

        profile_data.append({
            'inputs': inputs_row,
            'output': output_row,
            'unnormalized_inputs': unnormalized_inputs_row,
            'flow_type': flow_type_row
        })

    return profile_data


# --- Main Execution ---
if __name__ == "__main__":
    all_data_points = []
    print("Calculating Stokes' Second Problem profiles...")

    total_cases = len(U0_values) * len(omega_values) * len(nu_values)
    total_time_steps_per_case = time_steps_per_period * num_periods_to_sample
    total_combinations = total_cases * total_time_steps_per_case


    with tqdm(total=total_combinations, desc="Calculating Profiles") as pbar:
        for U0 in U0_values:
            for omega in omega_values:
                period = 2 * np.pi / omega
                # Generate time steps within the steady-oscillating state
                start_time = num_periods_for_steady_state * period
                end_time = start_time + num_periods_to_sample * period
                time_steps_for_case = np.linspace(start_time, end_time, total_time_steps_per_case, endpoint=False)

                for nu in nu_values:
                    for t_val in time_steps_for_case:
                        profile_data = calculate_stokes2_profile(
                            U0, omega, nu, rho, t_val,
                            y_points_per_delta_s, max_delta_s_multiple, YPLUS_THRESHOLD
                        )
                        all_data_points.extend(profile_data)
                        pbar.update(1)

    if not all_data_points:
        print("No valid data points were generated overall. Exiting.")
        sys.exit(1)
    print(f"\nGenerated {len(all_data_points)} valid data points in total.")

    # --- Separate data into lists of dictionaries for each DataFrame ---
    inputs_list = [d['inputs'] for d in all_data_points]
    output_list = [d['output'] for d in all_data_points]
    unnormalized_inputs_list = [d['unnormalized_inputs'] for d in all_data_points]
    flow_type_list = [d['flow_type'] for d in all_data_points]

    # --- Create DataFrames ---
    print("\nCreating DataFrames...")
    inputs_df = pd.DataFrame(inputs_list)
    output_df = pd.DataFrame(output_list)
    unnormalized_inputs_df = pd.DataFrame(unnormalized_inputs_list)
    flow_type_df = pd.DataFrame(flow_type_list)

    # --- Save DataFrames to HDF5 ---
    os.makedirs(output_data_dir, exist_ok=True)
    output_filename = os.path.join(output_data_dir, f"Stokes2_data.h5") # Changed filename to reflect change
    print(f"\nSaving data to HDF5 file: {output_filename}")

    try:
        # Use 'w' mode for the first key, 'a' for subsequent keys
        inputs_df.to_hdf(output_filename, key='inputs', mode='w', format='fixed')
        output_df.to_hdf(output_filename, key='output', mode='a', format='fixed')
        unnormalized_inputs_df.to_hdf(output_filename, key='unnormalized_inputs', mode='a', format='fixed')
        # flow_type_df can be saved as 'table' format which is generally more flexible
        flow_type_df.to_hdf(output_filename, key='flow_type', mode='a', format='table')
        print("Data successfully saved.")
    except Exception as e:
        print(f"Error saving HDF5 file: {e}")
        sys.exit(1)

    print(f"\n--- Summary ---")
    print(f"Data successfully saved to: {output_filename}")
    print(f"Final Shapes:\n  Inputs: {inputs_df.shape}\n  Output: {output_df.shape}\n  Flow Type: {flow_type_df.shape}\n  Unnormalized Inputs: {unnormalized_inputs_df.shape}")
