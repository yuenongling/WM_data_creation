# ./Pipe_OscillatingPG_folder/create_PipePG_data.py
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from scipy.special import iv # Modified Bessel functions of the first kind I_nu(z)

# --- Path Adjustments ---
# Assuming data will be saved in a 'data' subdirectory relative to the script location
script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
output_data_dir = os.path.join(script_dir, "..", "data") # Example: Save in parent_dir/data

# --- Physical and Simulation Constants ---
# Ranges for parameters (adjust as needed)
G_amplitude_values = [0.1, 1.0] # Amplitude of pressure gradient (Delta P / L) (Pa/m)
omega_values = [1.0, 5.0, 10.0] # Angular frequency (rad/s)
nu_values = [1.0e-6, 1.5e-6] # Kinematic viscosity (m^2/s)
pipe_radius_values = [0.05, 0.1] # Pipe radius (m)
rho = 1.225 # Fluid density (kg/m^3)
mu = rho * nu_values[0] # Assuming a nominal viscosity for mu calc, nu is the primary parameter

# Simulation parameters
time_steps_per_period = 20 # Number of time steps to sample per oscillation period
num_periods_to_sample = 1 # Number of periods to sample data from the steady-oscillating state
r_points_per_radius = 100 # Number of radial points across the radius
yplus_threshold = 5.0 # Minimum y+ (distance from wall) to include

# Define the steps for k in the FS-like inputs (velocity and derivative at y + k*delta_p)
# Mapping y (distance from wall) to r (radial coordinate): y = a - r
# y+k*delta_p corresponds to r - k*delta_p
# We sample velocity/derivatives at r and offsets r - k*delta_p
# u1: r, u2: r-delta_p, u3: r-2*delta_p, u4: r-3*delta_p
# dudy1: r, dudy2: r-delta_p, dudy3: r-2*delta_p
k_values_u = [0, 1, 2, 3]
k_values_dudy = [0, 1, 2]


# --- Complex Analytical Solution Components ---

# Beta = sqrt(i * omega / nu)
def calculate_beta(omega, nu):
    """Calculates the complex parameter beta = sqrt(i * omega / nu)."""
    if omega <= 0 or nu <= 0: return 0.0j
    # sqrt(i) = exp(i*pi/4) = cos(pi/4) + i*sin(pi/4) = 1/sqrt(2) * (1 + i)
    sqrt_i = np.sqrt(0.5) * (1.0 + 1.0j)
    return sqrt_i * np.sqrt(omega / nu)

# Complex function f(r)
# f(r) = G / (i * rho * omega) * (1 - I_0(beta * r) / I_0(beta * a))
def complex_f(r, G_amp, omega, nu, rho, a):
    """Calculates the complex function f(r)."""
    # Corrected: Use 1j for the imaginary unit
    if abs(1j * rho * omega) < 1e-12: return 0.0j # Avoid division by zero
    beta = calculate_beta(omega, nu)
    beta_r = beta * r
    beta_a = beta * a
    try:
        I0_br = iv(0, beta_r)
        I0_ba = iv(0, beta_a)
        if abs(I0_ba) < 1e-12: return 0.0j # Avoid division by zero
        # Corrected: Use 1j for the imaginary unit
        return (G_amp / (1j * rho * omega)) * (1.0 - I0_br / I0_ba)
    except Exception as e:
        # print(f"Error calculating complex_f at r={r}, t={t}, G={G_amp}, o={omega}, n={nu}, a={a}: {e}")
        return 0.0j # Return zero if calculation fails


# Complex derivative f'(r) = df/dr
# f'(r) = - G * beta * I_1(beta * r) / (i * rho * omega * I_0(beta * a))
def complex_dfdr(r, G_amp, omega, nu, rho, a):
    """Calculates the complex derivative df/dr."""
    # Corrected: Use 1j for the imaginary unit
    if abs(1j * rho * omega) < 1e-12: return 0.0j # Avoid division by zero
    beta = calculate_beta(omega, nu)
    if abs(beta) < 1e-12: return 0.0j # Avoid division by zero
    beta_r = beta * r
    beta_a = beta * a
    try:
        I1_br = iv(1, beta_r)
        I0_ba = iv(0, beta_a)
        if abs(I0_ba) < 1e-12: return 0.0j # Avoid division by zero
        # Corrected: Use 1j for the imaginary unit
        return -G_amp * beta * I1_br / (1j * rho * omega * I0_ba)
    except Exception as e:
        # print(f"Error calculating complex_dfdr at r={r}, t={t}, G={G_amp}, o={omega}, n={nu}, a={a}: {e}")
        return 0.0j # Return zero if calculation fails

# Complex second derivative f''(r) = d2f/dr2
# f''(r) = - G * beta^2 / (i * rho * omega * I_0(beta * a)) * (I_0(beta * r) - I_1(beta * r) / (beta * r))
def complex_d2fdr2(r, G_amp, omega, nu, rho, a):
    """Calculates the complex second derivative d2f/dr2."""
    # Corrected: Use 1j for the imaginary unit
    if r < 1e-12 or abs(1j * rho * omega) < 1e-12: return 0.0j # Avoid division by zero near center or invalid params
    beta = calculate_beta(omega, nu)
    if abs(beta) < 1e-12: return 0.0j
    beta_r = beta * r
    beta_a = beta * a
    try:
        I0_br = iv(0, beta_r)
        I1_br = iv(1, beta_r)
        I0_ba = iv(0, beta_a)
        if abs(I0_ba) < 1e-12 or abs(beta_r) < 1e-12: return 0.0j # Avoid division by zero
        # Corrected: Use 1j for the imaginary unit
        return -G_amp * beta**2 / (1j * rho * omega * I0_ba) * (I0_br - I1_br / beta_r)
    except Exception as e:
        # print(f"Error calculating complex_d2fdr2 at r={r}, t={t}, G={G_amp}, o={omega}, n={nu}, a={a}: {e}")
        return 0.0j

# Complex third derivative f'''(r) = d3f/dr3
# f'''(r) = - G beta^3 / (i rho omega I_0(ba)) * [I_1(beta r) (1 + 2/(br)^2) - I_0(beta r) / (br)]
def complex_d3fdr3(r, G_amp, omega, nu, rho, a):
    """Calculates the complex third derivative d3f/dr3."""
    # Corrected: Use 1j for the imaginary unit
    if r < 1e-12 or abs(1j * rho * omega) < 1e-12: return 0.0j
    beta = calculate_beta(omega, nu)
    if abs(beta) < 1e-12: return 0.0j
    beta_r = beta * r
    beta_a = beta * a
    try:
        I0_br = iv(0, beta_r)
        I1_br = iv(1, beta_r)
        I0_ba = iv(0, beta_a)
        if abs(I0_ba) < 1e-12 or abs(beta_r) < 1e-12: return 0.0j
        term1 = I1_br * (1.0 + 2.0 / beta_r**2)
        term2 = I0_br / beta_r
        # Corrected: Use 1j for the imaginary unit
        return -G_amp * beta**3 / (1j * rho * omega * I0_ba) * (term1 - term2)
    except Exception as e:
        # print(f"Error calculating complex_d3fdr3 at r={r}, t={t}, G={G_amp}, o={omega}, n={nu}, a={a}: {e}")
        return 0.0j


# Real velocity and derivatives
def uz(r, t, G_amp, omega, nu, rho, a):
    """Calculates the real axial velocity uz(r, t)."""
    # Ensure r is within the valid domain [0, a]
    if not (0 <= r <= a): return 0.0
    f_val = complex_f(r, G_amp, omega, nu, rho, a)
    # Re{f * exp(i*omega*t)} = Re{(Xr + i*Xi)*(cos(wt) + i*sin(wt))} = Xr*cos(wt) - Xi*sin(wt)
    return f_val.real * np.cos(omega * t) - f_val.imag * np.sin(omega * t)

def duzdr(r, t, G_amp, omega, nu, rho, a):
    """Calculates the real derivative duz/dr."""
     # Ensure r is within the valid domain [0, a]
    if not (0 <= r <= a): return 0.0
    dfdr_val = complex_dfdr(r, G_amp, omega, nu, rho, a)
    return dfdr_val.real * np.cos(omega * t) - dfdr_val.imag * np.sin(omega * t)

def d2uzdr2(r, t, G_amp, omega, nu, rho, a):
    """Calculates the real second derivative d2uz/dr2."""
     # Ensure r is within the valid domain [0, a]
    if not (0 <= r <= a): return 0.0
    d2fdr2_val = complex_d2fdr2(r, G_amp, omega, nu, rho, a)
    return d2fdr2_val.real * np.cos(omega * t) - d2fdr2_val.imag * np.sin(omega * t)

def d3uzdr3(r, t, G_amp, omega, nu, rho, a):
    """Calculates the real third derivative d3uz/dr3."""
     # Ensure r is within the valid domain [0, a]
    if not (0 <= r <= a): return 0.0
    d3fdr3_val = complex_d3fdr3(r, G_amp, omega, nu, rho, a)
    return d3fdr3_val.real * np.cos(omega * t) - d3fdr3_val.imag * np.sin(omega * t)


# --- Helper Functions ---
def oscillating_pg_penetration_depth(omega, nu):
    """Calculates the penetration depth delta_p = sqrt(2*nu/omega)."""
    if omega <= 0 or nu <= 0: return 0.0
    return np.sqrt(2.0 * nu / omega)

def instantaneous_wall_shear_stress(t, G_amp, omega, nu, rho, a):
    """Calculates the instantaneous wall shear stress tau_w(t) = mu * (duz/dr)_r=a."""
    # tau_w(t) = rho * nu * (duz/dr)_r=a
    # (duz/dr)_r=a = Re{ f'(a) * exp(i*omega*t) }
    dfdr_a = complex_dfdr(a, G_amp, omega, nu, rho, a)
    real_dfdr_a_exp_iwt = dfdr_a.real * np.cos(omega * t) - dfdr_a.imag * np.sin(omega * t)
    return rho * nu * real_dfdr_a_exp_iwt

def instantaneous_friction_velocity(t, G_amp, omega, nu, rho, a):
    """Calculates the instantaneous friction velocity u_tau(t) = sqrt(|tau_w(t)|/rho)."""
    tau_w_val = instantaneous_wall_shear_stress(t, G_amp, omega, nu, rho, a)
    return np.sqrt(abs(tau_w_val) / rho)

def pressure_gradient_velocity_scale(G_amp, nu, rho):
    """Calculates a pressure gradient velocity scale ( (|G|/rho * nu)^(1/3) )."""
    if abs(G_amp) < 1e-12 or nu < 1e-12 or rho < 1e-12: return 0.0
    return (abs(G_amp) / rho * nu)**(1/3)


# --- Profile Calculation Function ---
def calculate_pipe_pg_profile(G_amp, omega, nu, rho, a, t_val, r_points_per_radius, yplus_threshold):
    """
    Calculates detailed boundary layer profile data for a given pipe flow case at a specific time.

    Args:
        G_amp (float): Amplitude of pressure gradient.
        omega (float): Angular frequency.
        nu (float): Kinematic viscosity.
        rho (float): Fluid density.
        a (float): Pipe radius.
        t_val (float): Current time.
        r_points_per_radius (int): Number of radial points per radius.
        yplus_threshold (float): Minimum y+ (distance from wall) to include.

    Returns:
        list: A list of dictionaries, where each dictionary contains data for a single y point,
              or an empty list if u_tau is zero or invalid parameters.
    """
    delta_p = oscillating_pg_penetration_depth(omega, nu)
    if a <= 1e-12 or delta_p <= 1e-12: return [] # Avoid division by zero or invalid cases

    u_tau_val = instantaneous_friction_velocity(t_val, G_amp, omega, nu, rho, a)
    if u_tau_val < 1e-9: # u_tau is effectively zero at certain times
         return [] # Skip this time step if u_tau is near zero

    # Define radial points to sample. Sample from center to wall.
    # Points are spaced from r=0 to r=a.
    num_r_points = int(r_points_per_radius)
    if num_r_points < 2: num_r_points = 2 # Ensure at least center and near wall
    # Sample r linearly. We will filter by y+ later.
    # Start slightly above 0 to avoid potential r=0 issues if not handled perfectly by Bessel functions
    # But the analytical solution for r=0 is finite, so 0.0 is fine.
    r_vals_full = np.linspace(0.0, a, num_r_points)

    profile_data = []

    # Calculate pressure gradient related velocity scale at this time
    # Pressure gradient is Re{ G_amp * exp(i*omega*t) }
    pg_instantaneous = G_amp * np.cos(omega * t_val)
    up_scale_magnitude = pressure_gradient_velocity_scale(G_amp, nu, rho)
    up_val = np.sign(pg_instantaneous) * up_scale_magnitude # Signed pressure gradient velocity scale

    for r in r_vals_full:
        y = a - r # Distance from the wall
        # Include points from the wall (y=0) up to the center (y=a)
        # The y+ filter handles the near-wall exclusion as needed.
        # Ensure y >= 0 due to linspace(0, a)
        if y < -1e-9: continue # Just a safety check for float comparisons

        # Calculate y+
        # Ensure y >= 0 before calculating y+ to avoid issues if u_tau_val is 0 and y is negative (shouldn't happen here)
        yplus = y * u_tau_val / nu if y >= 0 else 0.0
        if yplus < yplus_threshold:
            continue # Skip points below the y+ threshold

        # Calculate velocity and derivatives at r and offsets r - k*delta_p
        # r_k_vals are radial positions for the offsets
        r_k_vals = [r - k * delta_p for k in k_values_u] # Offsets towards the center

        # Ensure r_k_vals are within [0, a]. For points outside, use the analytical solution,
        # which correctly decays. The functions handle r>=0.
        uz_k = [uz(rk, t_val, G_amp, omega, nu, rho, a) for rk in r_k_vals]

        # Derivatives w.r.t y: du/dy = -du/dr, d2u/dy2 = d2u/dr2, etc.
        dudy_k_vals_r = [r - k * delta_p for k in k_values_dudy] # Radial positions for offsets
        # du/dy = -duz/dr
        dudy_k = [-duzdr(rk, t_val, G_amp, omega, nu, rho, a) for rk in dudy_k_vals_r]

        # Calculate normalized inputs (matching FS structure where applicable)
        # Use y = a - r as the distance from the wall
        # FS Inputs: u1_y_over_nu, up_y_over_nu, upn_y_over_nu, u2_y_over_nu, u3_y_over_nu, u4_y_over_nu, dudy1_y_pow2_over_nu, dudy2_y_pow2_over_nu, dudy3_y_pow2_over_nu
        inputs_row = {
            'u1_y_over_nu': uz_k[0] * y / nu, # uz(r) * y / nu
            'up_y_over_nu': up_val * y / nu, # Pressure gradient velocity scale * y / nu
            'upn_y_over_nu': 0.0, # No other pressure gradient analog in standard pipe flow
            'u2_y_over_nu': uz_k[1] * y / nu if len(uz_k) > 1 else 0.0, # uz(r-delta_p) * y / nu
            'u3_y_over_nu': uz_k[2] * y / nu if len(uz_k) > 2 else 0.0, # uz(r-2*delta_p) * y / nu
            'u4_y_over_nu': uz_k[3] * y / nu if len(uz_k) > 3 else 0.0, # uz(r-3*delta_p) * y / nu
            'dudy1_y_pow2_over_nu': dudy_k[0] * y**2 / nu if len(dudy_k) > 0 else 0.0, # du/dy(r) * y^2 / nu
            'dudy2_y_pow2_over_nu': dudy_k[1] * y**2 / nu if len(dudy_k) > 1 else 0.0, # du/dy(r-delta_p) * y^2 / nu
            'dudy3_y_pow2_over_nu': dudy_k[2] * y**2 / nu if len(dudy_k) > 2 else 0.0 # du/dy(r-2*delta_p) * y^2 / nu
        }

        # Calculate normalized output (matching FS structure)
        # FS Output: utau_y_over_nu
        output_row = {
            'utau_y_over_nu': yplus # u_tau * y / nu (i.e., y+)
        }

        # Store unnormalized inputs
        unnormalized_inputs_row = {
            'r': r,
            'y': y, # Distance from wall
            'u1': uz_k[0], 'u2': uz_k[1] if len(uz_k) > 1 else 0.0,
            'u3': uz_k[2] if len(uz_k) > 2 else 0.0, 'u4': uz_k[3] if len(uz_k) > 3 else 0.0,
            'dudy1': dudy_k[0] if len(dudy_k) > 0 else 0.0, # du/dy
            'dudy2': dudy_k[1] if len(dudy_k) > 1 else 0.0, # du/dy
            'dudy3': dudy_k[2] if len(dudy_k) > 2 else 0.0, # du/dy
            'nu': nu,
            'rho': rho,
            'utau': u_tau_val, # Instantaneous u_tau
            'G_amp': G_amp,
            'omega': omega,
            'a': a, # Pipe radius
            't': t_val,
            'delta_p': delta_p,
            'pg_instantaneous': pg_instantaneous, # Instantaneous real pressure gradient
            'up': up_val # Calculated 'up' velocity scale
            # Note: duzdr, d2uzdr2, d3uzdr3 (related to du/dy, d2u/dy2, d3u/dy3) at the specific 'r' point
            # could be added to unnormalized_inputs if needed, but sticking closer to FS structure.
            # d2udy2_at_y = d2uzdr2(r, t_val, G_amp, omega, nu, rho, a)
            # d3udy3_at_y = -d3uzdr3(r, t_val, G_amp, omega, nu, rho, a)
            # 'd2udy2': d2udy2_at_y,
            # 'd3udy3': d3udy3_at_y,
        }

        # Store flow type information (case parameters + time)
        flow_type_row = {
            'case_name': f"PipePG_G{G_amp}_O{omega}_N{nu}_A{a}",
            'G_amp': G_amp,
            'omega': omega,
            'nu': nu,
            'rho': rho,
            'a': a,
            't': t_val,
            'delta_p': delta_p,
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
    print("Calculating Oscillating Pipe Flow profiles...")

    total_cases = len(G_amplitude_values) * len(omega_values) * len(nu_values) * len(pipe_radius_values)
    total_time_steps_per_case = time_steps_per_period * num_periods_to_sample
    total_combinations = total_cases * total_time_steps_per_case


    with tqdm(total=total_combinations, desc="Calculating Profiles") as pbar:
        for G_amp in G_amplitude_values:
            for omega in omega_values:
                period = 2 * np.pi / omega
                # Time steps in the steady-oscillating state
                # Start time doesn't strictly matter for the analytical solution,
                # but sampling over a period covers the cycle.
                start_time = 0.0 # Start from t=0 for simplicity, as sol is for steady state
                end_time = start_time + num_periods_to_sample * period
                time_steps_for_case = np.linspace(start_time, end_time, total_time_steps_per_case, endpoint=False)

                for nu in nu_values:
                    for a in pipe_radius_values:
                        for t_val in time_steps_for_case:
                            profile_data = calculate_pipe_pg_profile(
                                G_amp, omega, nu, rho, a, t_val,
                                r_points_per_radius, yplus_threshold
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
    output_filename = os.path.join(output_data_dir, f"PipePG_data.h5")
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
