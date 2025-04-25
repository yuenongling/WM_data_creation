'''
Special script to interpolate the pressure close to the wall

Finished 2800

10595, 19000, 37000 unstable; need to be fixed
'''

import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator
import os
import time
import csv

def calculate_wall_normals(x_wall, y_wall):
    """Calculates unit normal vectors pointing away from the lower wall."""
    # Calculate slope using central differences (gradient)
    # np.gradient calculates dy/dx using central differences where possible
    slopes = np.gradient(y_wall, x_wall)
    
    # Tangent vector direction: (1, slope)
    # Normal vector direction: (-slope, 1) (points generally upwards for positive slope)
    # Normalize the normal vectors
    norm_magnitude = np.sqrt(slopes**2 + 1)
    
    # Avoid division by zero if norm_magnitude is somehow zero (flat horizontal wall)
    norm_magnitude[norm_magnitude == 0] = 1 
    
    normal_vec_x = -slopes / norm_magnitude
    normal_vec_y = 1 / norm_magnitude
    
    # Stack into an array of shape (N_wall_points, 2)
    normal_vectors = np.column_stack((normal_vec_x, normal_vec_y))
    return slopes, normal_vectors

def generate_normal_line_points(x_start, y_start, normal_vector, max_dist=0.5, num_points=100, stretch_factor=1.5):
    """Generates points along the wall-normal line."""
    # Generate distances along the normal, denser near the wall
    t = np.linspace(0, 1, num_points)**stretch_factor
    s_distances = t * max_dist # Actual distances from the wall
    
    # Calculate points: start_point + distance * normal_vector
    x_normal = x_start + s_distances * normal_vector[0]
    y_normal = y_start + s_distances * normal_vector[1]
    
    return x_normal, y_normal, s_distances

def create_interpolator(grid_points, data_values):
    """
    Creates an interpolator for the given data.
    Uses griddata for potentially unstructured/curvilinear input points.
    """
    print("Creating griddata interpolator...")
    # griddata is suitable for scattered or curvilinear data points
    # grid_points should be shape (N, 2) -> [(x1,y1), (x2,y2), ...]
    # data_values should be shape (N,)
    start_time = time.time()
    interpolator = lambda pts: griddata(grid_points, data_values, pts, method='linear', fill_value=np.nan)
    end_time = time.time()
    print(f"Interpolator created in {end_time-start_time:.2f} s.")
    return interpolator

def interpolate_data_on_line(interpolator, x_line, y_line):
    """Interpolates data onto the points defined by x_line, y_line."""
    line_points = np.column_stack((x_line, y_line))
    interpolated_values = interpolator(line_points)
    return interpolated_values

# --- Configuration ---

Re_all = [ 10595]
plot_to_validate = int(sys.argv[1]) if len(sys.argv) > 1 else 0

for Re in Re_all:

    pkl_filename = f'./stats/2Dfields_Re{Re}.pkl' 
    wall_data_filename = './stats/hill_grid.csv'
    output_profiles_filename = f'./stats/Cp_Re{Re}.dat'
    output_dir = "./stats/"

# Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

# --- Load Data from Pickle File ---
    print(f"Loading reshaped data from: {pkl_filename}")
    with open(pkl_filename, 'rb') as pkl_file:
        loaded_data = pickle.load(pkl_file)

    x_coords_2d = loaded_data['X']      # Shape (J, I) -> (2304, 385)
    y_coords_2d = loaded_data['Y']      # Shape (J, I)
    mean_p_2d = loaded_data['P'] # Shape (J, I)

    print(f"Loaded X shape: {x_coords_2d.shape}")
    print(f"Loaded Y shape: {y_coords_2d.shape}")
    print(f"Loaded mean_p shape: {mean_p_2d.shape}")

# --- Load Wall Geometry Data ---
    print(f"Loading wall geometry from: {wall_data_filename}")
# Use numpy.loadtxt, skipping comments (#) and reading only columns 0 (X) and 1 (Y0)
    wall_data = np.loadtxt(wall_data_filename, delimiter=',', skiprows=1, usecols=(0, 1))
    x_wall = wall_data[:, 0] # Streamwise coordinates of wall points
    y_wall = wall_data[:, 1] # Corresponding Y coordinates of the lower wall
    print(f"Loaded wall data shape: {wall_data.shape}")

# Flatten coordinate and velocity data for interpolation if using griddata
    points = np.column_stack((x_coords_2d.flatten(), y_coords_2d.flatten()))

# --- Calculate Wall Normals ---
    print("Calculating wall slopes and normal vectors...")
    wall_slopes, wall_normals = calculate_wall_normals(x_wall, y_wall)

# Sanity check: Plot wall normals
    if plot_to_validate:
        plt.plot(x_wall, y_wall)
        plt.axis('equal')
        for i in [0, 20,30,40,50, 100, 130, 190]:
            vec_x = np.linspace(0, 1, 10) * wall_normals[i,0]
            vec_y = np.linspace(0, 1, 10) * wall_normals[i,1]
            plt.plot(x_wall[i]+vec_x, y_wall[i]+vec_y, 'r-')
        plt.show()


# --- Create Interpolator for Mean U ---
# Create the interpolator once, as the underlying grid doesn't change
    mean_p_interpolator = create_interpolator(points, mean_p_2d)

# --- Select Streamwise Locations for Profiles ---
# Example: Select N locations evenly spaced along the wall data, avoiding edges
    num_profiles = 180
# indices = np.linspace(5, len(x_wall) - 6, num_profiles, dtype=int) # Avoid very start/end
    indices = np.round(np.linspace(0, len(x_wall) - 1, num_profiles)).astype(int) # Include start/end maybe

# x_wall_to_investigate = [0, 1, 2, 4, 8]
# indices = np.array([np.argmin(np.abs(x_wall - x)) for x in x_wall_to_investigate])
    selected_x_wall = x_wall[indices]
    p_all = np.zeros_like(selected_x_wall)

    print(f"\nSelected streamwise locations (X) for profiles: {selected_x_wall}")

# --- Generate and Interpolate Profiles ---
    velocity_profiles = {} # Dictionary to store results {x_location: (s_distances, u_profile)}
    num_norm_points = 1 # Number of points along each normal line
    max_normal_dist = 0.001 # Max distance to extend normal line (adjust based on channel height, Y1=2)

    print(f"Generating {num_profiles} profiles with {num_norm_points} points up to s={max_normal_dist}...")

    

    total_interp_time = 0
    for i, idx in enumerate(indices):
        x_loc = x_wall[idx]
        y_loc = y_wall[idx]
        normal_vec = wall_normals[idx]
        
        print(f"Processing profile at X = {x_loc:.4f} (Index {idx})...")
        
        # Generate points along the normal line starting from (x_loc, y_loc)
        x_norm, y_norm, s_dist = generate_normal_line_points(
            x_loc, y_loc, normal_vec, 
            max_dist=max_normal_dist, 
            num_points=num_norm_points
        )
        
        # Interpolate mean_u onto these points
        interp_start_time = time.time()
        p_profile = interpolate_data_on_line(mean_p_interpolator, x_norm, y_norm)
        interp_end_time = time.time()
        total_interp_time += (interp_end_time - interp_start_time)

        p_all[i] = p_profile
        
        # Store the results
        # velocity_profiles[x_loc] = {'s': s_dist, 'x_norm': x_norm, 'y_norm': y_norm, 'x_loc': x_loc, 'y_loc': y_loc, 'p': p_profile}
        print(f"  Interpolation done in {interp_end_time - interp_start_time:.3f} s.")

    print(f"\nFinished generating profiles. Total interpolation time: {total_interp_time:.2f} s.")

    if plot_to_validate:
        plt.plot(selected_x_wall, p_all, 'o-')
        plt.show()

# --- Save Profiles ---
    data = np.vstack((selected_x_wall, p_all))

    with open(output_profiles_filename, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['X', 'P'])
        for i in range(len(data[0])):
            spamwriter.writerow([data[0][i], data[1][i]])

    # print(f"Saving calculated profiles to: {output_profiles_filename}")
    # with open(output_profiles_filename, "w") as file:
    #     file.write(" ".join(data))

# # --- Plotting Example (Optional) ---
    if plot_to_validate:

        profile_keys = list(velocity_profiles.keys())
        for i, (x_loc, profile_data) in enumerate(velocity_profiles.items()):
            plt.plot( profile_data['u'], profile_data['s'] + profile_data['y_loc'],'o-')
            plt.title('x = {:.3f}'.format(x_loc))
            plt.xlabel('Mean U Velocity')
            plt.ylabel('y')
            plt.xlim(-0.25, 1.2)
            plt.ylim(0.0, 3.05)
            plt.show()
