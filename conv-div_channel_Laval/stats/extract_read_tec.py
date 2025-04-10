import tecplot as tp
from tecplot.constant import * # Imports constants like ReadDataOption, SaveAsFileFormat, etc.
import numpy as np # Useful for data manipulation
import os
# Define the input filename
input_filename = './conv-div-mean.dat' # Or .plt, .szplt

# Connect to Tecplot session
tp.session.connect()
tp.new_layout()  # Clears the current layout

print(f"Loading data from: {input_filename}")
try:
    # Load the dataset. ReadAll loads all variable data immediately.
    # Use ReadHeaderOnly if you only need metadata initially.
    dataset = tp.data.load_tecplot(input_filename)
    print("Data loaded successfully.")

    # --- Accessing the loaded data (Reading) ---
    # Get the first zone (index 0)
    zone = dataset.zone(0)
    print(f"Dataset contains {dataset.num_zones} zone(s) and {dataset.num_variables} variable(s).")
    print(f"Working with zone '{zone.name}' (index 0).")
    print(f"Zone dimensions (I, J, K): {zone.dimensions}")

    # Get specific variables
    var_x = dataset.variable('X')
    var_y = dataset.variable('Y')
    var_mean_u = dataset.variable('mean_u')
    var_mean_v = dataset.variable('mean_v')

    # Get data as NumPy arrays using the variable object
    x_data = zone.values(var_x)[:]
    y_data = zone.values(var_y)[:]
    mean_u_data = zone.values(var_mean_u)[:]
    mean_v_data = zone.values(var_mean_v)[:]

    # Or get data directly using variable name string
    # mean_u_data = zone.values('mean_u')[:]

    print(f"Read data for X (shape: {x_data.shape}), Y (shape: {y_data.shape}), mean_u (shape: {mean_u_data.shape}), mean_v (shape: {mean_v_data.shape})")
    # Note: The shape from zone.values()[:] is typically 1D (total_points,)
    # You can reshape it if needed:
    total_points = zone.num_points
    i, j, k = zone.dimensions
    mean_u = mean_u_data.reshape((k, j, i), order='F') # Reshape to (K, J, I)
    mean_v = mean_v_data.reshape((k, j, i), order='F') # Reshape to (K, J, I)
    X      = x_data.reshape((k, j, i), order='F')
    Y      = y_data.reshape((k, j, i), order='F')


except Exception as e:
    print(f"Error loading data: {e}")
    exit() # Exit if loading failed

# --- Save Combined Data to PKL File ---
import pickle as pkl
output_pkl_filename = 'flow_data.pkl'
# Create a dictionary to hold the reshaped arrays
data_to_save = {
    'X': X.squeeze(),  # Remove single-dimensional entries
    'Y': Y.squeeze(),
    'U': mean_u.squeeze(),
    'V': mean_v.squeeze()
}

# Save the dictionary to a pickle file
with open(output_pkl_filename, 'wb') as pkl_file:
    # Use pickle.dump to serialize the dictionary into the file
    # HIGHEST_PROTOCOL is generally recommended for efficiency
    pkl.dump(data_to_save, pkl_file, protocol=pkl.HIGHEST_PROTOCOL)
