import tecplot as tp
from tecplot.constant import * # Imports constants like ReadDataOption, SaveAsFileFormat, etc.
import numpy as np # Useful for data manipulation
import os
import re
# Define the input filename

# Connect to Tecplot session
tp.session.connect()
tp.new_layout()  # Clears the current layout

all_files = ['./WallHump-WideSpan-OriginalTopWallContour.dat']
for input_filename in all_files:
    print(f"Loading data from: {input_filename}")
    # Load the dataset. ReadAll loads all variable data immediately.
    # Use ReadHeaderOnly if you only need metadata initially.
    dataset = tp.data.load_tecplot(input_filename)
    print("Data loaded successfully.")

    # --- Accessing the loaded data (Reading) ---
    # Get the first zone (index 0)
    zone = dataset.zone(0)
    print(f"Dataset contains {dataset.num_zones} zone(s) and {dataset.num_variables} variable(s).")

    x_all = []
    y_all = []
    mean_u_all = []
    mean_v_all = []
    mean_p_all = []

    # Iterate over zones and concatenate

    for i in range(dataset.num_zones):

        zone = dataset.zone(i)

        print(f"Working with zone '{zone.name}' (index 0).")
        # print(f"Zone dimensions (I, J, K): {zone.dimensions}")

        # Get specific variables
        var_x = dataset.variable('x/c')
        var_y = dataset.variable('y/c')
        var_mean_u = dataset.variable('U/Uinf')
        var_mean_v = dataset.variable('V/Uinf')
        var_mean_p = dataset.variable('p/pinf')

        # Get data as NumPy arrays using the variable object
        x_data = zone.values(var_x)[:]
        y_data = zone.values(var_y)[:]
        mean_u_data = zone.values(var_mean_u)[:]
        mean_v_data = zone.values(var_mean_v)[:]
        mean_p_data = zone.values(var_mean_p)[:]

        # x_data = x_data.reshape((jdim, idim))
        # y_data = y_data.reshape((jdim, idim))
        # mean_u_data = mean_u_data.reshape((jdim, idim))
        # mean_v_data = mean_v_data.reshape((jdim, idim))

        x_all = np.concatenate((x_all, x_data))
        y_all = np.concatenate((y_all, y_data))
        mean_u_all = np.concatenate((mean_u_all, mean_u_data))
        mean_v_all = np.concatenate((mean_v_all, mean_v_data))
        mean_p_all = np.concatenate((mean_p_all, mean_p_data))

# --- Save Combined Data to PKL File ---
    import pickle as pkl
    output_pkl_filename = re.sub(r'\.dat$', '.pkl', input_filename)  # Change .dat to .pkl
# Create a dictionary to hold the reshaped arrays
    data_to_save = {
        'X': x_all.squeeze(),  # Remove single-dimensional entries
        'Y': y_all.squeeze(),
        'U': mean_u_all.squeeze(),
        'V': mean_v_all.squeeze(),
        'P': mean_p_all.squeeze()
    }

# Save the dictionary to a pickle file
    with open(output_pkl_filename, 'wb') as pkl_file:
        # Use pickle.dump to serialize the dictionary into the file
        # HIGHEST_PROTOCOL is generally recommended for efficiency
        pkl.dump(data_to_save, pkl_file, protocol=pkl.HIGHEST_PROTOCOL)
