import re
import numpy as np
import io # To treat the string as a file

def parse_tecplot_hill_data(tecplot_string):
    """
    Parses Tecplot ASCII data in BLOCK format to extract grid dimensions
    and the coordinates of the first J-line (assumed hill profile).

    Args:
        tecplot_string (str): A string containing the Tecplot data.

    Returns:
        tuple: A tuple containing:
            - x_coords (np.ndarray): 1D array of X-coordinates for the first J-line.
            - y_coords (np.ndarray): 1D array of Y-coordinates for the first J-line.
            - I (int): Number of points in the I direction.
            - J (int): Number of points in the J direction.
        Returns None if parsing fails.
    """
    I, J = None, None
    variables = []
    data_lines = []
    in_data_block = False

    # Use StringIO to treat the input string like a file
    f = io.StringIO(tecplot_string)

    I = 197
    J = 129
    variables = ["X", "Y"]
    # --- Read Header ---
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): # Skip empty lines and comments
             continue

        low_line = line.lower()

        if low_line.startswith('variables'):
            try:
                # Extract variable names, remove quotes
                var_str = low_line.split('=')[1]
                variables = [v.strip().replace('"', '') for v in var_str.split(',')]
            except IndexError:
                print("Error parsing VARIABLES line.")
                return None
            continue

        # Check for the start of the data block (often after DT=...)
        # A simple check: if I and J are found and the line contains numbers
        if I is not None and J is not None:
             # Check if the line looks like data (contains numbers, possibly E notation)
             if re.search(r'[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?', line):
                 in_data_block = True
                 # Don't continue, this line is the first data line

        if in_data_block:
             data_lines.append(line)

    # --- Check if essential info was found ---
    if I is None or J is None:
         print("Error: Failed to find ZONE line with I and J dimensions.")
         return None
    if not variables:
         print("Error: Failed to find VARIABLES line.")
         return None
    if not data_lines:
         print("Error: No data block found.")
         return None

    print(f"Found Grid Dimensions: I={I}, J={J}")
    print(f"Found Variables: {variables}")

    # --- Parse Data Block ---
    num_points_per_var = I * J
    num_vars_expected = len(variables) # Using the count from the VARIABLES line

    print(f"Expecting {num_points_per_var} points per variable.")

    # Concatenate all data lines and split into numbers
    all_data_str = " ".join(data_lines)
    all_numbers_str = all_data_str.split()

    try:
        # Convert all number strings to floats
        all_numbers = np.array([float(n) for n in all_numbers_str])
    except ValueError:
        print("Error: Non-numeric value found in data block.")
        return None

    total_expected_numbers = num_points_per_var * num_vars_expected
    if len(all_numbers) < total_expected_numbers:
        print(f"Error: Not enough data points found. Expected {total_expected_numbers}, got {len(all_numbers)}.")
        # Check if DT line might have been included - a common issue if check isn't robust
        print("Check if the 'DT=(...)' line was accidentally included in the data.")
        return None
    elif len(all_numbers) > total_expected_numbers:
        print(f"Warning: More data points found than expected ({len(all_numbers)} vs {total_expected_numbers}). Truncating.")
        all_numbers = all_numbers[:total_expected_numbers]

    # --- Extract X and Y data ---
    # Assuming x is the first variable, y is the second
    try:
         x_data_flat = all_numbers[0 : num_points_per_var]
         y_data_flat = all_numbers[num_points_per_var : 2 * num_points_per_var]
    except IndexError:
         print("Error slicing data for X and Y coordinates.")
         return None

    # Reshape data: Tecplot BLOCK format varies I fastest, then J.
    # Reshaping to (J, I) in C order places elements correctly for j=0 indexing.
    try:
        # x_grid = x_data_flat.reshape((J, I)) # C order (row-major)
        # y_grid = y_data_flat.reshape((J, I))
        # Hill profile corresponds to the first J-line (index 0)
        # x_hill = x_grid[0, :]
        # y_hill = y_grid[0, :]

        # Alternative using Fortran order reshape (column-major)
        x_grid_f = x_data_flat.reshape((I, J), order='F')
        y_grid_f = y_data_flat.reshape((I, J), order='F')
        # Hill profile corresponds to the first J-line (index 0)
        x_hill = x_grid_f[:, 0]
        y_hill = y_grid_f[:, 0]

    except ValueError as e:
        print(f"Error reshaping data arrays: {e}")
        return None

    print(f"Successfully extracted hill profile with {len(x_hill)} points.")

    return x_hill, y_hill, I, J


with open("./hill_grid.dat", 'r') as f:
    full_file_content = f.read()
x_hill, y_hill, I, J = parse_tecplot_hill_data(full_file_content)

with open("./hill_grid.csv", 'w') as f:
    f.write("X,Y\n")
    for x, y in zip(x_hill, y_hill):
        f.write(f"{x},{y}\n")

