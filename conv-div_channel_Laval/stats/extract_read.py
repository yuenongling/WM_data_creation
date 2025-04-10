import numpy as np
import re
import time # Optional: for timing the read process

def read_tecplot_ascii_selective(filename, target_variables=["X", "Y", "mean_u"]):
    """
    Reads a Tecplot ASCII BLOCK format file, extracting only specified variables.

    Args:
        filename (str): Path to the Tecplot ASCII file.
        target_variables (list): A list of variable names (strings) to extract.

    Returns:
        tuple: (title, extracted_variables, zone_info, data)
               - title (str): The title from the Tecplot file.
               - extracted_variables (list): The list of variables actually extracted.
               - zone_info (dict): Information about the zone (name, I, J, K, etc.).
               - data (dict): A dictionary where keys are the extracted variable names
                              and values are NumPy arrays of the data, reshaped.
                              Shape is typically (K, J, I).
    """
    print(f"Reading file: {filename}")
    print(f"Target variables: {target_variables}")
    start_time = time.time() # Optional: start timer

    with open(filename, 'r') as file:
        # --- Read Header ---
        # Read TITLE
        title_line = file.readline()
        title = title_line.split('=', 1)[1].strip().strip('"') if '=' in title_line else "No Title Found"

        # Read VARIABLES
        all_variables = []
        line = file.readline().strip()
        # Handle VARIABLES = "Var1" "Var2" ... format
        if line.upper().startswith('VARIABLES'):
            # Use regex to find all quoted variable names
            var_names = re.findall(r'"([^"]*)"', line.split('=', 1)[1])
            all_variables.extend(var_names)
            # Read subsequent lines if variables span multiple lines
            while True:
                current_pos = file.tell() # Remember position in case the next line isn't a variable
                line = file.readline().strip()
                if line.startswith('"'):
                     # Check if the entire line is just quoted strings (potential variable names)
                    more_vars = re.findall(r'"([^"]*)"', line)
                    if len(more_vars) > 0 and "".join(more_vars) == line.replace('"', ''):
                        all_variables.extend(more_vars)
                    else:
                         # Not just variable names, rewind and break
                         file.seek(current_pos)
                         break
                else:
                    # Line doesn't start with quote, assume end of variables list
                    file.seek(current_pos) # Rewind to process this line as ZONE info
                    break
        else:
            raise ValueError("Could not find VARIABLES line in Tecplot header.")

        print(f"Found variables in file: {all_variables}")

        # --- Read Zone Information ---
        zone_info = {}
        # The line we potentially rewound to should contain ZONE info or be the start of data
        line = file.readline().strip()
        while True:
            if line.upper().startswith('ZONE'):
                 # Check for T="Zone Name" format
                 match = re.search(r'T="([^"]*)"', line, re.IGNORECASE)
                 if match:
                     zone_info['name'] = match.group(1)
                 else: # Fallback if no T= part
                     zone_info['name'] = line.split('=', 1)[1].strip().strip('"') if '=' in line else "Unnamed Zone"

                 # Parse I, J, K potentially on the same ZONE line or next line
                 ijk_info = re.findall(r'([IJK])=(\d+)', line, re.IGNORECASE)
                 for key, value in ijk_info:
                     zone_info[key.upper()] = int(value)

            # Look for I, J, K on subsequent lines if not found yet
            elif 'I=' in line.upper() or 'J=' in line.upper() or 'K=' in line.upper():
                 ijk_info = re.findall(r'([IJK])=(\d+)', line, re.IGNORECASE)
                 for key, value in ijk_info:
                     zone_info[key.upper()] = int(value)
                 # Check for ZONETYPE, DATAPACKING etc. on the same line
                 other_params = re.findall(r'(\w+)=([\w\(\)]+)', line, re.IGNORECASE)
                 for key, value in other_params:
                      key_upper = key.upper()
                      if key_upper not in ['I', 'J', 'K']:
                         zone_info[key_upper] = value.strip().strip('"')

            elif '=' in line: # Handle other key=value pairs like DATAPACKING, ZONETYPE
                 parts = line.split('=', 1)
                 if len(parts) == 2:
                     key, value = parts
                     key_upper = key.strip().upper()
                     # Avoid overwriting I, J, K if they were parsed already
                     if key_upper not in ['I', 'J', 'K']:
                        zone_info[key_upper] = value.strip().strip('"')
            elif line and not line.startswith('DT='): # Assume end of header if line has content but isn't DT=
                # This line is likely the start of data, rewind to read it as data
                file.seek(file.tell() - len(line) - len(file.newlines[0])) # Adjust for line content + newline
                break
            elif not line: # End of file reached unexpectedly
                 raise EOFError("Reached end of file while reading ZONE header.")

            # Check if required info is present before reading next line
            if 'I' in zone_info and 'J' in zone_info and 'K' in zone_info and 'DATAPACKING' in zone_info:
                 # Check if the *next* line starts with numbers (likely data start)
                 current_pos = file.tell()
                 next_line_peek = file.readline().strip()
                 file.seek(current_pos) # Go back
                 # A simple check: does it start with a digit, minus, or plus?
                 if next_line_peek and (next_line_peek[0].isdigit() or next_line_peek[0] in ['-', '+', '.']):
                     line = next_line_peek # Consume the peeked line
                     break # Assume header finished, next lines are data

            line = file.readline().strip() # Read next line of header/zone info

        # Verify essential zone info
        if 'I' not in zone_info or 'J' not in zone_info or 'K' not in zone_info:
             raise ValueError("Could not determine I, J, K dimensions from ZONE info.")
        if zone_info.get('DATAPACKING', '').upper() != 'BLOCK':
             raise ValueError(f"Expected DATAPACKING=BLOCK, found {zone_info.get('DATAPACKING')}")

        i_dim = zone_info['I']
        j_dim = zone_info['J']
        k_dim = zone_info['K']
        total_points = i_dim * j_dim * k_dim
        print(f"Zone Info: I={i_dim}, J={j_dim}, K={k_dim} (Total Points per Var: {total_points})")

        # --- Read Data (BLOCK format) ---
        data = {}
        extracted_variables = []

        # Pre-allocate numpy arrays only for target variables for efficiency
        for var in target_variables:
            if var in all_variables:
                data[var] = np.zeros(total_points, dtype=np.float32) # Assuming SINGLE precision from DT=()
                extracted_variables.append(var)
            else:
                print(f"Warning: Target variable '{var}' not found in file header variables.")

        # Determine indices of target variables in the file's variable order
        target_indices = {all_variables.index(var) for var in extracted_variables}


        points_read_total = 0
        current_var_index = 0
        points_read_for_current_var = 0

        # Start reading from the position where the header parsing left off
        # The 'line' variable should hold the first line of data (or be empty if file ended)
        while current_var_index < len(all_variables):
            is_target = current_var_index in target_indices
            current_variable_name = all_variables[current_var_index]

            while points_read_for_current_var < total_points:
                if not line: # Check if we already hit EOF
                    raise EOFError(f"Reached end of file while reading data for variable '{current_variable_name}' (index {current_var_index}). Expected {total_points} points, read {points_read_for_current_var}.")

                values_str = line.split()
                num_values_on_line = len(values_str)

                # Check for potential over-read (more values than needed for current var)
                if points_read_for_current_var + num_values_on_line > total_points:
                   num_to_take = total_points - points_read_for_current_var
                   values_to_process = values_str[:num_to_take]
                   # Store remaining values for the next variable block? Tecplot ASCII usually doesn't split vars mid-line
                   # If this happens, the format might be unusual or corrupted. For now, assume full lines per var block.
                   print(f"Warning: Line may contain data for next variable block. Processing only first {num_to_take} values for {current_variable_name}")
                else:
                    values_to_process = values_str
                    num_to_take = num_values_on_line

                # Store data only if it's a target variable
                if is_target:
                    try:
                         # Efficiently convert and store the slice
                         data_slice = np.array([float(v) for v in values_to_process], dtype=np.float32)
                         start_idx = points_read_for_current_var
                         end_idx = start_idx + num_to_take
                         data[current_variable_name][start_idx:end_idx] = data_slice
                    except ValueError as e:
                        print(f"Error converting value to float on line: '{line}'")
                        print(f"Problematic values: {values_to_process}")
                        raise e
                    except IndexError as e:
                         print(f"Index error storing data for {current_variable_name}. Start: {start_idx}, End: {end_idx}, Array size: {data[current_variable_name].size}")
                         raise e


                points_read_for_current_var += num_to_take
                points_read_total += num_to_take

                # Read next line only if we haven't finished the current variable block
                if points_read_for_current_var < total_points:
                    line = file.readline().strip()


            # Finished reading data for the current variable block
            print(f"Finished reading variable: {current_variable_name} (Index {current_var_index})")
            current_var_index += 1
            points_read_for_current_var = 0 # Reset for the next variable

            # If the last read consumed only part of the line (unlikely but possible)
            # The remaining part of 'line' needs processing for the *new* current_var_index
            # However, standard Tecplot ASCII BLOCK usually aligns data blocks with line breaks.
            # If not, the logic needs adjustment to handle partial lines across variable blocks.
            # For simplicity, we assume the next variable starts on a new line read.
            if current_var_index < len(all_variables):
                 line = file.readline().strip() # Read the first line for the *next* variable


    # --- Reshape Data ---
    # Reshape the extracted data using Fortran order (I varies fastest)
    for var in extracted_variables:
        try:
             # Reshape to (K, J, I) which is common, adjust if needed
            data[var] = data[var].reshape((k_dim, j_dim, i_dim), order='F')
        except ValueError as e:
            print(f"Error reshaping variable '{var}'. Expected {total_points} elements, found {data[var].size}. Dimension mismatch? (K={k_dim}, J={j_dim}, I={i_dim})")
            raise e

    end_time = time.time() # Optional: end timer
    print(f"Finished reading. Time taken: {end_time - start_time:.2f} seconds")

    return title, extracted_variables, zone_info, data

# Create a dummy Tecplot file matching the user's header structure
# Make sure this file exists or change the path
filename = './conv-div-mean.dat'

# # Use the dimensions and variables from the user's example
# i_dim, j_dim, k_dim = 385, 2304, 1
# total_points = i_dim * j_dim * k_dim
# all_vars = [
#     "X", "Y", "mean_u", "mean_v", "mean_w", "dx_mean_u", "dx_mean_v",
#     "dx_mean_w", "dy_mean_u", "dy_mean_v", "dy_mean_w", "dz_mean_u",
#     "dz_mean_v", "dz_mean_w", "reynolds_stress_uu", "reynolds_stress_uv",
#     "reynolds_stress_uw", "reynolds_stress_vv", "reynolds_stress_vw",
#     "reynolds_stress_ww"
# ]
#
# Now read the dummy file selectively
target_vars_to_read = ["X", "Y", "mean_u"]
title, extracted_vars, zone_info, extracted_data = read_tecplot_ascii_selective(filename, target_variables=target_vars_to_read)

X = extracted_data["X"]
Y = extracted_data["Y"]
mean_u = extracted_data["mean_u"]

import matplotlib.pyplot as plt
sc=plt.contourf(X[0], Y[0], mean_u[0],levels=100)
plt.colorbar(sc)
plt.title("Mean Velocity")
plt.show()
