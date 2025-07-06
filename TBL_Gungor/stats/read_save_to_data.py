import pandas as pd
import re # For more robust parsing of key-value pairs
import os
import numpy as np
import h5py

def parse_dns_file(filepath):
    """
    Parses a single DNS16_position file.

    Args:
        filepath (str): The path to the file.

    Returns:
        dict: A dictionary containing 'header_data' (a dict) and 
              'table_data' (a pandas DataFrame). Returns None if file not found.
    """
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return None

    header_data = {}
    table_lines = []
    column_names = None
    reading_table_data = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:  # Skip empty lines
                continue

            if line.startswith('%'):
                # Try to extract key-value pairs from comment lines
                # Example: % Re_tau = 4.435745e+02
                match_kv = re.match(r"%\s*([^=]+?)\s*=\s*([0-9eE.+-]+)", line)
                if match_kv:
                    key = match_kv.group(1).strip()
                    try:
                        value = float(match_kv.group(2).strip())
                        header_data[key] = value
                    except ValueError:
                        # If conversion to float fails, store as string or ignore
                        # print(f"Could not convert value for key '{key}' to float: {match_kv.group(2).strip()}")
                        header_data[key] = match_kv.group(2).strip() # Store as string
                    continue # Move to next line after processing key-value

                # Check for the data table header line
                # Example: % U V u2 v2 w2 uv y
                # A simple heuristic: starts with %, contains 'U', 'V', and 'y' (common columns)
                # and does not contain "===" or "***" (decorative lines)
                if "U" in line and "V" in line and "y" in line and \
                   "===" not in line and "***" not in line:
                    # Extract column names, removing the leading '%' and stripping whitespace
                    potential_cols = [col.strip() for col in line.lstrip('%').split()]
                    # Check if these look like valid column names (alphanumeric, underscores)
                    if all(re.match(r"^[A-Za-z0-9_]+$", col) for col in potential_cols):
                        column_names = potential_cols
                        reading_table_data = True # Next non-comment lines are data
                        # print(f"Found column names: {column_names}")
                    continue # Move to next line

                # Other comment lines are ignored for now
                # print(f"Ignoring comment: {line}")

            elif reading_table_data and column_names:
                # This should be a data line
                try:
                    # Split by whitespace and convert to float
                    data_values = [float(x) for x in line.split()]
                    if len(data_values) == len(column_names):
                        table_lines.append(data_values)
                    else:
                        print(f"Warning: Mismatch in data columns for line in {filepath}: {line}")
                        print(f"Expected {len(column_names)} columns, got {len(data_values)}")
                except ValueError:
                    print(f"Warning: Could not parse data line in {filepath}: {line}")
            
    # Create DataFrame from the collected table lines
    table_df = pd.DataFrame()
    if table_lines and column_names:
        table_df = pd.DataFrame(table_lines, columns=column_names)
    elif reading_table_data and not column_names:
        print(f"Warning: Started reading table data but no column names found for {filepath}.")
    elif reading_table_data and not table_lines:
        print(f"Warning: Found column names but no data lines for {filepath}.")


    return {
        'filepath': filepath,
        'header_data': header_data,
        'table_data': table_df
    }

# --- Main script execution ---
all_dns_data = []
num_files = 8

year = [16, 22, 23]

for y in year:
    output_file = f"dns_data_20{y}.h5"

    print(f"Processing {num_files} DNS files...")
    for i in range(1, num_files + 1):
        filename = f"./databases/DNS20{y}/DNS{y}_position_{i}" # Assuming no file extension like .txt or .dat
        parsed_data = parse_dns_file(filename)
        if parsed_data:
            all_dns_data.append(parsed_data)
            print(f"Successfully processed: {filename}")
        else:
            print(f"Failed to process or skipped: {filename}")

    print("\n--- Data parsing complete ---")

# Now you can access the data for each file:
    if all_dns_data:
        for i, data_dict in enumerate(all_dns_data):
            print(f"\n--- Data for: {data_dict['filepath']} ---")
            
            print("Header Data:")
            for key, value in data_dict['header_data'].items():
                print(f"  {key}: {value}")
            
            print("\nTable Data (first 5 rows):")
            if not data_dict['table_data'].empty:
                print(data_dict['table_data'].head())
            else:
                print("  Table data is empty.")

        # Example: Accessing specific data
        # Get Re_tau from the first file
        if all_dns_data[0]['header_data']:
                print(f"\nRe_tau for {all_dns_data[0]['filepath']}: {all_dns_data[0]['header_data'].get('Re_tau')}")

        # Get the 'U' column from the table data of the second file (if it exists)
        if len(all_dns_data) > 1 and not all_dns_data[1]['table_data'].empty:
            print(f"\n'U' column for {all_dns_data[1]['filepath']}:\n{all_dns_data[1]['table_data']['U'].head()}")

        # --- Prepare data for saving to NPZ ---
        print(f"\nPreparing data for saving to {output_file}...")

        with pd.HDFStore(output_file, mode='w') as store: # mode='w' overwrites
            for i, data_entry in enumerate(all_dns_data):
                filepath = data_entry['filepath']
                header_dict = data_entry['header_data']
                table_df = data_entry['table_data']

                # Derive a unique key for HDF5 store
                filename_base = os.path.basename(filepath)
                match = re.search(r'position_(\d+)', filename_base, re.IGNORECASE)
                if match:
                    position_id = match.group(1)
                    group_key_base = f"position_{position_id}"
                else:
                    group_key_base = re.sub(r'[^a-zA-Z0-9_]', '_', filename_base)

                # Store DataFrame
                if not table_df.empty:
                    store.put(f"{group_key_base}/table_data", table_df, format='table', data_columns=True)
                
                # Store header_data as metadata/attributes on the group or as a separate series/df
                # Option 1: Store header as attributes (simpler for small dicts)
                if header_dict:
                    # HDFStore can store metadata with a DataFrame.
                    # We can create a small Series/DataFrame for headers or attach to the table_data if it exists
                    if not table_df.empty:
                        # This attaches metadata to the '/table_data' node
                        store.get_storer(f"{group_key_base}/table_data").attrs.metadata = header_dict
                    else: # If no table_data, store header as a separate Series
                        header_series = pd.Series(header_dict)
                        store.put(f"{group_key_base}/header_data", header_series)
                
                # Store filepath as an attribute (example)
                if not table_df.empty: # Attach to table if it exists
                    store.get_storer(f"{group_key_base}/table_data").attrs.filepath_str = str(filepath)
                elif header_dict: # Attach to header series if it exists
                    store.get_storer(f"{group_key_base}/header_data").attrs.filepath_str = str(filepath)
                # Or store filepath as a separate tiny dataset/series

        print(f"Data saved to {output_file}")

        # --- Example Loading HDF5 ---
        reconstructed_data_hdf = {}
        with pd.HDFStore(output_file, mode='r') as store:
            for key in store.keys(): # Keys are like '/position_1/table_data'
                print(f"Loading key: {key}")
                if key.endswith('/table_data'):
                    group_prefix = key.rsplit('/', 1)[0] # e.g., /position_1
                    df = store[key]
                    headers = store.get_storer(key).attrs.metadata if 'metadata' in store.get_storer(key).attrs else {}
                    fpath = store.get_storer(key).attrs.filepath_str if 'filepath_str' in store.get_storer(key).attrs else "Unknown"
                    # Store it...
                elif key.endswith('/header_data'): # if header was stored separately
                    pass # handle this case

    else:
        print("No data was successfully processed.")
