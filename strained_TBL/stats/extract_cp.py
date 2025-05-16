import pandas as pd
import pickle
import io
import re

def parse_dat_file_content_pstat(file_content):
    """
    Parses the DAT file content, extracts metadata and tabular data,
    converts x and z to integers, and organizes Pstat by Z-station.
    """
    lines = file_content.strip().split('\n')
    
    metadata = {}
    description_lines = []
    
    # Regex patterns for metadata extraction
    meta_patterns = {
        "filename": r"Filename:\s*(.*)",
        "dynamic_pressure": r"Dinamic pressure\s+([\d.]+)",
        "acquisition_time": r"Acquisition time\s+([\d.]+)",
        "acquisition_frequency": r"Acquisition frequency\s+([\d.]+)",
        "number_of_samples": r"Number of samples\s+(\d+)",
        "barometric_pressure": r"Barometric pressure\s+([\d.]+)",
        "air_temperature_start": r"Air temperature \(start\)\s+([\d.]+)",
        "re_1m_10e6": r"Re 1m /10\.E6\s+([\d.]+)",
        "air_density_start": r"Air density \(start\)\s+([\d.]+)",
        "air_kin_viscosity_start": r"Air kin\. viscosity \(start\)\s+([\d.E_+-]+)",
        "date": r"Date\s+(.*)",
        "time": r"Time\s+(.*)"
    }
    
    data_header_marker = "x [mm]    z [mm]    Pstat[Pa]"
    data_section_start_line_index = -1

    for i, line in enumerate(lines):
        if data_header_marker in line:
            data_section_start_line_index = i + 1
            break
        
        stripped_line = line.strip()
        if not stripped_line:
            continue

        matched_meta = False
        for key, pattern in meta_patterns.items():
            match = re.search(pattern, stripped_line, re.IGNORECASE)
            if match:
                value_str = match.group(1).strip()
                try:
                    if '.' in value_str or 'E' in value_str.upper():
                        metadata[key] = float(value_str)
                    else:
                        metadata[key] = int(value_str)
                except ValueError:
                    metadata[key] = value_str
                matched_meta = True
                break
        
        if not matched_meta:
            description_lines.append(stripped_line)

    if description_lines:
        metadata["description"] = "\n".join(description_lines)

    if data_section_start_line_index == -1:
        raise ValueError("Data header marker ('x [mm]    z [mm]    Pstat[Pa]') not found.")

    actual_data_lines_start = data_section_start_line_index
    while actual_data_lines_start < len(lines) and not lines[actual_data_lines_start].strip():
        actual_data_lines_start += 1
        
    data_block_str = "\n".join(lines[actual_data_lines_start:])
    
    df = pd.DataFrame()
    if data_block_str.strip():
        try:
            df = pd.read_csv(
                io.StringIO(data_block_str),
                delim_whitespace=True,
                names=['x_orig', 'z_orig', 'Pstat[Pa]'], # Use temp names for original float x, z
                header=None,
                skip_blank_lines=True,
                dtype={'Pstat[Pa]': float, 'x_orig': float, 'z_orig': float}
            )
            # Convert x and z to integers
            # Using .round().astype(int) for robustness if there are minor float inaccuracies
            # but for values like 0.000, 25.000, astype(int) directly is fine.
            # If they are truly always like X.000, astype(int) is sufficient.
            df['x [mm]'] = df['x_orig'].astype(int)
            df['z [mm]'] = df['z_orig'].astype(int)
            df = df[['x [mm]', 'z [mm]', 'Pstat[Pa]']] # Keep only the desired columns

        except pd.errors.EmptyDataError:
            print("Warning: No data found after headers.")
        except Exception as e:
            print(f"Error reading data block with pandas: {e}")
            print(f"Data block content (first 500 chars): '{data_block_str[:500]}...'")
            raise
    
    # Organize data by Z-station
    data_by_z_station = {}
    if not df.empty:
        # Ensure 'z [mm]' is integer type for unique station identification
        unique_z_stations = sorted(df['z [mm]'].unique())
        for z_station in unique_z_stations:
            # Select relevant columns for the specific z_station
            station_df = df[df['z [mm]'] == z_station][['x [mm]', 'Pstat[Pa]']].copy()
            station_df.reset_index(drop=True, inplace=True)
            data_by_z_station[z_station] = station_df # z_station is already an integer here
            
    output_data = {
        "metadata": metadata,
        "data_by_z_station": data_by_z_station
    }
    
    return output_data

def save_to_pickle(data_to_save, output_filename="output_data_pstat.pkl"):
    """Saves the data to a pickle file."""
    with open(output_filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"Data saved to {output_filename}")

# --- Main execution ---
with open("./cp_conv.dat", "r") as f:
    file_content = f.read()
parsed_data = parse_dat_file_content_pstat(file_content)

print("--- Parsed Metadata ---")
if parsed_data["metadata"]:
    for key, value in parsed_data["metadata"].items():
        print(f"{key}: {value}")
else:
    print("No metadata extracted.")

print("\n--- Data for sample Z-stations (showing Pstat[Pa]) ---")
if parsed_data["data_by_z_station"]:
    # Print data for the first Z-station found as an example
    first_z_key = next(iter(parsed_data["data_by_z_station"])) # Get first key
    print(f"\nZ-station: {first_z_key} (type: {type(first_z_key)})")
    print(parsed_data["data_by_z_station"][first_z_key].head())
    print("Data types in Z-station DataFrame:")
    print(parsed_data["data_by_z_station"][first_z_key].dtypes)

else:
    print("No data by Z-station found.")

output_filename = "cp_conv.pkl"
save_to_pickle(parsed_data, output_filename)

print(f"\n--- Verifying pickle file: {output_filename} ---")
with open(output_filename, 'rb') as f:
    loaded_data = pickle.load(f)

print("\nLoaded Filename from Pickle:", loaded_data.get("metadata", {}).get("filename", "N/A"))

if loaded_data.get("data_by_z_station"):
    first_z_key_loaded = list(loaded_data["data_by_z_station"].keys())[0]
    print(f"\nData for Z = {first_z_key_loaded} (type: {type(first_z_key_loaded)}) from Pickle (first 5 rows):")
    print(loaded_data["data_by_z_station"][first_z_key_loaded].head())
    print("Data types in loaded Z-station DataFrame:")
    print(loaded_data["data_by_z_station"][first_z_key_loaded].dtypes)

else:
    print("No Z-station data in loaded pickle file.")
