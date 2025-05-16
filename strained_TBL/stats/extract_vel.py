import pandas as pd
import pickle
import io
import re

def parse_p3s_file_content(file_content):
    """
    Parses the P3S DAT file content, extracts main metadata,
    and data for each x-station including its local metadata and velocity/yaw profile.
    x and z coordinates in sub-headers are converted to integers.
    y coordinates in data tables remain float.
    """
    main_metadata = {}
    description_lines = []
    data_by_x_station = {}

    # Regex patterns for main metadata extraction
    meta_patterns_main = {
        "filename": r"Filename:\s*(.*)",
        "dynamic_pressure": r"Dinamic pressure\s+([\d.]+)",
        "acquisition_time": r"Acquisition time\s+([\d.]+)",
        "acquisition_frequency": r"Acquisition frequency\s+([\d.]+)",
        "number_of_samples": r"Number of samples\s+(\d+)",
        "barometric_pressure": r"Barometric pressure\s+([\d.]+)",
        "air_temperature_start": r"Air temperature \(start\)\s+([\d.]+)",
        "re_1m_10e6": r"Re 1m /10\.E6\s+([\d.]+)",
        "air_density_start": r"Air density \(start\)\s+([\d.]+)",
        # CORRECTED LINE BELOW:
        "air_kin_viscosity_start": r"Air kin\. viscosity \(start\)\s+([\d.E_+-]+)",
        "date": r"Date\s+(.*)",
        "time": r"Time\s+(.*)"
    }

    # Regex patterns for block-specific sub-header
    block_meta_patterns = {
        "x_mm": r"x\s*\[mm\]\s*=\s*([\d.-]+)",
        "z_mm": r"z\s*\[mm\]\s*=\s*([\d.-]+)",
        "temperature_C": r"temperature\s*\[C\]\s*=\s*([\d.-]+)",
        "density_kg_m3": r"density\s*\[kg/m\*\*3\]\s*=\s*([\d.-]+)"
    }

    # Separator for data blocks
    block_separator_regex = r"^\*{10,}\s*$"  # Line with 10 or more asterisks

    all_lines = file_content.strip().split('\n')

    # --- Parse Main Header ---
    main_header_end_index = -1
    for i, line in enumerate(all_lines):
        if re.match(block_separator_regex, line.strip()):
            main_header_end_index = i
            break
        
        stripped_line = line.strip()
        if not stripped_line:
            continue

        matched_meta = False
        for key, pattern in meta_patterns_main.items():
            match = re.search(pattern, stripped_line, re.IGNORECASE)
            if match:
                value_str = match.group(1).strip()
                try:
                    if '.' in value_str or 'E' in value_str.upper():
                        main_metadata[key] = float(value_str)
                    else:
                        main_metadata[key] = int(value_str)
                except ValueError:
                    main_metadata[key] = value_str # Store as string if conversion fails
                matched_meta = True
                break
        if not matched_meta:
            description_lines.append(stripped_line)
    
    if description_lines:
        main_metadata["description"] = "\n".join(d for d in description_lines if d)

    if main_header_end_index == -1:
        raise ValueError("Could not find the main header separator (line of asterisks).")

    # --- Parse Data Blocks ---
    blocks_section_str = "\n".join(all_lines[main_header_end_index:])
    block_segments = re.split(block_separator_regex, blocks_section_str, flags=re.MULTILINE)

    for segment in block_segments:
        segment = segment.strip()
        if not segment:
            continue

        current_block_lines = segment.split('\n')
        block_specific_metadata = {}
        data_lines_for_df_str = []
        
        data_table_header_str = "y [mm]   Q  [m/s]  yaw[deg]"
        
        parsing_data_table = False
        for line in current_block_lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            if parsing_data_table:
                # Check if the line contains a valid number of columns (3 for P3S data)
                if len(stripped_line.split()) == 3:
                     data_lines_for_df_str.append(stripped_line)
                else:
                    # print(f"Skipping malformed data line: '{stripped_line}' in block for x={block_specific_metadata.get('x_mm')}")
                    pass 
                continue

            matched_block_meta = False
            for key, pattern in block_meta_patterns.items():
                match = re.search(pattern, stripped_line)
                if match:
                    value_str = match.group(1).strip()
                    try:
                        if key in ["x_mm", "z_mm"]: 
                            block_specific_metadata[key] = int(float(value_str))
                        else: 
                            block_specific_metadata[key] = float(value_str)
                    except ValueError:
                        print(f"Warning: Could not convert block metadata '{key}': '{value_str}'")
                        block_specific_metadata[key] = value_str
                    matched_block_meta = True
                    break
            
            if matched_block_meta:
                continue

            if data_table_header_str in stripped_line:
                parsing_data_table = True
                continue

        if 'x_mm' not in block_specific_metadata:
            continue
        
        if not data_lines_for_df_str:
            print(f"Warning: No data found for x_mm = {block_specific_metadata['x_mm']}.")
            continue

        data_io = io.StringIO("\n".join(data_lines_for_df_str))
        try:
            df_block = pd.read_csv(
                data_io,
                delim_whitespace=True,
                names=['y [mm]', 'Q [m/s]', 'yaw[deg]'],
                header=None,
                dtype={'y [mm]': float, 'Q [m/s]': float, 'yaw[deg]': float}
            )
            velocity_profile_df = df_block.reset_index(drop=True)
        except pd.errors.EmptyDataError:
            print(f"Warning: Empty data table for x_mm = {block_specific_metadata['x_mm']}.")
            continue
        except Exception as e:
            print(f"Error parsing data table for x_mm = {block_specific_metadata['x_mm']}: {e}")
            print(f"Data that caused error: \n---\n{data_io.getvalue()}\n---")
            continue
            
        x_station_key = block_specific_metadata['x_mm'] 
        
        station_package = {
            "z_mm": block_specific_metadata.get('z_mm'),
            "temperature_C": block_specific_metadata.get('temperature_C'),
            "density_kg_m3": block_specific_metadata.get('density_kg_m3'),
            "velocity_profile": velocity_profile_df
        }
        data_by_x_station[x_station_key] = station_package

    output_data = {
        "main_metadata": main_metadata,
        "data_by_x_station": data_by_x_station
    }
    return output_data


def save_to_pickle(data_to_save, output_filename="output_p3s_data.pkl"):
    """Saves the data to a pickle file."""
    with open(output_filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"Data saved to {output_filename}")

# --- Main execution ---
#

z_list =[ -200, -150, -100, 0]

for z in z_list:

    fname = f"vel_z{z}_conv.dat"
# --- Main execution ---
    with open(fname, "r") as f:
        file_content = f.read()

    parsed_p3s_data = parse_p3s_file_content(file_content)

    print("--- Main Metadata (P3S) ---")
    if parsed_p3s_data["main_metadata"]:
        for key, value in parsed_p3s_data["main_metadata"].items():
            print(f"{key}: {value}")
    else:
        print("No main metadata extracted.")

    print("\n--- Data for X-stations (Velocity Profile - P3S) ---")
    if parsed_p3s_data["data_by_x_station"]:
        for x_station, station_data in parsed_p3s_data["data_by_x_station"].items():
            print(f"\nX-Station: {x_station} (type: {type(x_station)})")
            print(f"  Z [mm]: {station_data['z_mm']}")
            print(f"  Temperature [C]: {station_data['temperature_C']}")
            print(f"  Density [kg/m3]: {station_data['density_kg_m3']}")
            print("  Velocity Profile (first 5 rows):")
            print(station_data['velocity_profile'].head())
            print("  Velocity Profile dtypes:")
            print(station_data['velocity_profile'].dtypes)
            if len(parsed_p3s_data["data_by_x_station"]) > 1: # Stop after printing one for brevity
                break
    else:
        print("No data by X-station found.")

    output_filename_p3s = f"vel_z{z}_conv.pkl"
    save_to_pickle(parsed_p3s_data, output_filename_p3s)

    print(f"\n--- Verifying pickle file: {output_filename_p3s} ---")
    with open(output_filename_p3s, 'rb') as f:
        loaded_p3s_data = pickle.load(f)

    print("\nLoaded Filename from Pickle:", loaded_p3s_data.get("main_metadata", {}).get("filename", "N/A"))

    if loaded_p3s_data.get("data_by_x_station"):
        first_x_key = list(loaded_p3s_data["data_by_x_station"].keys())[0]
        print(f"\nData for X-station = {first_x_key} from Pickle:")
        print(f"  Z [mm]: {loaded_p3s_data['data_by_x_station'][first_x_key]['z_mm']}")
        print("  Velocity Profile (first 5 rows):")
        print(loaded_p3s_data['data_by_x_station'][first_x_key]['velocity_profile'].head())
    else:
        print("No X-station data in loaded pickle file.")
