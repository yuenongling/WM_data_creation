import pandas as pd
import pickle
import io
import re

def parse_general_metadata(lines):
    """Parses the initial block of metadata common to files."""
    metadata = {}
    description_lines = []
    
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
    
    header_end_marker = "*****************************************************"
    header_lines_processed = 0

    for i, line in enumerate(lines):
        if header_end_marker in line:
            header_lines_processed = i + 1
            break # Stop after the first block of asterisks
        
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
        
        if not matched_meta and "Filename:" not in line: # Avoid adding "Filename:" to description
            description_lines.append(stripped_line)
    
    if description_lines:
        metadata["description"] = "\n".join(description_lines)
        
    return metadata, header_lines_processed


def parse_cfc_data_file_content(file_content):
    """
    Parses the CFC_DAT file content.
    Data is organized by Z-station, with each Z-station containing X and Cf.
    """
    lines = file_content.strip().split('\n')
    
    # 1. Parse general metadata (header block)
    general_metadata, start_of_data_blocks_index = parse_general_metadata(lines)
    
    # Data blocks are separated by "*****************************************************"
    # The actual content for data blocks starts after the general metadata part.
    remaining_content = "\n".join(lines[start_of_data_blocks_index:])
    
    # Split the remaining content into individual data blocks
    # Each block starts with "y [mm] =", "x [mm] =", etc.
    # and ends before the next "*****************************************************" or end of file.
    data_blocks_raw = re.split(r'\n\s*\*+\s*\n', remaining_content) # Split by separator lines

    all_data_records = [] # To store (x_val, z_val, cf_val, block_meta) tuples

    for block_str in data_blocks_raw:
        block_str = block_str.strip()
        if not block_str:
            continue
            
        block_lines = block_str.split('\n')
        
        block_metadata = {}
        data_table_start_index = -1
        
        # Parse block-specific metadata (y, x, temperature, density)
        for i, line in enumerate(block_lines):
            line = line.strip()
            if "z [mm]" in line and "Cf*1000[-]" in line: # This is the header of the data table
                data_table_start_index = i + 1
                break
            
            match_y = re.match(r"y\s*\[mm\]\s*=\s*([\d.-]+)", line, re.IGNORECASE)
            if match_y:
                block_metadata['y_mm'] = float(match_y.group(1))
                continue
            
            match_x = re.match(r"x\s*\[mm\]\s*=\s*([\d.-]+)", line, re.IGNORECASE)
            if match_x:
                block_metadata['x_mm_block'] = float(match_x.group(1)) # x value for this whole block
                continue

            match_temp = re.match(r"temperature\s*\[C\]\s*=\s*([\d.-]+)", line, re.IGNORECASE)
            if match_temp:
                block_metadata['temperature_C'] = float(match_temp.group(1))
                continue

            match_density = re.match(r"density\s*\[kg/m\*\*3\]\s*=\s*([\d.-]+)", line, re.IGNORECASE)
            if match_density:
                block_metadata['density_kg_m3'] = float(match_density.group(1))
                continue

        if data_table_start_index == -1:
            # print(f"Warning: Could not find data table header in block:\n{block_str[:200]}...")
            continue

        # Extract the tabular data (z, Cf*1000) for this block
        table_data_str = "\n".join(block_lines[data_table_start_index:])
        if not table_data_str.strip():
            continue
            
        try:
            block_df = pd.read_csv(
                io.StringIO(table_data_str),
                delim_whitespace=True,
                names=['z_orig', 'Cf*1000'],
                header=None,
                skip_blank_lines=True,
                dtype=float
            )
            
            # Add the x_mm from this block to each row and convert z to integer
            x_val_for_block = int(block_metadata.get('x_mm_block', float('nan'))) # Convert block X to int
            
            for _, row in block_df.iterrows():
                z_val = int(row['z_orig']) # Convert z to integer
                cf_val = row['Cf*1000'] / 1000.0 # Store actual Cf, not Cf*1000
                
                # Store y, block_temp, block_density if needed later, or directly with x,z,Cf
                # For now, just collecting essential data for the final structure
                all_data_records.append({
                    'x [mm]': x_val_for_block,
                    'z [mm]': z_val,
                    'Cf': cf_val,
                    'y [mm]': block_metadata.get('y_mm'), # y is same for all z in a block
                    # Optionally add block-specific temp and density if desired per (x,z) point
                    # 'block_temperature_C': block_metadata.get('temperature_C'),
                    # 'block_density_kg_m3': block_metadata.get('density_kg_m3'),
                })

        except pd.errors.EmptyDataError:
            print(f"Warning: Empty data table in block starting with x={block_metadata.get('x_mm_block')}")
        except Exception as e:
            print(f"Error processing block (x={block_metadata.get('x_mm_block')}):\n{e}")
            print(f"Table data string (first 100 chars): '{table_data_str[:100]}...'")
            continue # Skip this block on error
            
    # Convert the list of records to a DataFrame for easier grouping
    if not all_data_records:
        print("Warning: No data records were extracted from the blocks.")
        # Create an empty DataFrame with expected columns if no data
        full_df = pd.DataFrame(columns=['x [mm]', 'z [mm]', 'Cf', 'y [mm]'])
    else:
        full_df = pd.DataFrame(all_data_records)

    # Organize data by Z-station (integer z)
    # Each z-station will have a DataFrame of x and Cf values
    data_by_z_station = {}
    if not full_df.empty:
        unique_z_stations = sorted(full_df['z [mm]'].unique())
        for z_station_val in unique_z_stations: # z_station_val is already an integer
            station_df = full_df[full_df['z [mm]'] == z_station_val][['x [mm]', 'Cf']].copy()
            # Sort by x [mm] within each z-station for consistency
            station_df.sort_values(by='x [mm]', inplace=True)
            station_df.reset_index(drop=True, inplace=True)
            data_by_z_station[z_station_val] = station_df
            
    output_data = {
        "metadata": general_metadata,
        "data_by_z_station": data_by_z_station,
        # Optionally, you could also store the y-value if it's constant and relevant
        # "y_value_mm": all_data_records[0]['y [mm]'] if all_data_records else None
    }
    
    return output_data


def save_to_pickle_cfc(data_to_save, output_filename="output_data_cfc.pkl"):
    """Saves the data to a pickle file."""
    with open(output_filename, 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"Data saved to {output_filename}")


# --- Main execution ---

with open("./cf_conv.dat", "r") as f:
    file_content = f.read()
parsed_cfc_data = parse_cfc_data_file_content(file_content)

# --- Print some results for verification ---
# print("--- Main Metadata ---")
# if parsed_cfc_data["main_metadata"]:
#     for key, value in parsed_cfc_data["main_metadata"].items():
#         print(f"{key}: {value}")
# else:
#     print("No main metadata extracted.")

# print("\n--- Data for X-stations (Cf*1000) ---")
# if parsed_cfc_data["data_by_x_station"]:
#     for x_station, station_data in parsed_cfc_data["data_by_x_station"].items():
#         print(f"\nX-Station: {x_station} (type: {type(x_station)})")
#         print(f"  Y [mm]: {station_data['y_mm']}")
#         print(f"  Temperature [C]: {station_data['temperature_C']}")
#         print(f"  Density [kg/m3]: {station_data['density_kg_m3']}")
#         print("  Cf Profile (first 5 rows):")
#         print(station_data['cf_profile'].head())
#         print("  Cf Profile dtypes:")
#         print(station_data['cf_profile'].dtypes)
#         if len(parsed_cfc_data["data_by_x_station"]) > 1: # Stop after printing one for brevity
#             break 
# else:
#     print("No data by X-station found.")

# Save to pickle
output_filename_cfc = "cf_conv.pkl"
save_to_pickle_cfc(parsed_cfc_data, output_filename_cfc)
