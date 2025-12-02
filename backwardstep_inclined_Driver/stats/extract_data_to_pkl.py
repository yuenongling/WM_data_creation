import re
import pandas as pd
import io
import pickle

def parse_driver_seegmiller_data(data_text):
    """
    Parses the Driver and Seegmiller experimental data text.

    Args:
        data_text (str): A string containing the entire data file content.

    Returns:
        dict: A dictionary where keys are X/H values (float) and values are
              dictionaries containing station metadata and a pandas DataFrame
              named 'data' with the tabular measurements.
    """
    lines = data_text.strip().split('\n')

    all_station_data = {}
    current_station_info = None
    data_rows_for_current_station = []
    parsing_data_table = False

    # Define column names based on the header:
    # LR   Y/H   U/Ur   V/Ur   uu    vv    uv    uuu   uvv   vuu   vvv
    column_names = ["LR", "Y/H", "U/Ur", "V/Ur", "uu", "vv", "uv", "uuu", "uvv", "vuu", "vvv"]

    # Regex patterns
    station_pattern = re.compile(r"Station=\s*(\S+)")
    meta_pattern = re.compile(
        r"X/H=\s*([-\d.]+)\s+Tunnel Run#=\s*(\d+)\s+Ue/Uref=\s*([-\d.]+)\s+Date=([\d.]+)\s+Time=([\d.]+)"
    )
    alpha_pattern = re.compile(r"alpha=(\d+)\s*deg")
    data_header_pattern = re.compile(r"^\s*LR\s+Y/H\s+U/Ur")
    # Data rows typically start with an integer (LR) and have the correct number of columns
    # This simple check verifies it starts with a digit and has roughly the right structure
    data_row_start_pattern = re.compile(r"^\s*\d+\s+[-\d.]+")

    def finalize_current_station():
        nonlocal current_station_info, data_rows_for_current_station, parsing_data_table, all_station_data
        if current_station_info and 'X/H' in current_station_info and data_rows_for_current_station:
            data_string = "\n".join(data_rows_for_current_station)
            df = pd.read_csv(io.StringIO(data_string), delim_whitespace=True, names=column_names, header=None)
            
            # Convert columns to numeric types
            for col in df.columns:
                if col == 'LR':
                    df[col] = pd.to_numeric(df[col], errors='coerce', downcast='integer')
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            current_station_info['data'] = df
            all_station_data[current_station_info['X/H']] = current_station_info.copy()
        
        # Reset for the next station
        current_station_info = None
        data_rows_for_current_station = []
        parsing_data_table = False

    for i, line_text_original in enumerate(lines):
        line_text_stripped = line_text_original.strip()

        # New station detection
        station_match = station_pattern.search(line_text_original) # Check original for indentation context
        if station_match:
            finalize_current_station() # Finalize previous station before starting new one

            current_station_info = {'Station_ID': station_match.group(1)}
            
            # Expect metadata on the next lines
            if i + 1 < len(lines):
                meta_line = lines[i+1].strip()
                meta_match = meta_pattern.search(meta_line)
                if meta_match:
                    current_station_info['X/H'] = float(meta_match.group(1))
                    current_station_info['Tunnel_Run'] = int(meta_match.group(2))
                    current_station_info['Ue/Uref'] = float(meta_match.group(3))
                    current_station_info['Date'] = meta_match.group(4)
                    current_station_info['Time'] = meta_match.group(5)
            
            if i + 2 < len(lines):
                alpha_line = lines[i+2].strip()
                alpha_match = alpha_pattern.search(alpha_line)
                if alpha_match:
                    current_station_info['alpha_deg'] = int(alpha_match.group(1))
            continue # Done with this line, move to the next

        # If we are processing a station (current_station_info is not None)
        if current_station_info:
            if data_header_pattern.search(line_text_original):
                parsing_data_table = True
                continue # Header found, data starts on next line

            if parsing_data_table:
                if data_row_start_pattern.search(line_text_original) and len(line_text_stripped.split()) == len(column_names):
                    data_rows_for_current_station.append(line_text_stripped)
                elif line_text_stripped == "" and data_rows_for_current_station: 
                    # Empty line often signifies end of data block for this station
                    finalize_current_station()
                elif data_rows_for_current_station: # If it's not a data row and not empty, it might be end of block
                    # This condition is tricky because the next line might be a new station header,
                    # which is handled at the top. So if we reach here, it's likely an unexpected line.
                    # We'll assume it means the end of the current data block.
                    finalize_current_station()
                    # The current non-data, non-empty line will be re-evaluated by the next loop iteration.
                    # If it was a new station header, it will be caught.

    # Finalize the last station in the file
    finalize_current_station()

    return all_station_data

filename = "./bkst-ldv-06.dat"
with open(filename, 'r') as file:
    data = file.read().rstrip()
all_station_data = parse_driver_seegmiller_data(data)

import pickle
with open("deg6_station_data.pkl", "wb") as f:
    pickle.dump(all_station_data, f)
