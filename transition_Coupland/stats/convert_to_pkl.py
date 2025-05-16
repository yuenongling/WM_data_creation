import re
import numpy as np
import pickle as pkl

def parse_hotwire_data_columnar(text_data):
    """
    Parses the Rolls-Royce hot wire anemometry test data into a columnar format.

    Args:
        text_data (str): The string containing the entire dataset.

    Returns:
        dict: A dictionary where keys are integer axial positions (e.g., 95).
              Each value is a dictionary with 'metadata' (dict) and 'data' (dict).
              The 'data' dict has column headers as keys and NumPy arrays of values.
    """
    all_runs_data = {}
    run_text_blocks = text_data.split("RUN NUMBER ")[1:]

    for run_block_content in run_text_blocks:
        full_run_block = "RUN NUMBER " + run_block_content.strip()
        lines = full_run_block.splitlines()

        current_run_metadata = {}
        # This will temporarily store data as a list of row dictionaries
        temp_row_based_data = []
        table_column_headers = []
        parsing_table_data = False
        axial_position_key = None

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            found_meta_pairs = re.findall(r"([\w\s().%*#-]+?)\s*=\s*([^=\n]+?)(?=\s{2,}[\w\s().%*#-]+\s*=|\s*$)", line)

            if found_meta_pairs and not parsing_table_data:
                for key, value_str in found_meta_pairs:
                    key = key.strip()
                    value_str = value_str.strip()
                    current_run_metadata[key] = value_str
                    
                    numeric_part_match = re.match(r'([-\d\.XE()\*]+)', value_str)
                    parsed_num_val = None
                    if numeric_part_match:
                        num_str = numeric_part_match.group(1)
                        try:
                            if "X10**" in num_str:
                                parsed_num_val = float(num_str.replace("X10**", "E"))
                            elif "X10(" in num_str:
                                parsed_num_val = float(num_str.replace("X10(", "E").replace(")", ""))
                            elif "%" in num_str:
                                parsed_num_val = float(num_str.replace("%", ""))
                            else:
                                parsed_num_val = float(num_str)
                            current_run_metadata[key + "_val"] = parsed_num_val
                        except ValueError:
                            pass

                    if key == "AXIAL POSITION" and parsed_num_val is not None:
                        axial_position_key = int(parsed_num_val)
                continue

            if line.startswith("Y ") and "Y/DEL" in line and "YPLUS" in line:
                table_column_headers = [h.strip() for h in line.split()]
                parsing_table_data = True
                continue

            if parsing_table_data and line.startswith("MM ") and "M/S" in line:
                continue

            if parsing_table_data and table_column_headers:
                try:
                    if not re.match(r'^\s*[-\d\.]', line):
                        parsing_table_data = False
                        continue
                    
                    row_values_str = line.split()
                    if len(row_values_str) == len(table_column_headers):
                        row_data = {}
                        for i, header in enumerate(table_column_headers):
                            row_data[header] = float(row_values_str[i])
                        temp_row_based_data.append(row_data)
                    # else:
                        # print(f"Warning: Column/data mismatch in run {axial_position_key}, line: {line}")
                except ValueError:
                    # print(f"Warning: Could not parse data row in run {axial_position_key}, line: {line}")
                    parsing_table_data = False
                    continue
        
        # --- Transform row-based data to column-based data ---
        columnar_data_for_run = {}
        if temp_row_based_data and table_column_headers:
            for header in table_column_headers:
                # Extract all values for this header from the list of row dictionaries
                column_values = [row[header] for row in temp_row_based_data]
                columnar_data_for_run[header] = np.array(column_values, dtype=float)
        
        if axial_position_key is not None:
            all_runs_data[axial_position_key] = {
                "metadata": current_run_metadata,
                "data": columnar_data_for_run # Store the columnar data
            }
        else:
            if current_run_metadata or temp_row_based_data:
                 print(f"Warning: Could not determine axial position for a run block. Metadata: {list(current_run_metadata.keys())[:3]}")

    return all_runs_data

cases = ['t3am', 't3a', 't3b' , 't3c1', 't3c2', 't3c3', 't3c4', 't3c5']

for case in cases:

# The provided text data
    with open(case + 's.dat', "r") as file:
        raw_data = file.read()

    parsed_column_data = parse_hotwire_data_columnar(raw_data)

    with open(f"{case}.pkl", "wb") as file:
        pkl.dump(parsed_column_data, file)

