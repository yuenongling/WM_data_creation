import pandas as pd
import numpy as np
import re
import argparse
import os

def extract_cf_cp_data(filename, cp=False):
    """Extract x and cf from cf.exp.dat"""
    with open(filename, 'r') as f:
        file_content = f.read()
    
    lines = file_content.strip().split('\n')
    
    # Skip comment lines (starting with #)
    data_lines = [line for line in lines if not line.strip().startswith('#')]
    
    # Find the variables line and skip it
    for i, line in enumerate(data_lines):
        if 'variables' in line:
            data_lines = data_lines[i+1:]
            break
    
    # Parse data
    x_values = []
    cf_values = []
    
    for line in data_lines:
        # Split by whitespace and skip empty lines
        values = line.strip().split()

        # Skip zone lines
        if line.strip().startswith('ZONE'):
            continue

        if len(values) >= 2:
            x = float(values[0])
            # Handle scientific notation (e.g., 2.88e-3)
            cf = float(values[1])
            
            x_values.append(x)
            cf_values.append(cf)
    
    if cp:
        return pd.DataFrame({'x': x_values, 'cp': cf_values})
    else:
        return pd.DataFrame({'x': x_values, 'cf': cf_values})

def extract_profile_data(filename):
    """Extract y and u from profiles.exp.dat"""
    with open(filename, 'r') as f:
        file_content = f.read()
    
    lines = file_content.strip().split('\n')
    
    # Skip comment lines (starting with #)
    data_lines = [line for line in lines if not line.strip().startswith('#')]
    
    # Find the variables line
    variables_line = None
    for i, line in enumerate(data_lines):
        if 'variables' in line:
            variables_line = line
            break
    
    # Extract variable indices for y and u
    y_index = 1  # Default index if not found
    u_index = 0  # Default index if not found
    
    # Parse data
    y_values = []
    u_values = []
    station_values = []  # To keep track of which station the data comes from
    
    reading_data = False
    current_station = None
    
    for line in data_lines:
        if 'VARIABLES' in line:
            reading_data = True
            continue
            
                   # Capture station information from zone lines
        if line.strip().startswith('ZONE'):
            zone_match = re.search(r'T="([^"]+)"', line)

            if zone_match:
                # Extract the station text and then just keep the x/H value
                station_text = zone_match.group(1)
                # Look for patterns like "exp, x/H=-4" and extract just the number
                # x_h_match = re.search(r'x/H=([+-]?\d+)', station_text)
                x_h_match = re.search(r'exp x=([-+]?\d+\.\d+)\s*m', station_text)
                if x_h_match:
                    current_station = x_h_match.group(1)
                else:
                    current_station = station_text
            continue 

        if reading_data and current_station:
            # Split by whitespace and skip empty lines
            values = line.strip().split()
            if len(values) > max(y_index, u_index):
                y = float(values[y_index])
                u = float(values[u_index])
                
                y_values.append(y)
                u_values.append(u)
                station_values.append(current_station)
    
    return pd.DataFrame({
        'station': station_values,
        'y': y_values, 
        'u': u_values
    })

def main():
    parser = argparse.ArgumentParser(description='Extract data from experimental files')
    parser.add_argument('--cf_file', required=True, help='Path to the cf.exp.dat file')
    parser.add_argument('--cp_file', required=True, help='Path to the cp.expnew.dat file')
    parser.add_argument('--profile_file', required=True, help='Path to the profiles.exp.dat file')
    parser.add_argument('--output', default='extracted_data', help='Output file prefix (without extension)')
    
    args = parser.parse_args()
    
    # Extract data from each file
    cf_data = extract_cf_cp_data(args.cf_file)
    cp_data = extract_cf_cp_data(args.cp_file, cp=True)
    profile_data = extract_profile_data(args.profile_file)
    
    # Save to CSV files
    cf_data.to_csv(f"./stats/{args.output}_cf.csv", index=False)
    cp_data.to_csv(f"./stats/{args.output}_cp.csv", index=False)
    profile_data.to_csv(f"./stats/{args.output}_profile.csv", index=False)
    
    # # Save to a combined Excel file with different sheets
    # with pd.ExcelWriter(f"{args.output}.xlsx") as writer:
    #     cf_data.to_excel(writer, sheet_name='CF_Data', index=False)
    #     cp_data.to_excel(writer, sheet_name='CP_Data', index=False)
    #     profile_data.to_excel(writer, sheet_name='Profile_Data', index=False)
    
    print(f"Data extracted and saved to {args.output}.xlsx and CSV files")
    
    # Print summary statistics
    print("\nSummary:")
    print(f"CF data: {len(cf_data)} points")
    print(f"CP data: {len(cp_data)} points")
    print(f"Profile data: {len(profile_data)} points from {profile_data['station'].nunique()} stations")

if __name__ == "__main__":
    main()
