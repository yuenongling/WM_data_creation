import tecplot as tp
import numpy as np
import sys
import pickle as pkl
import os

print("--- Starting Comprehensive Data Extraction ---")

# --- Helper Functions (for dPdx calculation) ---
def gaussian_kernel(x, xi, bandwidth):
    """ Gaussian kernel function. """
    return np.exp(-0.5 * ((x - xi) / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))

def local_gaussian_smooth(x, y, bandwidth=0.1):
    """ Perform local Gaussian smoothing on the data. """
    x = np.array(x)
    y = np.array(y)
    y_smooth = np.zeros_like(y)
    for i, xi in enumerate(x):
        weights = gaussian_kernel(x, xi, bandwidth)
        weights /= np.sum(weights) # Normalize
        y_smooth[i] = np.sum(weights * y)
    return y_smooth

 

# Switch case for selecting the dataset parameters
header_lines = 2
Re = 936000

# Define file paths
dataset_path = f'./WallHump-WideSpan-OriginalTopWallContour.dat'
cp_path = f'./Cp.dat'
cf_path = f'./Cf.dat'
output_pickle_path = f'./data/tecplot_data.pkl' # New name for clarity

# --- Data Loading and Pre-calculation ---

# 1. Load Surface Data (using NumPy)
Cp_data = np.genfromtxt(cp_path, delimiter=' ', skip_header=header_lines)
Cf_data = np.genfromtxt(cf_path, delimiter=' ', skip_header=header_lines)
# Assuming: Col 0: x, Col 1: Cp, Col 3: delta, Last Col: Cf

# 3. Load Volumetric Data (using Tecplot)
extracted_data = {}
try:
    print("Connecting to Tecplot session...")
    tp.session.connect(port=7600) # Adjust port if needed
    tp.new_layout()

    print(f"Loading Tecplot dataset: {dataset_path}...")
    dataset = tp.data.load_tecplot(dataset_path)
    print("Tecplot dataset loaded.")

    zone = dataset.zone(0)
    print(f"Accessing zone 0: {zone.name}")

    max_I = zone.dimensions[0]
    max_J = zone.dimensions[1]
    print(f"Zone dimensions: I={max_I}, J={max_J}")

    # Extract relevant volume variables into NumPy arrays
    print("Extracting volumetric variables (y, U)...")
    # Reshape assuming Fortran order reading into C order (J, I)
    y_vol = zone.values('y')[:].reshape((max_J, max_I))
    U_vol = zone.values('U')[:].reshape((max_J, max_I))
    V_vol = zone.values('V')[:].reshape((max_J, max_I))
    print("Variable extraction complete.")

    # Store all necessary data
    extracted_data = {
        'case_name': case_name,
        'Re': Re,
        'surface_data': {
            'x': x_surf,
            'Cp': Cp_surf,
            'Cf': Cf_surf,
            'delta': delta_surf,
            'dPdx_smooth': dPdx_smooth, # Store the smoothed version used for 'up'
            'up': up_calc,             # Store the calculated 'up'
            # 'raw_surface_data': data_surface # Optional
        },
        'volume_data': {
            'y': y_vol,
            'U': U_vol
        },
        'dimensions': {
            'I': max_I,
            'J': max_J
        }
    }

except Exception as e:
    print(f"\nError during Tecplot interaction or data extraction: {e}")
    print("Please ensure Tecplot is running and configured for connections.")
    sys.exit(1)
finally:
    # Optional: Disconnect if library supports it explicitly
     pass

# --- Save Data ---
if extracted_data:
    try:
        print(f"Saving extracted data to: {output_pickle_path}")
        with open(output_pickle_path, 'wb') as f:
            pkl.dump(extracted_data, f)
        print("Data saved successfully.")
    except Exception as e:
        print(f"Error saving data to Pickle file: {e}")
        sys.exit(1)
else:
    print("No data was extracted, skipping save.")

print("--- Comprehensive Data Extraction Finished ---")
