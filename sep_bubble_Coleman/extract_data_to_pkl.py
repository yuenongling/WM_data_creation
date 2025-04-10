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

# --- Configuration ---
if len(sys.argv) < 2:
    print("Usage: python extract_data_comprehensive.py <CaseName>")
    print("Example: python extract_data_comprehensive.py A")
    sys.exit(1)

case_name = sys.argv[1]

# Switch case for selecting the dataset parameters
match case_name:
    case 'A':
        header_lines = 11
        Re = 40000
    case 'B':
        header_lines = 11
        Re = 40000
    case 'C':
        header_lines = 15
        Re = 80000
    case _:
        print(f"Error: Unknown case '{case_name}'. Valid cases: A, B, C.")
        sys.exit(1)

# Define file paths
dataset_path = f'./stats/Qofxy_Case{case_name}_xavg.dat'
dataset_surface_path = f'./stats/Qofx_Case{case_name}_xavg.dat'
output_pickle_path = f'./data/tecplot_data_Case{case_name}.pkl' # New name for clarity

# Check if input files exist
if not os.path.exists(dataset_path):
    print(f"Error: Volumetric data file not found: {dataset_path}")
    sys.exit(1)
if not os.path.exists(dataset_surface_path):
    print(f"Error: Surface data file not found: {dataset_surface_path}")
    sys.exit(1)

print(f"Processing Case: {case_name} (Re={Re})")
print(f"Loading surface data: {dataset_surface_path}")
print(f"Loading volumetric data: {dataset_path}")

# --- Data Loading and Pre-calculation ---

# 1. Load Surface Data (using NumPy)
try:
    data_surface = np.genfromtxt(dataset_surface_path, delimiter=' ', skip_header=header_lines)
    # Assuming: Col 0: x, Col 1: Cp, Col 3: delta, Last Col: Cf
    x_surf = data_surface[:, 0]
    Cp_surf = data_surface[:, 1]
    delta_surf = data_surface[:, 3] # Make sure this is the correct column index
    Cf_surf = data_surface[:, -1]
    print("Surface data loaded successfully using NumPy.")
except Exception as e:
    print(f"Error loading surface data with NumPy: {e}")
    sys.exit(1)

# 2. Calculate Pressure Gradient (dPdx) and Friction Velocity (up)
print("Calculating dPdx and up from surface data...")
# NOTE: Calculate pressure gradients d(Cp)/dx
dPdx = np.gradient(Cp_surf, x_surf)
# Smooth the gradient
dPdx_smooth = local_gaussian_smooth(x_surf, dPdx, bandwidth=0.1) # Adjust bandwidth as needed

# NOTE: Calculate up/u_inf = sign(dPdx) * (0.5 * |dPdx/Re|)^(1/3)
# Avoid division by zero or issues with Cf=0 if used directly
# The formula used in the original script: sign(dPdx) * (0.5 * abs(dPdx)/Re)**(1/3)
# Ensure dPdx_smooth is used if smoothing is desired
with np.errstate(divide='ignore', invalid='ignore'): # Suppress potential warnings for 0/0 etc.
     up_calc = np.sign(dPdx_smooth) * (0.5 * np.abs(dPdx_smooth) / Re)**(1/3)
     up_calc = np.nan_to_num(up_calc) # Replace NaN with 0 if any occurred

print("dPdx and up calculated.")


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
