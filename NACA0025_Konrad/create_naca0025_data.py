import scipy.io
import h5py
import numpy as np
import pandas as pd
import os
import sys

region = sys.argv[1] if len(sys.argv) > 1 else 'Turbulent'

# 1. Specify the path to your input .mat file containing the structured arrays
input_mat_file = f'./stats/airfoil_bl_pi_fine_{region}_parameters.mat'  # <--- CHANGE THIS TO YOUR ACTUAL MAT FILE NAME

# 2. Specify the desired path for the output .h5 file
output_h5_file = f'../data/naca0025_{region}_Konrad_data.h5' # <--- CHANGE THIS IF YOU WANT A DIFFERENT OUTPUT NAME



def extract_cell_array_column(cell_array, col_index):
    """Helper function to extract a column from a MATLAB cell array loaded by scipy."""
    # (This function remains the same as the previous version)
    num_rows = cell_array.shape[0]
    extracted_col = []
    for i in range(num_rows):
        try:
            element = cell_array[i, col_index]
            while isinstance(element, np.ndarray) and element.size == 1:
                element = element.item()
            if isinstance(element, np.bytes_):
                 element = element.decode('utf-8') # Or appropriate encoding
            elif isinstance(element, np.str_):
                 element = str(element)
            extracted_col.append(element)
        except IndexError:
            extracted_col.append(None)
        except Exception as e:
            print(f"Warning: Error processing cell array element at row {i}, col {col_index}: {e}")
            extracted_col.append(None)
    return extracted_col


def convert_mat_to_h5_pandas(mat_file_path, h5_file_path):
    """
    Converts a .mat file containing specific structured arrays ('inputs',
    'output', 'unnormalized_inputs', 'flow_type') to an .h5 file using
    Pandas DataFrames. Maps array columns to DataFrame columns based on
    predefined order and saves each DataFrame under a corresponding key in HDF5.

    Args:
        mat_file_path (str): The path to the input .mat file.
        h5_file_path (str): The path where the output .h5 file will be saved.

    Raises:
        FileNotFoundError: If the .mat file does not exist.
        KeyError: If required variables ('inputs', etc.) are not found.
        Exception: For other potential errors.
    """
    print(f"Starting conversion using Pandas from '{mat_file_path}' to '{h5_file_path}'...")

    # --- 1. Check if input file exists ---
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f"Error: Input MAT file not found at '{mat_file_path}'")

    # --- Ensure clean output file ---
    # to_hdf in 'a' mode appends, so delete existing file first for a fresh start
    if os.path.exists(h5_file_path):
        print(f"Removing existing HDF5 file: {h5_file_path}")
        os.remove(h5_file_path)

    try:
        # --- 2. Load the .mat file ---
        print(f"Loading MAT file: {mat_file_path}")
        mat_data = scipy.io.loadmat(mat_file_path)
        print("MAT file loaded successfully.")

        # --- 3. Define target DataFrame column names in the desired order ---
        # (These lists remain the same as the previous version)
        input_keys = [
            'u1_y_over_nu', 'up_y_over_nu', 'u2_y_over_nu',
            'u3_y_over_nu', 'u4_y_over_nu', 'dudy1_y_pow2_over_nu',
            'dudy2_y_pow2_over_nu', 'dudy3_y_pow2_over_nu'
        ]
        output_keys = ['utau_y_over_nu']
        unnorm_keys = [
            'y', 'u1', 'nu', 'utau', 'up',
            'u2', 'u3', 'u4', 'dudy1', 'dudy2', 'dudy3'
        ]
        flow_type_keys = ['case_name', 'nu', 'x', 'delta', 'Ue']

        # --- 4. Verify required variables exist in MAT file ---
        required_mat_vars = ['inputs', 'output', 'unnormalized_inputs', 'flow_type']
        for var in required_mat_vars:
            if var not in mat_data:
                raise KeyError(f"Error: Required variable '{var}' not found in MAT file '{mat_file_path}'. Found keys: {list(mat_data.keys())}")
        print("Found required MAT variables: ", required_mat_vars)

        # --- 5. Create Pandas DataFrames ---
        print("\nCreating Pandas DataFrames...")

        # Inputs DataFrame
        data_array = mat_data['inputs']
        num_cols = data_array.shape[1]
        cols = input_keys[:num_cols] # Select names only for available columns
        inputs_df = pd.DataFrame(data_array, columns=cols)
        print(f"Created inputs_df with shape {inputs_df.shape} and columns: {inputs_df.columns.tolist()}")

        # Output DataFrame
        data_array = mat_data['output']
        num_cols = data_array.shape[1]
        cols = output_keys[:num_cols]
        output_df = pd.DataFrame(data_array, columns=cols)
        print(f"Created output_df with shape {output_df.shape} and columns: {output_df.columns.tolist()}")

        # Unnormalized Inputs DataFrame
        data_array = mat_data['unnormalized_inputs']
        num_cols = data_array.shape[1]
        cols = unnorm_keys[:num_cols]
        unnorm_df = pd.DataFrame(data_array, columns=cols)
        print(f"Created unnorm_df with shape {unnorm_df.shape} and columns: {unnorm_df.columns.tolist()}")

        # Flow Type DataFrame (from cell array)
        data_array = mat_data['flow_type']
        num_cols = data_array.shape[1]
        cols = flow_type_keys[:num_cols]
        flow_type_dict = {}
        print(f"Processing 'flow_type' cell array ({data_array.shape[0]}x{num_cols}) for DataFrame...")
        for i, key in enumerate(cols):
             print(f"  - Extracting column {i} for key '{key}'...")
             flow_type_dict[key] = extract_cell_array_column(data_array, i)
        flow_type_df = pd.DataFrame(flow_type_dict)
        print(f"Created flow_type_df with shape {flow_type_df.shape} and columns: {flow_type_df.columns.tolist()}")


        # --- 6. Save DataFrames to HDF5 ---
        print(f"\nSaving DataFrames to HDF5 file: {h5_file_path}")

        # Use format='table' for potential querying and better performance with strings/mixed types
        # Use mode='a' (append) because we write multiple distinct keys/DataFrames to the same file.
        # We deleted the file earlier, so the first 'a' effectively creates it.
        # Specify data_columns=True for flow_type_df if you plan to query on its columns later.

        inputs_df.to_hdf(h5_file_path, key='inputs', mode='a', format='table', complib='blosc', complevel=5)
        print("Saved '/inputs' DataFrame.")

        output_df.to_hdf(h5_file_path, key='output', mode='a', format='table', complib='blosc', complevel=5)
        print("Saved '/output' DataFrame.")

        unnorm_df.to_hdf(h5_file_path, key='unnormalized_inputs', mode='a', format='table', complib='blosc', complevel=5)
        print("Saved '/unnormalized_inputs' DataFrame.")

        # For flow_type, explicitly setting string columns might improve storage/querying
        # `data_columns=True` makes all columns queryable index columns.
        flow_type_df.to_hdf(h5_file_path, key='flow_type', mode='a', format='table', data_columns=True, complib='blosc', complevel=5)
        print("Saved '/flow_type' DataFrame.")


        print(f"\nConversion complete. HDF5 file saved to '{h5_file_path}'")

        # Optional: Verify content
        # with pd.HDFStore(h5_file_path, mode='r') as store:
        #     print("\nHDF5 Store contains keys:", store.keys())
        #     # print("\nFirst 5 rows of /inputs:\n", store['inputs'].head())


    except FileNotFoundError as e:
         print(f"Error: {e}")
         raise
    except KeyError as e:
        print(f"Error: {e}")
        # No partial file to clean as we delete at the start
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # No partial file to clean
        raise


convert_mat_to_h5_pandas(input_mat_file, output_h5_file)
