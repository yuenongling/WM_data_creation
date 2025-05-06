import scipy.io
import pickle
import numpy as np
import os

def convert_mat_to_pickle(mat_file, pickle_file=None):
    """
    Convert a MATLAB .mat file to a Python pickle file.
    
    Parameters:
    -----------
    mat_file : str
        Path to the input MATLAB .mat file
    pickle_file : str, optional
        Path to the output pickle file. If None, will use the same name as input but with .pkl extension
    
    Returns:
    --------
    dict
        Dictionary containing the data that was saved to the pickle file
    """
    print(f"Loading MATLAB file: {mat_file}")
    
    # Load the MATLAB file
    try:
        mat_data = scipy.io.loadmat(mat_file)
        print(f"Successfully loaded {len(mat_data.keys())} variables")
    except Exception as e:
        print(f"Error loading MATLAB file: {e}")
        
        # Try hdf5 format for MATLAB 7.3+ files
        try:
            import h5py
            print("Trying to load as an HDF5 file (MATLAB 7.3+)...")
            f = h5py.File(mat_file, 'r')
            
            # Convert HDF5 objects to numpy arrays
            mat_data = {}
            for key in f.keys():
                if key.startswith('_'):  # Skip private vars
                    continue
                    
                value = f[key]
                if isinstance(value, h5py.Dataset):
                    mat_data[key] = np.array(value)
                else:
                    print(f"Skipping non-Dataset object {key}")
                    
            print(f"Successfully loaded {len(mat_data.keys())} variables from HDF5 format")
            f.close()
        except ImportError:
            print("h5py not installed, cannot load MATLAB 7.3+ files")
            raise
        except Exception as e2:
            print(f"Error loading as HDF5: {e2}")
            raise
    
    # Convert MATLAB cell arrays to Python lists
    cleaned_data = {}
    for key, value in mat_data.items():
        if key.startswith('_'):  # Skip MATLAB metadata
            continue
            
        # Handle cell arrays
        if isinstance(value, np.ndarray) and value.dtype == np.object_:
            try:
                # Convert all cell array elements to native Python objects
                if value.ndim == 1:
                    cleaned_value = [convert_element(value[i]) for i in range(value.shape[0])]
                elif value.ndim == 2:
                    cleaned_value = [[convert_element(value[i, j]) for j in range(value.shape[1])] 
                                     for i in range(value.shape[0])]
                else:
                    cleaned_value = value  # Keep higher-dimensional arrays as is
                
                cleaned_data[key] = cleaned_value
            except Exception as e:
                print(f"Warning: Could not convert cell array {key}: {e}")
                cleaned_data[key] = value
        else:
            cleaned_data[key] = value
    
    # Handle the special 'flow_type' field which might be a cell array of strings and numbers
    if 'flow_type' in cleaned_data:
        try:
            flow_type = cleaned_data['flow_type']
            # If it's a list of lists (from cell array)
            if isinstance(flow_type, list) and isinstance(flow_type[0], list):
                processed_flow_type = []
                for row in flow_type:
                    processed_row = []
                    for item in row:
                        # Convert MATLAB strings (arrays of chars) to Python strings
                        if isinstance(item, np.ndarray) and item.dtype.kind in ['U', 'S']:
                            if item.size == 1:
                                processed_row.append(str(item.item()))
                            else:
                                processed_row.append(str(item))
                        else:
                            processed_row.append(item)
                    processed_flow_type.append(processed_row)
                cleaned_data['flow_type'] = processed_flow_type
        except Exception as e:
            print(f"Warning: Could not process flow_type: {e}")
    
    # Determine output filename if not provided
    if pickle_file is None:
        base_name = os.path.splitext(mat_file)[0]
        pickle_file = f"{base_name}.pkl"
    
    # Save to pickle
    print(f"Saving to pickle file: {pickle_file}")
    with open(pickle_file, 'wb') as f:
        pickle.dump(cleaned_data, f)
    
    print(f"Conversion complete. Saved to {pickle_file}")
    return cleaned_data

def convert_element(element):
    """Helper function to convert a MATLAB cell array element to Python object"""
    if isinstance(element, np.ndarray):
        if element.dtype.kind in ['U', 'S']:  # String or Unicode
            if element.size == 1:
                return str(element.item())
            else:
                return str(element)
        elif element.size == 1:  # Single value
            return element.item()
        else:
            return element
    else:
        return element

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert MATLAB .mat file to Python pickle')
    parser.add_argument('mat_file', help='Path to the input MATLAB .mat file')
    parser.add_argument('--output', '-o', help='Path to the output pickle file (optional)')
    
    args = parser.parse_args()
    
    convert_mat_to_pickle(args.mat_file, args.output)
