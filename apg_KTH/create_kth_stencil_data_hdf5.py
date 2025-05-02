from scipy.io import loadmat
import numpy as np
import pandas as pd
import os
from scipy.interpolate import RegularGridInterpolator, interp1d
import time

# Paste this entire block into a Google Colab cell and run it.

def process_object_array_to_2d_float(obj_array, pad_value=np.nan):
  """
  Converts a NumPy array of objects (where objects are lists or arrays)
  into a 2D NumPy array of floats, attempting to preserve the first dimension.

  It determines the maximum length (N) among all convertible inner lists/arrays.
  Shorter lists/arrays are padded to length N using the specified pad_value.
  If an inner list/array cannot be converted to numeric types (float),
  the entire corresponding row in the output array will be filled with pad_value.

  Args:
    obj_array (np.ndarray): A NumPy array with dtype=object (shape like (M,)).
                            Each element is expected to be a list or array-like
                            structure containing potentially numeric data.
    pad_value (float, optional): The value used for padding shorter sequences
                                 and for rows that fail conversion.
                                 Defaults to np.nan.

  Returns:
    np.ndarray: A 2D NumPy array of floats with shape (M, N), where M is
                the length of the original obj_array and N is the maximum
                length found in the convertible inner elements. Returns an
                empty 2D array (shape (M, 0)) if no elements could be
                processed. Returns an array of shape (0, 0) if the input
                array is empty.

  Raises:
      TypeError: If the input obj_array is not a 1D NumPy array.
  """
  if not isinstance(obj_array, np.ndarray):
      raise TypeError("Input must be a NumPy array.")

  if obj_array.ndim != 1:
    raise TypeError(f"Input must be a 1D NumPy array, but got shape {obj_array.shape}")

  if obj_array.size == 0:
      # Handle empty input array immediately
      return np.empty((0, 0), dtype=float)

  processed_rows = []
  max_len = 0
  success = False # Flag to track if any row was successfully processed

  # --- First Pass: Convert and find max length ---
  for item in obj_array:
    try:
      # Attempt conversion to a 1D float numpy array
      # Check for None explicitly before converting to array
      if item is None:
          raise TypeError("Item is None")

      # Ensure item is somewhat array-like before passing to np.array
      # This basic check prevents errors with completely unsuitable types
      if not hasattr(item, '__len__') and not isinstance(item, np.ndarray):
          raise TypeError(f"Item '{item}' is not array-like.")

      np_item = np.array(item, dtype=float)

      # Ensure it's 1D, handle potential multi-dimensional items if needed (flatten?)
      if np_item.ndim == 0: # Handle scalar items if needed
           np_item = np.array([np_item.item()]) # Convert scalar to 1D array
      elif np_item.ndim > 1:
         # Option 1: Flatten multi-dimensional items (Uncomment to enable)
         # print(f"Warning: Item '{item}' was not 1D, flattening.")
         # np_item = np_item.flatten()
         # Option 2: Treat as error (current implementation)
         raise ValueError(f"Inner item resulted in {np_item.ndim}D array, expected 1D.")

      processed_rows.append(np_item)
      max_len = max(max_len, len(np_item))
      success = True # Mark that at least one row worked
    except (ValueError, TypeError) as e:
      # Handle conversion errors (non-numeric) or type errors (not array-like, None)
      print(f"Info: Could not process item '{item}'. It will be represented by NaNs. Reason: {e}")
      processed_rows.append(None) # Store None as a placeholder for rows that failed
      # We don't update max_len for failed rows


  # If no rows could be processed at all, return an array filled with pad_value
  if not success:
       print("Warning: No inner elements could be successfully converted to float arrays.")
       # Determine shape based on input M and derived N (which is 0 here)
       # Return shape (M, 0) as max_len is 0
       return np.full((len(obj_array), 0), pad_value, dtype=float)


  # --- Second Pass: Pad and Stack ---
  final_rows = []
  for row_data in processed_rows:
    if row_data is None:
      # Create a row of pad_values for failed conversions
      padded_row = np.full(max_len, pad_value, dtype=float)
    else:
      # Pad successful conversions if necessary
      pad_width = max_len - len(row_data)
      if pad_width > 0:
        # np.pad is flexible: ((before, after),) for each axis
        padded_row = np.pad(row_data, (0, pad_width), mode='constant', constant_values=pad_value)
      elif pad_width < 0:
        # This case might happen if flattening was enabled and an item exceeded max_len
        padded_row = row_data[:max_len] # Truncate
      else:
         # No padding needed if already max_len
        padded_row = row_data
    final_rows.append(padded_row)

  # Stack the finalized rows into a 2D array
  # Using np.array is generally safer than vstack when you have a list of arrays
  if not final_rows: # Should not happen if input size > 0, but as safeguard
       return np.empty((len(obj_array), 0), dtype=float)

  result_array = np.array(final_rows, dtype=float)

  return result_array

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path

WM_DATA_PATH = import_path()

datapath = os.path.join(WM_DATA_PATH, 'data')

APG_MAT_FILE = os.path.join(WM_DATA_PATH, 'apg_KTH', 'data', 'APG.mat')

# --- Load the APG data ---
try:
    results = loadmat(APG_MAT_FILE, squeeze_me=True)
except FileNotFoundError:
    print(f"Error: APG.mat file not found at {APG_MAT_FILE}")
    results = None
except Exception as e:
    print(f"Error loading APG.mat file: {e}")
    results = None


subcases = ['b1n', 'b2n', 'm13n', 'm16n', 'm18n']

UP_FRAC = 0.2
DOWN_FRAC = 0.01

# --- Main Data Processing Loop ---
if results is not None:
    for subcase in subcases:
        print(f"Processing subcase: {subcase}")

        all_inputs_data = []
        all_output_data = []
        all_flow_type_data = []
        all_unnormalized_inputs_data = []

        result = results[subcase]

        Cf = result['Cf'].astype(np.float64)  # Cf = 2(utau / Ue)^2
        U = process_object_array_to_2d_float(result['U'])
        y = process_object_array_to_2d_float(result['y'])
        P = process_object_array_to_2d_float(result['P'])
        Ue = result['Ue'].astype(np.float64)
        theta = result['theta'].astype(np.float64)
        nu = result['nu'].astype(np.float64)
        delta99 = result['delta99'].astype(np.float64)
        beta = result['beta'].astype(np.float64)
        xa = result['x'].astype(np.float64)

        utau = np.array([np.sqrt(Cf[i] / 2) * Ue[i] for i, _ in enumerate(Cf)])
        dPdx = result['beta'] * utau**2 / result['deltas']  # NOTE: get back dPdx from beta
        up = np.sign(dPdx) * (abs(nu * dPdx)) ** (1 / 3)

        albert = theta / Ue**2 * dPdx

        # Pre-calculate dU/dy profiles for all x-locations
        dUdy_profiles_list = []
        for idx_grad in range(len(xa)):
                y_profile = y[idx_grad]
                U_profile = U[idx_grad]

                if len(y_profile) < 2:
                    dUdy_profiles_list.append(np.full_like(U_profile, np.nan)) # Cannot calculate gradient
                    continue

                # Calculate gradient
                dUdy_profile = np.gradient(U_profile, y_profile)
                dUdy_profiles_list.append(dUdy_profile)
        dUdy_profiles_list = np.array(dUdy_profiles_list)

        # Pre-calculate upn profiles for all x-locations
        upn_profiles_list = []
        for idx_upn in range(len(xa)):
                y_profile = y[idx_upn]
                P_profile = P[idx_upn]
                nu_at_x = nu[idx_upn]

                if len(y_profile) == 0:
                    upn_profiles_list.append(np.full_like(P_profile, np.nan))
                    continue

                # Calculate delta_p relative to P at the first available y point
                delta_p_profile = P_profile - P_profile[0]

                # Calculate upn profile at this x-location
                with np.errstate(divide='ignore', invalid='ignore'):
                    upn_profile_at_x = np.sign(delta_p_profile) * (abs(nu_at_x * delta_p_profile))**(1/3)

                upn_profiles_list.append(upn_profile_at_x)
        upn_profiles_list = np.array(upn_profiles_list)

        # Create 1D interpolator for U, P and dUdy
        U_interp = RegularGridInterpolator((xa, y[0,:]), U,
                                 bounds_error=False, fill_value=None)
        P_interp = RegularGridInterpolator((xa, y[0,:]), U,
                                 bounds_error=False, fill_value=None)
        dUdy_interp = RegularGridInterpolator((xa, y[0,:]), dUdy_profiles_list,
                                 bounds_error=False, fill_value=None)
        upn_interp = RegularGridInterpolator((xa, y[0,:]), upn_profiles_list,
                                 bounds_error=False, fill_value=None)
        up_interp  = interp1d(xa, up, bounds_error=False, fill_value=None)
        utau_interp = interp1d(xa, utau, bounds_error=False, fill_value=None)


        # Loop through original streamwise locations (idx corresponds to xa index)
        for idx in range(len(xa)):
            # Apply original filters based on x-location
            if xa[idx] > 2000:
                continue

            if idx % 100 == 0:
                print(f"Processing x-location {xa[idx]}...")
                start_time = time.time()

            # Get profiles and scalar values at the current original x-location (idx)
            y_i = y[idx] # y profile at current x
            U_i = U[idx] # U profile at current x
            delta99_i = delta99[idx]
            nu_i = nu[idx]
            x_i = xa[idx]
            utau_i = utau[idx]
            up_i = up[idx] # up scalar value at current x

            # Determine the y-range based on delta99 at the original x (idx)
            bot_index = np.where((y_i >= DOWN_FRAC * delta99_i) & (y_i <= UP_FRAC * delta99_i))[0]

            if len(bot_index) == 0:
                continue

            # Ensure the lowest selected y_val is non-zero for Pi_9 calculation denominator
            if y_i[bot_index[0]] == 0:
                    bot_index = bot_index[1:]
                    if len(bot_index) == 0:
                        continue

            # Loop through the selected y-indices for the current original x-location (idx)
            for j in bot_index:
                y_val = y_i[j] # The y-coordinate for THIS ROW's features

                # Define the three target x-locations relative to the original x and the current y_val
                x_val_original = xa[idx]
                target_x_current = x_val_original
                target_x_upstream = x_val_original - 2 * y_val
                target_x_downstream = x_val_original + 2 * y_val

                # Define the locations to process (target x-value and their corresponding suffixes)
                locations_to_process = {
                    'current': '', # No suffix for current
                    'upstream': '_upstream',
                    'downstream': '_downstream'
                }

                combined_inputs_dict = {}
                combined_unnorm_dict = {}

                input_bases = [
                    'u1_y_over_nu', 'up_y_over_nu', 'u2_y_over_nu', 'u3_y_over_nu',
                    'u4_y_over_nu', 'dudy1_y_pow2_over_nu', 'dudy2_y_pow2_over_nu',
                    'dudy3_y_pow2_over_nu', 'upn_y_over_nu',
                ]
                unnorm_bases = [
                        'y', 'u1', 'nu', 'utau', 'up', 'upn', 'u2', 'u3', 'u4', 'dudy1', 'dudy2', 'dudy3', 'x'
                ]

                # --- Process data for each of the three specified locations ---
                for loc_name, suffix in locations_to_process.items():
                    if loc_name == 'current':
                            x_target = target_x_current
                    elif loc_name == 'upstream':
                            x_target = target_x_upstream
                    elif loc_name == 'downstream':
                            x_target = target_x_downstream

                    # --- Data Extraction by 2D Interpolation at (x_target, y_val) and scaled y ---
                    # Interpolate scalar values defined on the xa grid
                    nu_at_xtarget = np.unique(nu)
                    up_at_xtarget = up_interp(x_target)

                    # Use 2D interpolators for U, dU/dy, upn at (x_target, y_val * 2^k)
                    # Need to handle potential NaNs from interpolation (e.g., outside convex hull)
                    U1_at_point = U_interp((x_target, y_val * 2**0))
                    U2_at_point = U_interp((x_target, y_val * (2**1+1)))
                    U3_at_point = U_interp((x_target, y_val * (2**2+1)))
                    U4_at_point = U_interp((x_target, y_val * (2**3+1)))

                    dudy1_at_point = dUdy_interp((x_target, y_val * 2**0))
                    dudy2_at_point = dUdy_interp((x_target, y_val * (2**1+1)))
                    dudy3_at_point = dUdy_interp((x_target, y_val * (2**2+1)))

                    upn_at_point = upn_interp((x_target, y_val * 2**0))


                    # Calculate Pi values for this specific location (x_target, y_val)
                    # Check for NaNs in interpolated values before calculating Pi
                    pi_1 = U1_at_point * y_val / nu_at_xtarget
                    pi_2 = up_at_xtarget * y_val / nu_at_xtarget
                    pi_3 = U2_at_point * y_val / nu_at_xtarget
                    pi_4 = U3_at_point * y_val / nu_at_xtarget
                    pi_5 = U4_at_point * y_val / nu_at_xtarget
                    pi_6 = dudy1_at_point * y_val**2 / nu_at_xtarget
                    pi_7 = dudy2_at_point * y_val**2 / nu_at_xtarget
                    pi_8 = dudy3_at_point * y_val**2 / nu_at_xtarget
                    pi_9 = upn_at_point * y_val / nu_at_xtarget


                    # Populate combined inputs dictionary
                    combined_inputs_dict[input_bases[0] + suffix] = pi_1
                    combined_inputs_dict[input_bases[1] + suffix] = pi_2
                    combined_inputs_dict[input_bases[2] + suffix] = pi_9 # Use pi_9 for upn_y_over_nu
                    combined_inputs_dict[input_bases[3] + suffix] = pi_3
                    combined_inputs_dict[input_bases[4] + suffix] = pi_4
                    combined_inputs_dict[input_bases[5] + suffix] = pi_5
                    combined_inputs_dict[input_bases[6] + suffix] = pi_6
                    combined_inputs_dict[input_bases[7] + suffix] = pi_7
                    combined_inputs_dict[input_bases[8] + suffix] = pi_8


                    # Populate combined unnormalized inputs dictionary
                    combined_unnorm_dict[unnorm_bases[0] + suffix] = y_val # y* for the row
                    combined_unnorm_dict[unnorm_bases[1] + suffix] = U1_at_point
                    combined_unnorm_dict[unnorm_bases[2] + suffix] = nu_at_xtarget
                    combined_unnorm_dict[unnorm_bases[3] + suffix] = utau_i
                    combined_unnorm_dict[unnorm_bases[4] + suffix] = up_at_xtarget
                    combined_unnorm_dict[unnorm_bases[5] + suffix] = upn_at_point
                    combined_unnorm_dict[unnorm_bases[6] + suffix] = U2_at_point
                    combined_unnorm_dict[unnorm_bases[7] + suffix] = U3_at_point
                    combined_unnorm_dict[unnorm_bases[8] + suffix] = U4_at_point
                    combined_unnorm_dict[unnorm_bases[9] + suffix] = dudy1_at_point
                    combined_unnorm_dict[unnorm_bases[10] + suffix] = dudy2_at_point
                    combined_unnorm_dict[unnorm_bases[11] + suffix] = dudy3_at_point
                    combined_unnorm_dict[unnorm_bases[12] + suffix] = x_target # Store the target x-coord


                # --- After processing all three locations for this (idx, j) point ---
                # Calculate Output Feature - uses values *ONLY* from the original point (idx)
                # Output is y+ (utau * y / nu)
                output_dict = {
                    'utau_y_over_nu': utau_i * y_val / nu_i # Output is y+ for the original point
                }

                # Collect Flow Type Information - uses values *ONLY* from the original point (idx)
                flow_type_dict = {
                    'case_name': 'apg_kth',
                    'nu': nu_i, # Using nu at the original x as reference value
                    'x': x_val_original, # The x-coordinate of the original point
                    'delta': delta99_i,
                    'albert': albert[idx]
                }

                all_inputs_data.append(combined_inputs_dict)
                all_output_data.append(output_dict)
                all_unnormalized_inputs_data.append(combined_unnorm_dict)
                all_flow_type_data.append(flow_type_dict)


            if idx % 100 == 0:
                # Print elapsed time for this x-location
                end_time = time.time()
                print(f"Processed x-location {x_i} in {end_time - start_time:.2f} seconds.")


        if not all_inputs_data:
            print(f"No data was generated for subcase {subcase}.")
        else:
            print(f"\nConcatenating data for subcase {subcase}...")
            inputs_df = pd.DataFrame(all_inputs_data)
            output_df = pd.DataFrame(all_output_data)
            flow_type_df = pd.DataFrame(all_flow_type_data)
            unnormalized_inputs_df = pd.DataFrame(all_unnormalized_inputs_data)

            # --- Filter rows with NaN in inputs_df ---
            print(f"Original shapes before NaN filter for {subcase}: Inputs={inputs_df.shape}, Output={output_df.shape}, Flow Type={flow_type_df.shape}, Unnormalized Inputs={unnormalized_inputs_df.shape}")
            rows_before_filter = inputs_df.shape[0]

            valid_rows_mask = inputs_df.notna().all(axis=1)

            inputs_df_filtered = inputs_df[valid_rows_mask].copy()
            output_df_filtered = output_df[valid_rows_mask].copy()
            flow_type_df_filtered = flow_type_df[valid_rows_mask].copy()
            unnormalized_inputs_df_filtered = unnormalized_inputs_df[valid_rows_mask].copy()

            rows_after_filter = inputs_df_filtered.shape[0]
            print(f"Filtered out {rows_before_filter - rows_after_filter} rows containing NaN in inputs for {subcase}.")
            print(f"Filtered shapes for {subcase}: Inputs={inputs_df_filtered.shape}, Output={output_df_filtered.shape}, Flow Type={flow_type_df_filtered.shape}, Unnormalized Inputs={unnormalized_inputs_df_filtered.shape}")



            os.makedirs(datapath, exist_ok=True)

            output_filename = os.path.join(datapath, f'apg_{subcase}_data_stencils.h5')
            print(f"\nSaving filtered data to HDF5 file: {output_filename}")
            try:
                inputs_df_filtered.to_hdf(output_filename, key='inputs', mode='w', format='fixed')
                output_df_filtered.to_hdf(output_filename, key='output', mode='a', format='fixed')
                unnormalized_inputs_df_filtered.to_hdf(output_filename, key='unnormalized_inputs', mode='a', format='fixed')
                flow_type_df_filtered.to_hdf(output_filename, key='flow_type', mode='a', format='table')
                print("Data successfully saved.")
            except Exception as e:
                print(f"Error saving HDF5 file for {subcase}: {e}")
                pass

            print(f"Final Filtered Shapes for {subcase}:")
            print(f"  Inputs: {inputs_df_filtered.shape}")
            print(f"  Output: {output_df_filtered.shape}")
            print(f"  Flow Type: {flow_type_df_filtered.shape}")
            print(f"  Unnormalized Inputs: {unnormalized_inputs_df_filtered.shape}")

else:
    print("Skipping data generation due to error loading APG.mat file.")

print("\nFinished processing all subcases.")


# Ad-hoc fix for object dtype columns in inputs_df
unnorm_bases = [
        'y', 'u1', 'nu', 'utau', 'up', 'upn', 'u2', 'u3', 'u4', 'dudy1', 'dudy2', 'dudy3', 'x'
]
subcases_to_fix = ['b1n', 'b2n', 'm13n', 'm16n', 'm18n']
for case in subcases_to_fix:
    output_filename = os.path.join(datapath, f'apg_{case}_data_stencils.h5')
    inputs_df = pd.read_hdf(output_filename, key='inputs')
    output_df = pd.read_hdf(output_filename, key='output')
    unnormalized_inputs_df = pd.read_hdf(output_filename, key='unnormalized_inputs')
    flow_type_df = pd.read_hdf(output_filename, key='flow_type')

    print(f"\nChecking for object dtype columns in inputs_df for {case}...")
    for col in inputs_df.columns:
        # Check if the column dtype is 'object' (often indicates mixed types or non-numeric)
        if inputs_df[col].dtype == 'object':
            # Inspect the first non-null value to see if it's a list/array
            first_val = inputs_df[col].dropna().iloc[0]
            if isinstance(first_val, (list, np.ndarray)):
                print(f"Converting column: {col}") # Optional: Print which columns are being converted
                inputs_df[col] = inputs_df[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x)
                inputs_df[col] = pd.to_numeric(inputs_df[col], errors='coerce')

    print(f"\nChecking for object dtype columns in unnormalized_inputs_df for {case}...")
    for col in unnormalized_inputs_df.columns:
        # Check if the column dtype is 'object' (often indicates mixed types or non-numeric)
        if unnormalized_inputs_df[col].dtype == 'object':
            # Inspect the first non-null value to see if it's a list/array
            first_val = unnormalized_inputs_df[col].dropna().iloc[0]
            if isinstance(first_val, (list, np.ndarray)):
                print(f"Converting column: {col}") # Optional: Print which columns are being converted

                if unnormalized_inputs_df[col].dropna().iloc[0].shape == ():
                    unnormalized_inputs_df[col] = unnormalized_inputs_df[col].apply(lambda x: x.item() if isinstance(x, (list, np.ndarray)) else x)
                else:
                    unnormalized_inputs_df[col] = unnormalized_inputs_df[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x)
                unnormalized_inputs_df[col] = pd.to_numeric(unnormalized_inputs_df[col], errors='coerce')

    # Reorder unnorm columns before output
    ordered_unnorm_bases = unnorm_bases.copy()
    for i in ['_upstream', '_downstream']:
        ordered_unnorm_bases += [f"{base}{i}" for base in unnorm_bases]
    unnormalized_inputs_df = unnormalized_inputs_df[ordered_unnorm_bases]

    print(f"\nSaving modified data back to HDF5 file: {output_filename}")
    inputs_df.to_hdf(output_filename, key='inputs', mode='w', format='fixed') # Use mode='w'
    output_df.to_hdf(output_filename, key='output', mode='a', format='fixed') # Use mode='a'
    unnormalized_inputs_df.to_hdf(output_filename, key='unnormalized_inputs', mode='a', format='fixed') # Use mode='a'
    flow_type_df.to_hdf(output_filename, key='flow_type', mode='a', format='table') # Use mode='a'

    print("Data successfully saved.")
