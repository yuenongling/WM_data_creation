'''
For this specific script, need to run from the TBL directory

python TBL/create_TBL_data_stencil.py

This script processes the TBL data for different angles and creates a dataset including one upstream and one downstream point for each original point. (adapted from create_TBL_data.py)

The dataset is saved in HDF5 format.
'''
import os
from scipy.io import loadmat
import sys
# import matplotlib.pyplot as plt # Not used
import numpy as np
import pickle as pkl
import pandas as pd
from utils import * # Assuming utils contains nu_dict and read_npz_stats

# --- Set up paths and constants ---
sys.path.append('../')
from data_processing_utils import import_path
WM_DATA_PATH = import_path(load_bfm_path=False) # Ensure the BFM_PATH and subdirectories are in the system path
savedatapath = os.path.join(WM_DATA_PATH, 'data')

# Y-fraction limits for selecting points in attached flow
UP_FRAC = 0.20
DOWN_FRAC = 0.01
# Y-fraction limits for selecting points in separated flow
UP_FRAC_SEP = 0.05
DOWN_FRAC_SEP = 0.005

TBL_PATH = "/home/yuenongling/Codes/BL/TBLS"
fpath_format = TBL_PATH + "/data/postnpz_20250321/TBL_Retheta_670_theta_{angle}deg_medium_avg_slice.npz"
# Load stats_BL once outside the angle loop
stats_BL = pkl.load(open(TBL_PATH + '/stats/20250321/stats_Re670_C_filtered_gauss.pkl', 'rb'))

# --- Retheta and some other setups ---
Retheta = 670
nu = nu_dict[Retheta]
angle_list = [-4, -3, -2, -1, 5, 10, 15, 20]
n_diff = 10 # For original downsampling criteria

# --- Helper function to sample value from a 1D profile along y ---
def sample_y_profile(y_profile, y_grid, target_y):
    """
    Samples a value from a 1D profile (defined on y_grid) at a target y-value.
    Uses linear interpolation.
    """
    if y_grid.size == 0 or y_profile.size == 0:
        return np.nan
    # np.interp requires y_grid to be increasing. Assume ym is increasing.
    return np.interp(target_y, y_grid, y_profile)

# --- Helper function to check if an x-index passes the original filtering ---
def passes_original_filter_by_index(check_idx, angle, xm, Cf, n_diff, current=True):
    """
    Checks if a given x-index corresponds to a location that would have
    been selected by the original filtering logic.

    current: Bool, if True, specifies that the check is for the current x location.
    """
    if not (0 <= check_idx < len(xm)):
        return False # Invalid index

    if not current:
        return True # If not current station, always pass

    x_check = xm[check_idx]
    Cf_check = Cf[check_idx]

    # Original x range filter
    x_range_ok = False
    if angle < 0:
        if 1.4 <= x_check <= 5.5:
            x_range_ok = True
    else: # angle >= 0
        if 1.4 <= x_check <= 7:
             # Check for the specific angle=20 gap
            if not (angle == 20 and (x_check > 5 and x_check < 6)):
                x_range_ok = True

    if not x_range_ok:
        return False

    # Original downsampling/separation filter (keep point if it's every n_diff OR near separation)
    if check_idx % n_diff == 0 or abs(Cf_check) < 0.0005:
        return True
    else:
        return False

# --- Main data processing loop ---
for i, angle in enumerate(angle_list):
    print(f"Processing angle: {angle} degrees")

    # Lists to collect data for the current angle
    all_inputs_data = []
    all_output_data = []
    all_flow_type_data = []
    all_unnormalized_inputs_data = []

    # Read in the boundary layer data for the current angle
    # Umean_raw is assumed to have shape (len(xm), len(ym)).
    # Pmean is assumed to have shape (len(xm), len(ym)).
    # Other returned variables are not used.
    x, y, z_unused, xm, ym, zm_unused, Umean_raw, V_unused, Pmean, tau_x_unused, up_stats, xg_unused, yg_unused = read_npz_stats(fpath_format.format(angle=angle), nu, filter=True)

    # Access stats data for the current angle (these are 1D arrays indexed by xm)
    delta_stats = stats_BL[angle].Delta_edge
    Cf_stats = stats_BL[angle].Cf
    up_stats = stats_BL[angle].up
    utau_stats = stats_BL[angle].utau_x

    # Loop through original streamwise locations (idx corresponds to xm index)
    # These 'idx' points are the base points for each row of the dataset.
    for idx in range(len(xm)):

        if idx % 20 != 0:
            continue

        print(f"Processing idx: {idx} (x: {xm[idx]})")

        # Only consider original base points that pass the filtering
        if not passes_original_filter_by_index(idx, angle, xm, Cf_stats, n_diff):
             continue

        # Determine the y-range based on delta and separation status *at the original x (idx)*
        # This determines the y-values (ym[j]) for which we create a row.
        delta_idx = delta_stats[idx]
        if Cf_stats[idx] < 0: # Near or after separation
            # Use separation fractions
            # Find lowest U velo index at this x location
            idx_min = np.argmin(Umean_raw[idx, :])

            # Find y indices using separation fractions, but clamp to relevant range
            idy_low_target_y = DOWN_FRAC_SEP * delta_idx
            idy_high_target_y = UP_FRAC_SEP * delta_idx

            # Find indices closest to target y values using binary search (assuming ym is sorted)
            idy_low = np.searchsorted(ym, idy_low_target_y, side='left')
            idy_high = np.searchsorted(ym, idy_high_target_y, side='right') - 1

            # Clamp idy_high to be at most the index of the minimum velocity
            idy_high = np.min([idy_high, idx_min])

            # Clamp idy_low based on idx_min and ensure non-negative
            idy_low = np.min([idy_low, idx_min // 2])
            idy_low = max(0, idy_low)
            idy_high = max(0, idy_high) # Ensure idy_high is not negative if range is empty

            if idy_low >= idy_high:
                 continue # Skip this x location if the valid y range is empty or invalid

        else: # Attached flow
             # Use attached flow fractions
             idy_low_target_y = DOWN_FRAC * delta_idx
             idy_high_target_y = UP_FRAC * delta_idx

             # Find indices closest to target y values
             idy_low = np.searchsorted(ym, idy_low_target_y, side='left')
             idy_high = np.searchsorted(ym, idy_high_target_y, side='right') - 1

             # Ensure indices are within ym bounds
             idy_low = max(0, idy_low)
             idy_high = min(len(ym) - 1, idy_high)

             if idy_low > idy_high:
                  continue # Skip if the valid y range is empty

        # Ensure idy_low corresponds to a non-zero y value to avoid division by zero in Pi_9 calculation
        if ym[idy_low] == 0:
             idy_low += 1 # Move up to the next point if the lowest selected y is zero
             if idy_low > idy_high:
                 continue # Skip if adjusting idy_low made the range invalid

        # Loop through the selected y-indices for the current original x-location (idx)
        # Each iteration of this loop creates ONE row in the output DataFrames
        for j in range(idy_low, idy_high + 1):
            y_val = ym[j] # The wall distance y* for THIS ROW's features

            # Define the three target x-locations relative to the original x and the current y_val
            x_val_original = xm[idx]
            target_x_current = x_val_original
            target_x_upstream = x_val_original - 2 * y_val
            target_x_downstream = x_val_original + 2 * y_val

            # Define the locations to process (target x-value and their corresponding suffixes)
            locations_to_process = [
                ('current', target_x_current),
                ('upstream', target_x_upstream),
                ('downstream', target_x_downstream)
            ]

            # Initialize dictionaries for combined inputs and unnormalized values for this row (this (idx, j) point)
            combined_inputs_dict = {}
            combined_unnorm_dict = {}

            # Define the base names for inputs and unnormalized values
            input_bases = [
                'u1_y_over_nu', 'up_y_over_nu', 'u2_y_over_nu', 'u3_y_over_nu',
                'u4_y_over_nu', 'dudy1_y_pow2_over_nu', 'dudy2_y_pow2_over_nu',
                'dudy3_y_pow2_over_nu', 'upn_y_over_nu',
            ]
            # Including 'x' in unnorm_bases to store the actual x coordinate of the sample point
            unnorm_bases = [
                'y', 'u1', 'nu', 'utau', 'up', 'u2', 'u3', 'u4', 'dudy1', 'dudy2', 'dudy3', 'upn', 'x'
            ]

            # --- Process data for each of the three specified locations ---
            for loc_name, x_target in locations_to_process:

                # print(f"  Processing location: {loc_name} (x_target: {x_target})")

                # --- Validity Checks for the target x-location ---
                # 1. Check if the target x is within the overall bounds of xm (prevents extreme extrapolation)
                # 2. Find the *closest* grid index to the target x
                # 3. Check if this *closest grid index* passes the original filtering logic.
                # This ensures we only sample from regions that would have been considered "valid"
                # in the original sampling strategy.
                closest_idx_to_target_x = np.argmin(np.abs(xm - x_target))
                is_valid_location = (xm[0] <= x_target <= xm[-1])
                                     # passes_original_filter_by_index(closest_idx_to_target_x, angle, xm, Cf_stats, n_diff)

                # Also check if y_val is zero, although handled by idy_low logic, double-check for safety
                if y_val == 0:
                     is_valid_location = False # Cannot calculate Pi_9 if y is zero

                if is_valid_location:
                    # print(f"  Processing location: {loc_name} (x_target: {x_target})")
                    # --- Data Extraction by Interpolation at (x_target, ym[j] etc.) ---
                    # Interpolate U profile along y at the target x-location
                    U_interp_profile_at_xtarget = np.array([np.interp(x_target, x, Umean_raw[:, k]) for k in range(len(ym))])

                    # Calculate interpolated dU/dy profile along y at the target x-location
                    dUdy_interp_profile_at_xtarget = np.gradient(U_interp_profile_at_xtarget, ym)

                    # Interpolate pressure values at the target x for y=0 and y=ym[j]
                    P_interp_wall_at_xtarget = np.interp(x_target, xm, Pmean[1:-1, 0])
                    P_interp_y_at_xtarget = np.interp(x_target, xm, Pmean[1:-1, j])

                    # Interpolate stats variables at the target x
                    utau_at_xtarget = np.interp(x_target, xm, utau_stats)
                    up_at_xtarget = np.interp(x_target, xm, up_stats)
                    # delta_at_xtarget = np.interp(x_target, xm, delta_stats) # Not needed for Pi calculation

                    # --- Sample interpolated profiles at specific y-levels ---
                    # Values at the point y = ym[j] (k=1 case)
                    u1_at_point = sample_y_profile(U_interp_profile_at_xtarget, ym, y_val) # Should be equivalent to U_interp_profile_at_xtarget[j] if y_val is in ym
                    dudy1_at_point = sample_y_profile(dUdy_interp_profile_at_xtarget, ym, y_val) # Should be equivalent to dUdy_interp_profile_at_xtarget[j]


                    # Values at scaled y locations (k>1 cases), using sampling helper
                    u2_at_point = sample_y_profile(U_interp_profile_at_xtarget, ym, y_val * (2**1+1))
                    u3_at_point = sample_y_profile(U_interp_profile_at_xtarget, ym, y_val * (2**2+1))
                    u4_at_point = sample_y_profile(U_interp_profile_at_xtarget, ym, y_val * (2**3+1))
                    dudy2_at_point = sample_y_profile(dUdy_interp_profile_at_xtarget, ym, y_val * (2**1+1))
                    dudy3_at_point = sample_y_profile(dUdy_interp_profile_at_xtarget, ym, y_val * (2**2+1))

                    # Pressure gradient terms at the target x and point j
                    # Calculate delta_p using interpolated Pmean values. y_val is guaranteed non-zero.
                    delta_p_at_point = (P_interp_y_at_xtarget - P_interp_wall_at_xtarget) / y_val
                    upn_at_point = np.sign(delta_p_at_point) * (abs(delta_p_at_point) * nu )**(1/3)

                    # --- Calculate Pi values for this specific location (x_target, j) ---
                    # Note: nu and y_val are constant for the set of inputs for this row
                    pi_1 = u1_at_point * y_val / nu
                    pi_2 = up_at_xtarget * y_val / nu # up is interpolated at x_target
                    pi_3 = u2_at_point * y_val / nu
                    pi_4 = u3_at_point * y_val / nu
                    pi_5 = u4_at_point * y_val / nu
                    pi_6 = dudy1_at_point * y_val**2 / nu
                    pi_7 = dudy2_at_point * y_val**2 / nu
                    pi_8 = dudy3_at_point * y_val**2 / nu
                    pi_9 = upn_at_point * y_val / nu

                    # Populate combined inputs dictionary
                    if loc_name == 'current':
                        suffix = '' # No suffix for the original point
                    else:
                        suffix = '_' + loc_name # e.g., '_current', '_upstream', '_downstream'
                    combined_inputs_dict[input_bases[0] + suffix] = pi_1
                    combined_inputs_dict[input_bases[1] + suffix] = pi_2
                    combined_inputs_dict[input_bases[2] + suffix] = pi_3
                    combined_inputs_dict[input_bases[3] + suffix] = pi_4
                    combined_inputs_dict[input_bases[4] + suffix] = pi_5
                    combined_inputs_dict[input_bases[5] + suffix] = pi_6
                    combined_inputs_dict[input_bases[6] + suffix] = pi_7
                    combined_inputs_dict[input_bases[7] + suffix] = pi_8
                    combined_inputs_dict[input_bases[8] + suffix] = pi_9

                    # Populate combined unnormalized inputs dictionary
                    combined_unnorm_dict[unnorm_bases[0] + suffix] = y_val # y* for the row
                    combined_unnorm_dict[unnorm_bases[1] + suffix] = u1_at_point
                    combined_unnorm_dict[unnorm_bases[2] + suffix] = nu   # nu is constant
                    combined_unnorm_dict[unnorm_bases[3] + suffix] = utau_at_xtarget # utau interpolated at x_target
                    combined_unnorm_dict[unnorm_bases[4] + suffix] = up_at_xtarget # up interpolated at x_target
                    combined_unnorm_dict[unnorm_bases[5] + suffix] = u2_at_point
                    combined_unnorm_dict[unnorm_bases[6] + suffix] = u3_at_point
                    combined_unnorm_dict[unnorm_bases[7] + suffix] = u4_at_point
                    combined_unnorm_dict[unnorm_bases[8] + suffix] = dudy1_at_point
                    combined_unnorm_dict[unnorm_bases[9] + suffix] = dudy2_at_point
                    combined_unnorm_dict[unnorm_bases[10] + suffix] = dudy3_at_point
                    combined_unnorm_dict[unnorm_bases[11] + suffix] = upn_at_point
                    combined_unnorm_dict[unnorm_bases[12] + suffix] = x_target # Store the target x-coord

                else:
                    # Location is invalid or filtered region, populate with NaNs for this location's features
                    suffix = '_' + loc_name
                    for base in input_bases:
                         combined_inputs_dict[base + suffix] = np.nan
                    for base in unnorm_bases:
                         # For y and nu, we can still store the value for the row
                         if base == 'y':
                             combined_unnorm_dict[base + suffix] = y_val
                         elif base == 'nu':
                             combined_unnorm_dict[base + suffix] = nu
                         elif base == 'x':
                             # Store the target x even if invalid
                              combined_unnorm_dict[base + suffix] = x_target
                         else:
                              combined_unnorm_dict[base + suffix] = np.nan # Other values are unknown/invalid


            # --- After processing all three locations for this (idx, j) point ---
            # Calculate Output Feature - uses values *ONLY* from the original point (idx)
            utau_original = utau_stats[idx]
            pi_out_original = utau_original * y_val / nu # Output is y+ for the original point
            output_dict = {
               'utau_y_over_nu': pi_out_original
            }

            # Collect Flow Type Information - uses values *ONLY* from the original point (idx)
            delta_original = delta_stats[idx]
            flow_type_dict = {
               'case_name': 'TBL',
               'nu': nu,
               'x': x_val_original, # The x-coordinate of the original point
               'delta': delta_original,
            }

            # Append the combined dictionaries to the lists
            all_inputs_data.append(combined_inputs_dict)
            all_output_data.append(output_dict) # This dict only has one key
            all_unnormalized_inputs_data.append(combined_unnorm_dict)
            all_flow_type_data.append(flow_type_dict)

    # After loops for idx and j finish for the current angle:
    # Convert the lists of dictionaries into DataFrames
    print(f"\nConcatenating data for angle {angle}...")
    # Ensure all columns are present even if some locations were always invalid for some rows
    inputs_df = pd.DataFrame(all_inputs_data)
    output_df = pd.DataFrame(all_output_data)
    flow_type_df = pd.DataFrame(all_flow_type_data)
    unnormalized_inputs_df = pd.DataFrame(all_unnormalized_inputs_data)

      # --- Filter rows with NaN in inputs_df ---
    print(f"Original shapes for angle {angle}: Inputs={inputs_df.shape}, Output={output_df.shape}, Flow Type={flow_type_df.shape}, Unnormalized Inputs={unnormalized_inputs_df.shape}")
    rows_before_filter = inputs_df.shape[0]

    # Find rows where NO element is NaN
    # inputs_df.notna() returns a boolean DataFrame
    # .all(axis=1) checks if ALL values in each row are True (i.e., not NaN)
    valid_rows_mask = inputs_df.notna().all(axis=1)

    # Use the boolean mask to select only the valid rows from all DataFrames
    inputs_df_filtered = inputs_df[valid_rows_mask].copy()
    output_df_filtered = output_df[valid_rows_mask].copy()
    flow_type_df_filtered = flow_type_df[valid_rows_mask].copy()
    unnormalized_inputs_df_filtered = unnormalized_inputs_df[valid_rows_mask].copy()

    rows_after_filter = inputs_df_filtered.shape[0]
    print(f"Filtered out {rows_before_filter - rows_after_filter} rows containing NaN in inputs.")
    print(f"Filtered shapes for angle {angle}: Inputs={inputs_df_filtered.shape}, Output={output_df_filtered.shape}, Flow Type={flow_type_df_filtered.shape}, Unnormalized Inputs={unnormalized_inputs_df_filtered.shape}")

    # Save DataFrames to HDF5 file
    # Use a new filename reflecting the interpolation approach and stencil locations
    output_filename = os.path.join(savedatapath, f'TBL_{angle}_data_stencils.h5')
    print(f"\nSaving data to HDF5 file: {output_filename}")
    # Use fixed format for better performance with numerical data
    inputs_df.to_hdf(output_filename, key='inputs', mode='w', format='fixed')
    output_df.to_hdf(output_filename, key='output', mode='a', format='fixed')
    unnormalized_inputs_df.to_hdf(output_filename, key='unnormalized_inputs', mode='a', format='fixed')
    # Use table format for flow_type if it contains strings, to keep them
    flow_type_df.to_hdf(output_filename, key='flow_type', mode='a', format='table')
    print("Data successfully saved.")

    # Print summary shapes
    print(f"Final Shapes for angle {angle}:")
    print(f"  Inputs: {inputs_df.shape}")
    print(f"  Output: {output_df.shape}")
    print(f"  Flow Type: {flow_type_df.shape}")
    print(f"  Unnormalized Inputs: {unnormalized_inputs_df.shape}")

print("\nFinished processing all angles.")
