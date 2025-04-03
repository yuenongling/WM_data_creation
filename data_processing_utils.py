# data_processing_utils.py
import numpy as np
import os
import sys

def import_path(load_bfm_path=True):
    """
    Adds the BFM_PATH and its subdirectories to the system path for module imports.
    This is necessary for accessing custom modules in the BFM project.
    """
    # Get the BFM_PATH from environment variables
    WM_DATA_PATH = os.environ.get("WM_DATA_PATH")
    
    if load_bfm_path:
        # Ensure necessary paths are included
        BFM_PATH = os.environ.get("BFM_PATH")
        sys.path.append(BFM_PATH)
        sys.path.append(os.path.join(BFM_PATH, "NNOpt"))
        sys.path.append(os.path.join(BFM_PATH, "NNOpt/post_proc"))

    return WM_DATA_PATH


def find_k_y_values(y, U_all, y_all, k=2):
    """
    Given a profile U_all(y_all) and target y locations, finds the U value
    at locations (2*k+1)*y using linear interpolation.

    Args:
        y (np.ndarray): Array of target y-coordinates where interpolation is needed.
        U_all (np.ndarray): Array of velocity values for the full profile.
        y_all (np.ndarray): Array of y-coordinates corresponding to U_all.
        k (int, optional): Multiplier factor for target y locations. Defaults to 2 (for U at 5y).
                           k=1 corresponds to 3y, k=3 corresponds to 7y, etc.

    Returns:
        np.ndarray: Array of interpolated U values at locations (2*k+1)*y.
    """
    # Ensure y_all and U_all have the same length
    if len(U_all) != len(y_all):
        raise ValueError("U_all and y_all must have the same length.")
        
    # Ensure y_all is sorted for interpolation - this improves robustness
    sort_indices = np.argsort(y_all)
    y_all_sorted = y_all[sort_indices]
    U_all_sorted = U_all[sort_indices]

    # Calculate target y locations for interpolation
    target_y = (2 * k + 1) * y

    # Interpolate, handling potential extrapolation by using the boundary values
    # np.interp requires xp to be increasing, hence the sorting above.
    U_at_ky = np.interp(target_y, y_all_sorted, U_all_sorted, 
                        left=U_all_sorted[0], right=U_all_sorted[-1])
                        
    return U_at_ky
