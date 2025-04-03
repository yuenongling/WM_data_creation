import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.signal import savgol_filter
from datetime import datetime

nu_dict = {}
nu_dict[670] = 5.2848e-06 
nu_dict[300] = 1.0570e-05
#############################################################
# NOTE: Define a dataclass to store the flow results
@dataclass
class Stats_BL:
    # NOTE: Boundary layer extraction method info
    edge_detect_method: str
    # NOTE: All relevant boundary layer parameters
    x: np.ndarray
    y: np.ndarray
    Delta_edge: np.ndarray
    U_edge: np.ndarray
    delta_star: np.ndarray
    theta: np.ndarray
    Retheta: np.ndarray
    utau_x: np.ndarray
    Retau_x: np.ndarray
    dPedx: np.ndarray
    Delta_edge_idx: np.ndarray
    vor_z: np.ndarray
    beta: np.ndarray
    K: np.ndarray
    Cf: np.ndarray
    up: np.ndarray
    # NOTE: Resolution parameters
    dx_plus: np.ndarray
    dy_plus: np.ndarray
    dz_plus: np.ndarray
    dx_pres: np.ndarray
    dy_pres: np.ndarray
    dz_pres: np.ndarray

    def plot_self_stats(self, y_data_name: str, ax, angle, x_cut=7, mid=True, filter=False, color=None, alpha=1):
        y_data = self.__getattribute__(y_data_name)
        x_data = self.x

        # If mid is True, plot in the middle of the cell
        if mid:
            x_plot = (x_data[2:] + x_data[:-2]) / 2
        else:
            x_plot = x_data

        # Apply Savitzky-Golay filter
        if filter:
            try:
                # data_plot= savgol_filter(y_data, 301, 2)
                data_plot = local_gaussian_smooth(x_data, y_data, bandwidth=0.1)
            except:
                # data_plot = np.zeros_like(y_data)
                # data_plot[:3500] = savgol_filter(y_data[:3500], 301, 2)
                # data_plot[3500:] = y_data[3500:]
                data_plot = local_gaussian_smooth(x_data[1:], y_data, bandwidth=0.1)
        else:
            data_plot = y_data

        # Plot up to x = x_cut
        idx = np.where(x_plot <= x_cut)[0]

        linestyle = '--' if angle < 0 else '-'
        if color is None:
            ax.plot(x_plot[idx[:-1]], data_plot[idx[:-1]], linestyle, alpha=alpha, label=fr"$\alpha = {{{angle}}}^{{\circ}}$")
        else:
            ax.plot(x_plot[idx[:-1]], data_plot[idx[:-1]], linestyle, color=color, alpha=alpha, label=fr"$\alpha = {{{angle}}}^{{\circ}}$")

#############################################################

def compute_Omega_z_xloc(U, V, x, y, xm, ym):
    """
    Compute Omega_z for 2D velocity field with specific indexing
    
    Parameters:
    -----------
    U : numpy.ndarray
        x-component of velocity (2D)
    V : numpy.ndarray
        y-component of velocity (2D)
    x, y : numpy.ndarray
        Grid coordinates
    xm, ym : numpy.ndarray
        Midpoint coordinates
    
    Returns:
    --------
    Omega_z : numpy.ndarray
        Vorticity in z-direction
    """
    # Compute grid edges
    xg = np.concatenate([
        [x[0] - (xm[0] - x[0])], 
        xm, 
        [x[-1] + x[-1] - xm[-1]]
    ])
    
    yg = np.concatenate([
        [y[0] - (ym[0] - y[0])], 
        ym, 
        [y[-1] + y[-1] - ym[-1]]
    ])
    
    # Allocate storage for Oij with extra dimensions to match MATLAB indexing
    nxg = len(x) + 1
    nyg = len(y) + 1
    Oij_ = np.zeros((nxg, nyg))
    Omega_z = np.zeros((nxg-1, nyg-1))
    
    # Compute vorticity (curl)
    for i in range(1, nxg):
        for j in range(1, nyg):
            # (dV/dx - dU/dy)
            Oij_[i,j] = (V[i,j-1] - V[i-1,j-1]) / (xg[i] - xg[i-1]) - \
                          (U[i-1,j] - U[i-1,j-1]) / (yg[j] - yg[j-1])
    
    # Extract Omega_z with specific indexing
    Omega_z = Oij_[1:nxg-1, 1:nyg-1]
    
    return xg, yg, Omega_z


def compute_Omega_z_2d(U, V, x, y, xm, ym):
    """
    Compute Omega_z for 2D velocity field with specific indexing
    
    Parameters:
    -----------
    U : numpy.ndarray
        x-component of velocity (2D)
    V : numpy.ndarray
        y-component of velocity (2D)
    x, y : numpy.ndarray
        Grid coordinates
    xm, ym : numpy.ndarray
        Midpoint coordinates
    
    Returns:
    --------
    Omega_z : numpy.ndarray
        Vorticity in z-direction
    """
    # Compute grid edges
    xg = np.concatenate([
        [x[0] - (xm[0] - x[0])], 
        xm, 
        [x[-1] + x[-1] - xm[-1]]
    ])
    
    yg = np.concatenate([
        [y[0] - (ym[0] - y[0])], 
        ym, 
        [y[-1] + y[-1] - ym[-1]]
    ])

    # # Calculate the weights for interpolation
    # weight_y_0 = ( yg[1:] - y ) / ( yg[1:] - yg[:-1]  )
    # weight_y_1 = 1 - weight_y_0
    
    # Allocate storage for Oij with extra dimensions to match MATLAB indexing
    nxg = len(x) + 1
    nyg = len(y) + 1
    Oij_ = np.zeros((nxg, nyg))
    Oij_temp = np.zeros((nxg, nyg))
    Omega_z = np.zeros((nxg-1, nyg-1))
    
    # Compute vorticity (curl)
    for i in range(1, nxg):
        for j in range(1, nyg):
            # (dV/dx - dU/dy)
            Oij_[i,j] = (V[i,j-1] - V[i-1,j-1]) / (xg[i] - xg[i-1]) - \
                          (U[i-1,j] - U[i-1,j-1]) / (yg[j] - yg[j-1])

    for i in range(1, nxg-1):
        for j in range(1, nyg-1):
            # Interpolate to cell center (Note that y is stretched)
            Oij_temp[i,j] = 0.25 * (Oij_[i,j] + Oij_[i+1,j] +  \
                Oij_[i,j+1] + Oij_[i+1,j+1])
    
    # Extract Omega_z with specific indexing
    Omega_z = Oij_temp[1:nxg-1, 1:nyg-1]
    
    return xg, yg, Omega_z

def compute_Omega_z_dUdyonly(U, V, x, y, xm, ym):
    """
    Compute Omega_z for 2D velocity field with specific indexing
    
    Parameters:
    -----------
    U : numpy.ndarray
        x-component of velocity (2D)
    V : numpy.ndarray
        y-component of velocity (2D)
    x, y : numpy.ndarray
        Grid coordinates
    xm, ym : numpy.ndarray
        Midpoint coordinates
    
    Returns:
    --------
    Omega_z : numpy.ndarray
        Vorticity in z-direction
    """
    # Compute grid edges
    xg = np.concatenate([
        [x[0] - (xm[0] - x[0])], 
        xm, 
        [x[-1] + x[-1] - xm[-1]]
    ])
    
    yg = np.concatenate([
        [y[0] - (ym[0] - y[0])], 
        ym, 
        [y[-1] + y[-1] - ym[-1]]
    ])

    # # Calculate the weights for interpolation
    # weight_y_0 = ( yg[1:] - y ) / ( yg[1:] - yg[:-1]  )
    # weight_y_1 = 1 - weight_y_0
    
    # Allocate storage for Oij with extra dimensions to match MATLAB indexing
    nxg = len(x) + 1
    nyg = len(y) + 1
    Oij_ = np.zeros((nxg, nyg))
    Oij_temp = np.zeros((nxg, nyg))
    Omega_z = np.zeros((nxg-1, nyg-1))
    
    # Compute vorticity (curl)
    for i in range(1, nxg):
        for j in range(1, nyg):
            # (dV/dx - dU/dy)
            Oij_[i,j] = (U[i-1,j] - U[i-1,j-1]) / (yg[j] - yg[j-1])

    for i in range(1, nxg-1):
        for j in range(1, nyg-1):
            # Interpolate to cell center (Note that y is stretched)
            Oij_temp[i,j] = 0.25 * (Oij_[i,j] + Oij_[i+1,j] +  \
                Oij_[i,j+1] + Oij_[i+1,j+1])
    
    # Extract Omega_z with specific indexing
    Omega_z = Oij_temp[1:nxg-1, 1:nyg-1]

    return xg, yg, Omega_z

def read_npz_file(file_path):
    """
    Read and explore the contents of an NPZ file.
    
    :param file_path: Path to the .npz file
    """
    try:
        # Load the NPZ file
        npz_data = np.load(file_path, allow_pickle=True)
        
        # Create a dictionary to store the arrays
        data_dict = {}
        
        # Populate the dictionary with arrays
        for array_name in npz_data.files:
            data_dict[array_name] = npz_data[array_name]
        
        return data_dict    

    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except Exception as e:
        print(f"Error reading NPZ file: {e}")
    finally:
        # Close the file if it was opened
        if 'npz_data' in locals():
            npz_data.close()

def polynomial_smooth(x, y, degree=3):
    """
    Fit a polynomial of given degree to the data and return the smoothed values.
    
    Parameters:
    x (array-like): The independent variable values.
    y (array-like): The dependent variable values.
    degree (int): The degree of the polynomial for smoothing (default is 3).
    
    Returns:
    y_smooth (array): The smoothed dependent variable values.
    poly_func (poly1d): The polynomial function used for smoothing.
    """
    # Polynomial fitting
    coefficients = np.polyfit(x, y, degree)
    
    # Create a polynomial function from the coefficients
    poly_func = np.poly1d(coefficients)
    
    # Generate smoothed y values
    y_smooth = poly_func(x)
    
    return y_smooth, poly_func

def gaussian_kernel(x, xi, bandwidth):
    """
    Gaussian kernel function.
    
    Parameters:
    x (float): The point at which the kernel is evaluated.
    xi (float): The center of the kernel.
    bandwidth (float): The bandwidth of the kernel (controls the spread).
    
    Returns:
    float: The value of the Gaussian kernel at x.
    """
    return np.exp(-0.5 * ((x - xi) / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))

def local_gaussian_smooth(x, y, bandwidth=0.1):
    """
    Perform local Gaussian smoothing on the data.
    
    Parameters:
    x (array-like): The independent variable values.
    y (array-like): The dependent variable values.
    bandwidth (float): The bandwidth of the Gaussian kernel (default is 1.0).
    
    Returns:
    y_smooth (array): The smoothed dependent variable values.
    """
    x = np.array(x)
    y = np.array(y)
    y_smooth = np.zeros_like(y)
    
    for i, xi in enumerate(x):
        # Compute the Gaussian weights for all points
        weights = gaussian_kernel(x, xi, bandwidth)
        
        # Normalize the weights
        weights /= np.sum(weights)
        
        # Compute the smoothed value as a weighted average
        y_smooth[i] = np.sum(weights * y)
    
    return y_smooth


def moving_average_smooth(x, y, window_size=5):
    """
    Apply moving average smoothing to the data.
    
    Parameters:
    x (array-like): The independent variable values.
    y (array-like): The dependent variable values.
    window_size (int): The size of the sliding window for averaging (default is 5).
    
    Returns:
    y_smooth (array): The smoothed dependent variable values.
    """
    y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='same')
    return y_smooth

# NOTE: Plot the mean velocity profile at different x-stations
def Plot_U_mean(file_path: str, x_stations: tuple = [1,2,3,4,5]):
    '''
    parameters:
        file_path: string, path to the NPZ file
    '''
    # NOTE: Read the saved NPZ file
    data_dict = read_npz_file(file_path)
    x = data_dict['x']
    y = data_dict['y']
    ym = (y[1:] + y[:-1])/2
    U = data_dict['Umean']
    for x_st in x_stations:
        idx = np.where(x >= x_st)[0]
        if len(idx) > 0:
            idx = idx[0]
            plt.plot(U[idx, 1:-1]*0.3 + x_st, ym, label=f"x = {x}", color='k')
    plt.show()

# NOTE: Plot the mean velocity profile at different x-stations (locally)
def Plot_U_mean_local(file_path: str, x_stations: tuple = [1,2,3,4,5]):
    '''
    parameters:
        file_path: string, path to the NPZ file
    '''
    # NOTE: Read the saved NPZ file
    data_dict = read_npz_file(file_path)
    x = data_dict['x']
    y = data_dict['y']
    ym = (y[1:] + y[:-1])/2
    U = data_dict['Umean']
    for x_st in x_stations:
        idx = np.where(x >= x_st)[0]
        if len(idx) > 0:
            idx = idx[0]
            plt.plot(U[idx, 1:-1]/np.max(U[idx, 1:-1]), ym, color='k')
            plt.title(f"x = {x_st:.3f}")
            plt.xlabel(r"$U/U_e$")
            plt.ylabel(r"$y/Y$")
        plt.show()

# NOTE: Plot the mean velocity profile in plus units 
def Plot_U_mean_plus(file_path: str, x_stations: tuple | np.ndarray = [1,2,3,4,5], keppa=0.412, A=5.29, Re=670):
    '''
    parameters:
        file_path: string, path to the NPZ file

    optional:
        x_stations: tuple or np.ndarray, x-stations to plot
        keppa : float, von Karman constant
        A : float, constant in the log-law

    '''
    nu = nu_dict[Re]
    # NOTE: Read the saved NPZ file
    data_dict = read_npz_file(file_path)
    x = data_dict['x']
    y = data_dict['y']
    ym = (y[1:] + y[:-1])/2
    U = data_dict['Umean']
    # NOTE: Calculate the wall shear stress and friction velocity
    tau_x = nu * (U[:,1]-U[:,0]) / (2*ym[0])
    utau_x = np.sqrt(abs(tau_x))
    fig, ax = plt.subplots()
    for x_st in x_stations:
        idx = np.where(x >= x_st)[0]
        if len(idx) > 0:
            idx = idx[0]
            yplus = ym * utau_x[idx] / nu
            Uplus = U[idx, 1:-1] / utau_x[idx]
            ax.semilogx(yplus, Uplus, label=f"x = {x[idx]:.3f}")
    # Plot the log-law
    yplus = np.logspace(1, 4)
    Up = np.log(yplus) / keppa + A
    ax.plot(yplus, Up, label="Log-law", color='k', linestyle='--')
    ax.set_xlabel(r"$y^+$")
    ax.set_ylabel(r"$U^+$")
    plt.legend()
    plt.show()

# NOTE: For convenience, we will use the date as the folder name that will be used to save the stats
# Format the date into YYYYMMDD
def get_current_date():
    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d")
    return formatted_date

def read_npz_stats(file_path: str, nu: float, filter: bool = False):
    '''
    parameters:
        file_path: string, path to the NPZ file

        edge_detect_method: string, method to detect the edge of the boundary layer. Options are 'F' for -y*Omega_z and 'O' for Omega_z
            F: Threshold on the F = -y*Omega_z (Default threshold is 0.02)
            O: Threshold on the Omega_z (Default threshold is 0.002)
            C: Spalart and Coleman's method (Integrate to infinity)

        filter: bool, apply Savitzky-Golay filter to the data
        filter_VP: bool, apply Savitzky-Golay filter to the V and P data

        mid: bool, whether to use vor_z at the mid-point of the cell or not

    return:
        Stats_BL: dataclass, containing all the boundary layer parameters
    '''

    # NOTE: Read the saved NPZ file
    data_dict = read_npz_file(file_path)
    x = data_dict['x']
    y = data_dict['y']
    z = data_dict['z']
    xm = (x[1:] + x[:-1])/2
    ym = (y[1:] + y[:-1])/2 # This is also valid for stretching grids
    zm = (z[1:] + z[:-1])/2
    U = data_dict['Umean']
    V = data_dict['Vmean']
    P = data_dict['Pmean']
    # NOTE: Get Nx and Ny (cell center counts)
    Nx = len(xm)
    Ny = len(ym)
    # Compute grid edges
    xg = np.concatenate([
        [x[0] - (xm[0] - x[0])], 
        xm, 
        [x[-1] + x[-1] - xm[-1]]
    ])
    
    yg = np.concatenate([
        [y[0] - (ym[0] - y[0])], 
        ym, 
        [y[-1] + y[-1] - ym[-1]]
    ])

    # NOTE: Calculate the wall shear stress and friction velocity
    tau_x = nu * (U[:,1]-U[:,0]) / (2*ym[0])

    # NOTE: Calculate local dx+, dy+ and dz+
    dx = np.diff(x)[0]
    dy = np.diff(y)
    dz = np.diff(z)[0]

    # NOTE: Pressure gradients
    dPedx = np.zeros((Nx,), dtype=float)
    # WARNING: Apply a smoothing filter to the pressure gradient
    if filter:
        # Apply Savitzky-Golay filter
        Psm = local_gaussian_smooth(xg, P[:, 1], bandwidth=0.2)
        for i in range(Nx-1):
            # WARNING: Use the pressure gradient at the wall
            dPedx[i] = (Psm[i+1] - Psm[i-1]) / (2*dx)
    else:
        for i in range(Nx-1):
            # WARNING: Use the pressure gradient at the wall
            dPedx[i] = (P[i+1, 1] - P[i-1, 1]) / (2*dx)

    # NOTE: Pressure-gradient based velocity scale
    up = np.sign(dPedx) * (abs(dPedx) * nu )**(1/3)

    return x, y, z, xm, ym, zm, U, V, P, tau_x, up, xg, yg

