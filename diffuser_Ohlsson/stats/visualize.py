import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# --- Set up paths and constants ---
from data_processing_utils import find_k_y_values, import_path
WM_DATA_PATH = import_path()  # Ensure the BFM_PATH and subdirectories are in the system path
datapath = os.path.join(WM_DATA_PATH, 'data')
statspath = os.path.join(WM_DATA_PATH, 'diffuer_Ohlsson', 'stats')
UP_FRAC = 0.2
DOWN_FRAC = 0.01
Re = 10000 # Bulk Reynolds number
nu = 1.0 / Re  # Kinematic viscosity

FILE_FORMAT = lambda x: statspath + f'/profiles/C13.2_z0500_x{x}_KTH_DNS.txt'

L = 15
x_all = np.array([-2, 0, 2, 4, 6, 8, 10, 12, 14])

Cf = np.loadtxt(statspath + "/cf.dat", delimiter=',')
dpdx = np.loadtxt(statspath + "/dpdx.dat", delimiter=',')
# dpdx is 100 times the real value
dpdx[:,1] = dpdx[:,1] / 100

dpdx_interpolated = np.interp(x_all/L, dpdx[:, 0], dpdx[:, 1])
Cf_interpolated = np.interp(x_all/L, Cf[:, 0], Cf[:, 1])

# Test interpolation
plt.plot(dpdx[:, 0], dpdx[:, 1], label='Original dpdx')
plt.plot(x_all/L, dpdx_interpolated, 'x', label='Interpolated dpdx')
plt.show()
plt.plot(Cf[:, 0], Cf[:, 1], label='Original Cf')
plt.plot(x_all/L, Cf_interpolated, 'x', label='Interpolated Cf')
plt.show()

# Initialize lists to collect data
all_inputs_data = []
all_output_data = []
all_flow_type_data = []
all_unnormalized_inputs_data = []

for i, x in enumerate(x_all):

    profile = np.loadtxt(FILE_FORMAT(x), comments='#', usecols=(1, 3, 5))

    y = profile[:,0]
    U = profile[:,1]
    W = profile[:,2]
    Umag = np.sqrt(profile[:, 1]**2 + profile[:, 2]**2)
    tauw_i = Cf_interpolated[i] / 2
    utau_i = np.sqrt(tauw_i)
    dpdx_i = dpdx_interpolated[i]
    up_i = np.sign(dpdx_i) * (abs(dpdx_i * nu)) ** (1 / 3)

    # We only care about the lower half of the domain
    # Take max U and find the y values
    max_index = np.argmax(U)
    delta = y[max_index] # boundary layer thickness

    # Plot the data for visual inspection
    plt.plot(y, Umag, label='Umag')
    plt.plot(y, U, label='U')
    plt.plot(y, W, label='W')
    plt.xlabel("y")
    plt.ylabel("Velocity Component")
    plt.axvline(y[max_index], color='r', linestyle='--', label='y_max')
    plt.title(f'x = {float(x)/L:.3f}')
    plt.legend()
    plt.show()
