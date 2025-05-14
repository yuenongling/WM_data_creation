import numpy as np

#########################################
Cp_data = np.loadtxt('./data_ref_cp.csv', delimiter=',')
x_cp = Cp_data[:, 0]
Cp = Cp_data[:, 1]

contour_file = './0012.dat'
contour_coords = np.loadtxt(
        contour_file
    )
contour_coords = contour_coords[contour_coords[:,1]>0]

contour_cooods_cp = np.interp(x_cp, contour_coords[:, 0], contour_coords[:, 1])
contour_coords = np.stack((x_cp, contour_cooods_cp), axis=1)

# 2. Calculate the arc length parameter 's'
diffs = np.diff(contour_coords, axis=0)      # Differences between consecutive points
segment_lengths = np.linalg.norm(diffs, axis=1) # Length of each line segment
# Cumulative arc length, starting at s=0 for the first point
s = np.insert(np.cumsum(segment_lengths), 0, 0) # s will have shape (N,)

# 3. Calculate dp/ds using numpy.gradient
# np.gradient computes the derivative numerically, handling spacing defined by 's'
dpds = np.gradient(Cp, s, axis=0) * 2

nu = 1 / 5000
up = np.sign(dpds) * (np.abs(dpds * nu)) ** (1/3)

# # Along the airfoil surface, calculate dpds
# pressure_gradient = np.gradient(interpolated_P, s_distance, axis=0)

#
# # Project the velocity onto the local tangent
# corrected_num_points = num_points // 2 - 1
# velocity = []
# for i in range(0, corrected_num_points):
#     # Each point has num_points points along the local tangent
#     # from i*num_points to (i+1)*num_points are the points along this local tangent
#     tangent = tangents_unit[i]
#     velocity_vector = np.array([interpolated_U[i*Num_points_along_normal :(i+1)*Num_points_along_normal ],
#                             interpolated_V[i*Num_points_along_normal :(i+1)*Num_points_along_normal ]]).T
#     # Project velocity onto tangent
#     projection_velocity = np.dot(velocity_vector, tangent)
#     velocity.append(projection_velocity)
#
#     plt.plot(s_distance[i], projection_velocity, 'r-', alpha=0.5, label='Projected Velocity')
#
#

import pickle as pkl
with open('./NACA0012_Upper_NoLE_dpds.pkl', 'wb') as f:
    pkl.dump({
            'x': x_cp,
            'dpds': dpds,
            'up': up,
    }, f)

