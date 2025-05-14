import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
SHIFT = 0.25  # Shift for visualization
num_points = 40            # Number of equally spaced points desired

#########################################
# Find normal vectors
contour_file = './0012.dat'
contour_coords = np.loadtxt(
        contour_file
    )
diffs = np.diff(contour_coords, axis=0)
segment_lengths = np.linalg.norm(diffs, axis=1)
s_original = np.insert(np.cumsum(segment_lengths), 0, 0)
total_length = s_original[-1]
s_target = np.linspace(total_length*0.05, total_length*0.95, num_points)
coords_to_interp = contour_coords
s_interp_domain = s_original
interp_x = np.interp(s_target, s_interp_domain, coords_to_interp[:, 0])
interp_y = np.interp(s_target, s_interp_domain, coords_to_interp[:, 1])
interp_points = np.vstack((interp_x, interp_y)).T

# 5. Calculate Tangents at Interpolated Points
# Use np.gradient to estimate derivatives (dx/ds, dy/ds) which form the tangent vector
# Note: gradient needs consistent spacing in the independent variable, which s_target provides
tx = np.gradient(interp_points[:, 0], s_target)
ty = np.gradient(interp_points[:, 1], s_target)
tangents = np.vstack((tx, ty)).T

# Normalize tangents (optional, but good practice)
tangent_norms = np.linalg.norm(tangents, axis=1)[:, np.newaxis]
tangents_unit = tangents / tangent_norms

# 6. Calculate Normals (Rotate Tangent by 90 degrees)
# Normal n = [-ty, tx]
normals = np.vstack((-tangents_unit[:, 1], tangents_unit[:, 0])).T

# 7. Ensure Outward Normal Direction (Heuristic for NACA 0012)
# Assumes points are ordered roughly trailing edge -> top -> leading edge -> bottom -> trailing edge
# Outward normals point away from the chord line (y=0)
# If y > 0, normal's y component should be > 0
# If y < 0, normal's y component should be < 0
for i in range(num_points):
    point_y = interp_points[i, 1]
    normal_y = normals[i, 1]
    # Check top surface
    if point_y > 1e-6 and normal_y < 0:
        normals[i] *= -1 # Flip normal
    # Check bottom surface
    elif point_y < -1e-6 and normal_y > 0:
        normals[i] *= -1 # Flip normal
    # Handle points very close to y=0 (leading/trailing edge approximation) - may need manual check
    elif abs(point_y) < 1e-6:
         # At leading edge (min x), normal should point approx -x direction
         # At trailing edge (max x), normal depends on convention (often avg of top/bottom)
         # This simple heuristic might not be perfect exactly at LE/TE
         pass # Keep calculated normal for now

# 8. Normalize the Final Normals (already done implicitly by using unit tangents, but recalculate for safety)
final_normal_norms = np.linalg.norm(normals, axis=1)[:, np.newaxis]
# Avoid division by zero if a normal is somehow zero
zero_norms = final_normal_norms < 1e-9
final_normal_norms[zero_norms] = 1.0 # Prevent NaN, result will be [0,0] anyway

normals_unit = normals / final_normal_norms

# 10. Optional: Visualization
plt.figure(figsize=(10, 5))
contour_coords[:,0] -= SHIFT
interp_points[:,0] -= SHIFT
plt.plot(contour_coords[:, 0], contour_coords[:, 1], 'k-', label='Original Airfoil Points')
plt.scatter(interp_points[:, 0], interp_points[:, 1], c='red', s=50, zorder=5, label=f'{num_points} Spaced Points')
plt.axis('equal') # Crucial for correct aspect ratio

# 11. Find points to interpolate velocity/pressure data
# NOTE: Only include points in the upper half of the airfoil (y > 0)
Num_points_along_normal = 30  # Number of points along the normal direction for interpolation
MAX_NORMAL_LENGTH = 0.05  # Length of the normal line for visualization
MIN_NORMAL_LENGTH = 0.010  # Minimum length of the normal line for visualization
points_to_interp = []
s_distance = []
tangents_unit = tangents_unit[num_points//2+1:, :]
for i in range(num_points//2+1, num_points):
    print(f"Point {i+1}: X={interp_points[i, 0]:.4f}, Y={interp_points[i, 1]:.4f}, Normal={normals_unit[i]}")
    Current_length = (interp_points[i, 0]+SHIFT) / (np.max(np.abs(interp_points[:, 0])+SHIFT)) * MAX_NORMAL_LENGTH + MIN_NORMAL_LENGTH
    points_along_normal = np.linspace(0, 1, Num_points_along_normal)[:, np.newaxis] * normals_unit[i] * Current_length + interp_points[i]
    s_distance.append(
        np.linalg.norm(points_along_normal - interp_points[i], axis=1)
        )
    points_to_interp.append(
        points_along_normal
    )
    plt.plot(points_along_normal[:, 0], points_along_normal[:, 1], 'g-', alpha=0.8, label='Normal Line')
plt.show()
# s_distance = np.concatenate(s_distance)

#########################################

# --- Load velocity/pressure data and interpolate ---
from scipy.interpolate import griddata
data = np.loadtxt('./NACA0012_Upper_NoLE.csv', delimiter=',', skiprows=1)
P = data[:, 0]  # Assuming P is the first column
U = data[:, 1]  # Assuming U is the third column
V = data[:, 2]  # Assuming U is the third column
X = data[:, 3]  # Assuming X is the fourth column
Y = data[:, 4]  # Assuming Y is the fifth column

points_to_interp = np.array(points_to_interp).reshape(-1, 2)
plt.plot(points_to_interp[:, 0], points_to_interp[:, 1], 'ro', markersize=2, label='Interpolation Points')
plt.show()

plt.scatter(X,Y);plt.scatter(points_to_interp[:, 0], points_to_interp[:, 1],color='red');plt.show()

interpolated_U = griddata(
    (X, Y), U,
    (points_to_interp[:, 0], points_to_interp[:, 1]),
    method='linear'
)
interpolated_V = griddata(
    (X, Y), V,
    (points_to_interp[:, 0], points_to_interp[:, 1]),
    method='linear'
)

# # Local tangent 
# for i in range(len(interp_points)):
#     tangent = tangents_unit[i]
#     plt.quiver(interp_points[i, 0], interp_points[i, 1],
#                tangent[0], tangent[1], color='blue', scale=20, width=0.004)
#     plt.quiver(interp_points[i, 0], interp_points[i, 1],
#                normals_unit[i, 0], normals_unit[i, 1], color='green', scale=20, width=0.004)
# plt.show()

# Project the velocity onto the local tangent
corrected_num_points = num_points // 2 - 1
velocity = []
for i in range(0, corrected_num_points):
    # Each point has num_points points along the local tangent
    # from i*num_points to (i+1)*num_points are the points along this local tangent
    tangent = tangents_unit[i]
    velocity_vector = np.array([interpolated_U[i*Num_points_along_normal :(i+1)*Num_points_along_normal ],
                            interpolated_V[i*Num_points_along_normal :(i+1)*Num_points_along_normal ]]).T
    # Project velocity onto tangent
    projection_velocity = np.dot(velocity_vector, tangent)
    velocity.append(projection_velocity)

    plt.plot(s_distance[i], projection_velocity, 'r-', alpha=0.5, label='Projected Velocity')


interpolated_P = griddata(
    (X, Y), P,
    (points_to_interp[:, 0], points_to_interp[:, 1]),
    method='linear'
)
P_to_save = []
for i in range(0, corrected_num_points):
    P_to_save.append(
            interpolated_P[i*Num_points_along_normal :(i+1)*Num_points_along_normal]
    )

import pickle as pkl
with open('./NACA0012_Upper_NoLE_data.pkl', 'wb') as f:
    pkl.dump({
             'U': velocity,
             's_dist': s_distance,
            'P': P_to_save,
             'x': interp_points[num_points//2+1:, 0]+SHIFT,
             'y': interp_points[num_points//2+1:, 1],
    }, f)
