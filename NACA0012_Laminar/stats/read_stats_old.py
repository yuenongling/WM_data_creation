import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
# Replace with the actual path to your Ensight case file
ensight_case_file = '~/Documents/CFD/NACA0012/laminar/fine/centerline/centerline.case'

# Replace with the actual name of your velocity variable in the Ensight file
# Common names: 'U', 'Velocity', 'VELOCITY', 'vector', 'U_avg', 'UMean' etc.
# You might need to inspect the reader or loaded mesh to find the exact name.
velocity_variable_name = 'U' # <--- CHANGE THIS IF NEEDED

# Replace with the name or index of the part representing the surface
# If your data is a single block, this might not be needed.
# If it's MultiBlock, you might access like: mesh['surface_part_name']
surface_part_name_or_index = None # <--- CHANGE THIS IF NEEDED (e.g., 'wall', 0)
# --------------------

# 1. Read the Ensight data
# PyVista automatically detects the Ensight reader based on the '.case' extension
# This might return a single mesh or a MultiBlock dataset if the case has multiple parts
try:
    data = pv.read(ensight_case_file)
except Exception as e:
    print(f"Error reading file: {e}")
    print("Trying with generic EnSight reader explicitly...")
    try:
        reader = pv.get_reader(ensight_case_file)
        # You can inspect available arrays before reading fully:
        print("Available Point Data Arrays:", reader.point_array_names)
        print("Available Cell Data Arrays:", reader.cell_array_names)
        data = reader.read()
    except Exception as e2:
        print(f"Failed with explicit reader too: {e2}")
        exit()

# 2. Select the relevant mesh/surface part
if isinstance(data, pv.MultiBlock):
    print("Data is MultiBlock. Available blocks:", list(data.keys()))
    if surface_part_name_or_index is None:
        # If no specific part name given, try to find a suitable block
        # Or potentially combine blocks if the surface spans multiple parts
        # For this example, let's assume the first block is the one we want
        if len(data) > 0:
            surface_mesh = data[0]
            print(f"Using the first block: {data.get_block_name(0)}")
        else:
            print("Error: MultiBlock dataset is empty.")
            exit()
    elif isinstance(surface_part_name_or_index, str):
         if surface_part_name_or_index in data.keys():
             surface_mesh = data[surface_part_name_or_index]
             print(f"Using block: {surface_part_name_or_index}")
         else:
            print(f"Error: Block '{surface_part_name_or_index}' not found in MultiBlock dataset.")
            exit()
    elif isinstance(surface_part_name_or_index, int):
        if surface_part_name_or_index < data.n_blocks:
            surface_mesh = data[surface_part_name_or_index]
            print(f"Using block index: {surface_part_name_or_index} (Name: {data.get_block_name(surface_part_name_or_index)})")
        else:
             print(f"Error: Block index {surface_part_name_or_index} out of range.")
             exit()
    else:
        print("Error: Invalid surface_part_name_or_index specified.")
        exit()

    # If needed, convert geometry (e.g., extract surface from volume)
    # surface_mesh = surface_mesh.extract_surface() # Uncomment if needed

elif isinstance(data, pv.DataSet):
    # If it's a single dataset (e.g., PolyData, UnstructuredGrid)
    surface_mesh = data
    print("Data is a single mesh.")
else:
    print(f"Error: Loaded data type {type(data)} not directly handled in this example.")
    exit()

# 3. Access the Velocity Data
# Check if the velocity data is point data or cell data
if velocity_variable_name in surface_mesh.point_data:
    velocity_vectors = surface_mesh.point_data[velocity_variable_name]
    data_location = 'PointData'
elif velocity_variable_name in surface_mesh.cell_data:
    velocity_vectors = surface_mesh.cell_data[velocity_variable_name]
    data_location = 'CellData'
    # If velocity is on cells, you might need to convert it to points first
    # for certain operations or visualizations
    print("Warning: Velocity is on cells. Consider converting to points if needed.")
    # surface_mesh = surface_mesh.cell_data_to_point_data()
    # velocity_vectors = surface_mesh.point_data[velocity_variable_name]
    # data_location = 'PointData' # after conversion
else:
    print(f"Error: Velocity variable '{velocity_variable_name}' not found.")
    print("Available Point Data:", list(surface_mesh.point_data.keys()))
    print("Available Cell Data:", list(surface_mesh.cell_data.keys()))
    exit()

print(f"Successfully loaded velocity vectors '{velocity_variable_name}' from {data_location}.")
print(f"Velocity array shape: {velocity_vectors.shape}") # Should be (n_points/n_cells, 3)

# --- Optional: Extract U component (assuming velocity is [Ux, Uy, Uz]) ---
u_velocity_component = velocity_vectors[:, 0]
print(f"Extracted U-velocity component (shape: {u_velocity_component.shape})")
# You can add this as a separate array if needed:
# surface_mesh['U_scalar'] = u_velocity_component

# Assuming you have 'surface_mesh' and 'u_velocity_component' from the previous code

# --- Visualization ---
plotter = pv.Plotter()
plotter.add_mesh(surface_mesh, scalars=u_velocity_component, cmap='coolwarm', scalar_bar_args={'title': 'U Velocity'})
# You can also add glyphs (arrows) for velocity
# plotter.add_mesh(surface_mesh.glyph(orient='Velocity', scale='U_scalar', factor=0.1), color='blue') # If you added 'U_scalar'

#########################################
# Find normal vectors
contour_file = './0012.dat'
contour_coords = np.loadtxt(
        contour_file
    )
num_points = 10            # Number of equally spaced points desired
diffs = np.diff(contour_coords, axis=0)
segment_lengths = np.linalg.norm(diffs, axis=1)
s_original = np.insert(np.cumsum(segment_lengths), 0, 0)
total_length = s_original[-1]
s_target = np.linspace(0, total_length, num_points)
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
plt.plot(contour_coords[:, 0], contour_coords[:, 1], 'k-', label='Original Airfoil Points')
plt.scatter(interp_points[:, 0], interp_points[:, 1], c='red', s=50, zorder=5, label=f'{num_points} Spaced Points')
# Plot normals as arrows (quiver plot)
plt.quiver(interp_points[:, 0], interp_points[:, 1], normals_unit[:, 0], normals_unit[:, 1],
           color='blue', scale=20, width=0.004, label='Normal Vectors') # Adjust scale for visibility
plt.title('NACA 0012 Airfoil with Equally Spaced Points and Normals')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.axis('equal') # Crucial for correct aspect ratio
plt.grid(True)
plt.legend()
plt.show()
#########################################


# add y equals 0
# Translate and rotate the contour coordinates to match the mesh
contour_coords[:,0] -= 0.25
interp_points[:,0] -= 0.25
angle = -10
contour_coords[:,0] = contour_coords[:,0] * np.cos(np.radians(angle)) - contour_coords[:,1] * np.sin(np.radians(angle))
contour_coords[:,1] = contour_coords[:,0] * np.sin(np.radians(angle)) + contour_coords[:,1] * np.cos(np.radians(angle))
interp_points[:,0] = interp_points[:,0] * np.cos(np.radians(angle)) - interp_points[:,1] * np.sin(np.radians(angle))
interp_points[:,1] = interp_points[:,0] * np.sin(np.radians(angle)) + interp_points[:,1] * np.cos(np.radians(angle))
normals_unit[:,0] = normals_unit[:,0] * np.cos(np.radians(angle)) - normals_unit[:,1] * np.sin(np.radians(angle))
normals_unit[:,1] = normals_unit[:,0] * np.sin(np.radians(angle)) + normals_unit[:,1] * np.cos(np.radians(angle))


#
contour_coords = np.array([[contour_coords[i, 0], 0, contour_coords[i, 1]] for i in range(len(contour_coords))])
contour_polydata = pv.lines_from_points(contour_coords, close=False) # Set close=True if it's a closed contour
# Add the contour line
plotter.add_mesh(
    contour_polydata,
    color='red',       # Choose a distinct color
    line_width=5,      # Make the line thicker
    render_lines_as_tubes=True, # Optional: makes lines look smoother
    name='contour_line' # Give it a name
)

###########################
arrow_length_factor = 0.05 # You might need to adjust this value
points_3d = np.array([[interp_points[i, 0], 0, interp_points[i, 1]] for i in range(len(interp_points))])
normals_3d = np.array([[normals_unit[i, 0], 0, normals_unit[i, 1]] for i in range(len(normals_unit))])
contour_3d = np.array([[contour_coords[i, 0], 0, contour_coords[i, 1]] for i in range(len(contour_coords))])

# Create point cloud
point_cloud = pv.PolyData(points_3d)
point_cloud['original_normals'] = normals_3d

plotter.add_mesh(point_cloud, color='red', point_size=8, render_points_as_spheres=True, label=f'{num_points} Points')

# Glyphs for original normals
glyphs_original = point_cloud.glyph(
    orient='original_normals',  # Use the vector data named 'original_normals'
    scale=False,                # Don't scale by vector magnitude (already unit)
    factor=arrow_length_factor
)
plotter.add_mesh(glyphs_original, color='blue', label='Original Normals')
###########################



plotter.show_axes()
plotter.enable_anti_aliasing() # Makes lines look smoother
plotter.show()
