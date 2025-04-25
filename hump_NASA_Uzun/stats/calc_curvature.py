import numpy as np

def local_radius_of_curvature(points):
    """
    Calculates the local radius of curvature for a series of 2D points.

    Args:
        points: A list of tuples or lists, where each inner element represents
                the (x, y) coordinates of a point.

    Returns:
        A list of radius of curvature values for each point. Returns None for
        the first and last points as the three-point method requires neighbors.
    """
    n = len(points)
    if n < 3:
        return [None] * n

    radii = [None] * n

    for i in range(1, n - 1):
        p1 = np.array(points[i - 1])
        p2 = np.array(points[i])
        p3 = np.array(points[i + 1])

        # Calculate the lengths of the sides of the triangle
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)

        # Calculate the area of the triangle using Heron's formula
        s = (a + b + c) / 2
        area_squared = s * (s - a) * (s - b) * (s - c)

        # Avoid division by zero if the points are collinear
        if area_squared <= 0:
            radii[i] = float('inf')  # Indicate infinite radius for a straight line
        else:
            area = np.sqrt(area_squared)
            radius = (a * b * c) / (4 * area)
            radii[i] = radius

    return radii

points = np.loadtxt("./CoordsCleanup.dat")
radii = local_radius_of_curvature(points)

# Remove nan and None values from radii
radii = np.array(radii, dtype=float)
points_radii = np.array(points[~np.isnan(radii),0], dtype=float)
radii = radii[~np.isnan(radii)]

import matplotlib.pyplot as plt
plt.plot(points[:, 0], points[:, 1], label='Path')
plt.axis("equal")
plt.show()

plt.semilogy(points_radii, 0.12/radii, label='Radius of Curvature')
plt.show()
