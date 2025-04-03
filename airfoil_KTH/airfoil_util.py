import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Union


def naca_airfoil_curvature(x_c: Union[float, List[float], np.ndarray], airfoil_code: str) -> Union[Dict[str, float], Dict[str, Union[List[float], np.ndarray]]]:
    """
    Calculate the local curvature and radius of curvature for a NACA 4-digit airfoil.
    
    Parameters:
    -----------
    x_c : float or List[float]
        Position(s) along the chord line (0 to 1). Can be a single value or a list of values.
    airfoil_code : str
        NACA 4-digit airfoil designation (e.g., '4412', '0012')
    
    Returns:
    --------
    Dict[str, float] or Dict[str, List[float]]
        Dictionary containing 'curvature' and 'radius' values at the specified x/c position(s).
        If x_c is a list, returns lists of values.
    """
    # Validate airfoil code
    if not (len(airfoil_code) == 4 and airfoil_code.isdigit()):
        raise ValueError("Airfoil code must be a 4-digit NACA designation")
    
    # Parse the NACA 4-digit code
    m = int(airfoil_code[0]) / 100  # Maximum camber
    p = int(airfoil_code[1]) / 10   # Position of maximum camber
    t = int(airfoil_code[2:]) / 100  # Maximum thickness
    
    # Handle list of x/c values
    if isinstance(x_c, (list, tuple, np.ndarray)):
        # Validate all inputs
        for x in x_c:
            if not 0 <= x <= 1:
                raise ValueError(f"x/c value {x} must be between 0 and 1")
        
        # Calculate results for each x/c value
        curvatures = []
        radii = []
        
        for x in x_c:
            if m == 0:
                result = calculate_symmetric_curvature(x, t)
            else:
                result = calculate_cambered_curvature(x, m, p)
            
            curvatures.append(result['curvature'])
            radii.append(result['radius'])
        
        # Convert to numpy array if input was a numpy array
        if isinstance(x_c, np.ndarray):
            return {
                'curvature': np.array(curvatures),
                'radius': np.array(radii),
                'x_c': x_c
            }
        else:
            return {
                'curvature': curvatures,
                'radius': radii,
                'x_c': list(x_c)  # Include the x/c values for reference
            }
    
    # Handle single x/c value
    else:
        # Validate input
        if not 0 <= x_c <= 1:
            raise ValueError("x/c must be between 0 and 1")
        
        # For symmetric airfoil (m=0), curvature calculation is different
        if m == 0:
            return calculate_symmetric_curvature(x_c, t)
        else:
            return calculate_cambered_curvature(x_c, m, p)


def calculate_symmetric_curvature(x: float, t: float) -> Dict[str, float]:
    """
    Calculate curvature for a symmetric NACA airfoil (00XX series).
    
    Parameters:
    -----------
    x : float
        Position along the chord (0 to 1)
    t : float
        Maximum thickness as a fraction of chord
    
    Returns:
    --------
    Dict[str, float]
        Dictionary with 'curvature' and 'radius' values
    """
    # Avoid the singularity at x=0 (leading edge)
    if x < 0.001:
        x = 0.001
    
    # Calculate the first derivative of the thickness function
    dyt_dx = (t/0.2) * (0.2969/(2*np.sqrt(x)) - 0.1260 - 2*0.3516*x + 3*0.2843*x**2 - 4*0.1015*x**3)
    
    # Calculate the second derivative of the thickness function
    d2yt_dx2 = (t/0.2) * (-0.2969/(4*x**(3/2)) - 2*0.3516 + 6*0.2843*x - 12*0.1015*x**2)
    
    # Calculate curvature using the formula: |y''| / (1 + (y')^2)^(3/2)
    curvature = np.abs(d2yt_dx2) / (1 + dyt_dx**2)**(3/2)
    
    # Radius of curvature is the reciprocal of the curvature
    radius = 1 / curvature if curvature > 0 else float('inf')
    
    return {'curvature': curvature, 'radius': radius}


def calculate_cambered_curvature(x: float, m: float, p: float) -> Dict[str, float]:
    """
    Calculate curvature for a cambered NACA airfoil.
    
    Parameters:
    -----------
    x : float
        Position along the chord (0 to 1)
    m : float
        Maximum camber as a fraction of chord
    p : float
        Position of maximum camber as a fraction of chord
    
    Returns:
    --------
    Dict[str, float]
        Dictionary with 'curvature' and 'radius' values
    """
    # Calculate the first derivative of the camber line
    if x <= p:
        dyc_dx = (2*m / p**2) * (p - x)
        d2yc_dx2 = -2*m / p**2
    else:
        dyc_dx = (2*m / (1-p)**2) * (p - x)
        d2yc_dx2 = -2*m / (1-p)**2
    
    # Calculate curvature using the formula: |y''| / (1 + (y')^2)^(3/2)
    curvature = np.abs(d2yc_dx2) / (1 + dyc_dx**2)**(3/2)
    
    # Radius of curvature is the reciprocal of the curvature
    radius = 1 / curvature if curvature > 0 else float('inf')
    
    return {'curvature': curvature, 'radius': radius}


def naca_shape(airfoil_code: str, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the coordinates for a NACA 4-digit airfoil.
    
    Parameters:
    -----------
    airfoil_code : str
        NACA 4-digit airfoil designation
    num_points : int, optional
        Number of points to generate (default: 100)
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple containing (x_upper, y_upper, x_lower, y_lower) coordinates
    """
    # Parse the NACA 4-digit code
    m = int(airfoil_code[0]) / 100
    p = int(airfoil_code[1]) / 10
    t = int(airfoil_code[2:]) / 100
    
    # Generate x-coordinates with cosine spacing for better resolution at leading edge
    beta = np.linspace(0, np.pi, num_points)
    x = 0.5 * (1 - np.cos(beta))
    
    # Initialize arrays for upper and lower surface coordinates
    x_upper = np.zeros(num_points)
    y_upper = np.zeros(num_points)
    x_lower = np.zeros(num_points)
    y_lower = np.zeros(num_points)
    
    for i in range(num_points):
        # Thickness distribution
        yt = (t/0.2) * (0.2969*np.sqrt(x[i]) - 0.1260*x[i] - 
                         0.3516*x[i]**2 + 0.2843*x[i]**3 - 0.1015*x[i]**4)
        
        # Camber line and its slope
        if m > 0:
            if x[i] <= p:
                yc = (m / p**2) * (2*p*x[i] - x[i]**2)
                dyc_dx = (2*m / p**2) * (p - x[i])
            else:
                yc = (m / (1-p)**2) * ((1-2*p) + 2*p*x[i] - x[i]**2)
                dyc_dx = (2*m / (1-p)**2) * (p - x[i])
            
            theta = np.arctan(dyc_dx)
            
            # Upper and lower surface coordinates
            x_upper[i] = x[i] - yt * np.sin(theta)
            y_upper[i] = yc + yt * np.cos(theta)
            x_lower[i] = x[i] + yt * np.sin(theta)
            y_lower[i] = yc - yt * np.cos(theta)
        else:
            # For symmetric airfoils
            x_upper[i] = x[i]
            y_upper[i] = yt
            x_lower[i] = x[i]
            y_lower[i] = -yt
    
    return x_upper, y_upper, x_lower, y_lower


def plot_airfoil_with_curvature(airfoil_code: str, x_c_values: List[float] = None) -> None:
    """
    Plot the airfoil shape and curvature/radius at specified points.
    
    Parameters:
    -----------
    airfoil_code : str
        NACA 4-digit airfoil designation
    x_c_values : List[float], optional
        List of x/c positions to calculate and display curvature
    """
    if x_c_values is None:
        x_c_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Generate airfoil shape
    x_upper, y_upper, x_lower, y_lower = naca_shape(airfoil_code)
    
    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot airfoil shape
    ax1.plot(x_upper, y_upper, 'b-', label='Upper surface')
    ax1.plot(x_lower, y_lower, 'r-', label='Lower surface')
    ax1.set_xlabel('x/c')
    ax1.set_ylabel('y/c')
    ax1.set_title(f'NACA {airfoil_code} Airfoil Profile')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')
    
    # Calculate curvature and radius at specified points
    x_values = np.array(x_c_values)
    curvature = np.array([naca_airfoil_curvature(x, airfoil_code)['curvature'] for x in x_values])
    radius = np.array([naca_airfoil_curvature(x, airfoil_code)['radius'] for x in x_values])
    
    # Plot curvature
    ax2.plot(x_values, curvature, 'g-o')
    ax2.set_xlabel('x/c')
    ax2.set_ylabel('Curvature')
    ax2.set_title(f'NACA {airfoil_code} Local Curvature')
    ax2.grid(True)
    
    # Plot radius of curvature
    ax3.plot(x_values, radius, 'm-o')
    ax3.set_xlabel('x/c')
    ax3.set_ylabel('Radius of Curvature')
    ax3.set_title(f'NACA {airfoil_code} Radius of Curvature')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()


def generate_curvature_table(airfoil_code: str, x_c_values: List[float] = None) -> None:
    """
    Generate a table of curvature and radius values at specified x/c positions.
    
    Parameters:
    -----------
    airfoil_code : str
        NACA 4-digit airfoil designation
    x_c_values : List[float], optional
        List of x/c positions to calculate curvature and radius
    """
    if x_c_values is None:
        x_c_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    print(f"NACA {airfoil_code} Airfoil Curvature and Radius of Curvature:")
    print("-" * 60)
    print(f"{'x/c':^10}{'Curvature':^20}{'Radius of Curvature':^25}")
    print("-" * 60)
    
    for x_c in x_c_values:
        result = naca_airfoil_curvature(x_c, airfoil_code)
        print(f"{x_c:^10.3f}{result['curvature']:^20.6f}{result['radius']:^25.6f}")


# Add a convenience function to query multiple x/c positions
def query_multiple_positions(x_c_list: Union[List[float], np.ndarray], airfoil_code: str, as_dataframe: bool = False) -> Union[Dict[str, Union[List[float], np.ndarray]], object]:
    """
    Query curvature and radius at multiple x/c positions for a given airfoil.
    
    Parameters:
    -----------
    x_c_list : List[float] or numpy.ndarray
        List or array of positions along the chord (0 to 1)
    airfoil_code : str
        NACA 4-digit airfoil designation
    as_dataframe : bool, optional
        If True, returns result as a pandas DataFrame (if pandas is available)
        
    Returns:
    --------
    Dict[str, Union[List[float], numpy.ndarray]] or pandas.DataFrame
        Results containing x/c positions, curvature values, and radius values.
        If input was a numpy array, returns numpy arrays for values.
    """
    results = naca_airfoil_curvature(x_c_list, airfoil_code)
    
    if as_dataframe:
        try:
            import pandas as pd
            df = pd.DataFrame({
                'x_c': results['x_c'],
                'curvature': results['curvature'],
                'radius': results['radius']
            })
            return df
        except ImportError:
            print("Warning: pandas not available. Returning dictionary instead.")
            return results
    else:
        return results


# Example usage
if __name__ == "__main__":
    # 1. Query curvature at a specific point
    x_c = 0.4
    airfoil_code = "4412"
    result = naca_airfoil_curvature(x_c, airfoil_code)
    print(f"NACA {airfoil_code} at x/c = {x_c}:")
    print(f"Curvature: {result['curvature']:.6f}")
    print(f"Radius of curvature: {result['radius']:.6f}")
    
    # 2. Query multiple positions using a list
    x_positions = [0.1, 0.2, 0.3, 0.4, 0.5]
    multi_results = naca_airfoil_curvature(x_positions, "4412")
    print("\nMultiple positions (list) for NACA 4412:")
    for i, x in enumerate(x_positions):
        print(f"x/c = {x:.2f}: Curvature = {multi_results['curvature'][i]:.6f}, Radius = {multi_results['radius'][i]:.6f}")
    
    # 3. Query multiple positions using numpy array
    x_array = np.linspace(0.1, 0.9, 9)  # [0.1, 0.2, ..., 0.9]
    array_results = naca_airfoil_curvature(x_array, "0012")
    print("\nMultiple positions (numpy array) for NACA 0012:")
    print(f"x/c values: {x_array}")
    print(f"Curvature values: {array_results['curvature']}")
    print(f"Radius values: {array_results['radius']}")
    
    # 4. Use the convenience function (with optional pandas DataFrame output)
    print("\nUsing the convenience function:")
    df_results = query_multiple_positions([0.1, 0.3, 0.5, 0.7, 0.9], "0012", as_dataframe=True)
    print(df_results)  # Will be a DataFrame if pandas is available, otherwise a dict
    
    # 5. Using numpy array with the convenience function
    print("\nUsing numpy array with the convenience function:")
    array_results = query_multiple_positions(np.array([0.05, 0.15, 0.25, 0.35, 0.45]), "4412")
    print(f"Input type: {type(array_results['x_c'])}")
    print(f"Output type: {type(array_results['curvature'])}")
    
    # 4. Generate table of values
    print("\nNACA 4412 Curvature Table:")
    generate_curvature_table("4412")
    
    print("\nNACA 0012 Curvature Table:")
    generate_curvature_table("0012")
    
    # 5. Plot the airfoil with curvature
    plot_airfoil_with_curvature("4412")
