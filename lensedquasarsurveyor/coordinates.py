import numpy as np

def spherical_to_tangent_plane(RA, Dec, RA0, Dec0):
    """
    Convert spherical coordinates (RA, Dec) to tangent plane coordinates (x, y).

    This function uses a gnomonic projection from spherical coordinates
    (Right Ascension and Declination) to Cartesian coordinates (x, y) on the
    tangent plane in arcseconds.

    Parameters:
    - RA (float): Right Ascension of the point to be converted (in degrees).
    - Dec (float): Declination of the point to be converted (in degrees).
    - RA0 (float): Right Ascension of the tangent point (in degrees).
    - Dec0 (float): Declination of the tangent point (in degrees).

    Returns:
    tuple (x, y):
    - x (float): x-coordinate on the tangent plane (in arcseconds).
    - y (float): y-coordinate on the tangent plane (in arcseconds).


    Example:
    >>> spherical_to_tangent_plane(31.1, 32.1, 31, 32)
    (a tuple of floats in arseconds, distances from (31,32) on the tangent plane)
    """

    RA_rad  = np.deg2rad(RA)
    Dec_rad = np.deg2rad(Dec)
    RA0_rad = np.deg2rad(RA0)
    Dec0_rad = np.deg2rad(Dec0)

    delta_RA = RA_rad - RA0_rad

    den = np.sin(Dec0_rad) * np.sin(Dec_rad) + np.cos(Dec0_rad) * np.cos(Dec_rad) * np.cos(delta_RA)

    x = np.cos(Dec_rad) * np.sin(delta_RA) / den
    y = (np.cos(Dec0_rad) * np.sin(Dec_rad) - np.sin(Dec0_rad) * np.cos(Dec_rad) * np.cos(delta_RA)) / den

    # radians to degrees, then multiply by 3600 for arcseconds
    return np.degrees(np.arctan(x)) * 3600, np.degrees(np.arctan(y)) * 3600

