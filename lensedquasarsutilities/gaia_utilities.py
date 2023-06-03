import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia


from lensedquasarsutilities.formatting import get_J2000_name


def get_positionangles_separations(ra, dec, searchradiusarcsec, name=None):

    """
    Queries Gaia for detections within `searchradiusarcsec` arcseconds of the given coordinates, 
    and calculates the separation and position angle between the first two detections.

    Parameters
    ----------
    ra : float
        The right ascension of the target coordinates in degrees.
    dec : float
        The declination of the target coordinates in degrees.
    searchradiusarcsec : float
        The search radius around the given coordinates in arcseconds.
    name : str, optional
        The name of the target. If None, a J2000 name is generated from the coordinates.

    Returns
    -------
    dict
        A dictionary with the following keys:
            - 'name': The name of the target.
            - 'separation': The separation between the first two detections in arcseconds.
            - 'position_angle': The position angle from the first detection to the second in degrees.
            - 'magnitudes': A list of dictionaries, each containing the ra, dec, and g, bp, rp magnitudes for a detection.
    None
        If less than two detections are found.

    Raises
    ------
    Exception
        If there is an error querying Gaia.
    """

    # If no name is provided, generate a J2000 name
    if name is None:
        name = get_J2000_name(ra, dec)

    # Set the Gaia data release
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

    # Set the query area
    radius = u.Quantity(searchradiusarcsec, u.arcsec)

    # Create a SkyCoord object
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')

    # Query Gaia
    result = Gaia.query_object_async(coordinate=coord, radius=radius)

    # Check if there are at least two detections
    if len(result) >= 2:
        # Get the SkyCoord objects of the first two detections
        coord1 = SkyCoord(ra=result[0]['ra'], dec=result[0]['dec'], unit=(u.degree, u.degree))
        coord2 = SkyCoord(ra=result[1]['ra'], dec=result[1]['dec'], unit=(u.degree, u.degree))

        # Calculate the separation in arcseconds
        separation = coord1.separation(coord2).arcsecond

        # Calculate the position angle
        position_angle = coord1.position_angle(coord2).degree

        # Retrieve the magnitudes
        magnitudes = []
        for i in range(2):
            magnitudes.append({
                'ra': result[i]['ra'],
                'dec': result[i]['dec'],
                'g': result[i]['phot_g_mean_mag'],
                'bp': result[i]['phot_bp_mean_mag'],
                'rp': result[i]['phot_rp_mean_mag']
            })

        return {
            'name': name,
            'separation': separation,
            'position_angle': position_angle,
            'magnitudes': magnitudes
        }
    else:
        return None



