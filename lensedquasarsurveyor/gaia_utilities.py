import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astropy.table import vstack

from lensedquasarsurveyor.formatting import get_J2000_name


def find_gaia_stars_around_coords(ra, dec, radiusarcsec):
    """

    :param ra:  float, degrees
    :param dec:  float, degrees
    :param radiusarcsec:  float, arcseconds
    """
    coord = SkyCoord(ra=ra, dec=dec,
                     unit=(u.degree, u.degree),
                     frame='icrs')
    radius = radiusarcsec * u.arcsec
    r = Gaia.query_object_async(coordinate=coord, radius=radius)

    return r


def get_similar_stars(ra, dec, threshold_distance, mag_estimate=None, verbose=False, toobright=16.5):
    """

    :param ra:  float, degrees
    :param dec:  float, degrees
    :param threshold_distance:  float, arcseconds
    :param mag_estimate: float, the typical magnitude of the stars we are looking for.
    :param verbose: bool, default False
    :param toobright: float, stars under this g-mag are not included.
    :return:  a list of RAs and a list of Decs of the stars we found.

    """
    available = find_gaia_stars_around_coords(ra, dec, threshold_distance)
    available.sort('dist')

    # so, if we do not provide a magnitude estimate, then we query gaia around our coordinates:
    # hopefully we will find the magnitude of at least the brightest image of our lensed quasar.
    if not mag_estimate:
        veryclose = available[available['dist'] < 2. * u.arcsec.to('degree')]  # at most 2 arcsec away

        if len(veryclose) == 0:
            raise RuntimeError('No magnitude estimate given, and no gaia detection to get it from.')
        veryclose.sort('phot_g_mean_mag')
        mag_estimate = veryclose[0]['phot_g_mean_mag']

    # ok, now we just look at what else we have:
    available = available[available['dist'] > 3. * u.arcsec.to('degree')]

    if verbose:
        print(f"Found {len(available)} potential stars.")
        print(f"Magnitudes: {[round(e['phot_g_mean_mag'],2) for e in available]}.")

    nottoobright = available[(toobright < available['phot_g_mean_mag'])]
    if len(nottoobright) == 0:
        return 0., ([], [])

    good = nottoobright[(nottoobright['phot_g_mean_mag'] < mag_estimate)]
    decent = nottoobright[(nottoobright['phot_g_mean_mag'] >= mag_estimate)]
    # say we are satisfied with one good, or a few faint ones.
    score = 0.3 * len(decent) + len(good)
    found = vstack([good, decent])

    if verbose:
        print(f"We have {len(good)} stars with {mag_estimate} > mag > {toobright}, and {len(decent)} fainter ones.")

    # just return the coordinates
    ras = [g['ra'] for g in found]
    decs = [g['dec'] for g in found]

    return score, (ras, decs)


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


if __name__ == "__main__":
    # ras, decs = get_similar_stars(320.6075, -16.357, 50, verbose=True)
    ra, dec = 40.0768, -2.1474
    threshold_distance = 100
    score, (ras, decs) = get_similar_stars(ra, dec, threshold_distance, mag_estimate=20.14, verbose=False, toobright=17.3)