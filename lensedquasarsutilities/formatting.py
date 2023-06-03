from astropy.coordinates import SkyCoord
from astropy import units as u

def get_J2000_name(ra_deg, dec_deg, N=4):
    """    
    Formatter converting coordinates to a name.

        ra_deg: float, RA in degrees
        dec_deg: float, DEC in degrees
        N: integer, default 4, number of digits in each coordinate in the name.

    returns:
        name: string, the formatted name.

    """


    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)

    # Format the coordinates in hours for RA and degrees for DEC, with zero-padding
    ra_formatted = coord.ra.to_string(u.hour, sep='', precision=2, pad=True)[:N]
    dec_formatted = coord.dec.to_string(u.deg, sep='', precision=2, alwayssign=True, pad=True)[:N+1]

    # Combine the formatted RA and DEC into the 'JHHMM-DDMM' format
    name = f"J{ra_formatted}{dec_formatted}"
    return name


def format_coord_deg2hmsdd(ra, dec):
    # from ra, dec in degrees to the usual HH:MM:SS DD:MM:SS
    coo = SkyCoord(ra*u.deg, dec*u.deg)
    rastr = coo.ra.to_string(unit=u.hourangle, sep=":", precision=2, pad=True)
    decstr = coo.dec.to_string(sep=":", precision=2, alwayssign=True, pad=True)
    return rastr, decstr
