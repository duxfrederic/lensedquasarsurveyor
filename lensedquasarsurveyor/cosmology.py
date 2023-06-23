from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


def get_physical_separation_pair(ang_sep_arcsec, redshift, H0=70, Omega0=0.3):
    """
    get the physical separation of a quasar pair given their angular separation on the sky and their redshift.
    :param ang_sep_arcsec: float, angular separation in arcsec
    :param redshift: float, cosmological redshift z
    :param H0: float, fiducial value for the cosmology, default 70.0
    :param Omega0: float, fiducial value for the cosmology, default 0.3
    :return: astropy unit quantity, in kiloparsec
    """
    cosmo = FlatLambdaCDM(H0=H0, Om0=Omega0)

    z = redshift

    ang_sep_rad = ang_sep_arcsec * u.arcsec.to('radian')

    d_A = cosmo.angular_diameter_distance(z)

    sep = d_A * ang_sep_rad

    sep = sep.to(u.kiloparsec)
    return sep


if __name__ == '__main__':
    print(get_physical_separation_pair(0.90, 3.09, H0=70, Omega0=0.3))
