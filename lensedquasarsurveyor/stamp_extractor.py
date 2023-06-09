import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.units as u
from astropy.coordinates import SkyCoord

from lensedquasarsurveyor.config import band_header_keyword


def extract_stamps(cutoutfile, ra, dec, survey, cutout_size=6):
    """

    :param cutoutfile: string or path object
    :param ra: float, degrees
    :param dec: float, degrees
    :param survey: string, necessary, so we can know through which keyword we can get the filter in the fits header.
    :param cutout_size: float, arcseconds
    :return: dictionary of numpy arrays, per filter
    """
    kwfilter = band_header_keyword[survey]
    hdulist = fits.open(cutoutfile)

    cutout_size = (cutout_size, cutout_size) * u.arcsec
    coord = SkyCoord(ra*u.deg, dec*u.deg)

    bands = set([hdulist[i].header[kwfilter] for i in range(1, len(hdulist), 2)])
    cutouts = {}
    for band in bands:
        datas, noises, wcs_headers = [], [], []
        for i in range(1, len(hdulist), 2):
            if hdulist[i].header[kwfilter] == hdulist[i+1].header[kwfilter] == band:
                wcs = WCS(hdulist[i].header)
                datacutout = Cutout2D(hdulist[i].data, coord, cutout_size, wcs=wcs, mode='partial')
                noisecutout = Cutout2D(hdulist[i+1].data, coord, cutout_size, wcs=wcs, mode='partial')
                datas.append(datacutout.data)
                noises.append(noisecutout.data)

                # let's also carry the WCS of the cutouts, will be useful for the intial guess of
                # the positions of the lensed images later.
                wcs_header = datacutout.wcs.to_header()
                wcs_header_string = wcs_header.tostring()
                wcs_headers.append(wcs_header_string)

        datas, noises = np.array(datas), np.array(noises)
        # sometimes the surveys will give us a weird floating point number, big endian type or stuff.
        # we don't want that.
        datas = datas.astype(np.float32)
        noises = noises.astype(np.float32)
        cutouts[band] = (datas, noises, wcs_headers)

    return cutouts


if __name__ == '__main__':
    from lensedquasarsurveyor.legacysurvey_utilities import download_legacy_survey_cutout
    RA, DEC = 320.6075, -16.357
    ff = download_legacy_survey_cutout(RA, DEC, 99, verbose=True)

    cutout = extract_stamps(ff, RA, DEC, 'legacysurvey', 10)
