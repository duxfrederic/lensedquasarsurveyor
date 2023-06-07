from lensedquasarsutilities.legacysurvey_utilities import download_legacy_survey_cutout
from lensedquasarsutilities.panstarrs_utilities import download_panstarrs_cutout
from lensedquasarsutilities.hsc_utilities import download_hsc_cutout
from lensedquasarsutilities import config


def get_cutouts_file(ra, dec, size, survey, downloaddir=None, filename=None, verbose=False):
    """
    Downloads a fits file containing a hdulist:
      pos 0: dummy hdu
      pos 1: band 1, stack
      pos 2: weights (noisemap) of stack of band 1
      pos 3: band 2, stack
      pos 4: weights (noisemap) of stack of band 2
      etc.
    THE HDUS NEED TO CONTAIN SOME WCS INFORMATION SO THE STAMP EXTRACTION CAN WORK


    :param ra: float, degrees
    :param dec: float, degrees
    :param size: float, arcsec
    :param survey: 'legacysurvey' or 'panstarrs' or ..........
    :param downloaddir: string, where to put the data
    :param filename: optional, give a special name to the resulting fits file
    :param verbose: bool, default False
    :return: the path to where the fits file was saved.
    """
    survey = survey.lower()
    if survey == 'legacysurvey':
        download_func = download_legacy_survey_cutout
    elif survey == 'panstarrs':
        download_func = download_panstarrs_cutout
    elif survey == 'hsc':
        download_func = download_hsc_cutout

    else:
        message = f"Haven't implemented your choice of survey, the ones I have are {config.supported_surveys}"
        raise NotImplementedError(message)

    savepath = download_func(ra, dec, size, downloaddir=downloaddir, filename=filename, verbose=verbose)
    return savepath


if __name__ == '__main__':
    raj, decj = 159.3665, 0.3057

    print(get_cutouts_file(raj, decj, 100, downloaddir='/tmp/', survey='hsc', verbose=True))

