"""
Here we download all the legacy survey available at a certain location, and save it in a fits file containing
pairs of data, noisemap.
"""

import urllib.request
from pathlib import Path
import numpy as np
from astropy.io import fits

from lensedquasarsutilities.formatting import get_J2000_name


def download_legacy_survey_cutout(ra, dec, size, downloaddir=None, filename=None, verbose=False):
    """
    Legacy survey downloader

    :param ra: degrees
    :param dec: degrees
    :param size: arcsec
    :param downloaddir: string, where to put the data
    :param filename: optional, give a special name to the resulting fits file
    :return: savepath where the data was saved
    """
    if not filename:
        filename = f"cutouts_legacy_survey_{get_J2000_name(ra, dec)}_size_{size:.0f}.fits"
    if not downloaddir:
        downloaddir = '.'

    # the pixel size in legacy survey cutouts:
    ls_pixel_size = 0.262
    # convert to pixels, as the API wants pixels.
    size_pix = int(size / ls_pixel_size)

    template = "https://www.legacysurvey.org/viewer/cutout.fits?ra={RA}&dec={DEC}&layer=ls-dr10&size={SIZE}&subimage"
    url = template.format(RA=ra, DEC=dec, SIZE=size_pix)
    if verbose:
        print('downloading from', url)

    downloaddir = Path(downloaddir)

    savepath = downloaddir / filename
    if savepath.exists():
        print('Already downloaded at', savepath)
        return savepath

    try:
        urllib.request.urlretrieve(url, savepath)
        print(f"File saved at: {savepath}")
    except Exception as e:
        print(f"An error occurred while downloading the file: {e}")

    # ADD GAUSSIAN NOISE, eliminate first hdu which has no data.
    estimate_and_add_gaussian_noise(savepath, verbose=verbose)
    return savepath


def estimate_and_add_gaussian_noise(filename, num_iterations=3, sigma_threshold=3.0, verbose=False):
    """
    The data downloaded from the get_legacy_survey_cutout is as follows:
     hdulist[0]: some general info
     hdulist[odd number]: data in werid flux unit, nanomaggie?
     hdulist[even number]: inverse variance map without gaussian noise of the data in the previous hdulist position.

     Here we open each inverse variance map, and add to them the gaussian noise contribution.
     Then we save the fits file again.
    """
    hdulist = fits.open(filename)

    # eliminate cutouts with incomplete data:
    newlist = fits.HDUList()
    # the "index hdu":
    newlist.append(hdulist[0])
    # eliminate weird shapes, would be too hard to work with anyways:
    for band_idx in range(1, len(hdulist), 2):
        d = hdulist[band_idx].data
        invv = hdulist[band_idx+1].data
        if d.shape[0] == d.shape[1] == invv.shape[0] == invv.shape[0]:
            newlist.append(hdulist[band_idx])
            newlist.append(hdulist[band_idx+1])

    for band_idx in range(1, len(newlist), 2):
        band_data = newlist[band_idx].data
        band_inverse_var = newlist[band_idx+1].data

        if verbose:
            print('layer', band_idx, 'of fits data: shapes', band_data.shape, band_inverse_var.shape)

        # estimate background noise
        for _ in range(num_iterations):
            mean = np.nanmean(band_data)
            std = np.nanstd(band_data)
            clip_min = mean - sigma_threshold * std
            clip_max = mean + sigma_threshold * std
            band_data = np.clip(band_data, clip_min, clip_max)

        noise = np.nanstd(band_data)

        var_with_noise = 1. / band_inverse_var + noise**2
        noisemap = var_with_noise**0.5

        newlist[band_idx+1].data = noisemap

    newlist.writeto(filename, overwrite=True)

    hdulist.close()

    if verbose:
        print("Gaussian noise estimated and added to the inverse variance maps.")


def create_weighted_stack(filename, band, verbose=False):
    hdulist = fits.open(filename)

    bands = set([hdulist[i].header['band'] for i in range(1, len(hdulist), 2)])
    if band not in bands:
        print(f'{band} not available. Available bands: {bands}')
        return

    stack = None
    weight_sum = None
    count = 0
    for band_idx in range(1, len(hdulist), 2):
        band_data = hdulist[band_idx].data
        band_inverse_var = hdulist[band_idx+1].data
        header = hdulist[band_idx].header
        if header['BAND'] == band:
            weight = band_inverse_var**0.5

            if stack is None:
                stack = band_data * weight
                weight_sum = weight
            else:
                stack += band_data * weight
                weight_sum += weight
            count += 1
    weighted_stack = stack / weight_sum
    noisemap = 1. / weight_sum

    hdulist.close()
    if verbose:
        print(f"Weighted stack created from {count} cutouts for band {band}.")
    return weighted_stack, noisemap


if __name__ == "__main__":
    # J 2122-1621
    ff = download_legacy_survey_cutout(320.6075, -16.357, 99, verbose=True)

    stacki, nmap = create_weighted_stack(ff, 'g')