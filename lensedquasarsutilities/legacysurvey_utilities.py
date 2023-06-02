import urllib.request
from pathlib import Path
import numpy as np
from astropy.io import fits


def download_legacy_survey_cutout(ra, dec, size, downloaddir=None, filename=None):
    TEMPLATE = "https://www.legacysurvey.org/viewer/cutout.fits?ra={RA}&dec={DEC}&layer=ls-dr10&size={SIZE}&subimage"
    url = TEMPLATE.format(RA=ra, DEC=dec, SIZE=size)

    if not filename:
        filename = f"cutout_legacy_survey_ra_{ra}_dec_{dec}_size_{size}.fits" 
    if not downloaddir:
        downloaddir = '.'
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


    # ADD GAUSSIAN NOISE
    estimate_and_add_gaussian_noise(savepath)
    return savepath



def estimate_and_add_gaussian_noise(filename, num_iterations=3, sigma_threshold=3.0, verbose=True):
    """
    The data downloaded from the get_legacy_survey_cutout is as follows:
     hdulist[0]: some general info
     hdulist[odd number]: data in werid flux unit, nanomaggie?
     hdulist[even number]: inverse variance map without gaussian noise of the data in the previous hdulist position.

     Here we open each inverse variance map, and add to them the gaussian noise contribution.
     Then we save the fits file again.
    """
    hdulist = fits.open(filename, mode='update')


    for band_idx in range(1, len(hdulist), 2):
        band_data = hdulist[band_idx].data
        band_inverse_var = hdulist[band_idx+1].data


        for _ in range(num_iterations):
            mean = np.nanmean(band_data)
            std = np.nanstd(band_data)
            clip_min = mean - sigma_threshold * std
            clip_max = mean + sigma_threshold * std
            band_data = np.clip(band_data, clip_min, clip_max)

        noise = np.nanstd(band_data)
        print(noise)
        inverse_var_with_noise = 1. / (1 / band_inverse_var + noise**2)

        hdulist[band_idx+1].data = inverse_var_with_noise

    hdulist.flush()
    hdulist.close()

    if verbose:
        print("Gaussian noise estimated and added to the inverse variance maps.")


def create_weighted_stack(filename, band, verbose=False):
    hdulist = fits.open(filename)

    bands = set([hdulist[i].header['band'] for i in range(1, len(hdulist), 2)])
    if not band in bands:
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
    filename = download_legacy_survey_cutout(320.6075, -16.357, 100)


    # ok, let's try
    from astropy.io import fits
    import matplotlib.pyplot as plt


    plt.figure()
    stack, noisemap = create_weighted_stack(filename, 'g')
    plt.imshow(stack / noisemap)
    plt.show()
