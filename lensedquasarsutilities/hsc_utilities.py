import astropy.units as u
import shutil
import urllib3
import os
from pathlib import Path
from astropy.io import fits

from lensedquasarsutilities.formatting import get_J2000_name
from lensedquasarsutilities.exceptions import HSCCredentialsNotInEnvironment, HSCNoData

template = "https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/cgi-bin/cutout?ra1={ra1:.5f}&dec1={dec1:.5f}"
template += "&ra2={ra2:.5f}&dec2={dec2:.5f}&type=coadd&image=on&mask=off&variance=on&filter={band}&rerun=pdr3_wide"

HSC_bands = ['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y']


def format_link(ra, dec, arcsec, band):
    ra1 = ra - arcsec * u.arcsec.to('degree') / 2.
    ra2 = ra + arcsec * u.arcsec.to('degree') / 2.
    dec1 = dec - arcsec * u.arcsec.to('degree') / 2.
    dec2 = dec + arcsec * u.arcsec.to('degree') / 2.

    link = template.format(ra1=ra1, dec1=dec1, ra2=ra2, dec2=dec2, band=band)

    return link


def download_single_cutout(ra, dec, band, cutoutsizearcsec, outfile):

    # check the credentials
    user = os.environ.get('HSCUSERNAME')
    password = os.environ.get('HSCPASSWORD')

    if (password is None) or (user is None):
        raise HSCCredentialsNotInEnvironment

    url = format_link(ra, dec, cutoutsizearcsec, band)
    http = urllib3.PoolManager()
    headers = urllib3.make_headers(basic_auth=f'{user}:{password}')
    response = http.request('GET', url, headers=headers, preload_content=False)
    with open(outfile, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

    response.release_conn()
    return outfile


def combine_fits(filelist, savepath):
    """
    Combine FITS files in the given directory into a single FITS file.

    :param filelist: list of paths or strings representing paths, the files to be combined
    :param savepath: string, path at which the combined FITS file should be saved.
    :return: None
    """
    # Create an empty HDU for position 0
    hdulist = fits.HDUList([fits.PrimaryHDU()])
    for path in filelist:
        try:
            # data
            bandhdulist = fits.open(path)
        except OSError:
            print(f"Problem with fits file at {path}")
            continue
        headerhdu, datahduheader = bandhdulist[0].header, bandhdulist[1].header
        datahduheader.update(headerhdu)
        datahdu = bandhdulist[1]
        datahdu.header = datahduheader
        hdulist.append(datahdu)

        headerhdu, varhduheader = bandhdulist[2].header, bandhdulist[3].header
        varhduheader.update(headerhdu)
        varhduheader['filter'] = datahduheader['filter']
        varhdu = bandhdulist[3]
        varhdu.header = varhduheader
        varhdu.data = varhdu.data**0.5
        hdulist.append(varhdu)
    if len(hdulist) == 1:
        # then we didn't collect anything ...
        raise HSCNoData("It seems we could download some files initially, but could not combine them.")
    hdulist.writeto(savepath, overwrite=True)

    # remove original files:
    for path in filelist:
        Path(path).unlink()


def download_hsc_cutout(ra, dec, size, downloaddir=None, filename=None, verbose=False):
    if not filename:
        filename = f"cutouts_HSC_{get_J2000_name(ra, dec)}_size_{size:.0f}.fits"
    if not downloaddir:
        downloaddir = '.'
    downloaddir = Path(downloaddir)
    savepath = downloaddir / filename
    if savepath.exists():
        print('Already downloaded at', savepath)
        return savepath

    # the pixel size in HSC cutouts:
    # hsc_pixel_size = 0.168
    # convert to pixels, as the API wants pixels.
    # aaaaaaaaah never mind, the HSC api wants arsecs.

    outfiles = []
    for band in HSC_bands:
        outname = f"{filename.replace('.fits', '')}_{band}.fits"
        outfile = Path(downloaddir) / outname
        try:
            download_single_cutout(ra, dec, band, size, outfile)
            outfiles.append(outfile)
            if verbose:
                print(f"Downloaded {band}-band HSC data at {ra,dec}.")
        except HSCCredentialsNotInEnvironment:
            raise HSCCredentialsNotInEnvironment  # yeah no won't let that slide bro
        except Exception as e:
            print(f"Problem with band {band}: {e}")

    if len(outfiles) == 0:
        raise HSCNoData(f"Could not get data in HSC at {ra}, {dec}.")

    combine_fits(outfiles, savepath)
    return savepath





if __name__ == "__main__":
    raj, decj = 159.3665, 0.3057

    print(download_hsc_cutout(raj, decj, 3, downloaddir='/tmp/', verbose=True))
