import urllib3
import shutil
from pathlib import Path
from astropy.io import fits

from time import time
from os.path import join, exists
from multiprocessing import Pool

from lensedquasarsutilities.formatting import get_J2000_name

download_target_url = "http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={ra}&dec={dec}&type=stack"


def unpack_line(line, headers):
    entry = {header: element for header, element in zip(headers, line.split(' '))}
    return entry


def parse_requests(url):
    """
        reads a list of file locations on the panstarrs public server and formats
        them as a python list.
    """
    http = urllib3.PoolManager()
    r = http.request('GET', url, preload_content=0)

    data = r.data.decode('utf-8')
    data = data.split('\n')
    headers = data[0].split(' ')
    data = data[1:]
    return [unpack_line(line, headers) for line in data if line]


def actual_download(entry):
    """
    will be passed to the threads in downlaodData.

    entry contains the download parameters.

    """
    (outimaget, basetarget, Nimage, size), entry = entry
    print(f"  Downloading images for epoch {entry['shortname']}")
    outimage = outimaget.format(filename=entry['shortname'])
    urlimage = basetarget.format(filename=entry['filename'],
                                 RA=entry['ra'],
                                 DEC=entry['dec'],
                                 width=size,
                                 shortname=entry['shortname']
                                 )
    # also generate the filenames and urls for the auxiliary data:

    outweight = outimage.replace('.fits', '.wt.fits')
    urlweight = urlimage.replace('.fits', '.wt.fits')

    # download and bin everything accordingly:
    outfiles = []
    try:
        for outfile, url in zip([outimage, outweight], [urlimage, urlweight]):
            if not exists(outfile):

                http = urllib3.PoolManager()
                response = http.request('GET', url, preload_content=False)

                with open(outfile, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)

                response.release_conn()
            outfiles.append(outfile)

    except Exception as e:
        print(f"      ----> error with image {entry['shortname']}: {e}")

    return outfiles


def download_panstarrs_cutout(ra, dec, size, downloaddir=None, filename=None, verbose=False):
    """
        downloads all the available panstarrs data that satisfies the requirements
        given in the parameters.

        :param ra: float: degrees
        :param dec: float: degrees
        :param size: float: arcseconds, desired size of the cutout.
        :param downloaddir: string, where to put the data: Default: None
        :param filename: optional, give a special name to the resulting fits file. Default: None
        :param verbose: display some messages? Default: False
        :return: savepath where the data was saved
    """
    if not filename:
        filename = f"cutouts_legacy_survey_{get_J2000_name(ra, dec)}_size_{size:.0f}.fits"
    if not downloaddir:
        downloaddir = '.'

    # the pixel size in Pan-STARRS cutouts:
    ps_pixel_size = 0.258
    # convert to pixels, as the API wants pixels.
    size_pix = int(size / ps_pixel_size)

    url = download_target_url.format(ra=ra, dec=dec)
    downloaddir = Path(downloaddir)
    downloaddir.mkdir(exist_ok=True)

    savepath = downloaddir / filename
    if savepath.exists():
        print('Already downloaded at', savepath)
        return savepath

    parsed_request = parse_requests(url)
    parsed_request_filtered = []
    channels = 'grizY'
    for channel in channels:
        parsed_request_filtered += [e for e in parsed_request if f".{channel}." in e['shortname']]
    parsed_request = parsed_request_filtered

    # preparing the urls to contact:
    outimaget = join(str(downloaddir), "{filename}")

    basetarget = "http://ps1images.stsci.edu/cgi-bin/fitscut.cgi?red={filename}"
    basetarget += "&&format=fits&x={RA}&y={DEC}&size={width}&wcs=1&imagename={shortname}"

    # 2 files (wt, image) for each stack:
    Nimage = len(parsed_request)
    if verbose:
        print(f"######### Downloading. There are {Nimage} stacks to download.")
    # since we are using multirpocessing and I don't want to write 200 lines,
    # give some overhead with the global info to each download:
    t0 = time()
    to_download = [((outimaget, basetarget, Nimage, size_pix), entry) for entry in parsed_request]

    pool = Pool(5)
    outfiles = pool.map(actual_download, to_download)
    pool.close()
    pool.join()

    if verbose:
        print(f"######## Download completed, took {time() - t0:.0f} seconds to complete.")
    outfiles = sum(outfiles, [])
    combine_fits(outfiles, savepath)
    return savepath


def combine_fits(filelist, savepath):
    """
    Combine FITS files in the given directory into a single FITS file.

    :param filelist: list of paths or strings representing paths, the files to be combined
    :param savepath: string, path at which the combined FITS file should be saved.
    :return: None
    """
    # Create an empty HDU for position 0
    hdulist = fits.HDUList([fits.PrimaryHDU()])

    for band in 'grizY':
        try:
            fim = [e for e in filelist if f'.{band}.unconv.fits' in e][0]
            fw = [e for e in filelist if f'.{band}.unconv.wt.fits' in e][0]
            hduim = fits.open(fim)
            hduw = fits.open(fw)
        except Exception as e:
            print(f'PROBLEM WITH BAND {band}: {e}. SKIPPING!')
            continue
        hdulist.append(hduim[0])
        hdulist.append(hduw[0])

        hduim.close()
        hduw.close()
        # if safe, remove the original files.
        Path(fim).unlink()
        Path(fw).unlink()

    hdulist.writeto(savepath, overwrite=True)


if __name__ == "__main__":

    RA, DEC = 320.6075, -16.357
    hsize = 100
    woutdir = '/tmp/'
    download_panstarrs_cutout(RA, DEC, hsize, woutdir)
