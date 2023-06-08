from pathlib import Path
import numpy as np

from lensedquasarsurveyor.submodules import f2n


def plot_psf(identifier, noise, stars, residuals, psf, lossplot, workdir):
    """
       used to check the estimation of the PSF, generates a nice plot.

    :param identifier: string
    :param noise: array (N, nx, ny)
    :param stars: array (N, nx, ny)
    :param residuals: array (N, nx, ny)
    :param psf: 2D array
    :param lossplot: bytes containing a matplotlib plot of the loss progress of the optimization
    :param workdir: string or Path, where we are working.
    :return:
    """
    psfsplotsdir = Path(workdir) / 'psf_plots'
    psfsplotsdir.mkdir(exist_ok=True, parents=True)

    tilesize = 256
    imsizeup = psf[0].shape[0]
    imsize = stars[0].shape[0]
    nbrpsf = stars.shape[0]

    totpsfimg = f2n.f2nimage(np.array(psf), verbose=False)

    pngpath = psfsplotsdir / (identifier + ".png")

    lossim = f2n.f2nimage(lossplot, verbose=False)
    lossim.setzscale(0, 255)
    lossim.makepilimage(scale="lin", negative=False)
    if tilesize != lossplot.shape[0]:
        lossim.upsample(tilesize / lossplot.shape[0])

    totpsfimg.setzscale('auto', 'auto')
    totpsfimg.makepilimage(scale="log", negative=False)
    totpsfimg.upsample(tilesize / imsizeup)
    totpsfimg.writetitle("Total PSF")

    txtendpiece = f2n.f2nimage(shape=(tilesize, tilesize), fill=0.0, verbose=False)
    txtendpiece.setzscale(0.0, 1.0)
    txtendpiece.makepilimage(scale="lin", negative=False)

    # The psf stars
    psfstarimglist = []
    for j in range(nbrpsf):
        f2nimg = f2n.f2nimage(stars[j], verbose=False)
        f2nimg.setzscale("auto", "auto")
        f2nimg.makepilimage(scale="log", negative=False)
        f2nimg.upsample(tilesize / imsize)
        psfstarimglist.append(f2nimg)

    psfstarimglist.append(txtendpiece)

    # The sigmas
    sigmaimglist = []
    for j in range(nbrpsf):
        f2nimg = f2n.f2nimage(noise[j], verbose=False)
        f2nimg.setzscale('auto', 'auto')
        f2nimg.makepilimage(scale="log", negative=False)
        f2nimg.upsample(tilesize / imsize)
        f2nimg.writetitle('noise map')
        sigmaimglist.append(f2nimg)

    sigmaimglist.append(lossim)

    # The residuals
    difnumlist = []
    for j in range(nbrpsf):
        f2nimg = f2n.f2nimage(residuals[j] / noise[j], verbose=False)
        f2nimg.setzscale('auto', 'auto')
        f2nimg.makepilimage(scale="lin", negative=False)
        f2nimg.upsample(tilesize / imsize)
        f2nimg.writetitle("rel. residuals")
        difnumlist.append(f2nimg)

    difnumlist.append(totpsfimg)

    f2n.compose([psfstarimglist, sigmaimglist, difnumlist], pngpath)

