from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
from jax.scipy.stats import norm
from jax import jit
from functools import partial
import jax.numpy as jnp
from jax import random
from jax.image import scale_and_translate
import numpyro
import numpyro.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel

from starred.utils.generic_utils import pad_and_convolve_fft, Downsample

from lensedquasarsurveyor.io import load_dict_from_hdf5
from lensedquasarsurveyor.gaia_utilities import find_gaia_stars_around_coords


def prepare_simple_lens_model_from_h5(h5file, imagecoords=None, debug=False, modeltype='simple'):
    """

    :param h5file: string or Path, path to an h5-file containing cutouts with WCS, PSFs, as prepared by
                                   download_and_extract and estimate_psf_from_extracted_h5.
    :param imagecoords: list of SkyCoords, where the PSFs of the lens are located. Default None, in this case we query
                                           gaia to get them.
    :param debug: bool, if True displays some plots along the way.
    :return: list of SimpleLensedQuasarModel instances, one per band
    """
    data = load_dict_from_hdf5(h5file)
    bands = list(data['lens'].keys())
    wcs_header = str(data['lens'][bands[0]]['0']['wcs_header'], encoding='ascii')
    wcs = WCS(wcs_header)

    if not imagecoords:
        # let us do this only once: query gaia to get the lensing positions
        height, width = data['lens'][bands[0]]['0']['data'].shape
        ra, dec = wcs.all_pix2world(height/2, width/2, 0)
        r = find_gaia_stars_around_coords(ra, dec, 5)
        if len(r) < 2:
            raise RuntimeError(f'Could not find two gaia detections on which to place PSFs around {ra}, {dec}.')
        ra1, dec1 = r[0]['ra'], r[0]['dec']
        ra2, dec2 = r[1]['ra'], r[1]['dec']
        imagecoords = [SkyCoord(ra1*u.deg, dec1*u.deg), SkyCoord(ra2*u.deg, dec2*u.deg)]

    # let's find the midpoint as an initial guess for a potential galaxy!
    coord1, coord2 = imagecoords
    pa = coord1.position_angle(coord2)
    sep = coord1.separation(coord2)
    coordg = coord1.directional_offset_by(pa, sep/2)  # very overkill compared to averaging RAs and Decs, but well ...

    # and let's go to our cutout coordinates.
    x1, y1 = skycoord_to_pixel(coord1, wcs)
    x2, y2 = skycoord_to_pixel(coord2, wcs)
    xg, yg = skycoord_to_pixel(coordg, wcs)

    models = {}
    for band in bands:
        # the '0' key below referes to the first available image. Subsequent ones would be denoted '1', '2', etc.
        # I have not decided yet on whether I want to allow the possibility for multiple images, or if
        # we should just stack everything we have ...
        # I think the latter, so I'll start building the infrastructure for a single image, but I'll keep the
        # possibility of having more than one image when constructing the h5 file just in case. Hence the '0'.
        lens = data['lens'][band]['0']['data']
        noise = data['lens'][band]['0']['noise']
        psf = data[band]['0']['psf']  # narrow psf eh
        upsampling_factor = data[band]['0']['psf_supersampling_factor']

        # messy part, go to supersampled model coordinates (smaller pixels, origin at the center instead of corner)
        if modeltype == 'mcs':
            sf = upsampling_factor
            offset = (sf * lens.shape[1] - 1) / 2.
            x1m, y1m = sf * x1 - offset, sf * y1 - offset
            x2m, y2m = sf * x2 - offset, sf * y2 - offset
            xgm, ygm = sf * xg - offset, sf * yg - offset
        elif modeltype == 'simple':
            offset = (lens.shape[1] - 1) / 2.
            x1m, y1m = x1 - offset, y1 - offset
            x2m, y2m = x2 - offset, y2 - offset
            xgm, ygm = xg - offset, yg - offset
        # point source initial guess
        A = 100*np.nanpercentile(lens, 99.5)
        ps_ini = [x1m, y1m, A, x2m, y2m, A]
        # galaxy initial guess
        #       [xgm, ygm, I_e,   r_e,  n,  ellip, theta ]
        g_ini = [xgm, ygm, 0.0001, 2., 2.5, 0.01, np.pi/2]
        # to be fair should be the same in every band, but let's think "ahead".
        initial_params_no_galaxy = [float(e) for e in ps_ini]
        initial_params_with_galaxy = [float(e) for e in (ps_ini+g_ini)]
        models[band] = SimpleLensedQuasarModel(lens, noise, psf, upsampling_factor,
                                               initial_params_no_galaxy, initial_params_with_galaxy, wcs)

    if debug:
        fig, axs = plt.subplots(3, len(bands), figsize=(2*len(bands), 5))
        axs = axs.reshape((3, len(bands)))
        for i, band in enumerate(bands):
            model = models[band]
            cdata = data['lens'][band]['0']['data']
            i0 = axs[0, i].imshow(cdata, origin='lower')
            mod = model.model_with_galaxy(model.initial_guess_with_galaxy)
            i1 = axs[1, i].imshow(mod, origin='lower')
            i2 = axs[2, i].imshow(cdata - mod, origin='lower')
            axs[0, i].set_title(f'{band}')
            # colorbars
            for j, im in enumerate([i0, i1, i2]):
                divider = make_axes_locatable(axs[j, i])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')

        for ax in axs.flatten():
            ax.plot([x1, x2, xg], [y1, y2, yg], 'x', color='red')
            ax.axis('off')

        plt.suptitle('debug: initial guess')
        plt.tight_layout()
        plt.show()

    return models


class SimpleLensedQuasarModel:
    def __init__(self, data, noisemap, narrowpsf, upsampling_factor,
                 initial_guess_no_galaxy, initial_guess_with_galaxy, wcs=None,
                 modeltype='simple'):
        """

        :param data: 2D np array
        :param noisemap: 2D np array
        :param narrowpsf: 2D np array
        :param upsampling_factor: int
        :param initial_guess_no_galaxy: list
        :param initial_guess_with_galaxy: list
        :param wcs: astropy WCS object
        :param modeltype: string, either 'simple' or 'mcs'. with 'simple', we just interpolate and scale the PSFs
                          onto the grid. With 'mcs', we interpolate 2D gaussians and sersics onto a finer grid
                          (PSF resolution), then convolve with the PSF and downsample to the image resolution.

        """


        shape = data.shape
        assert shape == noisemap.shape

        # here we scale!!
        scale = np.nanpercentile(data, 99.9)
        data /= scale
        noisemap /= scale

        self.scale = scale
        self.data = data
        self.noisemap = noisemap

        self.upsampling_factor = upsampling_factor
        self.wcs = wcs

        self.modeltype = modeltype

        if modeltype == 'mcs':
            self.model_no_galaxy = self.model_no_galaxy_mcs
            self.model_with_galaxy = self.model_with_galaxy_mcs

        elif modeltype == 'simple':
            self.model_no_galaxy = self.model_no_galaxy_simple
            self.model_with_galaxy = self.model_with_galaxy_simple

        else:
            raise AssertionError('modeltype is either mcs or simple')

        Nx, Ny = shape

        nx, ny = narrowpsf.shape
        if modeltype == 'mcs':
            Nx, Ny = upsampling_factor * Nx, upsampling_factor * Ny
            padx, pady = int((Nx - nx) / 2), int((Ny - ny) / 2)
            self.psf = jnp.pad(narrowpsf, ((padx, padx), (pady, pady)), constant_values=0.)
        elif modeltype == 'simple':
            self.psf = gaussian_filter(narrowpsf, 0.85)  # narrow psf --> psf by convolving with gaussian kernel fwhm=2

        x, y = np.arange(-Ny//2, Ny//2), np.arange(-Nx//2, Nx//2)
        self.X, self.Y = np.meshgrid(x, y)

        self.initial_guess_no_galaxy = initial_guess_no_galaxy
        self.initial_guess_with_galaxy = initial_guess_with_galaxy
        self.param_optim_with_galaxy = None
        self.param_optim_no_galaxy = None
        self.param_mediansampler_no_galaxy = None
        self.param_mediansampler_with_galaxy = None

    @partial(jit, static_argnums=(0,))
    def elliptical_sersic_profile(self, I_e, r_e, n, x0, y0, ellip, theta):
        # Ellipticity and orientation parameters
        q = 1 - ellip
        theta = jnp.radians(theta)
        xt = (self.X - x0) * jnp.cos(theta) + (self.Y - y0) * jnp.sin(theta)
        yt = (self.Y - y0) * jnp.cos(theta) - (self.X - x0) * jnp.sin(theta)
        # radius
        r = jnp.sqrt(xt ** 2 + (yt / q) ** 2)
        # sersicersic profile
        bn = 1.9992 * n - 0.3271  # approximation valid for 0.5 < n < 10
        return I_e * jnp.exp(-bn * ((r / r_e) ** (1 / n) - 1))

    def gaussian_psf(self, x0, y0, A):
        """Calculate the value of a 2D Gaussian PSF."""
        # fwhm 2 ~ sigma 0.85
        g = A * norm.pdf(self.X, loc=x0, scale=0.85) * norm.pdf(self.Y, loc=y0, scale=0.85)
        return g

    def translate_and_scale_psf(self, dx, dy, amplitude):
        outshape = self.X.shape
        supersampling = self.upsampling_factor
        scale = jnp.array((1. / supersampling, 1. / supersampling))
        inishape = self.psf.shape[0]
        # assuming square psf
        outoffset = (inishape / 2.) / supersampling
        outoffsetx = outoffset + (outshape[0] - inishape) / supersampling
        outoffsety = outoffset + (outshape[1] - inishape) / supersampling

        # assuming
        translation = jnp.array((dy + outoffsetx, dx + outoffsety))
        out = scale_and_translate(self.psf, outshape, (0, 1), scale, translation, method='bicubic')
        return amplitude * out

    def _create_model_no_galaxy(self, x1, y1, A1, x2, y2, A2):
        psf1 = self.gaussian_psf(x1, y1, A1)
        psf2 = self.gaussian_psf(x2, y2, A2)

        model = psf1 + psf2

        return model

    def _create_model_with_galaxy(self, x1, y1, A1, x2, y2, A2, xg, yg, I_e, r_e, n, ellip, theta):
        psfs = self._create_model_no_galaxy(x1, y1, A1, x2, y2, A2)

        sersic = self.elliptical_sersic_profile(I_e, r_e, n, xg, yg, ellip, theta)

        model = sersic + psfs

        return model

    def down_resolution(self, model):
        return Downsample(pad_and_convolve_fft(model, self.psf), self.upsampling_factor)

    def model_no_galaxy_mcs(self, params):
        _mod = self._create_model_no_galaxy(*params)
        return self.down_resolution(_mod)

    def model_with_galaxy_mcs(self, params):
        _mod = self._create_model_with_galaxy(*params)
        return self.down_resolution(_mod)

    def model_no_galaxy_simple(self, params):
        x1, y1, A1, x2, y2, A2 = params
        mod = self.translate_and_scale_psf(x1, y1, A1)
        mod += self.translate_and_scale_psf(x2, y2, A2)
        return mod

    def model_with_galaxy_simple(self, params):
        x1, y1, A1, x2, y2, A2, xg, yg, I_e, r_e, n, ellip, theta = params
        sersic = self.elliptical_sersic_profile(I_e, r_e, n, xg, yg, ellip, theta)
        return self.model_no_galaxy_simple([x1, y1, A1, x2, y2, A2]) + sersic

    @partial(jit, static_argnums=(0,))
    def residuals_with_galaxy(self, params):
        model = self.model_with_galaxy(params)
        return ((model - self.data) / self.noisemap).flatten()

    @partial(jit, static_argnums=(0,))
    def residuals_no_galaxy(self, params):
        model = self.model_no_galaxy(params)
        return ((model - self.data) / self.noisemap).flatten()

    def reduced_chi2_no_galaxy(self, params):
        residuals = self.residuals_no_galaxy(params)
        chi_squared = np.sum(residuals ** 2)
        dof = self.data.size - len(params)

        reduced_chi_squared = chi_squared / dof
        return reduced_chi_squared

    def reduced_chi2_with_galaxy(self, params):
        residuals = self.residuals_with_galaxy(params)
        chi_squared = np.sum(residuals ** 2)
        dof = self.data.size - len(params)

        reduced_chi_squared = chi_squared / dof
        return reduced_chi_squared

    def optimize_no_galaxy(self, initial_guess=None):
        if not initial_guess:
            initial_guess = self.initial_guess_no_galaxy
        res = least_squares(self.residuals_no_galaxy, initial_guess)
        self.param_optim_no_galaxy = res.x
        return res.x

    def optimize_with_galaxy(self, initial_guess=None):
        if not initial_guess:
            initial_guess = self.initial_guess_with_galaxy
        res = least_squares(self.residuals_with_galaxy, initial_guess)
        self.param_optim_with_galaxy = res.x
        return res.x

    def sample_with_galaxy(self, num_warmup=100, num_samples=200, num_chains=1):

        def numpyromodel(data, noise):
            # Flatten the images
            image_data_flat = data.flatten()
            image_uncertainties_flat = noise.flatten()
            # this basically allows us to use the numypro.plate context manager
            # below. Not very useful here, but indicates that each pixel
            # is independant. Some samplers can take advantage of this,
            # so let's do it this way.

            sizey, sizex = data.shape
            boundx, boundy = (sizex - 1.) / 2., (sizey - 1.) / 2.

            # Uniform priors on the entire field.
            x1 = numpyro.sample('x1', dist.Uniform(-boundx, boundx))
            y1 = numpyro.sample('y1', dist.Uniform(-boundy, boundy))
            A1 = numpyro.sample('A1', dist.Uniform(0., 20.))

            x2 = numpyro.sample('x2', dist.Uniform(-boundx, boundx))
            y2 = numpyro.sample('y2', dist.Uniform(-boundy, boundy))
            A2 = numpyro.sample('A2', dist.Uniform(0., 20.))

            xg = numpyro.sample('xg', dist.Uniform(-boundx, boundx))
            yg = numpyro.sample('yg', dist.Uniform(-boundy, boundy))
            I_e = numpyro.sample('I_e', dist.Uniform(0., 1.0))
            r_e = numpyro.sample('r_e', dist.Uniform(0.1, 5.))
            n = numpyro.sample('n', dist.Uniform(0.5, 10))
            ellip = numpyro.sample('ellip', dist.Uniform(0, 0.99))
            theta = numpyro.sample('theta', dist.Uniform(0, 2 * np.pi))

            # Model
            mod = self.model_with_galaxy([x1, y1, A1, x2, y2, A2, xg, yg, I_e, r_e, n, ellip, theta])

            # likelihood, gaussian errors
            with numpyro.plate('data', len(image_data_flat)):
                numpyro.sample('obs', dist.Normal(mod.flatten(), image_uncertainties_flat), obs=image_data_flat)

        # run MCMC
        nuts_kernel = numpyro.infer.NUTS(numpyromodel)
        mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, self.data, self.noisemap)
        mcmc.print_summary()
        keys = ['x1', 'y1', 'A', 'x2', 'y2', 'A2', 'xg', 'yg', 'I_e', 'r_e', 'n', 'ellip', 'theta']
        medians = {k: np.median(val) for k, val in mcmc.get_samples().items()}
        self.param_mediansampler_with_galaxy = [medians[k] for k in keys]
        return mcmc

    def sample_no_galaxy(self, num_warmup=500, num_samples=500, num_chains=1):

        def numpyromodel(data, noise):
            # Flatten the images
            image_data_flat = data.flatten()
            image_uncertainties_flat = noise.flatten()
            # this basically allows us to use the numypro.plate context manager
            # below. Not very useful here, but indicates that each pixel
            # is independant. Some samplers can take advantage of this,
            # so let's do it this way.

            sizey, sizex = data.shape
            boundx, boundy = (sizex - 1.) / 2., (sizey - 1.) / 2.

            # Uniform priors on the entire field.
            x1 = numpyro.sample('x1', dist.Uniform(-boundx, boundx))
            y1 = numpyro.sample('y1', dist.Uniform(-boundy, boundy))
            A1 = numpyro.sample('A1', dist.Uniform(0., 500.))

            x2 = numpyro.sample('x2', dist.Uniform(-boundx, boundx))
            y2 = numpyro.sample('y2', dist.Uniform(-boundy, boundy))
            A2 = numpyro.sample('A2', dist.Uniform(0., 500.))

            # Model
            mod = self.model_no_galaxy([x1, y1, A1, x2, y2, A2])

            # likelihood, gaussian errors
            with numpyro.plate('data', len(image_data_flat)):
                numpyro.sample('obs', dist.Normal(mod.flatten(), image_uncertainties_flat), obs=image_data_flat)

        # run MCMC
        nuts_kernel = numpyro.infer.NUTS(numpyromodel)
        mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, self.data, self.noisemap)
        mcmc.print_summary()
        keys = ['x1', 'y1', 'A', 'x2', 'y2', 'A2']
        medians = {k: np.median(val) for k, val in mcmc.get_samples().items()}
        self.param_mediansampler_no_galaxy = [medians[k] for k in keys]
        return mcmc

    def _plot_model(self, params, modelfunc):

        fig, axs = plt.subplots(1, 4, figsize=(8, 2))

        cdata = self.data
        i0 = axs[0].imshow(cdata, origin='lower')
        mod = modelfunc(params)
        i1 = axs[1].imshow(mod, origin='lower')
        i2 = axs[2].imshow((cdata - mod)/self.noisemap, origin='lower')
        i3 = axs[3].imshow(mod, origin='lower')
        # colorbars
        for j, im in enumerate([i0, i1, i2, i3]):
            divider = make_axes_locatable(axs[j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

        for ax in axs.flatten():
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_model_no_galaxy(self, params=None):
        if params is None:
            if self.param_mediansampler_no_galaxy is not None:
                params = self.param_mediansampler_no_galaxy
                print('Used median params from sampling')
            elif self.param_optim_no_galaxy is not None:
                params = self.param_optim_no_galaxy
                print('Used params from least-squares optimization')
            else:
                params = self.initial_guess_no_galaxy
                print('Used initial guess')

        self._plot_model(params, self.model_no_galaxy)

    def plot_model_with_galaxy(self, params=None):
        if params is None:
            if self.param_mediansampler_with_galaxy is not None:
                params = self.param_mediansampler_with_galaxy
                print('Used median params from sampling')
            elif self.param_optim_with_galaxy is not None:
                params = self.param_optim_with_galaxy
                print('Used params from least-squares optimization')
            else:
                params = self.initial_guess_with_galaxy
                print('Used initial guess')

        self._plot_model(params, self.model_with_galaxy)


"""
if __name__ == '__main__':
        # pass
        psf = lambda x, y, x0, y0, A: A * np.exp(-0.022 * (x - x0)**2 - 0.02 * (y - y0)**2)

        # grid of small pixels
        X, Y = x, y = np.meshgrid(np.linspace(-32, 32, 64), np.linspace(-32, 32, 64))

        narrow_psf = psf(x, y, 0, 0, 1)
        narrow_psf /= narrow_psf.sum()
        modc = SimpleLensedQuasarModel(X, X, narrow_psf, 2, [], [])
        params = [0, 6, 2, 0, -6, 3, 0, 0, 0.00, 5.0, 1.5, 0.5, 30]
        m = modc.model_with_galaxy(params)

        noise_scale = 0.001
        data = m
        data += np.random.normal(loc=0, scale=noise_scale, size=data.shape)
        plt.imshow(data)
        noisemap = noise_scale * np.ones_like(data)
        modc.data = data
        modc.noisemap = noisemap

        plt.show()

#"""