from copy import deepcopy

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter

from jax import jit
from jax import vmap
from functools import partial
import jax.numpy as jnp
from jax.scipy.signal import fftconvolve
from jax import random
from jax.image import scale_and_translate
from jaxopt import ScipyMinimize
import numpyro
import numpyro.distributions as dist


from lensedquasarsurveyor.io import load_dict_from_hdf5, save_dict_to_hdf5


def prepare_fitter_from_h5(h5file, filter_psf=True, verbose=False):
    """

    :param h5file: string or Path, path to an h5-file containing cutouts with WCS, PSFs, as prepared by
                                   download_and_extract and estimate_psf_from_extracted_h5.
    :param filter_psf: bool, default True. Supress the edges of the PSF. Can be a decent safety given our method of
                       psf modelling. If the PSF is completely out of control, might suppress useful regions. But
                       in this case this whole endeavor would be useless ...
    :param verbose: bool, default False
    :return: a DoublyLensedQuasarFitter, ready to try its best to fit the data with its `sample` method.
    """
    data = load_dict_from_hdf5(h5file)
    bands = list(data['lens'].keys())
    bands = sort_filters(bands)

    lensdata = []
    noisedata = []
    psfdata = []

    for band in bands:
        # the '0' key below referes to the first available image. Subsequent ones would be denoted '1', '2', etc.
        # I have not decided yet on whether I want to allow the possibility for multiple images, or if
        # we should just stack everything we have ...
        # I think the latter, so I'll start building the infrastructure for a single image, but I'll keep the
        # possibility of having more than one image when constructing the h5 file just in case. Hence the '0'.
        lens = data['lens'][band]['0']['data']
        noise = data['lens'][band]['0']['noise']
        psf = data[band]['0']['psf']  # narrow psf eh

        lensdata.append(lens)
        noisedata.append(noise)
        psfdata.append(psf)

    upsampling_factor = data[band]['0']['psf_supersampling_factor']
    if verbose:
        print(f"Preparing a multi band models using {len(bands)} bands, data size {lens.shape[1]} pixels.")
    lensdata, noisedata, psfdata = np.array(lensdata), np.array(noisedata), np.array(psfdata)
    model = DoublyLensedQuasarFitter(lensdata, noisedata, psfdata, upsampling_factor, bands, filter_psf)

    return model


def sort_filters(strings):
    order = ['g', 'r', 'i', 'z', 'y']
    return sorted(strings, key=lambda s: (order.index(s[0].lower()), s.lower()))


def psf_cleaner(psf, frac_suppressed=0.25):
    """
        filters the edges of the PSF by
        - fitting a 2D gaussian
        - marking the pixels below a certain fraction of the total luminosity
        - making those pixels 10 times smaller in the original PSF.

        :param psf: 2d array
        :param frac_suppressed: float, fraction of the pixels (determined by inspecting the levels of the
                                fitted 2d gaussian) to be suppressed.
    """

    def gaussian_2d(xy, amp, x0, y0, sigma_x, sigma_y):
        x, y = xy
        gauss = amp * np.exp(-(x - x0) ** 2 / (2 * sigma_x ** 2) - (y - y0) ** 2 / (2 * sigma_y ** 2))
        return gauss.ravel()

    def residuals(p, xy, data):
        return gaussian_2d(xy, *p) - data.ravel()

    # Get the x, y values for each data point
    x = np.arange(psf.shape[1])
    y = np.arange(psf.shape[0])
    x, y = np.meshgrid(x, y)

    max_loc = (psf.shape[1] - 1) / 2, (psf.shape[0] - 1) / 2,
    init_guess = [np.max(psf), max_loc[1], max_loc[0], psf.shape[1] * 0.15, psf.shape[0] * 0.15]

    popt = least_squares(residuals, init_guess, args=((x, y), psf)).x

    gaussian_2d_values = gaussian_2d((x, y), *popt).reshape(psf.shape)
    suppression_threshold = np.percentile(gaussian_2d_values, 100 * frac_suppressed)

    # suppress regions where the Gaussian is below the threshold
    suppressed_data = np.where(gaussian_2d_values < suppression_threshold, 0.2 * psf, psf)

    return suppressed_data


class DoublyLensedQuasarFitter:
    def __init__(self, data, noisemap, narrowpsf, upsampling_factor, bandnames, filter_psf):
        """
        A class that will hold your data in different bands (cutouts, noisemaps, psfs).
        Its end goal is using its `sample` method to blindly fit two PSFs to the data, and potentially a
        sersic profile for the lensing galaxy if the two PSFs are not enough. The astrometry of all sources and
        morphology of the sersic profile are common to all bands. Per band, we fit the fluxes and alignment offsets.

        Future work: multi resolution, use bands from multiple surveys at once. This will be slower since we won't
        be able to vectorize ...let's see.

        :param data: 2D np array
        :param noisemap: 2D np array
        :param narrowpsf: 2D np array
        :param upsampling_factor: int
        :param filter_psf: bool, default True. Supress the edges of the PSF. Can be a decent safety given our method of
                           psf modelling. If the PSF is completely out of control, might suppress useful regions. But
                           in this case this whole endeavor would be useless ...

        """

        sortedbandnames = sort_filters(bandnames)
        assert sortedbandnames == bandnames, "Your filter are not properly sorted! Sort them according to g,r,i,z,y"
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
        self.bands = bandnames

        nband, Nx, Ny = shape
        nband2, nx, ny = narrowpsf.shape
        assert nband == nband2 == len(bandnames)

        # narrow psf --> psf by convolving with gaussian kernel fwhm=2
        self.psf = np.array([gaussian_filter(e, 0.85) for e in narrowpsf])
        if filter_psf:
            self.psf = np.array([psf_cleaner(e) for e in self.psf])

        x, y = np.linspace(-(Ny-1)/2, (Ny-1)/2, Ny), np.linspace(-(Nx-1)/2, (Nx-1)/2, Nx)
        self.X, self.Y = np.meshgrid(x, y)

        self.param_optim_with_galaxy = None
        self.param_optim_no_galaxy = None
        self.param_mediansampler_no_galaxy = None
        self.param_mediansampler_with_galaxy = None
        self.mcmc = None

    def remove_band(self, bandname):
        """
        erases `bandname` out of existence from this model.

        :param bandname: string, name of the band to remove
        :return: Nothing
        """
        try:
            index = self.bands.index(bandname)
        except ValueError:
            print(f"{bandname} is not in our bands. Current bands: {self.bands}")
            return
        self.bands.pop(index)
        self.data = np.delete(self.data, index, 0)
        self.noisemap = np.delete(self.noisemap, index, 0)
        self.psf = np.delete(self.psf, index, 0)

    def elliptical_sersic_profile(self, I_e, r_e, x0, y0, n, ellip, theta):
        # ellipticity and orientation parameters
        q = 1 - ellip
        theta = jnp.radians(theta)
        xt = (self.X - x0) * jnp.cos(theta) + (self.Y - y0) * jnp.sin(theta)
        yt = (self.Y - y0) * jnp.cos(theta) - (self.X - x0) * jnp.sin(theta)
        # radius
        r = jnp.sqrt(xt ** 2 + (yt / q) ** 2)
        # sersic profile
        # let's fix n ...
        bn = 1.9992 * n - 0.3271  # approximation valid for 0.5 < n < 10
        profilewow = jnp.exp(-bn * ((r / r_e) ** (1. / n) - 1))
        profilewow /= profilewow.sum()
        profile = I_e * profilewow

        return profile

    def elliptical_sersic_profile_convolved(self, I_e, r_e, x0, y0, n, ellip, theta, psf):
        """
        this is what we call in the model.
        :param I_e: amplitude
        :param r_e: radius
        :param x0: x coord
        :param y0: y coord
        :param n: n parameter
        :param ellip: ellipticity
        :param theta: angle
        :param psf: the psf by which we convolve the profile.
        :return:
        """
        profile = self.elliptical_sersic_profile(I_e, r_e, x0, y0, n, ellip, theta)
        return fftconvolve(profile, psf, mode='same')

    def translate_and_scale_psf(self, dx, dy, amplitude, psf):
        """
          scales the PSF down to the resolution of the model, then translates it by dx, dy
          (could be x,y but since this is a translation, called them dx, dy) and multiplies the result by the
          amplitude. This is a way of adding the PSF to the model.
          This method is used in the `self.model*` functions.
        :param dx: float, x coordinate of the point source to be modelled
        :param dy: float, y coordinate of the point source to be modelled
        :param amplitude: some kind of flux.
        :param psf: the actual array to be interpolated onto the model.
        :return: the downscaled, translated and multiplied by amplitude PSF, ready to be added to the model.
        """
        outshape = self.X.shape
        supersampling = self.upsampling_factor
        scale = jnp.array((1. / supersampling, 1. / supersampling))
        inishape = psf.shape[1]
        # assuming square psf
        outoffset = (inishape / 2.) / supersampling
        outoffsetx = outoffset + (outshape[0] - inishape) / supersampling
        outoffsety = outoffset + (outshape[1] - inishape) / supersampling

        # assuming
        translation = jnp.array((dy + outoffsetx, dx + outoffsety))
        out = scale_and_translate(psf, outshape, (0, 1), scale, translation, method='bicubic')
        return amplitude * out

    @partial(jit, static_argnums=(0,))
    def model_no_galaxy(self, params):
        """
         interpolates the PSF at the positions and amplitudes specified by the `params` dictionary, in each band.
        :param params: dictionary of parameters
        :return: 3D array containing the model.
        """

        x1, y1, x2, y2 = params['positions']

        A1s = jnp.array([params[band][0] for band in self.bands])
        A2s = jnp.array([params[band][1] for band in self.bands])

        xs1 = jnp.array([x1 + params[f'offsets_{band}'][0] for band in self.bands])
        ys1 = jnp.array([y1 + params[f'offsets_{band}'][1] for band in self.bands])
        xs2 = jnp.array([x2 + params[f'offsets_{band}'][0] for band in self.bands])
        ys2 = jnp.array([y2 + params[f'offsets_{band}'][1] for band in self.bands])
        vecpsf = vmap(lambda x, y, a, psf: self.translate_and_scale_psf(x, y, a, psf), in_axes=(0, 0, 0, 0))
        p1 = vecpsf(xs1, ys1, A1s, self.psf)
        p2 = vecpsf(xs2, ys2, A2s, self.psf)

        model = p1 + p2
        return model

    @partial(jit, static_argnums=(0,))
    def model_with_galaxy(self, params):
        """
         interpolates the PSF at the positions and amplitudes specified by the `params` dictionary, then adds a sersic.
         In each band.
        :param params: dictionary of parameters
        :return: 3D array containing the model.
        """
        # first only point sources:
        model = self.model_no_galaxy(params)
        # next, prepare the sersic params:
        xg, yg = params['galparams']['positions']
        xgs = jnp.array([xg + params[f'offsets_{band}'][0] for band in self.bands])
        ygs = jnp.array([yg + params[f'offsets_{band}'][1] for band in self.bands])
        I_es = jnp.array([params['galparams'][f'I_e_{band}'] for band in self.bands])
        psfs = jnp.array([self.translate_and_scale_psf(-0.5, -0.5, 1., self.psf[i]) for i in range(len(self.bands))])
        r_e, n, ellip, theta = params['galparams']['morphology']
        # we'll have to produce one model per band, vectorize
        vecsersic = vmap(
            lambda x, y, ie, psf: self.elliptical_sersic_profile_convolved(ie, r_e, x, y, n, ellip, theta, psf),
            in_axes=(0, 0, 0, 0)
        )
        # add the sersics to our model, already containing the point sources.
        model += vecsersic(xgs, ygs, I_es, psfs)

        return model

    def residuals_with_galaxy(self, params):
        model = self.model_with_galaxy(params)
        return (model - self.data) / self.noisemap

    def residuals_no_galaxy(self, params):
        model = self.model_no_galaxy(params)
        return (model - self.data) / self.noisemap

    def count_parameters(self, obj):
        if isinstance(obj, dict):
            # if the object is a dictionary, apply this function to each item
            return sum(self.count_parameters(v) for v in obj.values())
        elif isinstance(obj, list):
            # if the object is a list, apply this function to each item
            return sum(self.count_parameters(v) for v in obj)
        elif isinstance(obj, np.ndarray):
            # if the object is a numpy array, return the number of elements
            return obj.size
        else:
            # else, assume it's a scalar and return 1
            return 1

    def reduced_chi2_no_galaxy(self, params):
        residuals = self.residuals_no_galaxy(params)
        chi_squared = np.sum(residuals**2)
        dof = self.data.size - self.count_parameters(params)

        reduced_chi_squared = chi_squared / dof
        return reduced_chi_squared

    def reduced_chi2_with_galaxy(self, params):
        residuals = self.residuals_with_galaxy(params)
        chi_squared = np.sum(residuals**2)
        dof = self.data.size - self.count_parameters(params)

        reduced_chi_squared = chi_squared / dof
        return reduced_chi_squared

    def arrayify(self, dic):
        for key, value in dic.items():
            if isinstance(value, dict):
                self.arrayify(value)
            elif isinstance(value, list):
                dic[key] = np.array(value, dtype=np.float32)
            elif isinstance(value, np.ndarray) or isinstance(value, jnp.ndarray):
                pass
            elif isinstance(value, float):
                pass
                #dic[key] = np.array([value], dtype=np.float32)

    def optimize_no_galaxy(self, initial_guess=None, method='Nelder-Mead'):
        if not initial_guess:
            if not self.param_mediansampler_no_galaxy:
                self.sample(num_samples=1000, num_warmup=1000, include_galaxy=False)
            initial_guess = self.param_mediansampler_no_galaxy

        opt = ScipyMinimize(method=method, fun=self.reduced_chi2_no_galaxy)
        # yeah, run it a few times
        initial_guess = deepcopy(initial_guess)
        self.arrayify(initial_guess)
        res = opt.run(initial_guess)
        self.param_optim_no_galaxy = res.params
        return res.params

    def optimize_with_galaxy(self, initial_guess=None, method='Nelder-Mead'):
        if not initial_guess:
            if not self.param_mediansampler_with_galaxy:
                self.sample(num_samples=1000, num_warmup=1000, include_galaxy=True, force_galaxy_between_points=True)
            initial_guess = self.param_mediansampler_with_galaxy

        opt = ScipyMinimize(method=method, fun=self.reduced_chi2_with_galaxy)
        # yeah, run it a few times
        initial_guess = deepcopy(initial_guess)
        self.arrayify(initial_guess)
        res = opt.run(initial_guess)
        self.param_optim_with_galaxy = res.params
        return res.params

    def sample(self, num_warmup=20_000, num_samples=10_000, num_chains=1,
               position_scale=10., positions_prior_type="box", max_band_offset=1.,
               include_galaxy=True, force_galaxy_between_points=True):
        """
        Trying to fit our data without initial guess with a sampler. Advice: use an insanely large number of steps,
        because no more half measures. It takes approximately 2 minutes on a GPU for 100_000 steps so yeah ...

        :param num_warmup: steps for warmup
        :param num_samples: steps for actual sampling
        :param num_chains: number of chain to do in parallel. if gpu, prob. 1 is best unless multiple gpus.
        :param position_scale: float, default 10 (pixels), scale of the allowed regions for the positions.
                               Either width of a centered box, or scale of a centered gaussian depending on
                               `position_prior_type`
        :param positions_prior_type: string, either "gaussian" or "box".
        :param max_band_offset: float, max translation allowed between frames.
        :param include_galaxy: bool, whether we include extra parameters for a Sersic profile in the model.
        :param force_galaxy_between_points: bool, default True, whether the galaxy is forced to lie somewhere
                                            between the lensed images or not.
        :return: numpyro.infer MCMC class used to do the sampling here

        params are updated in class, then you can use the plot functions which will use the medians of the
        class attributes. (self.param_mediansampler_no_galaxy)
        """
        def numpyromodel(data, noise):
            # Flatten the images
            image_data_flat = data.flatten()
            image_uncertainties_flat = noise.flatten()
            # this basically allows us to use the numypro.plate context manager
            # below. Not very useful here, but indicates that each pixel
            # is independant. Some samplers can take advantage of this,
            # so let's do it this way just in case it becomes useful.

            _, sizey, sizex = data.shape

            params = {}

            if positions_prior_type == 'box':
                bs = position_scale/2

                x1 = numpyro.sample('x1', dist.Uniform(-bs, bs))
                y1 = numpyro.sample('y1', dist.Uniform(-bs, bs))

                x2 = numpyro.sample('x2', dist.Uniform(-bs, bs))
                y2 = numpyro.sample('y2', dist.Uniform(-bs, bs))

            elif positions_prior_type == 'gaussian':
                x1 = numpyro.sample('x1', dist.Normal(loc=0., scale=position_scale))
                y1 = numpyro.sample('y1', dist.Normal(loc=0., scale=position_scale))

                x2 = numpyro.sample('x2', dist.Normal(loc=0., scale=position_scale))
                y2 = numpyro.sample('y2', dist.Normal(loc=0., scale=position_scale))

            params['positions'] = (x1, y1, x2, y2)
            # ok, now populate the band-dependant params:
            for band in self.bands:
                A1 = numpyro.sample(f'A1_{band}', dist.Uniform(-100., 200.))  # since we normalize our data, this range
                A2 = numpyro.sample(f'A2_{band}', dist.Uniform(-100., 200.))  # should be fine ...
                params[band] = (A1, A2)
                dx = numpyro.sample(f'dx_{band}', dist.Uniform(-max_band_offset, max_band_offset))
                dy = numpyro.sample(f'dy_{band}', dist.Uniform(-max_band_offset, max_band_offset))
                params[f'offsets_{band}'] = dx, dy

            if not include_galaxy:
                # then we stop here and use the simple model.
                mod = self.model_no_galaxy(params)

            # buuut more stuff if we add a galaxy.
            if include_galaxy:

                if force_galaxy_between_points:
                    # ok, let's condition xg, yg to lie somewhere between the point sources
                    angle = numpyro.deterministic("angle", jnp.arctan2(jnp.array([y2 - y1]), jnp.array([x2 - x1]))[0])
                    length = numpyro.deterministic("length", ((x2-x1)**2 + (y2-y1)**2)**0.5)
                    scale = numpyro.sample("scale", dist.Uniform(0.2, 0.8))  # "between" source 1 and source 2.
                    xg = numpyro.deterministic("xg",  x1 + jnp.cos(angle) * length * scale)
                    yg = numpyro.deterministic("yg",  y1 + jnp.sin(angle) * length * scale)
                else:
                    if positions_prior_type == 'box':
                        xg = numpyro.sample('xg', dist.Uniform(-bs, bs))
                        yg = numpyro.sample('yg', dist.Uniform(-bs, bs))
                    elif positions_prior_type == 'gaussian':
                        xg = numpyro.sample('xg', dist.Normal(loc=0., scale=position_scale))
                        yg = numpyro.sample('yg', dist.Normal(loc=0., scale=position_scale))

                params['galparams'] = {}
                params['galparams']['positions'] = (xg, yg)

                # sersic profile params
                r_e = numpyro.sample('r_e', dist.Uniform(0.5, 4.))
                ellip = numpyro.sample('ellip', dist.Uniform(0., 0.9))
                theta = numpyro.sample('theta', dist.Uniform(0, 2*np.pi))
                n = numpyro.sample('n', dist.Uniform(1.5, 6.5))
                params['galparams']['morphology'] = r_e, n, ellip, theta

                # band dependant params
                params['galparams']['I_e'] = {}
                for band in self.bands:
                    params['galparams'][f'I_e_{band}'] = numpyro.sample(f'I_e_{band}', dist.Uniform(0., 30.))

                mod = self.model_with_galaxy(params)

            # likelihood, gaussian errors
            with numpyro.plate('data', len(image_data_flat)):
                numpyro.sample('obs', dist.Normal(mod.flatten(), image_uncertainties_flat), obs=image_data_flat)

        # run MCMC, this barkerMH kernel seems to be working well.
        # NUTS was getting stuck too much.
        kernel = numpyro.infer.BarkerMH(numpyromodel)
        mcmc = numpyro.infer.MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, self.data, self.noisemap)
        mcmc.print_summary()
        self.unpack_params_mcmc(mcmc, include_galaxy)
        self.mcmc = mcmc
        return mcmc

    def unpack_params_mcmc(self, mcmc, include_galaxy):
        # and now the great unpacking.
        pps = {'galparams': {}}
        medians = {k: np.median(val) for k, val in mcmc.get_samples().items()}

        pps['positions'] = [medians[k] for k in ('x1', 'y1', 'x2', 'y2')]

        for band in self.bands:
            pps[band] = [medians[k] for k in (f'A1_{band}', f'A2_{band}')]
            pps[f'offsets_{band}'] = [medians[k] for k in (f'dx_{band}', f'dy_{band}')]

        if include_galaxy:
            for band in self.bands:
                pps['galparams'][f'I_e_{band}'] = medians[f'I_e_{band}']
            pps['galparams']['positions'] = [medians[k] for k in ('xg', 'yg')]
            pps['galparams']['morphology'] = [medians[k] for k in ('r_e', 'n', 'ellip', 'theta')]
            # store the medians
            self.param_mediansampler_with_galaxy = pps
        else:
            self.param_mediansampler_no_galaxy = pps

    def _plot_model(self, params, modelfunc):
        """
        :param params: dictionary of parameters.
        :param modelfunc: Which class method should we use to transform the parameters into a model?
        :return: matplotlib figure and axes.
        """
        nrow = len(self.bands)
        fig, axs = plt.subplots(nrow, 3, figsize=(6, 1.8*nrow))
        axs = axs.reshape((nrow, 3))
        mod = modelfunc(params)
        data = self.data
        noise = self.noisemap
        res = (data - mod) / noise
        x1, y1, x2, y2 = params['positions']

        offsx = (self.X.shape[1] - 1)/2.
        offsy = (self.X.shape[0] - 1) / 2.
        x1, x2 = x1 + offsx, x2 + offsx
        y1, y2 = y1 + offsy, y2 + offsy
        if 'galparams' in params and 'positions' in params['galparams']:
            xg, yg = params['galparams']['positions']
            xg += offsx
            yg += offsy

        for i, band in enumerate(self.bands):
            i0 = axs[i, 0].imshow(data[i], origin='lower')
            i1 = axs[i, 1].imshow(mod[i], origin='lower')
            i2 = axs[i, 2].imshow(res[i], origin='lower')
            dx, dy = params[f'offsets_{band}']

            for j, (im, title) in enumerate(zip([i0, i1, i2], ['data', 'model', 'norm. res.'])):
                divider = make_axes_locatable(axs[i, j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
                axs[i, j].axis('off')
                axs[i, j].set_title(f'{band} {title}')
                axs[i, j].plot([x1 + dx, x2 + dx], [y1 + dy, y2 + dy], 'x', color='red', ms=3, alpha=0.8)
                if 'galparams' in params and 'positions' in params['galparams']:
                    axs[i, j].plot([xg + dx], [yg + dy], 'x', color='orange', ms=3, alpha=0.8)
        plt.tight_layout()
        return fig, axs

    def _plot_model_color(self, params, modelfunc, band_indexes):
        """
          do a color plot of our model!
        :param params: dictionary of parameters
        :param modelfunc: class method to use to produce the models
        :param band_indexes: list, which bands should be used? must be a list of integers of length 3.
        :return: matplotlib figure, axes
        """

        fig, axs = plt.subplots(1, 3, figsize=(6, 2))

        cdata = np.moveaxis(self.data, 0, 2)
        scale = np.nanpercentile(cdata, 99.5)
        i0 = axs[0].imshow(cdata / scale, origin='lower')

        mod = modelfunc(params)
        mod = np.moveaxis(mod, 0, 2)
        i1 = axs[1].imshow(mod / scale, origin='lower')
        res = (cdata - mod) / np.moveaxis(self.noisemap, 0, 2)
        res -= np.nanmin(res)
        res /= np.nanmax(np.abs(res))
        i2 = axs[2].imshow(res, origin='lower')

        for j, (im, ax, title) in enumerate(zip([i0, i1, i2], axs, ['data', 'model', 'norm. res.'])):
            ax.set_title(title)
            ax.axis('off')

        plt.show()
        return fig, axs

    def plot_model_no_galaxy(self, params=None):
        """
        plots the data, model and residuals. Tries to use the galaxyless model and parameters.
        :param params: dictionary of parameters, default None (then uses the class attribute saved when running the
                       sampler)
        :return: matplotlib figure and axes.
        """
        if params is None:
            if self.param_mediansampler_no_galaxy is not None:
                params = self.param_mediansampler_no_galaxy
                print('Used median params from sampling')
            elif self.param_optim_no_galaxy is not None:
                params = self.param_optim_no_galaxy
                print('Used params from least-squares optimization')
            else:
                raise RuntimeError('Run an optimizer or sampler first')

        return self._plot_model(params, self.model_no_galaxy)

    def plot_model_with_galaxy(self, params=None):
        """
        plots the data, model and residuals. Tries to use the model and parameters with galaxy.
        :param params: dictionary of parameters, default None (then uses the class attribute saved when running the
                       sampler)
        :return: matplotlib figure and axes.
        """
        if params is None:
            if self.param_mediansampler_with_galaxy is not None:
                params = self.param_mediansampler_with_galaxy
                print('Used median params from sampling')
            elif self.param_optim_with_galaxy is not None:
                params = self.param_optim_with_galaxy
                print('Used params from least-squares optimization')
            else:
                raise RuntimeError('Run an optimizer or sampler first')

        return self._plot_model(params, self.model_with_galaxy)

    def view_data(self):
        """
        mostly a debug function, check the data, noisemaps and PSFs in every band.
        :return: nothing
        """
        psf = self.psf
        N = len(psf)
        data = self.data
        noisemap = self.noisemap
        fig, axs = plt.subplots(nrows=3, ncols=N)
        axs = axs.reshape((3, N))
        for i, (b, d, n, p) in enumerate(zip(self.bands, data, noisemap, psf)):
            axs[0, i].set_title(b)
            axs[0, i].imshow(d, origin='lower')
            axs[1, i].imshow(n, origin='lower')
            axs[2, i].imshow(p**0.5, origin='lower')
            for ax in axs[:, i]:
                ax.set_xticks([])
                ax.set_yticks([])
            axs[0, 0].set_ylabel('data')
            axs[1, 0].set_ylabel('noisemap')
            axs[2, 0].set_ylabel('PSF**0.5')

        plt.tight_layout()
        plt.show()

    def get_sersic_SED(self, params):

        # prepare the sersic params:
        xg, yg = params['galparams']['positions']
        xgs = jnp.array([xg + params[f'offsets_{band}'][0] for band in self.bands])
        ygs = jnp.array([yg + params[f'offsets_{band}'][1] for band in self.bands])
        I_es = jnp.array([params['galparams'][f'I_e_{band}'] for band in self.bands])
        r_e, n, ellip, theta = params['galparams']['morphology']
        # we'll have to produce one model per band, vectorize
        vecsersic = vmap(
            lambda x, y, ie, psf: self.elliptical_sersic_profile_convolved(ie, r_e, x, y, n, ellip, theta, psf),
            in_axes=(0, 0, 0, 0)
        )
        psfs = jnp.array([self.translate_and_scale_psf(-0.5, -0.5, 1., self.psf[i]) for i in range(len(self.bands))])
        sersics = vecsersic(xgs, ygs, I_es, psfs)
        fluxes = np.array([np.sum(sersics[i]) for i in range(len(self.bands))])
        magnitudes = -2.5 * np.log10(self.scale * fluxes)
        return magnitudes

    def get_ps_SEDs(self, params):
        x1, y1, x2, y2 = params['positions']

        A1s = jnp.array([params[band][0] for band in self.bands])
        A2s = jnp.array([params[band][1] for band in self.bands])

        xs1 = jnp.array([x1 + params[f'offsets_{band}'][0] for band in self.bands])
        ys1 = jnp.array([y1 + params[f'offsets_{band}'][1] for band in self.bands])
        xs2 = jnp.array([x2 + params[f'offsets_{band}'][0] for band in self.bands])
        ys2 = jnp.array([y2 + params[f'offsets_{band}'][1] for band in self.bands])
        vecpsf = vmap(lambda x, y, a, psf: self.translate_and_scale_psf(x, y, a, psf), in_axes=(0, 0, 0, 0))
        p1 = vecpsf(xs1, ys1, A1s, self.psf)
        p2 = vecpsf(xs2, ys2, A2s, self.psf)
        fluxes1 = np.array([np.sum(p1[i]) for i in range(len(self.bands))])
        fluxes2 = np.array([np.sum(p2[i]) for i in range(len(self.bands))])
        mags1 = -2.5 * np.log10(self.scale * fluxes1)
        mags2 = -2.5 * np.log10(self.scale * fluxes2)

        return mags1, mags2

    @staticmethod
    def get_separation(params, pixelsize):
        x1, y1, x2, y2 = params['positions']
        sep = (x1 - x2)**2 + (y1 - y2)**2
        return pixelsize * sep**0.5

    @staticmethod
    def convert_lists_to_arrays(data):
        """
        utility function used when saving the model into an hdf5 file.
        :param data: anything, if it's a list or a float, we'll make it an array that can be stored in an hdf5 file.
        :return: hdf5-i-fied data
        """
        if isinstance(data, dict):
            return {key: DoublyLensedQuasarFitter.convert_lists_to_arrays(value) for key, value in data.items()}
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, float):
            return np.array([data])
        elif data is None:
            return np.array([])
        else:
            return data

    @staticmethod
    def convert_arrays_to_lists(data):
        """
        reverse process of the sister function above.
        :param data: some array
        :return: de-hdf5-i-fied data
        """
        if isinstance(data, dict):
            return {key: DoublyLensedQuasarFitter.convert_arrays_to_lists(value) for key, value in data.items()}
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    def to_hdf5(self, filename):
        """
        saves the model to an hdf5 file that can be loaded later with `from_hdf5`.
        :param filename: string or Path, where should we save our model?
        :return: Nothing
        """
        # attributes to save
        attributes = {
            "data": self.data,
            "noisemap": self.noisemap,
            "psf": self.psf,
            "scale": np.array([self.scale]),
            "bands": np.array([np.string_(band) for band in self.bands]),  # encode strings as bytes
            "upsampling_factor": np.array([self.upsampling_factor]),
            "param_optim_no_galaxy": self.convert_lists_to_arrays(self.param_optim_no_galaxy),
            "param_optim_with_galaxy": self.convert_lists_to_arrays(self.param_optim_with_galaxy),
            "param_mediansampler_no_galaxy": self.convert_lists_to_arrays(self.param_mediansampler_no_galaxy),
            "param_mediansampler_with_galaxy": self.convert_lists_to_arrays(self.param_mediansampler_with_galaxy),
            "X": self.X,
            "Y": self.Y,
        }

        save_dict_to_hdf5(filename, attributes)

    @classmethod
    def from_hdf5(cls, filename):
        """
         Creates a new DoublyLensedQuasarFitter instance, restoring previously saved data and parameter.
        :param filename: string or Path, where the dump of the model is located.
        :return: a DoublyLensedQuasarFitter instance
        """
        # Load the attributes dict from the hdf5 file
        attributes = load_dict_from_hdf5(filename)

        # convert any ndarray attributes that are not data back to their original type
        for key, value in attributes.items():
            if isinstance(value, np.ndarray) and key not in ["data", "noisemap", "psf", "X", "Y"]:
                if key == "bands":  # decode bytes back to strings here ...
                    attributes[key] = [band.decode() for band in value]
                else:
                    if len(value) == 0:
                        attributes[key] = None
                    else:
                        attributes[key] = value[0]

        # convert any ndarray attributes that were list back to their original type, cuz we can.
        pnogalaxy = attributes["param_mediansampler_no_galaxy"]
        attributes["param_mediansampler_no_galaxy"] = cls.convert_arrays_to_lists(pnogalaxy)
        pwgalaxy = attributes["param_mediansampler_with_galaxy"]
        attributes["param_mediansampler_with_galaxy"] = cls.convert_arrays_to_lists(pwgalaxy)

        # new object without calling __init__
        new_instance = cls.__new__(cls)

        # adding the attributes
        for key, value in attributes.items():
            setattr(new_instance, key, value)
        new_instance.mcmc = None

        return new_instance


if __name__ == "__main__":

    # ff = '/tmp/test/cutouts_legacysurvey_J1037+0018_cutouts.h5'
    # ff = '/tmp/test/cutouts_panstarrs_J1037+0018_cutouts.h5'
    ff = "/tmp/test/cutouts_legacysurvey_J2122-1621_cutouts.h5"
    # ff = '/tmp/test/cutouts_panstarrs_J2122-1621_cutouts.h5'
    ff = '/scratch/diff_img_paper/survey_data_and_modelling/PSJ0557-2959/legacysurvey/cutouts_legacysurvey_J0557-2959_cutouts.h5'
    modelm = prepare_fitter_from_h5(ff)

    modelm.param_mediansampler_with_galaxy = {'galparams': {'positions': [0, 0], 'morphology': [1.0710126, 1., 0.0, 0.],
                                                            'I_e_g': 0.5, 'I_e_i': .1,
                                                            'I_e_r': 2.0, 'I_e_z': 5.0},
                                              'positions': [0., 1., 0., 1.], 'g': [0., 0.],
                                              'offsets_g': [0.0, 0.0], 'i': [1., 1.],
                                              'offsets_i': [0.0, 0.0], 'r': [1., 1.],
                                              'offsets_r': [0.0, 0.0], 'z': [1., 1.],
                                              'offsets_z': [0.0, 0.0]}
    modelm.param_mediansampler_no_galaxy = {'positions': [0., 1., 0., 0.], 'g': [1., 1.],
                                            'offsets_g': [0.0, 0.0], 'i': [1., 1.],
                                            'offsets_i': [0.0, 0.0], 'r': [1., 1.],
                                            'offsets_r': [0.0, 0.0], 'z': [1., 1.],
                                            'offsets_z': [0.0, 0.0]}

    # out = modelm.sample(num_warmup=1000, num_samples=500, position_scale=15.0, include_galaxy=False)
    # print(modelm.param_mediansampler_with_galaxy)
    # modelm.plot_model_no_galaxy()
    # plt.show()
    modelm.param_mediansampler_no_galaxy = None
    modelm.to_hdf5('/tmp/wow.h5')
    m2 = DoublyLensedQuasarFitter.from_hdf5('/tmp/wow.h5')
    m2.sample(include_galaxy=False)
    m2.plot_model_no_galaxy()
    m2.plot_model_with_galaxy()
