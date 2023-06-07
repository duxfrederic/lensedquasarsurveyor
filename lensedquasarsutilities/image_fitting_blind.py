import numpy as np
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
import numpyro
import numpyro.distributions as dist
from numpyro.infer import HMCECS, MCMC, NUTS


from lensedquasarsutilities.io import load_dict_from_hdf5


def prepare_model_from_h5(h5file):
    """

    :param h5file: string or Path, path to an h5-file containing cutouts with WCS, PSFs, as prepared by
                                   download_and_extract and estimate_psf_from_extracted_h5.
    :return: list of SimpleLensedQuasarModel instances, one per band
    """
    data = load_dict_from_hdf5(h5file)
    bands = list(data['lens'].keys())

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
    lensdata, noisedata, psfdata = np.array(lensdata), np.array(noisedata), np.array(psfdata)
    model = SimpleLensedQuasarModel(lensdata, noisedata, psfdata, upsampling_factor, bands)

    return model


class SimpleLensedQuasarModel:
    def __init__(self, data, noisemap, narrowpsf, upsampling_factor, bandnames):
        """

        :param data: 2D np array
        :param noisemap: 2D np array
        :param narrowpsf: 2D np array
        :param upsampling_factor: int

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
        self.bands = bandnames

        nband, Nx, Ny = shape
        nband2, nx, ny = narrowpsf.shape
        assert nband == nband2 == len(bandnames)

        # narrow psf --> psf by convolving with gaussian kernel fwhm=2
        self.psf = np.array([gaussian_filter(e, 0.85) for e in narrowpsf])

        x, y = np.linspace(-(Ny-1)/2, (Ny-1)/2, Ny), np.linspace(-(Nx-1)/2, (Nx-1)/2, Nx)
        self.X, self.Y = np.meshgrid(x, y)

        self.param_optim_with_galaxy = None
        self.param_optim_no_galaxy = None
        self.param_mediansampler_no_galaxy = None
        self.param_mediansampler_with_galaxy = None

    #@partial(jit, static_argnums=(0,))
    def elliptical_sersic_profile(self, I_e, r_e, x0, y0, n, ellip, theta):
        # x0 and y0 must be converted to model coordinates
        # x0 -= (self.X.shape[1] - 1) / 2.
        # y0 -= (self.X.shape[0] - 1) / 2.
        # Ellipticity and orientation parameters
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
        profile = self.elliptical_sersic_profile(I_e, r_e, x0, y0, n, ellip, theta)
        return fftconvolve(profile, psf, mode='same')

    #@partial(jit, static_argnums=(0,))
    def translate_and_scale_psf(self, dx, dy, amplitude, psf):
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
        model = self.model_no_galaxy(params)

        xg, yg = params['galparams']['positions']

        xgs = jnp.array([xg + params[f'offsets_{band}'][0] for band in self.bands])
        ygs = jnp.array([yg + params[f'offsets_{band}'][1] for band in self.bands])
        I_es = jnp.array([params['galparams'][f'I_e_{band}'] for band in self.bands])

        # vecpsf = vmap(lambda x, y, a, psf: self.translate_and_scale_psf(x, y, a, psf), in_axes=(0, 0, 0, 0))
        # p1 = vecpsf(xgs, ygs, I_es, self.psf)
        # return model + p1

        r_e, n, ellip, theta = params['galparams']['morphology']
        vecsersic = vmap(lambda x, y, ie, psf: self.elliptical_sersic_profile_convolved(ie, r_e, x, y, n, ellip, theta, psf),
                         in_axes=(0, 0, 0, 0))
        psfs = jnp.array([self.translate_and_scale_psf(-0.5, -0.5, 1., self.psf[i]) for i in range(len(self.bands))])
        model += vecsersic(xgs, ygs, I_es, psfs)

        # vecsersic = vmap(lambda x, y, ie: self.elliptical_sersic_profile(ie, r_e, x, y, n, ellip, theta),
        #                  in_axes=(0, 0, 0))
        # model += vecsersic(xgs, ygs, I_es)

        return model

    #@partial(jit, static_argnums=(0,))
    def residuals_with_galaxy(self, params):
        model = self.model_with_galaxy(params)
        return ((model - self.data) / self.noisemap).flatten()

    #@partial(jit, static_argnums=(0,))
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

    def optimize_no_galaxy(self, initial_guess):
        res = least_squares(self.residuals_no_galaxy, initial_guess)
        self.param_optim_no_galaxy = res.x
        return res.x

    def optimize_with_galaxy(self, initial_guess):
        res = least_squares(self.residuals_with_galaxy, initial_guess)
        self.param_optim_with_galaxy = res.x
        return res.x

    def sample_with_galaxy(self, num_warmup=500, num_samples=500, num_chains=1, position_scale=10.):

        def numpyromodel(data, noise):
            # Flatten the images
            image_data_flat = data.flatten()
            image_uncertainties_flat = noise.flatten()
            # this basically allows us to use the numypro.plate context manager
            # below. Not very useful here, but indicates that each pixel
            # is independant. Some samplers can take advantage of this,
            # so let's do it this way.

            _, sizey, sizex = data.shape

            params = {}

            bs = position_scale/2
            # more likely to be in the center, let's use centered gaussian priors
            x1 = numpyro.sample('x1', dist.Uniform(-bs, bs))
            y1 = numpyro.sample('y1', dist.Uniform(-bs, bs))

            x2 = numpyro.sample('x2', dist.Uniform(-bs, bs))
            y2 = numpyro.sample('y2', dist.Uniform(-bs, bs))

            # x1 = numpyro.sample('x1', dist.Normal(loc=0., scale=position_scale))
            # y1 = numpyro.sample('y1', dist.Normal(loc=0., scale=position_scale))
            #
            # x2 = numpyro.sample('x2', dist.Normal(loc=0., scale=position_scale))
            # y2 = numpyro.sample('y2', dist.Normal(loc=0., scale=position_scale))

            params['positions'] = (x1, y1, x2, y2)

            # ok, let's condition xg, yg to lie somewhere between the point sources
            angle = numpyro.deterministic("angle", jnp.arctan2(jnp.array([y2 - y1]), jnp.array([x2 - x1]))[0])
            length = numpyro.deterministic("length", ((x2-x1)**2 + (y2-y1)**2)**0.5)
            scale = numpyro.sample("scale", dist.Uniform(0.2, 0.8))  # "between" source 1 and source 2.
            xg = numpyro.deterministic("xg",  x1 + jnp.cos(angle) * length * scale)
            yg = numpyro.deterministic("yg",  y1 + jnp.sin(angle) * length * scale)

            # xg = numpyro.sample('xg', dist.Normal(loc=0., scale=position_scale))
            # yg = numpyro.sample('yg', dist.Normal(loc=0., scale=position_scale))

            params['galparams'] = {}
            params['galparams']['positions'] = (xg, yg)

            r_e = numpyro.sample('r_e', dist.Uniform(0.5, 4.))
            ellip = numpyro.sample('ellip', dist.Uniform(0., 0.9))
            theta = numpyro.sample('theta', dist.Uniform(0, 2*np.pi))
            n = numpyro.sample('n', dist.Normal(loc=4.0, scale=0.5))
            params['galparams']['morphology'] = r_e, n, ellip, theta

            params['galparams']['I_e'] = {}
            for i, band in enumerate(self.bands):
                A1 = numpyro.sample(f'A1_{band}', dist.Uniform(5., 140.))
                A2 = numpyro.sample(f'A2_{band}', dist.Uniform(5., 140.))
                params[band] = (A1, A2)
                params['galparams'][f'I_e_{band}'] = numpyro.sample(f'I_e_{band}', dist.Uniform(0., 30.))

                dx = numpyro.sample(f'dx_{band}', dist.Uniform(-0.1, 0.1))
                dy = numpyro.sample(f'dy_{band}', dist.Uniform(-0.1, 0.1))
                params[f'offsets_{band}'] = dx, dy

            mod = self.model_with_galaxy(params)

            # likelihood, gaussian errors
            with numpyro.plate('data', len(image_data_flat)):
                numpyro.sample('obs', dist.Normal(mod.flatten(), image_uncertainties_flat), obs=image_data_flat)

        # run MCMC
        nuts_kernel = numpyro.infer.BarkerMH(numpyromodel)
        mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, self.data, self.noisemap)
        mcmc.print_summary()

        pps = {'galparams': {}}
        medians = {k: np.median(val) for k, val in mcmc.get_samples().items()}

        pps['positions'] = [medians[k] for k in ('x1', 'y1', 'x2', 'y2')]
        if 'xg' not in medians:
            # then we forced the midpoint
            x1, y1, x2, y2 = pps['positions']
            xg, yg = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            pps['galparams']['positions'] = xg, yg
        else:
            pps['galparams']['positions'] = [medians[k] for k in ('xg', 'yg')]
        pps['galparams']['morphology'] = [medians[k] for k in ('r_e', 'n', 'ellip', 'theta')]

        for i, band in enumerate(self.bands):
            pps[band] = [medians[k] for k in (f'A1_{band}', f'A2_{band}')]
            pps['galparams'][f'I_e_{band}'] = medians[f'I_e_{band}']
            pps[f'offsets_{band}'] = [medians[k] for k in (f'dx_{band}', f'dy_{band}')]

        self.param_mediansampler_with_galaxy = pps
        return mcmc

    def sample_no_galaxy(self, num_warmup=500, num_samples=500, num_chains=1, position_scale=5.):
        """

        :param num_warmup: steps for warmup
        :param num_samples: steps for actual sampling
        :param num_chains: number of chain to do in parallel. if gpu, prob. 1 is best unless multiple gpus.
        :param position_scale: scale of the gaussian prior used for positions, in pixels.
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
            # so let's do it this way.

            _, sizey, sizex = data.shape

            params = {}

            # more likely to be in the center, let's use centere gaussian priors
            x1 = numpyro.sample('x1', dist.Normal(loc=0., scale=position_scale))
            y1 = numpyro.sample('y1', dist.Normal(loc=0., scale=position_scale))

            x2 = numpyro.sample('x2', dist.Normal(loc=0., scale=position_scale))
            y2 = numpyro.sample('y2', dist.Normal(loc=0., scale=position_scale))

            params['positions'] = (x1, y1, x2, y2)

            for i, band in enumerate(self.bands):
                A1 = numpyro.sample(f'A1_{band}', dist.Uniform(5., 140.))
                A2 = numpyro.sample(f'A2_{band}', dist.Uniform(5., 140.))
                params[band] = (A1, A2)

                dx = numpyro.sample(f'dx_{band}', dist.Uniform(-0.1, 0.1))
                dy = numpyro.sample(f'dy_{band}', dist.Uniform(-0.1, 0.1))
                params[f'offsets_{band}'] = dx, dy

            mod = self.model_no_galaxy(params)

            # likelihood, gaussian errors
            with numpyro.plate('data', len(image_data_flat)):
                numpyro.sample('obs', dist.Normal(mod.flatten(), image_uncertainties_flat), obs=image_data_flat)

        nuts_kernel = numpyro.infer.BarkerMH(numpyromodel)
        mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, self.data, self.noisemap)
        mcmc.print_summary()
        pps = {}
        medians = {k: np.median(val) for k, val in mcmc.get_samples().items()}
        pps['positions'] = [medians[k] for k in ('x1', 'y1', 'x2', 'y2')]

        for i, band in enumerate(self.bands):
            pps[band] = [medians[k] for k in (f'A1_{band}', f'A2_{band}')]
            pps[f'offsets_{band}'] = [medians[k] for k in (f'dx_{band}', f'dy_{band}')]

        self.param_mediansampler_no_galaxy = pps
        return mcmc

    def _plot_model(self, params, modelfunc):
        nrow = len(self.bands)
        fig, axs = plt.subplots(nrow, 3, figsize=(6, 2*nrow))
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
        if 'galparams' in params:
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
                axs[i, j].plot([x1+dx, x2+dx], [y1+dy, y2+dy], 'x', color='red')
                if 'galparams' in params:
                    axs[i, j].plot([xg+dx], [yg+dy], 'x', color='orange')
        plt.tight_layout()
        return fig, axs

    def _plot_model_color(self, params, modelfunc):

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


if __name__ == "__main__":

    # ff = '/tmp/test/cutouts_legacysurvey_J1037+0018_cutouts.h5'
    # ff = '/tmp/test/cutouts_panstarrs_J1037+0018_cutouts.h5'
    ff = "/tmp/test/cutouts_legacysurvey_J2122-1621_cutouts.h5"
    # ff = '/tmp/test/cutouts_panstarrs_J2122-1621_cutouts.h5'
    modelm = prepare_model_from_h5(ff)
    # param =  {'positions': (5, 5, -3, -5), 'g': (200., 105.), 'r': (100., 130.),
    #                                        'i': (90.0, 100.), 'z': (80.0, 100.)}
    # hi = modelm.model_no_galaxy(param)
    # hi2 = np.moveaxis(hi, 0, 2)
    # hi2 /= np.max(hi2)
    # plt.imshow(hi2, origin='lower')
    # plt.show()

    # param = {'positions': (0, 0, 0, 0), 'g': (200., 105.), 'r': (100., 130.),
    #                                        'i': (90.0, 100.), 'z': (80.0, 100.),
    #          'galparams': {'positions': (0., 0.), 'morphology': (10., 1.5, 0.1, 0.2),
    #                        'I_e': {'r': 0.0001, 'i': 0.005, 'g': 0.005, 'z': 0.006}
    #           }
    #          }
    #
    # hi2 = np.moveaxis(hi, 0, 2)
    # hi2 /= np.max(hi2)
    # plt.imshow(hi2, origin='lower')
    # plt.show()
    #
    # out = modelm.sample_with_galaxy(num_samples=100, num_warmup=100)
    # modelm.plot_model_with_galaxy()
    # modelm.plot_model_with_galaxy()

    modelm.param_mediansampler_with_galaxy = {'galparams': {'positions': [0, 0], 'morphology': [1.0710126, 1, 0.0,0],
                                                            'I_e_g.00000': 0.5, 'I_e_i.00000': .1, 'I_e_r.00000': 2.0, 'I_e_z.00000': 5.0},
                                              'positions': [0, 10, 0, -10], 'g.00000': [0., 0.],
                                              'offsets_g.00000': [0., 0.], 'i.00000': [1., 1.],
                                              'offsets_i.00000': [0., -0.0], 'r.00000': [1., 1.],
                                              'offsets_r.00000': [0.0, -0.0], 'z.00000': [1.0, 1.],
                                              'offsets_z.00000': [0.0, -0.0]}
    modelm.param_mediansampler_no_galaxy = {  'positions': [0, 10, 0, 0], 'g.00000': [1., 1.],
                                              'offsets_g.00000': [0., 0.], 'i.00000': [1., 1.],
                                              'offsets_i.00000': [0., -0.0], 'r.00000': [1., 1.],
                                              'offsets_r.00000': [0.0, -0.0], 'z.00000': [1.0, 1.],
                                              'offsets_z.00000': [0.0, -0.0]}

    # out = modelm.sample_no_galaxy(num_warmup=1000, num_samples=500, position_scale=10.0)
    # modelm.plot_model_no_galaxy()
    out = modelm.sample_with_galaxy(num_warmup=4000, num_samples=2500, position_scale=15.0)
    print(modelm.param_mediansampler_with_galaxy)
    modelm.plot_model_with_galaxy()
    plt.show()
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