from scipy.optimize import least_squares
from jax.scipy.stats import norm
from jax import jit
from functools import partial
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist

from starred.utils.generic_utils import pad_and_convolve_fft, Downsample


class SimpleLensedQuasarModel:
    def __init__(self, data, noisemap, narrowpsf, upsampling_factor, pixel_size=None):
        shape = data.shape
        assert shape == noisemap.shape

        self.data = data
        self.noisemap = noisemap
        self.psf = narrowpsf
        self.upsampling_factor = upsampling_factor
        self.pixel_size = pixel_size

        Nx, Ny = shape

        x, y = np.arange(-Ny//2, Ny//2), np.arange(-Nx//2, Nx//2)
        self.X, self.Y = np.meshgrid(x, y)

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

    def create_model_no_galaxy(self, x1, y1, A1, x2, y2, A2):
        psf1 = self.gaussian_psf(x1, y1, A1)
        psf2 = self.gaussian_psf(x2, y2, A2)

        model = psf1 + psf2

        return model

    def create_model_with_galaxy(self, x1, y1, A1, x2, y2, A2, xg, yg, I_e, r_e, n, ellip, theta):
        psfs = self.create_model_no_galaxy(x1, y1, A1, x2, y2, A2)

        sersic = self.elliptical_sersic_profile(I_e, r_e, n, xg, yg, ellip, theta)

        model = sersic + psfs

        return model

    def down_resolution(self, model):
        return Downsample(pad_and_convolve_fft(model, self.psf), self.upsampling_factor)

    @partial(jit, static_argnums=(0,))
    def residuals_with_galaxy(self, params):
        model = self.create_model_with_galaxy(*params)
        model = self.down_resolution(model)
        return ((model - data) / noisemap).flatten()

    @partial(jit, static_argnums=(0,))
    def residuals_no_galaxy(self, params):
        model = self.create_model_no_galaxy(*params)
        model = self.down_resolution(model)
        return ((model - data) / noisemap).flatten()

    def reduced_chi2_no_galaxy(self, params):
        residuals = self.residuals_no_galaxy(params)
        chi_squared = np.sum(residuals ** 2)
        dof = data.size - len(params)

        reduced_chi_squared = chi_squared / dof
        return reduced_chi_squared

    def reduced_chi2_with_galaxy(self, params):
        residuals = self.residuals_with_galaxy(params)
        chi_squared = np.sum(residuals ** 2)
        dof = data.size - len(params)

        reduced_chi_squared = chi_squared / dof
        return reduced_chi_squared

    def optimize_no_galaxy(self, initial_guess):
        res = least_squares(self.residuals_no_galaxy, initial_guess)
        return res.x

    def optimize_no_galaxy(self, initial_guess):
        res = least_squares(self.residuals_with_galaxy, initial_guess)
        return res.x

    def sample_with_galaxy(self, starting_point):

        def model(params):
            # Flatten the images
            image_data_flat = self.data.flatten()
            image_uncertainties_flat = self.noisemap.flatten()
            # this basically allows us to use the numypro.plate context manager
            # below. Not very useful here, but indicates that each pixel
            # is independant. Some samplers can take advantage of this,
            # so let's do it this way.

            # Unpack optimized parameters
            x1_opt, y1_opt, A1_opt, x2_opt, y2_opt, A2_opt, xg_opt, yg_opt, \
                I_e_opt, r_e_opt, n_opt, ellip_opt, theta_opt = params

            bounds = 5
            # Uniform priors centered around optimized parameters
            x1 = numpyro.sample('x1', dist.Uniform(x1_opt - bounds, x1_opt + bounds))
            y1 = numpyro.sample('y1', dist.Uniform(y1_opt - bounds, y1_opt + bounds))
            A1 = numpyro.sample('A1', dist.Uniform(A1_opt - bounds, A1_opt + bounds))

            x2 = numpyro.sample('x2', dist.Uniform(x2_opt - bounds, x2_opt + bounds))
            y2 = numpyro.sample('y2', dist.Uniform(y2_opt - bounds, y2_opt + bounds))
            A2 = numpyro.sample('A2', dist.Uniform(A2_opt - bounds, A2_opt + bounds))

            xg = numpyro.sample('xg', dist.Uniform(xg_opt - bounds, xg_opt + bounds))
            yg = numpyro.sample('yg', dist.Uniform(yg_opt - bounds, yg_opt + bounds))
            I_e = numpyro.sample('I_e', dist.Uniform(0.1 * I_e_opt, 10 * I_e_opt))
            r_e = numpyro.sample('r_e', dist.Uniform(0.1 * r_e_opt, 10 * r_e_opt))
            n = numpyro.sample('n', dist.Uniform(0.5, 10))
            ellip = numpyro.sample('ellip', dist.Uniform(0, 0.99))
            theta = numpyro.sample('theta', dist.Uniform(0, 2 * np.pi))

            # Model
            mod = self.create_model_with_galaxy(x1, y1, A1, x2, y2, A2, xg, yg, I_e, r_e, n, ellip, theta)
            mod = self.down_resolution(mod)

            # likelihood, gaussian errors
            with numpyro.plate('data', len(image_data_flat)):
                numpyro.sample('obs', dist.Normal(mod.flatten(), image_uncertainties_flat), obs=image_data_flat)

        # run MCMC
        nuts_kernel = numpyro.infer.NUTS(model)
        mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=300, num_samples=600)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, data, noisemap, starting_point)
        mcmc.print_summary()

    def sample_no_galaxy(self, starting_point):

        def model(params):
            # Flatten the images
            image_data_flat = self.data.flatten()
            image_uncertainties_flat = self.noisemap.flatten()
            # this basically allows us to use the numypro.plate context manager
            # below. Not very useful here, but indicates that each pixel
            # is independant. Some samplers can take advantage of this,
            # so let's do it this way.

            # Unpack optimized parameters
            x1_opt, y1_opt, A1_opt, x2_opt, y2_opt, A2_opt = params

            bounds = 5
            # Uniform priors centered around optimized parameters
            x1 = numpyro.sample('x1', dist.Uniform(x1_opt - bounds, x1_opt + bounds))
            y1 = numpyro.sample('y1', dist.Uniform(y1_opt - bounds, y1_opt + bounds))
            A1 = numpyro.sample('A1', dist.Uniform(A1_opt - bounds, A1_opt + bounds))

            x2 = numpyro.sample('x2', dist.Uniform(x2_opt - bounds, x2_opt + bounds))
            y2 = numpyro.sample('y2', dist.Uniform(y2_opt - bounds, y2_opt + bounds))
            A2 = numpyro.sample('A2', dist.Uniform(A2_opt - bounds, A2_opt + bounds))

            # Model
            mod = self.create_model_no_galaxy(x1, y1, A1, x2, y2, A2)
            mod = self.down_resolution(mod)

            # likelihood, gaussian errors
            with numpyro.plate('data', len(image_data_flat)):
                numpyro.sample('obs', dist.Normal(mod.flatten(), image_uncertainties_flat), obs=image_data_flat)

        # run MCMC
        nuts_kernel = numpyro.infer.NUTS(model)
        mcmc = numpyro.infer.MCMC(nuts_kernel, num_warmup=300, num_samples=600)
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, data, noisemap, starting_point)
        mcmc.print_summary()


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    psf = lambda x, y, x0, y0, A: A * np.exp(-0.2 * (x - x0)**2 - 0.17 * (y - y0)**2)

    # grid of small pixels
    X, Y = x, y = np.meshgrid(np.linspace(-32, 32, 128), np.linspace(-32, 32, 128))

    narrow_psf = psf(x, y, 0, 0, 1)
    narrow_psf /= narrow_psf.sum()
    modc = SimpleLensedQuasarModel(X, X, psf(X, Y, 0, 0, 1), 2)
    m = modc.create_model_with_galaxy(0, 10, 1, 0, -10, 1.5, 0, 0, 0.01, 5.0, 1.5, 0.5, 30)

    noise_scale = 0.0015
    data = modc.down_resolution(m)
    data += np.random.normal(loc=0, scale=noise_scale, size=data.shape)
    plt.imshow(data)
    noisemap = noise_scale * np.ones_like(data)
    modc.data = data
    modc.noisemap = noisemap

    plt.show()

