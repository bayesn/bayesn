import numpy as np
from astropy.cosmology import FlatLambdaCDM
from jax import numpy as jnp

class Distmod2Redshift:
    """
    Class to build an interpolator to convert distance modulus to comoving
    distance in `Mpc`. Choice of `h` is determined when calling the
    `__call__` method.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.28, zmin_interp=1e-8, zmax_interp=0.5,
                 npoints_interp=1000):
        cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        mu_grid = cosmo.distmod(z_grid).value

        self._mu_grid = jnp.array(mu_grid)
        self._log_z_grid = jnp.log(z_grid)
        self._f = lambda x: jnp.interp(x, self._mu_grid, self._log_z_grid)

    def __call__(self, mu, h=0.7324, return_log=False):
        if return_log:
            return self._f(mu + 5 * jnp.log10(h))

        return jnp.exp(self._f(mu + 5 * jnp.log10(h)))