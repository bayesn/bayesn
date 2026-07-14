"""
PDF Utilities.

Defines utility classes for constructing empirical
inverse CDF objects that are jax friendly.

Pushing U(0,1) RVs through these gives samples from
the corresponding PDFs.
"""

import numpy as np
import jax.random as jr
import jax.numpy as jnp
from numpyro.distributions.util import is_prng_key

class NoName1():
    """
    No Name distribution of the 1st kind.
    
    Based on Ben Goodrich's StanCon2020 talk (https://github.com/bgoodri/StanCon2020).
    This class can be used to define an approximation to a distribution's inverse CDF
    using a Cheyshev series.
    
    Can be constructed using a precomputed set of quantiles, or from a provided set of
    samples from the distribution.
    
    Not guaranteed to be monotonic.
    
    Parameters
    ----------
    probs: array-like, optional
        Array of values in [0,1] where the inverse CDF (quantile function)
        is known (if `method="quantiles"`) or to be computed (if `method="samples"`).
        Must be provided if `method="quantiles"`.
    vals: array-like, optional
        Array of parameter values corresponding to the inverse CDF evaluated at `probs`.
        Must be provided if `method="quantiles"`.
    samples: array-like, optional
        Array of samples whose inverse CDF is to be approximated.
        Must be provided if `method="samples"`.
    K: int, optional
        Order of polynomial if `method="samples"`.
    method: "quantiles" or "samples"
        Specifies the kind of input the ICDF is constructed from. Default is "quantiles".
        
    Methods
    -------
    clenshaw:
        Clenshaw routine for evaluating a Chebyshev series
    icdf:
        Method for computing approximate inverse CDF
    sample:
        Sample from the distribution by transforming uniform RVs
        
    Attributes
    ----------
    K: int
        Degree of Chebyshev polynomial
    c: array
        Coefficients in Chebyshev series
    """
    def __init__(self, probs=None, vals=None, samples=None, K=None, method="quantiles"):
        # check probs are within [0,1]
        if probs is not None and (np.any(probs < 0) or np.any(probs > 1)):
                raise ValueError("All probs must be in [0,1]!")
        # set K if quantiles are provided by user
        if method == "quantiles":
            # catch insufficient info from user
            if probs is None or vals is None:
                raise ValueError('If method == "quantiles", probs and vals must be provided as arguments!')
            self.K = len(probs)-1 # order of the polynomial
        # compute quantiles if samples are provided by user
        elif method == "samples":
            # catch insufficient info from user
            if samples is None:
                raise ValueError('If method == "samples", samples must be provided as an argument!')
            if probs is None and K is None:
                raise ValueError('If method == "samples", one of probs or K must be provided as an argument!')
            if probs is not None and K is not None and len(probs) != K+1:
                raise ValueError('Both probs and K provided in an ambiguous way. Found len(probs) = {:d} but K = {:d}!'.format(len(probs), K))
            # pick chebyshev nodes if probs is None
            if probs is None:
                self.K = K
                probs = 0.5*(np.polynomial.chebyshev.chebpts1(self.K+1) + 1.0)
            else:
                self.K = len(probs)-1
            vals = np.quantile(samples, probs)
        # something unexpected happened
        else:
            raise ValueError('method = {} not recognised! Provide one of "quantiles" or "samples"!'.format(method))
        # set up the locations in [-1, 1]
        x = 2.0*probs - 1.0
        # find coefficients of Chebyshev polynomials
        self.c = jnp.array(np.polynomial.chebyshev.chebfit(x, vals, self.K))
        
    def clenshaw(self, x):
        """
        Evaluate the Chebyshev series at a given x.
        
        Uses Clenshaw's algorithm.
        (https://en.wikipedia.org/wiki/Clenshaw_algorithm#Special_case_for_Chebyshev_series)
        
        Parameters
        ----------
        x: array-like
            Locations in [-1, 1] to compute the Chebyshev series at.
            
        Returns
        -------
        z: array-like
            Chebyshev series evaluated at `x`.
        """
        b = jnp.zeros((self.K+3, *x.shape))
        for k in range(self.K+1):
            b = b.at[-3-k,:].add(self.c[-1-k] + 2*x*b[-2-k] - b[-1-k])
        return 0.5*(self.c[0] + b[0] - b[2])
    
    def icdf(self, p):
        """
        Compute the inverse CDF at a given p.
        
        Parameters
        ----------
        p: array-like
            Values in [0,1] to evaluate the inverse CDF at.
            
        Returns
        -------
        z: array-like
            Inverse CDF evaluated at `p`.
        """
        return self.clenshaw(2.0*p - 1.0)
    
    def sample(self, key, sample_shape=()):
        """
        Sample from the approximated distribution.
        
        Parameters
        ----------
        p: array-like
            Values in [0,1] to evaluate the inverse CDF at.
            
        Returns
        -------
        z: array-like
            Inverse CDF evaluated at `p`.
        """
        assert is_prng_key(key)
        p = jr.uniform(key, shape=sample_shape)
        return self.icdf(p)
    
class MonoEICDF():
    """
    Monotonic empirical inverse CDF.
    
    Defined by interpolating a set of samples from a PDF.
    
    Uses linear interpolation (continuous, but not in derivative)
    or monotonic hermite cubic spline interpolation.
    
    Parameters
    ----------
    samples: array-like, optional
        Array of samples whose inverse CDF is to be approximated.
    kind: "linear" or "cubic"
        Selects linear or cubic interpolation of the empirical ICDF.
        Default is "linear".
        
    Methods
    -------
    icdf:
        Method for computing approximate inverse CDF
    sample:
        Sample from the distribution by transforming uniform RVs
        
    Attributes
    ----------
    pp: array
        Cumulative probabilities associated with `zp`.
    zp: array
        Sorted array of parameter values
    interpolator: callable
        Callable function that interpolates the empirical ICDF
        defined by `pp` and `zp`.
        If `kind="linear"` this is an alias for `jax.numpy.interp`.
        If `kind="cubic"` this is an `interpax.PchipInterpolator`
    """
    def __init__(self, samples, kind="linear"):
        # sort the samples to get the empirical ICDF
        self.pp = (1+jnp.arange(len(samples)))/(len(samples)+1)
        self.zp = jnp.sort(samples)
        # set up the interpolator
        if kind == "linear":
            self.interpolator = lambda p : jnp.interp(p, self.pp, self.zp)
        elif kind == "cubic":
            try:
                from interpax import PchipInterpolator
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError("To use cubic interpolation first run pip install interpax!") from e
            self.interpolator = PchipInterpolator(self.pp, self.zp)
        else:
            raise ValueError('kind must be either "linear or "cubic"!')
            
    def icdf(self, p):
        """
        Compute the inverse CDF at a given p.
        
        Parameters
        ----------
        p: array-like
            Values in [0,1] to evaluate the inverse CDF at.
            
        Returns
        -------
        z: array-like
            Inverse CDF evaluated at `p`.
        """
        return self.interpolator(p)
    
    def sample(self, key, sample_shape=()):
        """
        Sample from the approximated distribution.
        
        Parameters
        ----------
        p: array-like
            Values in [0,1] to evaluate the inverse CDF at.
            
        Returns
        -------
        z: array-like
            Inverse CDF evaluated at `p`.
        """
        assert is_prng_key(key)
        p = jr.uniform(key, shape=sample_shape)
        return self.icdf(p)