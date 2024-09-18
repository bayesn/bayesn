"""
BayeSN Spline Utilities. Defines a set of functions which carry out the
2D spline operations essential to BayeSN.
"""

import numpy as np
import jax.numpy as jnp

def invKD_irr(x):
	"""
	Compute K^{-1}D for a set of spline knots.

	For knots y at locations x, the vector, y'' of non-zero second
	derivatives is constructed from y'' = K^{-1}Dy, where K^{-1}D
	is independent of y, meaning it can be precomputed and reused for
	arbitrary y to compute the second derivatives of y.

	Parameters
	----------
	x : :py:class:`numpy.array`
		Numpy array containing the locations of the cubic spline knots.
	
	Returns
	-------
	KD : :py:class:`numpy.array`
		y independednt matrix whose product can be taken with y to
		obtain a vector of second derivatives of y.
	"""
	n = len(x)

	K = np.zeros((n-2,n-2))
	D = np.zeros((n-2,n))

	K[0,0:2] = [(x[2] - x[0])/3, (x[2] - x[1])/6]
	K[-1, -2:n-2] = [(x[n-2] - x[n-3])/6, (x[n-1] - x[n-3])/3]

	for j in np.arange(2,n-2):
		row = j - 1
		K[row, row-1:row+2] = [(x[j] - x[j-1])/6, (x[j+1] - x[j-1])/3, (x[j+1] - x[j])/6]
	for j in np.arange(1,n-1):
		row = j - 1
		D[row, row:row+3] = [1./(x[j] - x[j-1]), -(1./(x[j+1] - x[j]) + 1./(x[j] - x[j-1])), 1./(x[j+1] - x[j])]
	
	M = np.zeros((n,n))
	M[1:-1, :] = np.linalg.solve(K,D)
	return M

def cartesian_prod(x, y):
	"""
	Compute cartesian product of two vectors.

	Parameters
	----------
	x : :py:class:`numpy.array`
		First vector.
	x : :py:class:`numpy.array`
		Second vector.
	
	Returns
	-------
	z : :py:class:`numpy.array`
		Cartesian product of x and y.
	"""
	n_x = len(x)
	n_y = len(y)
	return np.array([np.repeat(x,n_y),np.tile(y,n_x)]).T

def spline_coeffs_irr(x_int, x, invkd, allow_extrap=True, hermite=False, x_extreme="gap"):
	"""
	Compute a matrix of spline coefficients.

	Given a set of knots at x, with values y, compute a matrix, J, which
	can be multiplied into y to evaluate the cubic spline at points
	x_int.

	Parameters
	----------
	x_int : :py:class:`numpy.array`
		Numpy array containing the locations which the output matrix will
		interpolate the spline to.
	x : :py:class:`numpy.array`
		Numpy array containing the locations of the spline knots.
	invkd : :py:class:`numpy.array`
		Precomputed matrix for generating second derivatives. Can be obtained
		from the output of ``invKD_irr``.
	allow_extrap : bool
		Flag permitting extrapolation. If True, the returned matrix will be
		configured to extrapolate linearly beyond the outer knots.
		Or to use a cubic Hermite spline if `hermite` is True. If False,
		values which fall out of bounds will raise ValueError.
	hermite : bool
		If True uses cubic Hermite spline interpolation to 
		fall back towards zero outside the knot range. If False
		uses linear extrapolation.
	x_extreme : "hsiao", "gap", or two-tuple
		Method for setting the outer knots if doing hermite
		extrapolation. The surface will go to zero at these points.
		If "hsiao" uses the limits of the Hsiao template: [1000, 25000]
		If "gap" extends beyond the existing knots by one step of
		size equal to the adjacent inter-knot space.
		If a two-tuple, uses the custom limits provided.
		Deafult is "gap"
	
	Returns
	-------
	J : :py:class:`numpy.array`
		y independednt matrix whose product can be taken with y to evaluate
		the spline at x_int.
	"""
	n_x_int = len(x_int)
	n_x = len(x)
	X = np.zeros((n_x_int,n_x))

	if hermite is True:
		if x_extreme == "hsiao":
			x_extreme = np.array([1000, 25000])
		elif x_extreme == "gap":
			x_extreme = np.array([2*x[0] - x[1], 2*x[-1] - x[-2]])

	if not allow_extrap and ((max(x_int) > max(x)) or (min(x_int) < min(x))):
		raise ValueError("Interpolation point out of bounds! " + 
			"Ensure all points are within bounds, or set allow_extrap=True.")
	
	for i in range(n_x_int):
		x_now = x_int[i]
		if x_now > max(x):
			if not hermite:
				h = x[-1] - x[-2]
				a = (x[-1] - x_now)/h
				b = 1 - a
				f = (x_now - x[-1])*h/6.0

				X[i,-2] = a
				X[i,-1] = b
				X[i,:] = X[i,:] + f*invkd[-2,:]
			elif x_now < x_extreme[1]:
				# hermite rule
				t = (x_now - x[-1])/(x_extreme[1] - x[-1])
				r = (x_extreme[1] - x[-1])/(x[-1] - x[-2])
				h00 = 2*t**3 - 3*t**2 + 1
				h10 = t**3 - 2*t**2 + t

				X[i,-1] = h00 + h10*r
				X[i,-2] = -h10*r
				X[i,:] = X[i,:] + h10*invkd[-2,:]*(x_extreme[1] - x[-1])*(x[-1] - x[-2])/6.0
			else:
				# if x is beyond the outer anchor,
				# leave the row as all zeros
				pass
		elif x_now < min(x):
			if not hermite:
				h = x[1] - x[0]
				b = (x_now - x[0])/h
				a = 1 - b
				f = (x_now - x[0])*h/6.0

				X[i,0] = a
				X[i,1] = b
				X[i,:] = X[i,:] - f*invkd[1,:]
			elif x_now > x_extreme[0]:
				# hermite rule
				t = (x_now - x_extreme[0])/(x[0] - x_extreme[0])
				r = (x[0] - x_extreme[0])/(x[1] - x[0])
				h01 = -2*t**3 + 3*t**2
				h11 = t**3 - t**2

				X[i,0] = h01 - h11*r
				X[i,1] = h11*r
				X[i,:] = X[i,:] - h11*invkd[1,:]*(x[1] - x[0])*(x[0] - x_extreme[0])/6.0
			else:
				# if x is beyond the outer anchor,
				# leave the row as all zeros
				pass
		else:
			q = np.where(x[0:-1] <= x_now)[0][-1]
			h = x[q+1] - x[q]
			a = (x[q+1] - x_now)/h
			b = 1 - a
			c = ((a**3 - a)/6)*h**2
			d = ((b**3 - b)/6)*h**2

			X[i,q] = a
			X[i,q+1] = b
			X[i,:] = X[i,:] + c*invkd[q,:] + d*invkd[q+1,:]

	return X


def spline_coeffs_irr_step(x_now, x, invkd):
	X = jnp.zeros_like(x)
	up_extrap = x_now > x[-1]
	down_extrap = x_now < x[0]
	interp = 1 - up_extrap - down_extrap

	h = x[-1] - x[-2]
	a = (x[-1] - x_now) / h
	b = 1 - a
	f = (x_now - x[-1]) * h / 6.0

	X = X.at[-2].set(X[-2] + a * up_extrap)
	X = X.at[-1].set(X[-1] + b * up_extrap)
	X = X.at[:].set(X[:] + f * invkd[-2, :] * up_extrap)

	h = x[1] - x[0]
	b = (x_now - x[0]) / h
	a = 1 - b
	f = (x_now - x[0]) * h / 6.0

	X = X.at[0].set(X[0] + a * down_extrap)
	X = X.at[1].set(X[1] + b * down_extrap)
	X = X.at[:].set(X[:] - f * invkd[1, :] * down_extrap)

	q = jnp.argmax(x_now < x) - 1
	h = x[q + 1] - x[q]
	a = (x[q + 1] - x_now) / h
	b = 1 - a
	c = ((a ** 3 - a) / 6) * h ** 2
	d = ((b ** 3 - b) / 6) * h ** 2

	X = X.at[q].set(X[q] + a * interp)
	X = X.at[q + 1].set(X[q + 1] + b * interp)
	X = X.at[:].set(X[:] + c * invkd[q, :] * interp + d * invkd[q + 1, :] * interp)

	return X