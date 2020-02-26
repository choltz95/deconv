"""
Recovery of neural spiking activity from a fluorescence trace 

@article{giovannucci2019caiman,
  title={CaImAn: An open source tool for scalable Calcium Imaging data Analysis},
  author={Giovannucci, Andrea and Friedrich, Johannes and Gunn, Pat and Kalfon, Jeremie and Brown, Brandon L and Koay, Sue Ann and Taxidis, Jiannis and Najafi, Farzaneh and Gauthier, Jeffrey L and Zhou, Pengcheng and Khakh, Baljit S and Tank, David W and Chklovskii, Dmitri B and Pnevmatikakis, Eftychios A},
  journal={eLife},
  volume={8},
  pages={e38173},
  year={2019},
  publisher={eLife Sciences Publications Limited}
}

Reference implementations:
https://github.com/flatironinstitute/CaImAn
https://github.com/liubenyuan/py-oopsi
https://github.com/zhoupc/OASIS_matlab
https://github.com/epnev/constrained-foopsi
"""
import sys
import numpy as np
import numpy.linalg as lp
import scipy.signal 
import scipy.linalg

from scipy.signal import lfilter
from scipy.sparse import spdiags, eye
from scipy.sparse.linalg.dsolve import linsolve

from math import log, sqrt, exp
import cvxpy as cvx

from warnings import warn 

from deconv.utils import *
from past.utils import old_div


class Deconv():
	def __init__(self, pars):
		pass

	def foopsi():
		pass

	def thresholded_foopsi():
		pass

	def constrained_foopsi(self, fluor, bl=None,  c1=None, g=None,  sn=None, p=None, bas_nonneg=True,
	                       noise_range=[.25, .5], noise_method='logmexp', lags=5, fudge_factor=1.,
	                       verbosity=False, **kwargs):
	    """
	    Args:
	        fluor: np.ndarray
	            One dimensional array containing the fluorescence intensities with
	            one entry per time-bin.
	        bl: [optional] float
	            Fluorescence baseline value. If no value is given, then bl is estimated
	            from the data.
	        c1: [optional] float
	            value of calcium at time 0
	        g: [optional] list,float
	            Parameters of the AR process that models the fluorescence impulse response.
	            Estimated from the data if no value is given
	        sn: float, optional
	            Standard deviation of the noise distribution.  If no value is given,
	            then sn is estimated from the data.
	        p: int
	            order of the autoregression model
	        bas_nonneg: bool
	            baseline strictly non-negative
	        noise_range:  list of two elms
	            frequency range for averaging noise PSD
	        noise_method: string
	            method of averaging noise PSD
	        lags: int
	            number of lags for estimating time constants
	        fudge_factor: float
	            fudge factor for reducing time constant bias
	        verbosity: bool
	             display optimization details
	    Returns:
	        c: np.ndarray float
	            The inferred denoised fluorescence signal at each time-bin.
	        bl, c1, g, sn : As explained above
	        sp: ndarray of float
	            Discretized deconvolved neural activity (spikes)
	        lam: float
	            Regularization parameter
	    """
	    lam = 0
	    if p is None:
	        raise Exception("You must specify the value of p")

	    if g is None or sn is None:
	        # Estimate noise standard deviation and AR coefficients if they are not present
	        g, sn = self.estimate_parameters(fluor, p=p, sn=sn, g=g, range_ff=noise_range,
	                                    method=noise_method, lags=lags, fudge_factor=fudge_factor)

	    c, bl, c1, g, sn, sp = self._contr_foopsi(fluor, g, sn, b=bl, c1=c1, bas_nonneg=bas_nonneg)

	    return c, bl, c1, g, sn, sp, lam

	def _contr_foopsi(self, fluor, g, sn, b=None, c1=None, bas_nonneg=True):
	    """Solves the deconvolution problem using the cvxpy package and the ECOS/SCS library.
	    Args:
	        fluor: ndarray
	            fluorescence trace
	        g: list of doubles
	            parameters of the autoregressive model, cardinality equivalent to p
	        sn: double
	            estimated noise level
	        b: double
	            baseline level. If None it is estimated.
	        c1: double
	            initial value of calcium. If None it is estimated.
	        bas_nonneg: boolean
	            should the baseline be estimated

	    Returns:
	        c: estimated calcium trace
	        b: estimated baseline
	        c1: esimtated initial calcium value
	        g: esitmated parameters of the autoregressive model
	        sn: estimated noise level
	        sp: estimated spikes
	    """

	    T = fluor.shape[0]

	    # construct deconvolution matrix  (sp = G*c)
	    G = scipy.sparse.dia_matrix((np.ones((1, T)), [0]), (T, T))

	    for i, gi in enumerate(g):
	        G = G + \
	            scipy.sparse.dia_matrix((-gi * np.ones((1, T)), [-1 - i]), (T, T))

	    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
	    gd_vec = np.max(gr)**np.arange(T)  # decay vector for initial fluorescence
	    gen_vec = G.dot(scipy.sparse.coo_matrix(np.ones((T, 1))))

	    c = cvx.Variable(T)  # calcium at each time step
	    constraints = []
	    if b is None:
	        flag_b = True
	        b = cvx.Variable(1)  # baseline value
	        if bas_nonneg:
	            b_lb = 0
	        else:
	            b_lb = np.min(fluor)
	        constraints.append(b >= b_lb)
	    else:
	        flag_b = False

	    if c1 is None:
	        flag_c1 = True
	        c1 = cvx.Variable(1)  # baseline value
	        constraints.append(c1 >= 0)
	    else:
	        flag_c1 = False

	    thrNoise = sn * np.sqrt(fluor.size)

 		"""
	    # threshold foopsi
        # minimize number of spikes
        objective = cvx.Minimize(cvx.norm(G * c, 1))
        constraints.append(G * c >= 0)
        constraints.append(
            cvx.norm(G * c >= thrNoise)  # constraints
        prob = cvx.Problem(objective, constraints)
        result = prob.solve(solver='ECOS')
		"""

	    """
	    # constrained foopsi
        # minimize number of spikes
        objective = cvx.Minimize(cvx.norm(G * c, 1))
        constraints.append(G * c >= 0)
        constraints.append(
            cvx.norm(-c + fluor - b - gd_vec * c1, 2) <= thrNoise)  # constraints
        prob = cvx.Problem(objective, constraints)
        result = prob.solve(solver='ECOS')
		"""

        """
        # constrained foopsi
        lam = old_div(sn, 500)
        constraints = constraints[:-1]
        objective = cvx.Minimize(cvx.norm(-c + fluor - b - gd_vec *
                                          c1, 2) + lam * cvx.norm(G * c, 1))
        prob = cvx.Problem(objective, constraints)
        result = prob.solve(solver='ECOS')
	    """
	    lam = old_div(sn, 500)
	    constraints = constraints[:-1]
	    objective = cvx.Minimize(cvx.norm(-c + fluor - b - gd_vec *
	                                  c1, 2) + lam * cvx.norm(G * c, 1))
	    prob = cvx.Problem(objective, constraints)
	    result = prob.solve(solver='ECOS')

	    if not (prob.status == 'optimal' or prob.status == 'optimal_inaccurate'):
	        print(('PROBLEM STATUS:' + prob.status))
	        sp = fluor
	        c = fluor
	        b = 0
	        c1 = 0
	        return c, b, c1, g, sn, sp

	    sp = np.squeeze(np.asarray(G * c.value))
	    c = np.squeeze(np.asarray(c.value))
	    if flag_b:
	        b = np.squeeze(b.value)
	    if flag_c1:
	        c1 = np.squeeze(c1.value)

	    return c, b, c1, g, sn, sp

	def estimate_parameters(self, fluor, p=2,range_ff=[0.25, 0.5],
	                        method='logmexp', lags=5, fudge_factor=1.):
	    """
	    Estimate noise standard deviation and AR coefficients if they are not present
	    Args:
	        p: positive integer
	            order of AR system
	    
	        lags: positive integer
	            number of additional lags where he autocovariance is computed
	    
	        range_ff : (1,2) array, nonnegative, max value <= 0.5
	            range of frequency (x Nyquist rate) over which the spectrum is averaged
	    
	        method: string
	            method of averaging: Mean, median, exponentiated mean of logvalues (default)
	    
	        fudge_factor: float (0< fudge_factor <= 1)
	            shrinkage factor to reduce bias
	    """
	   	if p < 1:
	        raise Exception("p >= 1")

        sn = GetSn(fluor, range_ff, method)
        g = self.estimate_time_constant(fluor, p, sn, lags, fudge_factor)

	    return g, sn

	def estimate_time_constant(self, fluor, p=2, sn=None, lags=5, fudge_factor=1.):
	    """
	    Estimate AR model parameters through the autocovariance function
	    Args:
	        fluor        : nparray
	            One dimensional array containing the fluorescence intensities with
	            one entry per time-bin.
	    
	        p            : positive integer
	            order of AR system
	    
	        sn           : float
	            noise standard deviation, estimated if not provided.
	    
	        lags         : positive integer
	            number of additional lags where he autocovariance is computed
	    
	        fudge_factor : float (0< fudge_factor <= 1)
	            shrinkage factor to reduce bias
	    Returns:
	        g       : estimated coefficients of the AR process
	    """

	    if sn is None:
	        sn = GetSn(fluor)

	    lags += p
	    xc = axcov(fluor, lags)
	    xc = xc[:, np.newaxis]

	    A = scipy.linalg.toeplitz(xc[lags + np.arange(lags)],
	                              xc[lags + np.arange(p)]) - sn**2 * np.eye(lags, p)
	    g = np.linalg.lstsq(A, xc[lags + 1:], rcond=None)[0]
	    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
	    gr = old_div((gr + gr.conjugate()), 2.)

	    gr[gr > 1] = 0.95 + np.random.normal(0, 0.01, np.sum(gr > 1))
	    gr[gr < 0] = 0.15 + np.random.normal(0, 0.01, np.sum(gr < 0))
	    g = np.poly(fudge_factor * gr)
	    g = -g[1:]

	    return g.flatten()


	def wiener(self, F, dt=0.020, iter_max=25, update=True):
	    # normalize
	    F = (F-F.mean())/np.abs(F).max() # fluor
	    T = F.shape[0]
	    gam = 1.0 - dt/1.0
	    gtol = 1e-4 # global tolerance

	    M = spdiags([-gam*np.ones(T), np.ones(T)],[-1,0],T,T) # conv mat
	    C = np.ones(T) # calcium 
	    n = M*C
	    lam = 1.0
	    llam = (lam*dt)
	    sig = 0.1*lp.norm(F) # 0.1 is arbitrary

	    D0 = F - C # assume a=1.0, b=0.0
	    D1 = n - llam

	    # likelihood
	    lik = np.dot(D0.T,D0)/(2*sig**2) + np.dot(D1.T,D1)/(2*llam) # eq 11.b

	    # See appendix B Vogelstein et. al., 2010.
	    for i in range(iter_max):
	        g = -(F-C)/sig**2 + (M.T*(M*C)-llam*(M.T*np.ones(T)))/llam
	        H = eye(T)/sig**2 + M.T*M/llam
	        d = linsolve.spsolve(H,g)
	        C = C - d
	        N = M*C
	        old_lik = lik
	        D0 = F - C
	        D1 = n - llam
	        lik = np.dot(D0.T,D0)/(2*sig**2) + np.dot(D1.T,D1)/(2*llam) 
	        if lik <= old_lik - gtol: # NR step until convergence
	            n = N
	            if update:
	                sig = np.sqrt(np.dot(D0.T,D0)/T)
	        else:
	            break
	    n = n/n.max()
	    return n, C


	def discretize(self, F, bins=[0.12], high_pass=True):
	    epsilon = 1e-3
	    if high_pass:
	        v = np.diff(F,axis=0)
	    else:
	        v = F[1:]
	    vmax = v.max() + epsilon
	    vmin = v.min() - epsilon
	    D = np.zeros(F.shape)
	    if np.isscalar(bins):
	        binEdges = np.linspace(vmin,vmax,bins+1)
	    else:
	        binEdges = np.array(bins)
	    D[1:] = np.digitize(v,binEdges)
	    D[0] = epsilon
	    D = D/D.max()
	    return D, v