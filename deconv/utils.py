
"""
@article{giovannucci2019caiman,
  title={CaImAn: An open source tool for scalable Calcium Imaging data Analysis},
  author={Giovannucci, Andrea and Friedrich, Johannes and Gunn, Pat and Kalfon, Jeremie and Brown, Brandon L and Koay, Sue Ann and Taxidis, Jiannis and Najafi, Farzaneh and Gauthier, Jeffrey L and Zhou, Pengcheng and Khakh, Baljit S and Tank, David W and Chklovskii, Dmitri B and Pnevmatikakis, Eftychios A},
  journal={eLife},
  volume={8},
  pages={e38173},
  year={2019},
  publisher={eLife Sciences Publications Limited}
}

https://github.com/flatironinstitute/CaImAn
"""

import numpy as np
import scipy
from scipy.signal import lfilter, detrend

from past.utils import old_div


def generate_data(N,T,dt=0.02,lam=0.1,tau=1.5,sigma=0.1):
    """
    generate fluor, calcium traces, spikes
    N: # neurons
    T     - # of time steps
    dt    - time step size
    lam   - firing rate = lam*dt
    tau   - decay time constant
    sigma - standard derivation of observation noise
    """
    S = np.random.poisson(lam=lam*dt,size=(N,T))
    gam = 1.0 - dt/tau
    C = lfilter([1.0],[1.0,-gam],S)
    F = C + sigma*np.random.randn(N,T) # a=1.0, b=0.0
    return S, C, F


def G_inv_mat(x, mode, NT, gs, gd_vec, bas_flag=True, c1_flag=True):
    """
    Fast computation of G^{-1}*x and G^{-T}*x
    """
    if mode == 1:
        b = lfilter(np.array([1]), np.concatenate([np.array([1.]), -gs]), x[:NT]
                    ) + bas_flag * x[NT - 1 + bas_flag] + c1_flag * gd_vec * x[-1]
    elif mode == 2:
        b = np.hstack((np.flipud(lfilter(np.array([1]), np.concatenate([np.array(
            [1.]), -gs]), np.flipud(x))), np.ones(bas_flag) * np.sum(x), np.ones(c1_flag) * np.sum(gd_vec * x)))

    return b

def GetSn(fluor, range_ff=[0.25, 0.5], method='logmexp'):
    """    
    Estimate noise power through the power spectral density over the range of large frequencies    
    Args:
        fluor    : nparray
            One dimensional array containing the fluorescence intensities with
            one entry per time-bin.
    
        range_ff : (1,2) array, nonnegative, max value <= 0.5
            range of frequency (x Nyquist rate) over which the spectrum is averaged  
    
        method   : string
            method of averaging: Mean, median, exponentiated mean of logvalues (default)
    Returns:
        sn       : noise standard deviation
    """

    ff, Pxx = scipy.signal.welch(fluor)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1, ind2)
    Pxx_ind = Pxx[ind]
    sn = {
        'mean': lambda Pxx_ind: np.sqrt(np.mean(old_div(Pxx_ind, 2))),
        'median': lambda Pxx_ind: np.sqrt(np.median(old_div(Pxx_ind, 2))),
        'logmexp': lambda Pxx_ind: np.sqrt(np.exp(np.mean(np.log(old_div(Pxx_ind, 2)))))
    }[method](Pxx_ind)

    return sn

def axcov(data, maxlag=5):
    """
    Compute the autocovariance of data at lag = -maxlag:0:maxlag
    Args:
        data : array
            Array containing fluorescence data
    
        maxlag : int
            Number of lags to use in autocovariance calculation
    Returns:
        axcov : array
            Autocovariances computed from -maxlag:0:maxlag
    """

    data = data - np.mean(data)
    T = len(data)
    bins = np.size(data)
    xcov = np.fft.fft(data, np.power(2, nextpow2(2 * bins - 1)))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag + 1)]])
    return np.real(old_div(xcov, T))


    def nnls(self, KK, Ky, s=None, mask=None, tol=1e-9, max_iter=None):
        """
        Solve non-negative least squares problem
        ``argmin_s || Ks - y ||_2`` for ``s>=0``
        Args:
            KK : array, shape (n, n)
                Dot-product of design matrix K transposed and K, K'K
            Ky : array, shape (n,)
                Dot-product of design matrix K transposed and target vector y, K'y
            s : None or array, shape (n,), optional, default None
                Initialization of deconvolved neural activity.
            mask : array of bool, shape (n,), optional, default (True,)*n
                Mask to restrict potential spike times considered.
            tol : float, optional, default 1e-9
                Tolerance parameter.
            max_iter : None or int, optional, default None
                Maximum number of iterations before termination.
                If None (default), it is set to len(KK).
        Returns:
            s : array, shape (n,)
                Discretized deconvolved neural activity (spikes)
        """

        if mask is None:
            mask = np.ones(len(KK), dtype=bool)
        else:
            KK = KK[mask][:, mask]
            Ky = Ky[mask]
        if s is None:
            s = np.zeros(len(KK))
            l = Ky.copy()
            P = np.zeros(len(KK), dtype=bool)
        else:
            s = s[mask]
            P = s > 0
            l = Ky - KK[:, P].dot(s[P])
        i = 0
        if max_iter is None:
            max_iter = len(KK)
        for i in range(max_iter):  # max(l) is checked at the end, should do at least one iteration
            w = np.argmax(l)
            P[w] = True

            try:
                mu = np.linalg.solve(KK[P][:, P], Ky[P])
            except:
                mu = np.linalg.solve(KK[P][:, P] + tol * np.eye(P.sum()), Ky[P])
                print(r'added $\epsilon$I to avoid singularity')
            while len(mu > 0) and min(mu) < 0:
                a = min(s[P][mu < 0] / (s[P][mu < 0] - mu[mu < 0]))
                s[P] += a * (mu - s[P])
                P[s <= tol] = False
                try:
                    mu = np.linalg.solve(KK[P][:, P], Ky[P])
                except:
                    mu = np.linalg.solve(KK[P][:, P] + tol * np.eye(P.sum()), Ky[P])
                    print(r'added $\epsilon$I to avoid singularity')
            s[P] = mu.copy()
            l = Ky - KK[:, P].dot(s[P])
            if max(l) < tol:
                break
        tmp = np.zeros(len(mask))
        tmp[mask] = s
        return tmp

def nextpow2(value):
    """
    Find exponent such that 2^exponent is equal to or greater than abs(value).
    Args:
        value : int
    Returns:
        exponent : int
    """

    exponent = 0
    avalue = np.abs(value)
    while avalue > np.power(2, exponent):
        exponent += 1
    return exponent