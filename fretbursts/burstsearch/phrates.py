
# FRETBursts - A single-molecule FRET burst analysis toolkit.
#
# Copyright (C) 2014-2016 Antonino Ingargiola <tritemio@gmail.com>
#
"""
This module provides functions to compute photon rates from timestamps
arrays. Different methods to compute rates are implemented::

1. Consecutive set of `m` timestamps ("sliding m-tuple")
2. Sliding window with fixed duration
3. KDE-based methods with Gaussian or Laplace distribution kernels.


**Time axis for the rates**

In case of using of "sliding m-tuple" method (1), rates can be only computed
for each consecutive set of `m` timestamps. The time-axis can be
computed from the mean timestamp in each m-tuple.

For the other methods (2 & 3), rates can be in priciple computed at each time
point. Practically, in most cases, the time points at which rates are computed
are defined by the same timestamps (or by timestamps in a related photon
stream). In other words, we don't normally use a uniformly sampled time axis
but we use a timestamps array as time axis for the rate.

Note that computing rates with a fixed sliding time window and sampling
the function by centering the window on each timestamp is equivalent to
a KDE-based rate computation using a rectangular kernel.

"""

import numpy as np
import numba
from math import exp, fabs


def kde_laplace_self(ph, tau):
    """Computes exponential KDE for each photon in `ph`.

    The rate function is evaluated for each element in `ph` (that's why
    name ends with ``_self``).
    This function computes the rate of timestamps in `ph`
    using a KDE and simmetric-exponential kernel (i.e. laplacian)::

        kernel = exp( -|t - t0| / tau)

    The rate is evaluated for each element in `ph` (that's why name ends
    with ``_self``).

    Arguments:
        ph (array): arrays of photon timestamps
        tau (float): time constant of the exponential kernel

    Returns:
        rates (array): unnormalized rates (just the sum of the
            exponential kernels). To obtain rates in Hz divide the
            array by `2*tau` (or other conventional x*tau duration).
        nph (array): number of photons in -5*tau..5*tau window
            for each timestamp. Proportional to the rate computed
            with KDE and rectangular kernel.
        """
    ph_size = ph.size
    ipos, ineg = 0, 0
    rates = np.zeros((ph_size,), dtype=np.float64)
    nph = np.zeros((ph_size,), dtype=np.int16)
    tau_lim = 5*tau
    for i, t in enumerate(ph):
        # Increment ipos until falling out of N*tau (tau_lim)
        # ipos is the first value *outside* the limit
        while ipos < ph_size and ph[ipos] - t < tau_lim:
            ipos += 1

        # Increment ineg until falling inside N*tau (tau_lim)
        # ineg is the first value *inside* the limit
        while t - ph[ineg] > tau_lim:
            ineg += 1

        delta_t = np.abs(ph[ineg:ipos] - t)
        rates[i] = np.exp(-delta_t / tau).sum()
        nph[i] = ipos - ineg
    return rates, nph


@numba.jit
def kde_laplace_self_numba(ph, tau):
    """Numba version of `kde_laplace_self()`
    """
    ph_size = ph.size
    ipos, ineg = 0, 0
    rates = np.zeros((ph_size,), dtype=np.float64)
    nph = np.zeros((ph_size,), dtype=np.int16)
    tau_lim = 5*tau
    for i, t in enumerate(ph):
        while ipos < ph_size and ph[ipos] - t < tau_lim:
            ipos += 1

        while t - ph[ineg] > tau_lim:
            ineg += 1

        for itx in range(ineg, ipos):
            rates[i] += exp(-fabs(ph[itx]-t)/tau)
            nph[i] += 1

    return rates, nph

@numba.jit
def kde_laplace_numba(timestamps, tau, time_axis=None):
    """Computes exponential KDE for `timestamps` evaluated at `time_axis`.

    When ``time_axis`` is None them ``timestamps`` is used also as time axis.
    """
    if time_axis is None:
        time_axis = timestamps
    t_size = time_axis.size
    timestamps_size = timestamps.size
    rates = np.zeros((t_size,), dtype=np.float64)
    nph = np.zeros((t_size,), dtype=np.int16)
    tau_lim = 5 * tau

    ipos, ineg = 0, 0  # indexes for timestamps
    for it, t in enumerate(time_axis):

        while ipos < timestamps_size and timestamps[ipos] - t < tau_lim:
            ipos += 1

        while t - timestamps[ineg] > tau_lim:
            ineg += 1

        for itx in range(ineg, ipos):
            rates[it] += exp(-fabs(timestamps[itx] - t)/tau)
            nph[it] += 1
    return rates, nph

@numba.jit
def kde_laplace_numba_nc(timestamps, tau, time_axis=None):
    """Computes exponential KDE for `timestamps` evaluated at `time_axis`.

    When ``time_axis`` is None them ``timestamps`` is used also as time axis.
    """
    if time_axis is None:
        time_axis = timestamps
    t_size = time_axis.size
    timestamps_size = timestamps.size
    rates = np.zeros((t_size,), dtype=np.float64)
    tau_lim = 5 * tau

    ipos, ineg = 0, 0  # indexes for timestamps
    for it, t in enumerate(time_axis):

        while ipos < timestamps_size and timestamps[ipos] - t < tau_lim:
            ipos += 1

        while t - timestamps[ineg] > tau_lim:
            ineg += 1

        for itx in range(ineg, ipos):
            rates[it] += exp(-fabs(timestamps[itx] - t)/tau)

    return rates


@numba.jit
def kde_laplace_numba2(timestamps, tau, time_axis=None):
    """Computes exponential KDE for `timestamps` evaluated at `time_axis`.

    This is an alternative version of :func:`kde_laplace_numba` not using
    the `abs` function.
    """
    if time_axis is None:
        time_axis = timestamps
    t_size = time_axis.size
    timestamps_size = timestamps.size
    rates = np.zeros((t_size,), dtype=np.float64)
    nph = np.zeros((t_size,), dtype=np.int16)
    tau_lim = 5 * tau

    ipos, ineg = 0, 0  # indexes for timestamps
    icenter = 0          # index for time_axis
    for it, t in enumerate(time_axis):

        while ipos < timestamps_size and timestamps[ipos] - t < tau_lim:
            ipos += 1

        while t - timestamps[ineg] > tau_lim:
            ineg += 1

        while timestamps[icenter] < t:
            icenter += 1
        # now timestamps[icenter] is >= t

        #  CASE 1: timestamps[icenter] > t
        #
        #        +                +            --> timestamps
        #              *          |            --> time_axis
        #              t      timestamps[icenter]
        #
        #  CASE 2: timestamps[icenter] = t
        #
        #      +       +                   +   --> timestamps
        #              *                       --> time_axis
        #              t = timestamps[icenter]

        # includes timestamps[icenter]
        for itx in range(icenter, ipos):
            rates[it] += exp(-(timestamps[itx] - t)/tau)
            nph[it] += 1

        # does not include timestamps[icenter]
        for itx in range(ineg, icenter):
            rates[it] += exp(-(t - timestamps[itx])/tau)
            nph[it] += 1

    return rates, nph

@numba.jit
def kde_gaussian_numba(timestamps, tau, time_axis=None):
    """Computes Gaussian KDE for `timestamps` evaluated at `time_axis`.
    """
    if time_axis is None:
        time_axis = timestamps
    t_size = time_axis.size
    timestamps_size = timestamps.size
    rates = np.zeros((t_size,), dtype=np.float64)
    tau_lim = 3 * tau   # 3 tau = 99.7 % of the Gaussian
    tau2 = 2 * (tau**2)

    ipos, ineg = 0, 0  # indexes for timestamps
    for it, t in enumerate(time_axis):

        while ipos < timestamps_size and timestamps[ipos] - t < tau_lim:
            ipos += 1

        while t - timestamps[ineg] > tau_lim:
            ineg += 1

        for itx in range(ineg, ipos):
            rates[it] += exp(-((timestamps[itx] - t)**2)/tau2)

    return rates
