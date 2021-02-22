import numpy as np
from numba import njit, prange


@njit(parallel=True)
def optimize_lile_helper_1d(d, numsamples):
    L1 = np.empty(len(d))
    L1[:] = np.NaN  # Pads the end with NaNs.
    for i in prange(len(d) - numsamples):
        L1[i] = np.sum(np.abs(np.diff(d[i:i + numsamples])))
    return list(L1)


@njit(parallel=True)
def optimize_lile_helper_2d(d, numsamples):
    L2 = np.empty((len(d), len(d[0])))
    L2[:] = np.NaN  # Pads the end with NaNs.
    for j in prange(len(d)):
        for i in prange(len(d[0]) - numsamples):
            L2[j][i] = np.sum(np.abs(np.diff(d[j][i:i + numsamples])))
    return L2


def lltransform(d, sfx):
    """Calculates line-length transform for a vector or a matrix.

    Parameters
    ----------
    d : float list
        data to perform line-length transform on
    sfx : int
        sampling frequency of data

    Returns
    -------
    float list (same size of data)
        each element in list is the line-length calculation
        at each sample calculated in a specified line-length window.
        end of this list is padded with Nans.

    Raises
    ------
    ValueError
        If `d` is greater than 2-dimensions.

    Notes
    ------
        Calculates line-length by adding the absolute value of the differences
        in data values over a discrete time window (llw).
        Uses variable 'numsamples' to determine how many samples per llw.
        Uses numba/@jit function decorator for optimal performance (parallel
        processing) in helper functions.
    """

    llw = 0.04  # default value
    d = np.array(d) # make sure data is in np array format

    # transpose data if needed, assumes larger dimension is time (# samples)
    if d.shape == 2 & len(d) > len(d[0]):
        td = np.transpose(d)
        flipped = True
    else:
        td = d
        flipped = False

    if len(td.shape) > 2:
        raise ValueError('Only accepts data in a 1-D or 2-D list.')

    numsamples = int(round(llw * sfx))  # calculate number of samples per transform window

    if len(td.shape) == 1:
        return optimize_lile_helper_1d(td, numsamples)
    else:
        return optimize_lile_helper_2d(td, numsamples)

