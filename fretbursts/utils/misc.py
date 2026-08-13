#
# FRETBursts - A single-molecule FRET burst analysis toolkit.
#
# Copyright (C) 2014 Antonino Ingargiola <tritemio@gmail.com>
#
"""
Misc utility functions
"""

import os
import sys

import numpy as np


def selection_mask(arr, values):
    """Return a boolean mask, True when `arr` is one element of `values`.

    This function generalizes the comparison `arr == values`
    where `values` can be a scalar or a sequence.
    If a sequence, the returned mask is True each time `arr` has an element
    in `values`.

    Arguments:
        arr (array): input array for which a mask is computed.
        values (scalar or sequence): one or more values to be selected in
            `arr`.
    
    Returns:
        Boolean mask same size as `arr`. True where `arr` has an element 
        in `values`.
    """
    values = np.atleast_1d(values)
    mask = arr == values[0]
    for value in values[1:]:
        mask += arr == value
    return mask



def _is_list_of_arrays(obj):
    return isinstance(obj, list) and np.all([isinstance(v, np.ndarray)
                                             for v in obj])

class HistData:
    """Stores histogram counts and bins and provides derived fields.

    Attributes:
        counts (array, ints): array of counts in each bin
        bins (array): array of bin edges. Size is size(counts) + 1.
        bincenters (array): array of bin  centers. Size is size(counts).
        pdf (array, floats): array of normalized counts (aka PDF)
    """
    def __init__(self, counts, bins):
        self.counts = counts
        self.bins = bins
        self.binwidth = bins[1] - bins[0]

    @property
    def bincenters(self):
        if not hasattr(self, '_bincenters'):
            self._bincenters = self.bins[:-1] + 0.5*self.binwidth
        return self._bincenters

    @property
    def pdf(self):
        if not hasattr(self, '_pdf'):
            self._pdf = np.array(self.counts, dtype=np.float64)
            self._pdf /= (self.counts.sum() * self.binwidth)
        return self._pdf


def clk_to_s(t_ck, clk_p=12.5*1e-9):
    """Convert clock cycles to seconds."""
    return t_ck*clk_p

def pprint(s, mute=False):
    """Print immediately, even if inside a busy loop."""
    if mute: return
    sys.stdout.write(s)
    sys.stdout.flush()

def deprecate(function, old_name, new_name):
    def deprecated_function(*args, **kwargs):
        pprint("Function name %s is deprecated, use %s instead.\n" %\
                (old_name, new_name))
        res = function(*args, **kwargs)
        return res
    return deprecated_function

def shorten_fname(f):
    """Return a path with only the last subfolder (i.e. measurement date)."""
    return '/'.join(f.split('/')[-2:])

def binning(times, bin_width_ms=1, max_num_bins=1e5, clk_p=12.5e-9):
    """Return the binned histogram of array times."""
    bin_width_clk = (bin_width_ms*1e-3)/clk_p
    num_bins = min(times.max()/bin_width_clk, max_num_bins)
    h = np.histogram(times[times<(num_bins*bin_width_clk)], bins=num_bins)
    return h


def mkdir_p(path):
    """Create the path if not existent, otherwise do nothing.
    If `path` exists, and is not a dir, raise an exception.
    """
    import errno
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise exc

# Magic bytes every HDF5 file starts with (HDF5 superblock signature).
_HDF5_SIGNATURE = b'\x89HDF\r\n\x1a\n'


def _download_is_valid(path):
    """Return whether the file at `path` looks like a complete download.

    Rejects empty files and, for HDF5 destinations, anything not starting with
    the HDF5 signature. A failed download (an error page, a redirect stub, or
    an interrupted transfer) otherwise lands on disk under the data file's
    name and only surfaces much later as an opaque reader error.
    """
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    if os.path.splitext(path)[1].lower() in ('.hdf5', '.h5'):
        with open(path, 'rb') as f:
            return f.read(len(_HDF5_SIGNATURE)) == _HDF5_SIGNATURE
    return True


def download_file(url, save_dir='./'):
    """Download a file from `url` saving it to disk.

    The file name is taken from `url` and left unchanged.
    The destination dir can be set using `save_dir`
    (Default: the current dir).

    An existing file is kept only if it passes a sanity check, so a previously
    failed download is retried instead of being trusted forever. The transfer
    goes to a temporary file and is moved into place only once complete, so an
    interrupted download never leaves a partial file at the destination.
    """
    # Check if local path already exist
    fname = url.split('/')[-1]
    print('URL:  %s' % url)
    print('File: %s\n ' % fname)

    path = '/'.join([os.path.abspath(save_dir), fname])
    if os.path.exists(path):
        if _download_is_valid(path):
            print('File already on disk: %s \nDelete it to re-download.' % path)
            return
        print(f'Incomplete or corrupted file on disk, re-downloading: {path}')

    from urllib.error import HTTPError, URLError
    from urllib.request import urlretrieve

    # Download the file
    def _report(blocknr, blocksize, size):
        current = blocknr*blocksize/2**20
        sys.stdout.write(
            f"\rDownloaded {current:4.1f} / {size/2**20:4.1f} MB")
    mkdir_p(save_dir)

    # Download to a temporary name so a failure cannot clobber a good file
    # nor leave a partial one that later looks like a finished download.
    tmp_path = path + '.part'
    try:
        urlretrieve(url, tmp_path, _report)
    except HTTPError as e:
        _remove_if_exists(tmp_path)
        print(f'\nURL not found (HTTP {e.code}): {url}')
        return
    except URLError as e:
        _remove_if_exists(tmp_path)
        print(f'\nWrong URL or no connection.\n\nError:\n{e}\n')
        return

    if not _download_is_valid(tmp_path):
        _remove_if_exists(tmp_path)
        print('\nThe download did not return a valid data file. '
              f'The URL may be out of date:\n  {url}')
        return

    os.replace(tmp_path, path)


def _remove_if_exists(path):
    try:
        os.remove(path)
    except OSError:
        pass

def _large_equal(val0, val1):
    if type(val0) != type(val1):
        return False
    elif isinstance(val0, np.ndarray):
        if np.any(val0.shape != val1.shape):
            return False
        else:
            return np.all(val0 == val1)
    elif isinstance(val0, (list, tuple)):
        if len(val0) == len(val1):
            return np.all([_large_equal(v0, v1) for v0, v1 in zip(val0, val1)])
        else:
            return False
    else:
        return val0 == val1


def dict_equal(*dicts):
    key0 = dicts[0].keys()
    comp = np.all([key0 == dct.keys() for dct in dicts])
    if comp:
        for key, val0 in dicts[0].items():
            for dct in dicts[1:]:
                if not _large_equal(val0, dct[key]):
                    comp = False
                    break
            if not comp:
                break
    return comp
