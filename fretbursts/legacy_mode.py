#
# FRETBursts - A single-molecule FRET burst analysis toolkit.
#
# Copyright (C) 2014 Antonino Ingargiola <tritemio@gmail.com>
#
"""
Load this module as::

    from legacy_names import *

to load a bunch of names once available when loading FRETBursts. This module
is provided as a compatibility layer to be used to run old scripts or notebooks.

Please don't import this module on new code.

**Long rationale explanation**

On April 2014 FRETBursts project structure has changed in order to become a
conventional python package. During this refactoring the default way to load
FRETBursts changed from::

    run fretbursts.py

to::

    from fretbursts import *

The latter imports the submodules and packages as before, but it only imports
a restricted number of core object names (i.e. Data(), dplot(), see
__init__.py for details).
The old behaviour was loading a big chunk of matplotlib.pyplot and
a lot of functions from burstlib.

Now the preferred way is to use always `plt.` to access the matplotlib.pyplot
functions and `bl.` to access any function in `burstlib`. This change makes
maintenance, unit testing and installation easier.

This module provides "most" of the old object names now removed as a way to run
old code with minimal effort. Please do not use this in new code.
"""

__all__ = [
        # Library modules and functions
        "np", "r_", "zeros", "arange", "size", "SS",
        "plt", "rcParams", "figure", "subplots", "plot", "hist",
        "xlabel", "ylabel", "grid", "title", "legend", "xlim", "ylim",
        "axhline", "axvline",
        "savefig", "rc", "gca", "gcf", "sca",

        # Local modules
        "loader", "select_bursts", "bl", "bg", "bpl", "bext",

        # Classes, functions, variables
        "data_dir", "Data", "Sel", "Sel_mask", "Sel_mask_apply",
        "gamma_correct_E", "gamma_uncorrect_E",

        # Deprecated names  (only for compatibility)
        "load_multispot8", "select_bursts_nda", "select_bursts_time",
        "bg_calc_exp", "bg_calc_exp_cdf", "b_start",

        # Generic fit functions
        "gaussian_fit_hist",
        "gaussian_fit_cdf",
        "two_gaussian_fit_hist",
        "two_gaussian_fit_hist_min",
        "two_gaussian_fit_hist_min_ab",
        "two_gaussian_fit_EM",
        "two_gauss_mix_pdf",
        "two_gauss_mix_ab",

        # Standalone plots or plots as a function of ch
        "bsavefig", "mch_plot_bg", "plot_alternation_hist",

        # Plots types used for 1ch of multi-ch plots through `dplot`
        "timetrace", "timetrace_da", "ratetrace", "ratetrace_da",
        "timetrace_alex", "timetrace_fret",
        "hist_width", "hist_size", "hist_fret", "kde_fret", "hist_fret_kde",
        "hist2d_alex", "hist_S", "hist_sbr",
        "hist_bg_fit_single", "hist_bg_fit", "hist_ph_delays", "hist_mdelays",
        "hist_mrates", "hist_rate_in_burst", "hist_burst_delays",
        "scatter_width_size", "scatter_rate_da", "scatter_fret_size",
        "scatter_fret_nd_na", "scatter_fret_width", "scatter_da",
        "scatter_naa_nt", "scatter_alex",

        # Wrapper functions that create a plot for each channel
        "dplot", "dplot_48ch", "dplot_8ch", "dplot_1ch",
        ]

import background as bg
import burst_plot as bpl
import burstlib as bl
import burstlib_ext as bext
import loader
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as SS
import select_bursts
from burst_plot import (
        # Standalone plots as a function of ch
        bsavefig,
        # Wrapper functions that create a plot for each channel
        dplot,
        dplot_1ch,
        dplot_8ch,
        dplot_48ch,
        hist2d_alex,
        hist_bg_fit,
        hist_bg_fit_single,
        hist_burst_delays,
        hist_fret,
        hist_fret_kde,
        hist_mdelays,
        hist_mrates,
        hist_ph_delays,
        hist_rate_in_burst,
        hist_S,
        hist_sbr,
        hist_size,
        hist_width,
        kde_fret,
        mch_plot_bg,
        plot_alternation_hist,
        ratetrace,
        ratetrace_da,
        scatter_alex,
        scatter_da,
        scatter_fret_nd_na,
        scatter_fret_size,
        scatter_fret_width,
        scatter_naa_nt,
        scatter_rate_da,
        scatter_width_size,
        # Plots types used for 1ch of multi-ch plots through `dplot`
        timetrace,
        timetrace_alex,
        timetrace_da,
        timetrace_fret,
)
from fit.gaussian_fitting import (
        gaussian_fit_cdf,
        gaussian_fit_hist,
        two_gauss_mix_ab,
        two_gauss_mix_pdf,
        two_gaussian_fit_EM,
        two_gaussian_fit_hist,
        two_gaussian_fit_hist_min,
        two_gaussian_fit_hist_min_ab,
)
from loader import load_multispot8
from matplotlib import rcParams
from matplotlib.pyplot import (
        axhline,
        axvline,
        figure,
        gca,
        gcf,
        grid,
        hist,
        legend,
        plot,
        rc,
        savefig,
        sca,
        subplots,
        title,
        xlabel,
        xlim,
        ylabel,
        ylim,
)
from numpy import arange, r_, size, zeros
from select_bursts import select_bursts_nda, select_bursts_time

from .burstlib import (
        Data,
        Sel,
        Sel_mask,
        Sel_mask_apply,
        b_start,
        bg_calc_exp,
        bg_calc_exp_cdf,
        gamma_correct_E,
        gamma_uncorrect_E,
)
from .path_def import data_dir

print(" - Compatibility legacy mode ON.")
