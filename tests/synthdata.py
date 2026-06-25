#
# FRETBursts - A single-molecule FRET burst analysis toolkit.
#
"""
Synthetic Photon-HDF5 data generators for the test-suite.

These produce small, self-contained Photon-HDF5 files (via ``phconvert``) that
mimic the structure of the real datasets the tests used to download from
figshare/zenodo. The goal is *structural* fidelity (measurement type, ALEX
alternation, nanotimes, multi-spot layout) and enough burst density that a
standard burst search finds bursts -- not physical realism.

The numeric assertions in the test-suite are invariant/relational (e.g.
``time_max == ph_times.max() * clk_p``, ``num_bursts > 0``), so synthetic data
is sufficient to exercise them without any network access.
"""

import numpy as np
import phconvert as phc


CLK_P = 12.5e-9          # timestamp unit (s) -- matches real usALEX data
ALEX_PERIOD = 4000       # alternation period in timestamp units

# Acquisition length. It must be long enough that background estimation (the
# tests use ``calc_bg(time_s=30)``) yields *at least two* background periods:
# several analyses (e.g. ``calc_mdelays_hist`` / ``hist_mdelays``) index a second
# period (``d.Lim[i][1]``) or drop the last one (``bg[...][0:-1]``), which is only
# valid with >=2 periods. Real datasets are long; 40 s gives 2 periods at 30 s.
DURATION_S = 40.0
BURST_RATE_HZ = 20       # bursts per second (keeps burst density duration-independent)
# Fraction of acceptor-excitation photons detected on the *donor* channel. Real
# data always has some donor signal during acceptor excitation (leakage / dark
# counts); without it the ``Aex/Dem`` photon selection is empty and functions
# reducing over it (e.g. ``hist_interphoton_single`` -> ``array.max()``) fail.
AEX_DEM_LEAK = 0.1


def _n_bursts(duration_s):
    """Number of bursts for a given duration, at a constant density."""
    return int(round(BURST_RATE_HZ * duration_s))


def _apply_aex_leak(rng, det, aex_mask, donor=0):
    """Reassign a fraction (``AEX_DEM_LEAK``) of acceptor-excitation photons to
    the donor channel, in place, so the ``Aex/Dem`` selection is non-empty."""
    aex_idx = np.where(aex_mask)[0]
    n_leak = int(round(AEX_DEM_LEAK * aex_idx.size))
    if n_leak:
        leak_idx = rng.choice(aex_idx, size=n_leak, replace=False)
        det[leak_idx] = donor


def _simulate_stream(rng, duration_s, bg_cps, n_bursts, burst_size,
                     burst_dur_s, clk_p=CLK_P):
    """Simulate a single photon stream (background + bursts).

    Returns integer timestamps (monotonically increasing) in ``clk_p`` units.
    """
    # Background: homogeneous Poisson process
    n_bg = rng.poisson(bg_cps * duration_s)
    t_bg = rng.uniform(0, duration_s, size=n_bg)

    # Bursts: dense clusters scattered over the acquisition
    centers = rng.uniform(burst_dur_s, duration_s - burst_dur_s, size=n_bursts)
    sizes = rng.poisson(burst_size, size=n_bursts)
    t_burst = np.concatenate([
        c + rng.normal(0, burst_dur_s / 4, size=s)
        for c, s in zip(centers, sizes)
    ]) if n_bursts else np.empty(0)
    t_burst = np.clip(t_burst, 0, duration_s)

    t = np.concatenate([t_bg, t_burst])
    t.sort()
    ts = np.round(t / clk_p).astype(np.int64)
    # Enforce strictly increasing timestamps (Photon-HDF5 / burst search expect it)
    ts = _make_strictly_increasing(ts)
    return ts


def _make_strictly_increasing(ts):
    """Bump any duplicate/decreasing timestamps so the array is strictly increasing."""
    ts = np.asarray(ts, dtype=np.int64)
    for i in range(1, ts.size):
        if ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + 1
    return ts


def _usalex_setup():
    return dict(
        num_pixels=2, num_spots=1, num_spectral_ch=2,
        num_polarization_ch=1, num_split_ch=1,
        modulated_excitation=True, lifetime=False,
        excitation_alternated=[True, True],
        excitation_cw=[True, True],
        excitation_wavelengths=[532e-9, 635e-9],
        detection_wavelengths=[580e-9, 680e-9],
    )


def make_usalex(path, seed=1, duration_s=DURATION_S, fret_eff=0.4,
                overwrite=True):
    """Write a single-spot us-ALEX Photon-HDF5 file at ``path``."""
    rng = np.random.default_rng(seed)
    ts = _simulate_stream(rng, duration_s, bg_cps=4000,
                          n_bursts=_n_bursts(duration_s),
                          burst_size=60, burst_dur_s=1e-3)

    # Alternation: first half of the period = donor excitation, second = acceptor
    d_on = (0, ALEX_PERIOD // 2)
    a_on = (ALEX_PERIOD // 2, ALEX_PERIOD)
    phase = ts % ALEX_PERIOD
    d_ex = (phase >= d_on[0]) & (phase < d_on[1])

    # Detector assignment: donor=0, acceptor=1
    det = np.ones(ts.size, dtype=np.uint8)        # default acceptor
    # During donor excitation, emit donor (1-E) or acceptor (E, via FRET)
    n_dex = int(d_ex.sum())
    det[d_ex] = (rng.random(n_dex) < fret_eff).astype(np.uint8)
    # Donor leakage during acceptor excitation (keeps Aex/Dem non-empty)
    _apply_aex_leak(rng, det, ~d_ex)

    data = {
        'description': 'Synthetic us-ALEX test data (FRETBursts test-suite).',
        'acquisition_duration': float(duration_s),
        'photon_data': {
            'timestamps': ts,
            'timestamps_specs': {'timestamps_unit': CLK_P},
            'detectors': det,
            'measurement_specs': {
                'measurement_type': 'smFRET-usALEX',
                'alex_period': ALEX_PERIOD,
                'alex_offset': 0,
                'alex_excitation_period1': np.array(d_on),
                'alex_excitation_period2': np.array(a_on),
                'detectors_specs': {
                    'spectral_ch1': np.atleast_1d(0),
                    'spectral_ch2': np.atleast_1d(1),
                },
            },
        },
        'setup': _usalex_setup(),
    }
    phc.hdf5.save_photon_hdf5(data, h5_fname=str(path), overwrite=overwrite,
                              close=True)
    return str(path)


TCSPC_NUM_BINS = 4096
TCSPC_UNIT = 50e-9 / TCSPC_NUM_BINS     # ~50 ns laser period spread over the bins
NS_CLK_P = 50e-9                        # macrotime unit for ns-ALEX
# Excitation windows in *nanotime* (TCSPC bin) units
NS_D_ON = (100, 2000)
NS_A_ON = (2148, 4000)


def _nsalex_setup():
    s = _usalex_setup()
    s.update(lifetime=True, excitation_cw=[False, False])
    return s


def make_nsalex(path, seed=1, duration_s=DURATION_S, fret_eff=0.4,
                overwrite=True):
    """Write a single-spot ns-ALEX (TCSPC) Photon-HDF5 file at ``path``."""
    rng = np.random.default_rng(seed)
    ts = _simulate_stream(rng, duration_s, bg_cps=4000,
                          n_bursts=_n_bursts(duration_s),
                          burst_size=60, burst_dur_s=1e-3, clk_p=NS_CLK_P)
    n = ts.size

    # PIE: each photon belongs to the donor or the acceptor excitation pulse
    donor_ex = rng.random(n) < 0.5
    det = np.ones(n, dtype=np.uint8)                       # default acceptor (1)
    nanotimes = np.empty(n, dtype=np.int64)

    # Acceptor-excitation photons: nanotime in A_ON window, detected on acceptor
    n_aex = int((~donor_ex).sum())
    nanotimes[~donor_ex] = rng.integers(NS_A_ON[0] + 1, NS_A_ON[1], size=n_aex)

    # Donor-excitation photons: nanotime in D_ON window; emit donor (1-E)/acceptor (E)
    n_dex = int(donor_ex.sum())
    nanotimes[donor_ex] = rng.integers(NS_D_ON[0] + 1, NS_D_ON[1], size=n_dex)
    det[donor_ex] = (rng.random(n_dex) < fret_eff).astype(np.uint8)
    # Donor leakage during acceptor excitation (keeps Aex/Dem non-empty)
    _apply_aex_leak(rng, det, ~donor_ex)

    data = {
        'description': 'Synthetic ns-ALEX test data (FRETBursts test-suite).',
        'acquisition_duration': float(duration_s),
        'photon_data': {
            'timestamps': ts,
            'timestamps_specs': {'timestamps_unit': NS_CLK_P},
            'detectors': det,
            'nanotimes': nanotimes,
            'nanotimes_specs': {
                'tcspc_unit': TCSPC_UNIT,
                'tcspc_num_bins': TCSPC_NUM_BINS,
                'tcspc_range': TCSPC_UNIT * TCSPC_NUM_BINS,
            },
            'measurement_specs': {
                'measurement_type': 'smFRET-nsALEX',
                'laser_repetition_rate': 20e6,
                'alex_excitation_period1': np.array(NS_D_ON),
                'alex_excitation_period2': np.array(NS_A_ON),
                'detectors_specs': {
                    'spectral_ch1': np.atleast_1d(0),
                    'spectral_ch2': np.atleast_1d(1),
                },
            },
        },
        'setup': _nsalex_setup(),
    }
    phc.hdf5.save_photon_hdf5(data, h5_fname=str(path), overwrite=overwrite,
                              close=True)
    return str(path)


def make_multispot(path, num_spots=8, seed=1, duration_s=DURATION_S,
                   fret_eff=0.4, overwrite=True):
    """Write a multi-spot, non-alternated smFRET Photon-HDF5 file at ``path``."""
    rng = np.random.default_rng(seed)
    data = {
        'description': 'Synthetic %d-spot smFRET test data (FRETBursts '
                       'test-suite).' % num_spots,
        'acquisition_duration': float(duration_s),
        'setup': dict(
            num_pixels=2 * num_spots, num_spots=num_spots, num_spectral_ch=2,
            num_polarization_ch=1, num_split_ch=1,
            modulated_excitation=False, lifetime=False,
            excitation_alternated=[False],
            excitation_cw=[True],
            excitation_wavelengths=[532e-9],
            detection_wavelengths=[580e-9, 680e-9],
        ),
    }
    for spot in range(num_spots):
        ts = _simulate_stream(rng, duration_s, bg_cps=4000,
                              n_bursts=_n_bursts(duration_s),
                              burst_size=60, burst_dur_s=1e-3)
        # Non-alternated: emission channel encodes FRET directly (donor=0, accept=1)
        det = (rng.random(ts.size) < fret_eff).astype(np.uint8)
        data['photon_data%d' % spot] = {
            'timestamps': ts,
            'timestamps_specs': {'timestamps_unit': CLK_P},
            'detectors': det,
            'measurement_specs': {
                'measurement_type': 'smFRET',
                'detectors_specs': {
                    'spectral_ch1': np.atleast_1d(0),
                    'spectral_ch2': np.atleast_1d(1),
                },
            },
        }
    phc.hdf5.save_photon_hdf5(data, h5_fname=str(path), overwrite=overwrite,
                              close=True)
    return str(path)


if __name__ == '__main__':
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else 'usalex'
    out = sys.argv[2] if len(sys.argv) > 2 else 'synth_%s.hdf5' % kind
    {'usalex': make_usalex, 'nsalex': make_nsalex,
     'multispot': make_multispot}[kind](out)
    print('wrote', out)
