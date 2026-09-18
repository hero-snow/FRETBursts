#!/usr/bin/env python
"""Cross-check FRETBursts (this fork) against smfBursts on one Photon-HDF5 file.

smfBursts (https://github.com/OpenSMFS/smfBursts) is the declared successor of
FRETBursts. This script runs the canonical us-ALEX pipeline in both packages with
identical parameters and compares the numbers at every stage:

  1. photon streams          (loader / alternation-period agreement)
  2. background rates        (per period, per stream)
  3. burst boundaries        (start/stop timestamps, exact and overlap matching)
  4. per-burst photon counts (raw: must be identical; corrected: tolerance)
  5. E / S per burst         (tolerance)
  6. burst selection         (size threshold, add_naa)
  7. E / S histograms + KDE peak of the selected population

Outputs (in --out): report.md, fretbursts_bursts.csv, smfbursts_bursts.csv,
compare.png. Exit status is 1 only when a *hard* check fails (photon streams or
raw counts on matched bursts disagree) -- those indicate a loader or burst-search
bug rather than a formula difference.

Parameter mapping (FRETBursts -> smfBursts), from the upstream
``us-ALEX_translation.ipynb`` shipped with smfBursts:

  loader.photon_hdf5 + alex_apply_period  -> photonHDF5.load + regularize_dets
  calc_bg(bg.exp_fit, time_s, 'auto', F_bg) -> make_bg(func=exp_mlefit, period,
                                                auto_threshold=True, F_bg)
  burst_search(L, m, F, Ph_sel('all'))    -> make_burst_search(m, F,
                                                streams=PhSel('0ex_1ex1em'))
                                             (no L: it is a gate on NphActive_raw;
                                              with L <= m it never removes anything)
  select_bursts.size(th1, add_naa=True)   -> make_geq_gate(NphActive_c, th1)

Known convention differences this script normalises or reports:
  * FRETBursts subtracts alex_offset from every timestamp, smfBursts does not.
    Bursts are therefore matched on photon *indices* (both packages keep the
    same photon array) and the constant timestamp shift is reported once.
  * FRETBursts' Ph_sel('all') on ALEX data includes the AexDem stream in both
    background estimation and burst search; smfBursts' translation notebook maps
    it to PhSel('0ex_1ex1em') (AexDem excluded). Use --smf-streams all to test
    the other reading.
  * smfBursts fuses bursts whose threshold-crossing windows overlap
    (fuse=0.0 default); FRETBursts does not fuse unless asked. Use --smf-fuse -1
    to disable fusing on the smfBursts side.
  * FRETBursts keeps a trailing partial background period; smfBursts drops it.

Usage:
    uv run --no-sync python tools/compare_smfbursts.py [DATA.hdf5] [--out DIR]
With no DATA argument the standard usALEX example dataset
(0023uLRpitc_NTP_20dT_0.5GndCl.hdf5) is downloaded into ./data/.
Requires the ``compare`` dependency group:  uv sync --group compare
"""

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', category=UserWarning)

DEFAULT_FILE = '0023uLRpitc_NTP_20dT_0.5GndCl.hdf5'
ZENODO_DOI = 'doi:10.5281/zenodo.20038738'          # smfBursts' test-data record
FIGSHARE_URL = ('https://ndownloader.figshare.com/files/2182601/'
                + DEFAULT_FILE)                       # FRETBursts' notebook URL

E_BINS = np.arange(-0.2, 1.2 + 0.03 / 2, 0.03)        # FRETBursts hist_fret default
S_BINS = E_BINS

STREAMS = ('DD', 'DA', 'AA', 'AD')                    # DexDem, DexAem, AexAem, AexDem

# Series colors (dataviz reference palette, categorical slots 1 and 2).
COLOR_A = '#2a78d6'
COLOR_B = '#eb6834'


# ---------------------------------------------------------------------------
# Data container shared by both pipelines
# ---------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    version: str
    clk_p: float
    n_ph: int
    stream_counts: dict           # {'DD': int, ...} photon-level
    n_periods: int
    bg: dict                      # {'DD': array(period), 'DA', 'AA', 'ALL'}
    istart: np.ndarray            # index of first photon
    istop: np.ndarray             # index one past the last photon
    start: np.ndarray             # clk units, first photon
    stop: np.ndarray              # clk units, last photon
    raw: dict                     # {'DD': int array per burst, 'DA', 'AA'}
    nd: np.ndarray                # corrected counts
    na: np.ndarray
    naa: np.ndarray
    E: np.ndarray
    S: np.ndarray
    size: np.ndarray              # corrected size used for selection
    sel: np.ndarray               # bool mask, selected bursts
    notes: list = field(default_factory=list)
    t0: int = 0                   # timestamp of the first photon (clk)

    @property
    def n_bursts(self):
        return self.start.size

    def frame(self):
        import pandas as pd
        return pd.DataFrame({
            'istart': self.istart, 'istop': self.istop,
            'start_clk': self.start, 'stop_clk': self.stop,
            'DD_raw': self.raw['DD'], 'DA_raw': self.raw['DA'], 'AA_raw': self.raw['AA'],
            'nd': self.nd, 'na': self.na, 'naa': self.naa,
            'E': self.E, 'S': self.S, 'size': self.size, 'selected': self.sel,
        })


def _counts_from_masks(mask, istart, istop_excl):
    """Photon counts of `mask` inside [istart, istop_excl) for every burst."""
    cs = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
    return cs[istop_excl] - cs[istart]


# ---------------------------------------------------------------------------
# FRETBursts pipeline
# ---------------------------------------------------------------------------
def run_fretbursts(fn, p):
    import fretbursts
    from fretbursts import Ph_sel, bg, loader, select_bursts

    d = loader.photon_hdf5(str(fn))
    d.leakage, d.dir_ex, d.gamma = p.leakage, p.dir_ex, p.gamma
    if not d.alternated:
        raise SystemExit('FRETBursts: file is not an ALEX measurement; '
                         'this script only handles us-ALEX data.')
    loader.alex_apply_period(d)

    d.calc_bg(bg.exp_fit, time_s=p.bg_period, tail_min_us='auto', F_bg=p.F_bg)
    d.burst_search(L=p.L, m=p.m, F=p.F, ph_sel=Ph_sel(p.fb_ph_sel))

    sel = {'DD': Ph_sel(Dex='Dem'), 'DA': Ph_sel(Dex='Aem'),
           'AA': Ph_sel(Aex='Aem'), 'AD': Ph_sel(Aex='Dem')}
    masks = {k: d.get_ph_mask(ich=0, ph_sel=s) for k, s in sel.items()}

    mb = d.mburst[0]
    istart, istop_excl = mb.istart, mb.istop + 1       # FRETBursts istop is inclusive
    raw = {k: _counts_from_masks(masks[k], istart, istop_excl) for k in ('DD', 'DA', 'AA')}

    bgd = {'DD': np.asarray(d.bg[Ph_sel(Dex='Dem')][0]),
           'DA': np.asarray(d.bg[Ph_sel(Dex='Aem')][0]),
           'AA': np.asarray(d.bg[Ph_sel(Aex='Aem')][0]),
           'ALL': np.asarray(d.bg[Ph_sel('all')][0])}

    size = d.burst_sizes_ich(ich=0, gamma=p.gamma, add_naa=True)
    selmask, _ = select_bursts.size(d, ich=0, th1=p.size_th, add_naa=True, gamma=p.gamma)

    return Result(
        name='FRETBursts', version=fretbursts.__version__, clk_p=d.clk_p,
        n_ph=int(d.ph_times_m[0].size),
        stream_counts={k: int(m.sum()) for k, m in masks.items()},
        n_periods=len(bgd['DD']), bg=bgd,
        istart=np.asarray(istart, np.int64), istop=np.asarray(istop_excl, np.int64),
        start=np.asarray(mb.start, dtype=np.int64), stop=np.asarray(mb.stop, dtype=np.int64),
        raw=raw, nd=np.asarray(d.nd[0], float), na=np.asarray(d.na[0], float),
        naa=np.asarray(d.naa[0], float), E=np.asarray(d.E[0], float),
        S=np.asarray(d.S[0], float), size=np.asarray(size, float),
        sel=np.asarray(selmask, bool), t0=int(d.ph_times_m[0][0]),
    )


# ---------------------------------------------------------------------------
# smfBursts pipeline
# ---------------------------------------------------------------------------
def run_smfbursts(fn, p):
    import smfbursts as smf

    raw_data = smf.photonHDF5.load(str(fn))
    data = smf.photonHDF5.regularize_dets(raw_data)

    bgd = smf.fretfactory.make_bg(data, func=smf.bg.exp_mlefit, period=float(p.bg_period),
                                  auto_threshold=True, F_bg=p.F_bg)
    bsd = smf.fretfactory.make_burst_search(bg=bgd['bg'], m=p.m, F=float(p.F),
                                            streams=smf.PhSel(p.smf_streams), fuse=p.smf_fuse,
                                            lk=p.leakage, dir_ex=p.dir_ex, gamma=p.gamma)
    bursts = bsd['bursts']

    def col(name, *args):
        return data.get_column(smf.Column(bursts, name, *args))

    istart, istop = col('istart'), col('istop')       # photon slice [istart, istop)
    # smfBursts' own 'start'/'stop' columns are window-threshold crossing times,
    # not photon timestamps (its 'Dur' column is photon-based though). FRETBursts
    # stores first/last photon timestamps, so derive those from the indices.
    times = data.times
    start, stop = times[istart], times[istop - 1]

    sel = {'DD': smf.PhSel('0ex0em'), 'DA': smf.PhSel('0ex1em'),
           'AA': smf.PhSel('1ex1em'), 'AD': smf.PhSel('1ex0em')}
    masks = {k: np.isin(data.dets, data.detdef.get_stream_ids(s)) for k, s in sel.items()}
    raw = {k: _counts_from_masks(masks[k], istart, istop) for k in ('DD', 'DA', 'AA')}
    # cross-check the independent count against smfBursts' own raw columns
    for k, key in (('DD', 'NphDD_raw'), ('DA', 'NphDA_raw'), ('AA', 'NphAA_raw')):
        own = data.get_column(bsd[key])
        if not np.array_equal(own, raw[k]):
            raise AssertionError(f'smfBursts: {key} != counts from istart/istop masks')

    bg_cols = {'DD': 'BgDD', 'DA': 'BgDA', 'AA': 'BgAA', 'ALL': 'BgAll'}
    bg = {k: np.asarray(data.get_column(bgd[c]), float) for k, c in bg_cols.items()}

    size = np.asarray(data.get_column(bsd['NphActive_c']), float)
    gate = smf.make_geq_gate(bsd['NphActive_c'], float(p.size_th))
    n_gated = data.get_column(bsd['E'], gate=gate).size
    selmask = size >= p.size_th
    notes = []
    if n_gated != selmask.sum():
        notes.append(f'gate count {n_gated} != mask count {selmask.sum()}')
    if p.L > p.m:
        selmask &= data.get_column(bsd['NphActive_raw']) >= p.L
        notes.append(f'L={p.L} > m={p.m}: applied as a gate on NphActive_raw')

    return Result(
        name='smfBursts', version=smf.__version__, clk_p=data.clk_p,
        n_ph=int(times.size),
        stream_counts={k: int(m.sum()) for k, m in masks.items()},
        n_periods=len(bg['DD']), bg=bg,
        istart=np.asarray(istart, np.int64), istop=np.asarray(istop, np.int64),
        start=np.asarray(start, np.int64), stop=np.asarray(stop, np.int64),
        raw=raw,
        nd=np.asarray(data.get_column(bsd['NphDD_c']), float),
        na=np.asarray(data.get_column(bsd['NphDA_c']), float),
        naa=np.asarray(data.get_column(bsd['NphAA_c']), float),
        E=np.asarray(data.get_column(bsd['E']), float),
        S=np.asarray(data.get_column(bsd['S']), float),
        size=size, sel=selmask, notes=notes, t0=int(times[0]),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def match_bursts(a, b):
    """Match bursts on photon indices.

    Returns (ia, ib) of bursts with identical [istart, istop), (ja, jb) of
    bursts sharing istart (a superset), and the number of bursts of each side
    that overlap at least one burst of the other side.
    """
    ka = a.istart * np.int64(1 << 24) + a.istop      # bursts are < 2^24 photons long
    kb = b.istart * np.int64(1 << 24) + b.istop
    _, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    _, ja, jb = np.intersect1d(a.istart, b.istart, return_indices=True)

    def n_overlapping(x, y):
        j = np.searchsorted(y.istart, x.istop, side='left') - 1
        j = np.clip(j, 0, y.n_bursts - 1)
        return int(np.sum((y.istop[j] > x.istart) & (y.istart[j] < x.istop)))

    return ia, ib, ja, jb, n_overlapping(a, b), n_overlapping(b, a)


def hist_l1(x, y, bins):
    hx, _ = np.histogram(x, bins=bins, density=True)
    hy, _ = np.histogram(y, bins=bins, density=True)
    w = np.diff(bins)
    return float(np.sum(np.abs(hx - hy) * w))


def kde_peak(x, lo=-0.2, hi=1.2):
    from scipy.stats import gaussian_kde
    x = x[(x > lo) & (x < hi)]
    if x.size < 10:
        return float('nan')
    grid = np.arange(lo, hi, 0.001)
    return float(grid[np.argmax(gaussian_kde(x)(grid))])


def rel_max(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return float('inf')
    den = np.maximum(np.abs(a), np.abs(b))
    den[den == 0] = 1
    return float(np.max(np.abs(a - b) / den)) if a.size else 0.0


class Report:
    def __init__(self):
        self.rows = []          # (section, check, A, B, diff, status)
        self.hard_fail = False

    def add(self, section, check, a, b, diff='', ok=None, hard=False):
        status = '' if ok is None else ('PASS' if ok else 'FAIL')
        if ok is False and hard:
            self.hard_fail = True
            status = 'FAIL (hard)'
        self.rows.append((section, check, a, b, diff, status))

    def markdown(self, ra, rb, args, fn):
        out = ['# FRETBursts vs smfBursts cross-check', '',
               f'- data: `{fn}`',
               f'- {ra.name} {ra.version}  |  {rb.name} {rb.version}',
               f'- params: m={args.m} F={args.F} L={args.L} bg_period={args.bg_period}s '
               f'F_bg={args.F_bg} size_th={args.size_th} leakage={args.leakage} '
               f'dir_ex={args.dir_ex} gamma={args.gamma}',
               f'- FRETBursts ph_sel={args.fb_ph_sel!r}; smfBursts streams={args.smf_streams!r} '
               f'fuse={args.smf_fuse}', '']
        for n in ra.notes + rb.notes:
            out.append(f'- note: {n}')
        out += ['', f'| section | check | {ra.name} | {rb.name} | diff | status |',
                '|---|---|---|---|---|---|']
        for sec, chk, a, b, diff, st in self.rows:
            out.append(f'| {sec} | {chk} | {a} | {b} | {diff} | {st} |')
        out += ['', 'Hard checks: photon streams and raw per-burst counts on matched bursts '
                    '(loader / burst-search identity). Everything else is reported with '
                    'tolerances and differences are expected to be small formula details.']
        return '\n'.join(out) + '\n'


def compare(ra, rb, args):
    rep = Report()
    def f6(x):
        return f'{x:.6g}'

    # 1. photons ------------------------------------------------------------
    rep.add('photons', 'clk_p', ra.clk_p, rb.clk_p, ok=ra.clk_p == rb.clk_p, hard=True)
    rep.add('photons', 'n_photons', ra.n_ph, rb.n_ph, ra.n_ph - rb.n_ph,
            ok=ra.n_ph == rb.n_ph, hard=True)
    for k in STREAMS:
        a, b = ra.stream_counts[k], rb.stream_counts[k]
        rep.add('photons', f'n_{k}', a, b, a - b, ok=a == b, hard=True)
    rep.add('photons', 'timestamp of first photon (clk)', ra.t0, rb.t0, ra.t0 - rb.t0)

    # 2. background -----------------------------------------------------------
    rep.add('background', 'n_periods', ra.n_periods, rb.n_periods,
            ra.n_periods - rb.n_periods, ok=ra.n_periods == rb.n_periods)
    n = min(ra.n_periods, rb.n_periods)
    for k in ('DD', 'DA', 'AA', 'ALL'):
        a, b = ra.bg[k][:n], rb.bg[k][:n]
        r = rel_max(a, b)
        rep.add('background', f'bg_{k} (Hz, mean over first {n} periods)',
                f6(a.mean()), f6(b.mean()), f'max rel {r:.2e}', ok=r < args.rtol_bg)

    # 3. bursts ---------------------------------------------------------------
    ia, ib, ja, jb, ov_a, ov_b = match_bursts(ra, rb)
    nm = ia.size
    nmax = max(ra.n_bursts, rb.n_bursts, 1)
    rep.add('bursts', 'n_bursts', ra.n_bursts, rb.n_bursts, ra.n_bursts - rb.n_bursts,
            ok=ra.n_bursts == rb.n_bursts)
    rep.add('bursts', 'identical photon range', nm, nm, f'{nm / nmax:.2%} of max',
            ok=nm / nmax >= args.match_frac)
    rep.add('bursts', 'same first photon', ja.size, jb.size, f'{ja.size / nmax:.2%} of max')
    if ja.size:
        dph = rb.istop[jb] - ra.istop[ja]
        dt = (rb.stop[jb] - rb.t0) - (ra.stop[ja] - ra.t0)
        p5, p50, p95 = np.percentile(dph, [5, 50, 95])
        rep.add('bursts', 'last photon of same-start bursts', '', '',
                f'{rb.name} minus {ra.name}: p5/p50/p95 = {p5:g}/{p50:g}/{p95:g} photons, '
                f'median {np.median(dt) * ra.clk_p * 1e3:.3f} ms', ok=p50 == 0)
    rep.add('bursts', 'overlapping some burst of other', ov_a, ov_b,
            f'unmatched: {ra.n_bursts - ov_a} / {rb.n_bursts - ov_b}')
    dup_a = ra.n_bursts - np.unique(ra.istart).size
    dup_b = rb.n_bursts - np.unique(rb.istart).size
    rep.add('bursts', 'bursts sharing a first photon (overlapping bursts)', dup_a, dup_b,
            '', ok=(dup_a == 0 and dup_b == 0))
    wa = np.median(ra.stop - ra.start) * ra.clk_p * 1e3
    wb = np.median(rb.stop - rb.start) * rb.clk_p * 1e3
    rep.add('bursts', 'median width (ms)', f'{wa:.3f}', f'{wb:.3f}', f'{wb - wa:+.3f}')
    ta = np.median(ra.raw['DD'] + ra.raw['DA'] + ra.raw['AA'])
    tb = np.median(rb.raw['DD'] + rb.raw['DA'] + rb.raw['AA'])
    rep.add('bursts', 'median raw DD+DA+AA', f'{ta:g}', f'{tb:g}', f'{tb - ta:+g}')

    # 4. raw counts on identical-range bursts (loader / index identity) -------
    for k in ('DD', 'DA', 'AA'):
        a, b = ra.raw[k][ia], rb.raw[k][ib]
        nbad = int(np.sum(a != b))
        rep.add('raw counts', f'{k} identical on {nm} identical-range bursts',
                int(a.sum()), int(b.sum()), f'{nbad} bursts differ',
                ok=(nbad == 0) if nm else None, hard=True)

    # 5. corrected counts, E, S on identical-range bursts ---------------------
    # E/S are compared on selected bursts only: below the size threshold a few
    # corrected counts near zero make the ratios arbitrarily sensitive. For E
    # the relevant size is the Dex one (nd + na), as in the FRETBursts tutorials.
    keep = ra.sel[ia] & rb.sel[ib]
    keep_dex = keep & ((ra.nd + ra.na)[ia] >= args.size_th)
    for label, a, b, tol, mask in (('nd', ra.nd, rb.nd, args.atol_counts, None),
                                   ('na', ra.na, rb.na, args.atol_counts, None),
                                   ('naa', ra.naa, rb.naa, args.atol_counts, None),
                                   ('size', ra.size, rb.size, args.atol_counts, None),
                                   ('E', ra.E, rb.E, args.atol_ratio, keep_dex),
                                   ('S', ra.S, rb.S, args.atol_ratio, keep)):
        sa, sb = (ia, ib) if mask is None else (ia[mask], ib[mask])
        what = f'{sa.size} identical-range' + ('' if mask is None else ', selected')
        if mask is keep_dex:
            what += f', nd+na>={args.size_th:g}'
        da = np.abs(a[sa] - b[sb])
        da = da[np.isfinite(da)]
        if not da.size:
            rep.add('corrected', label, '', '', f'no finite values ({what} bursts)')
            continue
        p50, p99, mx = np.percentile(da, [50, 99]).tolist() + [float(da.max())]
        rep.add('corrected', f'{label} (mean over {what} bursts)',
                f6(np.nanmean(a[sa])), f6(np.nanmean(b[sb])),
                f'abs diff median {p50:.3g}, p99 {p99:.3g}, max {mx:.3g}', ok=p99 <= tol)

    # 6. selection ------------------------------------------------------------
    na_sel, nb_sel = int(ra.sel.sum()), int(rb.sel.sum())
    rep.add('selection', f'n selected (size >= {args.size_th}, add_naa)', na_sel, nb_sel,
            na_sel - nb_sel, ok=na_sel == nb_sel)
    both = ra.sel[ia] == rb.sel[ib]
    rep.add('selection', 'same decision on identical-range bursts', int(both.sum()),
            int(both.sum()), f'{int((~both).sum())} disagree',
            ok=bool(both.all()) if nm else None)

    # 7. histograms of the selected population --------------------------------
    Ea, Eb = ra.E[ra.sel], rb.E[rb.sel]
    Sa, Sb = ra.S[ra.sel], rb.S[rb.sel]
    l1e, l1s = hist_l1(Ea, Eb, E_BINS), hist_l1(Sa, Sb, S_BINS)
    rep.add('histograms', 'E hist L1 distance (density)', '', '', f'{l1e:.4f}',
            ok=l1e < args.hist_l1)
    rep.add('histograms', 'S hist L1 distance (density)', '', '', f'{l1s:.4f}',
            ok=l1s < args.hist_l1)
    pa, pb = kde_peak(Ea), kde_peak(Eb)
    rep.add('histograms', 'E KDE peak', f'{pa:.4f}', f'{pb:.4f}', f'{abs(pa - pb):.4f}',
            ok=abs(pa - pb) <= args.atol_peak)
    pa, pb = kde_peak(Sa), kde_peak(Sb)
    rep.add('histograms', 'S KDE peak', f'{pa:.4f}', f'{pb:.4f}', f'{abs(pa - pb):.4f}',
            ok=abs(pa - pb) <= args.atol_peak)
    return rep, (ia[keep_dex], ib[keep_dex])


def plot(ra, rb, matched, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ia, ib = matched
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax in axs:
        ax.grid(True, color='#e5e5e5', linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    for ax, key, bins in ((axs[0], 'E', E_BINS), (axs[1], 'S', S_BINS)):
        va, vb = getattr(ra, key)[ra.sel], getattr(rb, key)[rb.sel]
        ax.hist(va, bins=bins, histtype='step', lw=2, color=COLOR_A,
                label=f'{ra.name} (n={va.size})')
        ax.hist(vb, bins=bins, histtype='step', lw=2, color=COLOR_B, ls='--',
                label=f'{rb.name} (n={vb.size})')
        ax.set_xlabel(key)
        ax.set_ylabel('bursts')
        ax.set_title(f'{key} histogram, selected bursts')
        ax.legend(frameon=False)

    ax = axs[2]
    if ia.size:
        d = rb.E[ib] - ra.E[ia]
        ax.scatter(ra.E[ia], d, s=8, color=COLOR_A, alpha=0.5, edgecolors='none')
        ax.axhline(0, color='#888888', lw=1)
        ax.set_xlim(-0.2, 1.2)
        ax.set_xlabel(f'E ({ra.name})')
        ax.set_ylabel(f'E ({rb.name}) - E ({ra.name})')
        ax.set_title(f'E difference, {ia.size} identical selected bursts')
        lim = max(0.01, float(np.percentile(np.abs(d), 99)) * 1.5)
        ax.set_ylim(-lim, lim)
        n_out = int(np.sum(np.abs(d) > lim))
        if n_out:
            ax.text(0.02, 0.97, f'{n_out} bursts outside +/-{lim:.3g}', transform=ax.transAxes,
                    va='top', fontsize=9, color='#555555')
    else:
        ax.text(0.5, 0.5, 'no matched bursts', ha='center', va='center',
                transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
def fetch_default(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    fn = data_dir / DEFAULT_FILE
    if fn.exists():
        return fn
    try:
        import pooch
        repo = pooch.create(path=str(data_dir), base_url=ZENODO_DOI)
        repo.load_registry_from_doi()
        return Path(repo.fetch(DEFAULT_FILE))
    except Exception as e:      # noqa: BLE001 - any pooch/network failure falls back
        print(f'[fetch] pooch/zenodo failed ({e!r}); trying figshare URL', file=sys.stderr)
        from fretbursts.utils.misc import download_file
        download_file(FIGSHARE_URL, save_dir=str(data_dir))
        return fn


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data', nargs='?', help='Photon-HDF5 us-ALEX file '
                    f'(default: download {DEFAULT_FILE} into --data-dir)')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out', default='compare_out')
    ap.add_argument('--m', type=int, default=10)
    ap.add_argument('--F', type=float, default=6.0)
    ap.add_argument('--L', type=int, default=10)
    ap.add_argument('--bg-period', type=float, default=30.0)
    ap.add_argument('--F-bg', type=float, default=1.7)
    ap.add_argument('--size-th', type=float, default=30.0)
    ap.add_argument('--leakage', type=float, default=0.11)
    ap.add_argument('--dir-ex', type=float, default=0.04)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--fb-ph-sel', default='all',
                    help="FRETBursts burst-search Ph_sel string (default 'all')")
    ap.add_argument('--smf-streams', default='0ex_1ex1em',
                    help="smfBursts burst-search PhSel string (default '0ex_1ex1em', "
                         "as in the upstream translation notebook; try 'all')")
    ap.add_argument('--smf-fuse', type=float, default=0.0,
                    help='smfBursts fuse (s): 0.0 = fuse overlapping windows (its '
                         'default), -1 = never fuse (closest to FRETBursts)')
    ap.add_argument('--no-plot', action='store_true')
    tol = ap.add_argument_group('tolerances')
    tol.add_argument('--rtol-bg', type=float, default=1e-3)
    tol.add_argument('--match-frac', type=float, default=0.99,
                     help='min fraction of bursts with identical start/stop')
    tol.add_argument('--atol-counts', type=float, default=0.5)
    tol.add_argument('--atol-ratio', type=float, default=0.01)
    tol.add_argument('--atol-peak', type=float, default=0.01)
    tol.add_argument('--hist-l1', type=float, default=0.05)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    fn = Path(args.data) if args.data else fetch_default(args.data_dir)
    if not fn.exists():
        raise SystemExit(f'data file not found: {fn}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f'[1/3] FRETBursts on {fn.name} ...', flush=True)
    ra = run_fretbursts(fn, args)
    print(f'[2/3] smfBursts on {fn.name} ...', flush=True)
    rb = run_smfbursts(fn, args)
    print('[3/3] comparing ...', flush=True)
    rep, matched = compare(ra, rb, args)

    ra.frame().to_csv(out / 'fretbursts_bursts.csv', index=False)
    rb.frame().to_csv(out / 'smfbursts_bursts.csv', index=False)
    md = rep.markdown(ra, rb, args, fn)
    (out / 'report.md').write_text(md, encoding='utf-8')
    if not args.no_plot:
        plot(ra, rb, matched, out / 'compare.png')
    print(md)
    print(f'outputs written to {out.resolve()}')
    return 1 if rep.hard_fail else 0


if __name__ == '__main__':
    sys.exit(main())
