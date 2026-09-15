"""Faithful Python port of phastimate (Zrenner et al. 2020, github.com/bnplab/phastimate).

Two functions, matching the MATLAB source in ./phastimate/:
  - phastimate()   -> pre-stimulus instantaneous phase (phastimate.m)
  - estimate_snr() -> 1/f-corrected spectral SNR of the target band (estimate_SNR.m)
"""

import numpy as np
from mne.time_frequency import psd_array_multitaper
from scipy.signal import filtfilt, firwin, hilbert


def design_bandpass(order, f_lo, f_hi, fs):
    """Windowed-sinc FIR bandpass = MATLAB designfilt('bandpassfir', ..., 'window')."""
    return firwin(order + 1, [f_lo, f_hi], pass_zero=False, fs=fs)


def _aryule(x, order):
    """AR Yule-Walker coefficients via Levinson-Durbin (matches MATLAB aryule).

    Returns a with a[0] == 1, so x[n] = -sum_{k>=1} a[k] x[n-k] + noise.
    """
    x = x - x.mean()
    r = np.correlate(x, x, "full")[len(x) - 1:][: order + 1] / len(x)  # biased autocorr
    a = np.array([1.0])
    e = r[0]
    for i in range(1, order + 1):
        k = -(r[i] + np.dot(a[1:i], r[i - 1:0:-1])) / e
        a = np.concatenate([a, [0.0]]) + k * np.concatenate([[0.0], a[::-1]])
        e *= 1 - k * k
    return a


def phastimate(data, b, edge, ar_order, hilbert_window, offset=0, iterations=None, return_trace=False):
    """Estimate the phase at the end of `data` (a 1-D window ending near the pulse).

    b               FIR bandpass coefficients (from design_bandpass)
    edge            filter-edge samples removed after filtfilt
    ar_order        Yule-Walker AR model order
    hilbert_window  samples used for the Hilbert transform
    offset          phase-index offset correction
    iterations      forward-predicted samples (None -> edge + ceil(hilbert_window/2))
    return_trace    if True, also return (real_core, predicted_future) sample arrays
    Returns (phase_rad, amplitude) or (phase_rad, amplitude, real_core, predicted_future).
    """
    if iterations is None:
        iterations = edge + int(np.ceil(hilbert_window / 2))
    data = data - data.mean()
    # b is FIR (a=1.0): impulse response is exactly len(b)-1 samples long, so that many
    # padded samples is the exact (not heuristic) amount needed to fully absorb the
    # filtfilt edge transient -- well below scipy's generic 3*len(b) IIR-safe default.
    core = filtfilt(b, 1.0, data, padlen=len(b) - 1)[edge:-edge]    # filter, drop edges
    coeffs = -_aryule(core, ar_order)[1:][::-1]                    # forward-predict weights
    pred = np.concatenate([core, np.zeros(iterations)])
    for i in range(len(core), len(core) + iterations):
        pred[i] = np.dot(coeffs, pred[i - ar_order:i])
    analytic = hilbert(pred[-hilbert_window:])
    idx = hilbert_window - iterations + edge + offset - 1          # 0-based phase index
    if 0 <= idx < len(analytic):
        phase, amplitude = np.angle(analytic[idx]), np.mean(np.abs(analytic))
    else:                                                          # e.g. iterations >> hilbert_window
        phase, amplitude = np.nan, np.mean(np.abs(analytic))
    if return_trace:
        return phase, amplitude, core, pred[len(core):]
    return phase, amplitude


def estimate_snr(signal, fs, band, fit_ranges, bandwidth, return_fit=False):
    """Band SNR in dB after subtracting the 1/f spectral background (estimate_SNR.m).

    fit_ranges is a sequence of (lo_hz, hi_hz) pairs: the background-only bins used to
    fit the 1/f trend. These must avoid `band` (and any other real oscillatory peak) --
    if a fit bin overlaps a real peak, the fit gets pulled toward it and then that same
    peak gets subtracted back out, suppressing the very thing you're trying to measure.
    The original estimate_SNR.m default (0.5-7 Hz, 35-65 Hz) is safe for its 8-13 Hz mu/
    alpha target but overlaps a 4-8 Hz theta target, so this isn't hardcoded here.

    bandwidth   multitaper spectral smoothing bandwidth in Hz (full width, not half).
                Replaces Welch's nperseg/noverlap: instead of averaging over short,
                chopped-up time segments (which coarsens frequency resolution -- the
                original estimate_SNR.m default nperseg=2*fs assumes several seconds of
                continuous data, which a single short epoch doesn't have), multitaper
                keeps the full signal length and averages several orthogonal tapers
                instead, so variance is reduced without sacrificing frequency resolution.

    return_fit  if True, also return a dict with the spectrum and fit diagnostics
                (f, log_p, slope, intercept, r2, fit_mask) for plotting/inspection.

    peak_freq/snr_db are the frequency/height of the highest bin inside `band` in the
    1/f-corrected spectrum -- not a `find_peaks`-style local maximum. With a short window,
    `band` can hold as few as 1-2 bins, too few for a strict local-max-with-prominence peak
    shape to ever form even when the band genuinely sits above the 1/f background. Taking
    the band max instead guarantees a real (possibly low or negative dB) value for every
    call; a trial with no true oscillation shows up as a low/negative snr_db rather than
    silently vanishing as nan -- filter on that downstream if a threshold is wanted.

    Returns (peak_frequency_hz, snr_db) or, with return_fit, (peak_frequency_hz, snr_db,
    fit_info). (nan, nan[, fit_info]) only if `band` falls entirely outside the 2-45 Hz
    range this function evaluates (e.g. a misconfigured `band`).
    """
    pxx, f = psd_array_multitaper(
        signal, sfreq=fs, fmin=2, fmax=45, bandwidth=bandwidth,
        adaptive=True, low_bias=True, normalization="length", verbose=False,
    )
    log_p = 10 * np.log10(pxx)
    log_f = np.log10(f)
    fit_mask = np.zeros_like(f, dtype=bool)
    for lo, hi in fit_ranges:
        fit_mask |= (f >= lo) & (f <= hi)
    slope, intercept = np.polyfit(log_f[fit_mask], log_p[fit_mask], 1)
    corrected = log_p - (slope * log_f + intercept)
    band_idx = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    if band_idx.size == 0:
        peak_freq, snr_db = np.nan, np.nan
    else:
        best = band_idx[np.argmax(corrected[band_idx])]
        peak_freq, snr_db = f[best], corrected[best]

    if not return_fit:
        return peak_freq, snr_db

    # R^2 computed on the fit bins only -- a good 1/f fit should NOT explain the target
    # band (that gap is the peak), so scoring it against the full spectrum would be wrong.
    fit_resid = log_p[fit_mask] - (slope * log_f[fit_mask] + intercept)
    ss_res = np.sum(fit_resid ** 2)
    ss_tot = np.sum((log_p[fit_mask] - log_p[fit_mask].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    fit_info = dict(f=f, log_p=log_p, slope=slope, intercept=intercept, r2=r2, fit_mask=fit_mask)
    return peak_freq, snr_db, fit_info
