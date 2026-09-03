"""
vtr_extraction.py
=================

Extraction of the gate-voltage Transition Region width (V_TR) and the
associated threshold voltages (V_T,on and V_T,off) from a linear-mode
(low V_DS) I_D-V_GS transfer curve of a FET, following:

    K. Jana et al., "Advancing the Supply Voltage Scalability of Oxide
    Semiconductor Transistors through Transition Region Engineering."
    (V_TR extraction scheme, Sec. V, Eqs. 6-9)

and the series-resistance (R_S/R_D) correction described in the companion
slide deck ("Impact of R_S/R_D on I_D,lin / Y", "Modified I'_DON decoupling
R_S/R_D", "Use of Y function", "Extracting R_S/R_D").

--------------------------------------------------------------------------
Physics summary
--------------------------------------------------------------------------
V_T,on  : extracted from the ABOVE-THRESHOLD (linear) regime by linearly
          extrapolating I_D-V_GS at the point of maximum transconductance
          (g_m) back to I_D = 0                                    (Eq. 9)

V_T,off : extracted from the SUBTHRESHOLD (exponential) regime by linearly
          extrapolating log10(I_D) vs V_GS at the point of steepest slope
          (minimum subthreshold swing SS_min) back to a reference current
          I_D0 = beta*eta_SS*phi_T^2*(1-exp(-V_DS/phi_T))          (Eqs. 7-8)

V_TR = V_T,on - V_T,off                                              (Eq. 6)

--------------------------------------------------------------------------
Series resistance (R_S, R_D)
--------------------------------------------------------------------------
Series resistance mainly distorts the ABOVE-THRESHOLD part of the curve
(subthreshold current is too small to drop appreciable voltage across
R_S/R_D), so V_T,off is essentially unaffected, while V_T,on (and hence
V_TR) can be substantially biased if R_S/R_D is ignored -- e.g. the
example in the slides shows extracted V_TR going from ~0.59 V (no R) down
to a spurious ~0.17 V once R_S/R_D is added and NOT corrected for.

Two correction strategies are implemented here:

1. Direct correction (requires known R_S, R_D):
        I_D' = I_D / (1 - I_D*(R_S+R_D)/V_DS)
   I_D' is used in place of I_D for the above-threshold (V_T,on) extraction.

2. Y-function method (does NOT require knowing R_S, R_D):
        Y = I_D / sqrt(g_m)
   Y is, to first order, independent of R_S/R_D, so it can be linearly
   extrapolated to Y = 0 to obtain V_T,on directly from RAW data. The
   extrapolation point is chosen where dY/dV_GS is maximal (where Y itself
   is most linear) -- NOT where g_m is maximal, since those are generally
   different points once R_S/R_D is present.

R_S+R_D can itself be *estimated* from the data once beta is known, via:
        R_S+R_D ~= V_DS/I_D,on - V_DS/(Y*beta)

--------------------------------------------------------------------------
Noise handling
--------------------------------------------------------------------------
Real transfer-curve data is noisy, and naive differencing (np.diff /
np.gradient) amplifies that noise badly -- especially for a second
derivative. To keep this robust:

  * All derivatives are computed with a Savitzky-Golay filter (which
    smooths and differentiates simultaneously) on a uniformly-resampled
    grid, rather than point-to-point finite differences.
  * All threshold-voltage extrapolations are done as a LOCAL LINEAR
    REGRESSION over a small window of points around the extremum of
    interest (not just "one slope value + one data point"), which further
    suppresses point-to-point noise while still tracking the local
    curvature correctly.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import pandas as pd
import csv

log_f = None
def print_file(*args, **kwargs):
    if log_f:
        print(*args, file=log_f, **kwargs)
        log_f.flush()
    else:
        print(*args, **kwargs)

# ----------------------------------------------------------------------------
# Physical constants
# ----------------------------------------------------------------------------
KB = 1.380649e-23      # J/K
Q = 1.602176634e-19    # C


def thermal_voltage(T=300.0):
    """Thermal voltage phi_T = kB*T/q, in volts."""
    return KB * T / Q


# ----------------------------------------------------------------------------
# Noise-robust resampling & differentiation helpers
# ----------------------------------------------------------------------------
def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def _is_uniform(x, rtol=1e-3):
    """Check whether x is (to within rtol) evenly spaced."""
    d = np.diff(x)
    return np.std(d) <= rtol * np.mean(np.abs(d))


def _uniform_resample(x, y, n=None):
    """
    Put (x, y) onto a uniform grid for Savitzky-Golay filtering.

    IMPORTANT for noisy data: if x is already (near-)uniformly spaced, we
    do NOT interpolate at all -- we use the data as-is. Interpolating noisy
    data (especially with cubic splines, and especially onto a FINER grid
    than the original) creates spurious ringing/overshoot between points
    that can be larger than the genuine noise, which then contaminates the
    derivative. We only resample (with plain LINEAR interpolation, at the
    ORIGINAL point density -- never oversampled) if the input grid is
    genuinely non-uniform, since Savitzky-Golay itself requires uniform
    spacing.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    mask = np.concatenate(([True], np.diff(x) > 0))  # drop duplicate x
    x, y = x[mask], y[mask]

    if _is_uniform(x) and (n is None or n == len(x)):
        return x, y

    n = n or len(x)
    xu = np.linspace(x[0], x[-1], n)
    f = interp1d(x, y, kind="linear")
    yu = f(xu)
    return xu, yu


def smooth(y, window_length=11, polyorder=3):
    """Savitzky-Golay smoothing with automatic window-size clamping."""
    y = np.asarray(y, dtype=float)
    wl = _odd(min(window_length, len(y) - 1)) if len(y) > 3 else len(y)
    wl = max(wl, 3)
    po = min(polyorder, wl - 1)
    return savgol_filter(y, wl, po)


def smooth_derivative(x, y, order=1, window_length=11, polyorder=3,
                       n_resample=None):
    """
    Noise-robust derivative of y wrt x, of the given `order`, computed by:
      1) resampling (x, y) onto a uniform grid (cubic interpolation),
      2) applying a Savitzky-Golay filter configured to directly output the
         requested derivative order on that uniform grid,
      3) interpolating the result back onto the original x locations.

    This is much less noise-sensitive than np.gradient / repeated np.diff,
    particularly for order=2 (needed e.g. for dGm/dVGS-type extractions).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xu, yu = _uniform_resample(x, y, n=n_resample)  # n_resample=None -> native resolution
    dx = xu[1] - xu[0]

    wl = _odd(min(window_length, len(xu) - 1))
    po = min(polyorder, wl - 1)
    po = max(po, order)  # savgol requires polyorder >= deriv order

    dyu = savgol_filter(yu, wl, po, deriv=order, delta=dx)
    f = interp1d(xu, dyu, kind="linear", fill_value="extrapolate")
    return f(x)


# ----------------------------------------------------------------------------
# Derived transistor quantities
# ----------------------------------------------------------------------------
def compute_gm(VGS, ID, window_length=11, polyorder=3, n_resample=None):
    """Transconductance g_m = dI_D/dV_GS (noise-robust)."""
    return smooth_derivative(VGS, ID, order=1, window_length=window_length,
                              polyorder=polyorder, n_resample=n_resample)


def compute_d2ID_dVGS2(VGS, ID, window_length=11, polyorder=4, n_resample=None):
    """
    Second derivative d^2I_D/dV_GS^2 (noise-robust), useful for alternative
    extraction schemes such as V_T,G'm or V_T,(Gm/ID)' in Table I of the
    manuscript. Computed directly via savgol(deriv=2) rather than by
    differentiating g_m a second time, to avoid compounding smoothing error.
    """
    return smooth_derivative(VGS, ID, order=2, window_length=window_length,
                              polyorder=polyorder, n_resample=n_resample)


def compute_SS(VGS, ID, gm=None, window_length=11, polyorder=3,
               n_resample=None, eps=1e-30):
    """
    Subthreshold swing SS = dV_GS / d(log10 I_D) = ln(10) * I_D / g_m   [V/dec]
    """
    ID = np.asarray(ID, dtype=float)
    if gm is None:
        gm = compute_gm(VGS, ID, window_length, polyorder, n_resample)
    gm_safe = np.where(np.abs(gm) < eps, eps, gm)
    return np.log(10.0) * ID / gm_safe


def compute_Y_function(ID, gm, eps=1e-30):
    """
    Y-function: Y = I_D / sqrt(g_m).

    To first order, Y = sqrt(beta*VDS)*(VGS - VTON - eta_SS/2*VDS) is
    INDEPENDENT of series resistance R_S/R_D, so it lets you extract V_T,on
    even when R_S/R_D are unknown.
    """
    ID = np.asarray(ID, dtype=float)
    gm = np.asarray(gm, dtype=float)
    gm_pos = np.clip(gm, eps, None)
    return ID / np.sqrt(gm_pos)


def correct_ID_series_R(ID, VDS, Rtot):
    """
    Remove the effect of series resistance R_tot = R_S + R_D from measured
    I_D (valid in the above-threshold/linear regime):

        I_D' = I_D / (1 - I_D * R_tot / V_DS)

    Points where the correction would blow up or invert (I_D*R_tot >= V_DS)
    are masked to NaN.
    """
    ID = np.asarray(ID, dtype=float)
    denom = 1.0 - ID * Rtot / VDS
    safe_denom = np.where(denom > 1e-6, denom, np.nan)
    return ID / safe_denom


# ----------------------------------------------------------------------------
# Local linear-regression extrapolation (robust to point-to-point noise)
# ----------------------------------------------------------------------------
def _local_linear_fit(x, y, idx, npts=5):
    """Least-squares line through `npts` points centered on idx."""
    n = len(x)
    half = npts // 2
    lo = max(0, idx - half)
    hi = min(n, lo + npts)
    lo = max(0, hi - npts)
    xs, ys = np.asarray(x[lo:hi]), np.asarray(y[lo:hi])
    finite = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[finite], ys[finite]
    if len(xs) < 2:
        return np.nan, np.nan
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, intercept = np.linalg.lstsq(A, ys, rcond=None)[0]
    return slope, intercept


def extrapolate_to_value(x, y, idx, target_y, npts=5):
    """
    Fit a local line to (x, y) around index `idx`, and return the x at which
    that line equals `target_y` (target_y=0 for V_T,on; target_y=log10(ID0)
    for V_T,off).
    """
    slope, intercept = _local_linear_fit(np.asarray(x), np.asarray(y), idx, npts)
    if not np.isfinite(slope) or slope == 0:
        return np.nan
    return (target_y - intercept) / slope


def rolling_ols_slope(x, y, window):
    """
    OLS slope of y vs x in a sliding window of `window` points centered at
    every index. The window genuinely SHRINKS near the array edges (rather
    than re-anchoring to a full-size window shifted inward) so that each
    reported slope reflects the actual local neighborhood of that index --
    important here because the argmax of this array is used to LOCATE a
    point, not just to estimate a single value, so a location bias at the
    edges would corrupt the result.

    This is used instead of a pointwise numerical derivative (e.g.
    Savitzky-Golay deriv=1) whenever `y` is ITSELF already a derived,
    noise-amplified quantity -- most notably the Y-function, Y = I_D/sqrt(g_m),
    which already involved one differentiation (g_m = dI_D/dV_GS). Computing
    a further pointwise derivative of Y (i.e. dY/dV_GS via a filter) compounds
    that noise amplification, effectively behaving like a second derivative
    of I_D. A windowed regression instead only ever performs ONE level of
    differencing -- the same principle already used for the final V_T
    extrapolations (`_local_linear_fit`) -- just swept across every
    candidate index to find where the local slope is largest.

    Use a wider `window` than you would for a plain first derivative to
    compensate for Y's already-elevated noise floor.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    window = max(3, min(int(window), n))
    half = window // 2
    slopes = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        xs, ys = x[lo:hi], y[lo:hi]
        finite = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[finite], ys[finite]
        if len(xs) < 2:
            continue
        xm = xs.mean()
        dx = xs - xm
        denom = np.dot(dx, dx)
        if denom <= 0:
            continue
        slopes[i] = np.dot(dx, ys - ys.mean()) / denom
    return slopes


# ----------------------------------------------------------------------------
# beta, eta_SS, ID0
# ----------------------------------------------------------------------------
def extract_beta(VDS, deriv, correct_series_R=False):
    """
    beta = mu*Ceff*(W/L), from g_m = beta*V_DS (above-threshold, R-free)
    => beta = g_m,max / V_DS.

    NOTE: if series resistance is significant, gm_max should come from
    R_S/R_D-corrected current (or you should rely on the Y-function path),
    otherwise beta will be systematically underestimated.
    """
    if not correct_series_R:
        return deriv / VDS
    else:
        return deriv**2 / VDS


def extract_eta_SS(SS_min, T=300.0):
    """Subthreshold non-ideality factor eta_SS from SS_min [V/dec]."""
    phiT = thermal_voltage(T)
    return SS_min / (np.log(10.0) * phiT)


def compute_ID0(beta, eta_SS, VDS, T=300.0):
    """
    Off-current pre-factor (MIT Virtual Source model), Eq. (8):
        I_D0 = beta * eta_SS * phi_T^2 * (1 - exp(-V_DS/phi_T))
    """
    phiT = thermal_voltage(T)
    return beta * eta_SS * phiT ** 2 * (1.0 - np.exp(-VDS / phiT))


# ----------------------------------------------------------------------------
# Series resistance (R_S+R_D) estimation from data (optional diagnostic)
# ----------------------------------------------------------------------------
def estimate_series_R(VGS, ID, VDS, beta, gm=None, idx=None,
                       window_length=11, polyorder=3, n_resample=None):
    """
    Estimate R_S+R_D at a chosen above-threshold operating point (default:
    the point of max g_m) using the Y-function relation (R-independent by
    construction, so no prior knowledge of R_S/R_D is needed):

        R_S+R_D ~= V_DS/I_D,on - V_DS/(Y*beta)
    """
    ID = np.asarray(ID, dtype=float)
    if gm is None:
        gm = compute_gm(VGS, ID, window_length, polyorder, n_resample)
    Y = compute_Y_function(ID, gm)
    if idx is None:
        idx = int(np.nanargmax(gm))
    ID_on, Y_on = ID[idx], Y[idx]
    return VDS / ID_on - VDS / (Y_on * beta)


# ----------------------------------------------------------------------------
# Region auto-detection
# ----------------------------------------------------------------------------
def _default_masks(ID, on_frac=0.5, off_frac=0.05):
    """
    Simple auto-detection of subthreshold vs. above-threshold index ranges,
    based on the fraction of max(I_D). Works for a typical monotonic
    transfer curve; pass explicit masks for anything more specific (e.g. a
    strict multi-decade exponential-only subthreshold window).
    """
    ID = np.asarray(ID, dtype=float)
    ID_max = np.nanmax(ID)
    above_mask = ID >= on_frac * ID_max
    sub_mask = (ID > 0) & (ID <= off_frac * ID_max)
    return sub_mask, above_mask


# ----------------------------------------------------------------------------
# Main extraction routine
# ----------------------------------------------------------------------------
def _exclude_edges(mask, n, margin):
    """Zero out `margin` points at each end of a boolean mask (avoids
    Savitzky-Golay boundary artifacts being mistaken for a true extremum)."""
    mask = np.array(mask, dtype=bool, copy=True)
    margin = max(0, min(margin, n // 2 - 1))
    if margin > 0:
        mask[:margin] = False
        mask[-margin:] = False
    return mask


def _largest_true_run(mask):
    """Keep only the longest contiguous run of True in a boolean mask,
    clearing everything else.

    An `ID >= ID_limit` mask is meant to isolate the device's "on" region,
    but noise can make an isolated point or two deep in the off-state
    (measurement noise floor) transiently exceed ID_limit as well, leaving a
    small disconnected island alongside the real turn-on run. Feeding that
    straight to a Savitzky-Golay filter (which assumes an evenly-spaced,
    contiguous grid) makes it bridge across the gap between the island and
    the real run, corrupting the derivative near both. A real Id-Vg sweep
    only has one genuine "on" region, so keeping just the longest run
    removes such islands without needing a separate noise-detection step.
    """
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask
    padded = np.concatenate(([0], mask.view(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    starts, ends = edges[0::2], edges[1::2]
    best = np.argmax(ends - starts)
    out = np.zeros_like(mask)
    out[starts[best]:ends[best]] = True
    return out


def extract_VTR(VGS, ID, VDS, VOV_limit=None, ID_limit=None, T=300.0,
                 correct_series_R=False, RS=None, RD=None,
                 sub_mask=None, above_mask=None,
                 on_frac=0.5, off_frac=0.05,
                 window_length=11, polyorder=3, n_resample=None,
                 npts_fit=5, npts_fit_Y=None, edge_margin=None, y_slope_window=None,
                 return_details=False):
    """
    Extract V_T,on, V_T,off and V_TR = V_T,on - V_T,off from a linear-mode
    (low V_DS, ideally V_DS ~ a few phi_T) I_D-V_GS transfer curve.

    Parameters
    ----------
    VGS, ID : array_like
        Gate-voltage sweep and corresponding measured drain current
        (monotonically increasing VGS; ID > 0). Any consistent current
        unit works (A, A/um, uA, ...) since beta/ID0 carry the same unit.
    VDS : float
        Drain bias used for the measurement (V); should be small.
    VOV_limit : float
        Maximum allowed V_OV = V_GS - V_T,off for the above-threshold region.
    T : float
        Temperature in Kelvin (for phi_T).
    correct_series_R : bool
        False (default): use the RAW measured I_D for both V_T,on and
            V_T,off (standard method -- V_T,on/V_TR may be biased if
            R_S+R_D is not negligible).
        True: correct the V_T,on extraction for series resistance:
            * If RS and RD are BOTH given (not None) -> apply the direct
              correction I_D' = I_D / (1 - I_D*(RS+RD)/VDS) and extrapolate
              I_D' instead of I_D ("Modified I'_DON" method).
            * If RS and/or RD are NOT given -> automatically fall back to
              the RS/RD-INDEPENDENT Y-function method
              (Y = I_D/sqrt(g_m)) which needs no resistance value at all.
        V_T,off is left unaffected in all cases (the subthreshold current
        is too small for R_S/R_D to matter).
    sub_mask, above_mask : boolean array_like, optional
        Explicit index masks for the subthreshold / above-threshold
        regions. Auto-detected from `on_frac`/`off_frac` if not given.
    on_frac, off_frac : float
        Fractions of max(ID) used for auto-detecting the above-threshold
        (>= on_frac*max) and subthreshold (<= off_frac*max) regions.
    window_length, polyorder : int
        Savitzky-Golay smoothing parameters for all derivatives (g_m, SS).
        Increase window_length for noisier data.
    n_resample : int, optional
        Only relevant if VGS is NOT uniformly spaced. In that case this many
        points are used for the linear-interpolation resample onto a uniform
        grid (default: native resolution, i.e. len(VGS); avoid oversampling,
        which amplifies noise). If VGS is already uniform (typical for a
        voltage sweep), no resampling/interpolation is performed at all.
    npts_fit : int
        Number of points used in each local linear-regression extrapolation.
        Increase for noisier data (trades noise rejection vs. extrapolation
        window nonlinearity).
    npts_fit_Y : int, optional
        Number of points used in the local linear-regression fit for the Y-function.
        If None, defaults to int(0.5 / VGS_step) (i.e. ~0.5 V window, in points).
    edge_margin : int, optional
        Number of points excluded from each end of the sweep when searching
        for the SS-minimum / g_m-maximum locations, to avoid mistaking a
        Savitzky-Golay boundary artifact for the true extremum. Default:
        max(window_length, npts_fit).
    y_slope_window : int, optional
        Window size (in points) for the rolling-OLS-slope scan used to find
        the most-linear region of the Y-function (only used when
        correct_series_R=True and RS/RD are not given). Y = I_D/sqrt(g_m) is
        already a derived, noise-amplified quantity (g_m being a first
        derivative), so locating its point of maximum slope needs a WIDER
        window than a plain first derivative would -- filtering Y a second
        time (e.g. via Savitzky-Golay deriv=1) compounds noise the way a
        second derivative of I_D would. Default: 3x window_length (odd-ized).
    return_details : bool
        If True, also returns a dict of intermediate quantities (gm, SS, Y,
        beta, eta_SS, ID0, region masks, indices, estimated R_tot, method
        used, etc.) useful for diagnostics / plotting.

    Returns
    -------
    VTON, VTOFF, VTR : float
    details : dict   (only if return_details=True)
    """
    VGS = np.asarray(VGS, dtype=float)
    ID = np.asarray(ID, dtype=float)
    ID = np.abs(ID)
    n = len(VGS)
    margin = edge_margin if edge_margin is not None else max(window_length, npts_fit)

    # --- region masks -----------------------------------------------------
    auto_sub, auto_above = _default_masks(ID, on_frac=on_frac, off_frac=off_frac)
    sub_mask = auto_sub if sub_mask is None else np.asarray(sub_mask, dtype=bool)
    above_mask = auto_above if above_mask is None else np.asarray(above_mask, dtype=bool)

    # exclude sweep-edge points from extremum search only (edge points can
    # still be used inside the local linear-regression fit windows)
    sub_mask_search = _exclude_edges(sub_mask, n, margin)
    above_mask_search = _exclude_edges(above_mask, n, margin)
    if not np.any(sub_mask_search):
        sub_mask_search = sub_mask
    if not np.any(above_mask_search):
        above_mask_search = above_mask

    # --- smoothed derivatives on RAW data (needed for SS regardless of
    #     correction choice, and for the Y-function / R_tot diagnostic) ----
    gm_raw = compute_gm(VGS, ID, window_length, polyorder, n_resample)
    SS = compute_SS(VGS, ID, gm=np.gradient(ID, VGS))
    Y = compute_Y_function(ID, gm_raw)

    # =======================================================================
    # V_T,off (subthreshold; always from raw ID, unaffected by RS/RD)
    # =======================================================================
    # Guard against points where the noisy g_m is spuriously <= 0 (typically
    # very-low-current, near-noise-floor artifacts): SS = ln(10)*ID/gm is
    # only physically meaningful for gm > 0, so such points must never be
    # selected as the "SS minimum" -- they are excluded from the search
    # (but NOT from the eventual local-linear-fit window, which uses
    # whatever real neighboring points fall in range).
    valid_gm = gm_raw > 0
    if not ID_limit is None:
        valid_gm &= (ID >= ID_limit)
    SS_sub = np.where(sub_mask_search & valid_gm, SS, np.inf)
    if not np.any(np.isfinite(SS_sub)):
        SS_sub = np.where(sub_mask & valid_gm, SS, np.inf)  # relax edge margin if needed
    idx_off = int(np.nanargmin(SS_sub))
    SS_min = SS[idx_off]

    gm_above_raw = np.where(above_mask, gm_raw, -np.inf)
    gm_above_raw_search = np.where(above_mask_search, gm_raw, -np.inf)

    yw = y_slope_window
    if yw is None:
        above_idx = np.where(above_mask_search)[0]
        span = (above_idx.max() - above_idx.min() + 1) if len(above_idx) > 0 else window_length
        # Wide enough to average over a good fraction of the
        # above-threshold region -- Y's noise floor is elevated (it
        # already involved one differentiation via g_m), so a narrow
        # window here re-introduces the very noise sensitivity we're
        # trying to avoid.
        yw = _odd(max(3 * window_length, span // 3, npts_fit))
    Y_slope_scan = rolling_ols_slope(VGS, Y, window=yw)
    y_margin = max(margin, yw // 2)
    above_mask_for_Y = _exclude_edges(above_mask, n, y_margin)
    if not np.any(above_mask_for_Y):
        above_mask_for_Y = above_mask_search
    Y_slope_above_search = np.where(above_mask_for_Y, Y_slope_scan, -np.inf)

    if not correct_series_R:
        beta_est = extract_beta(VDS, np.nanmax(gm_above_raw_search), correct_series_R)
    else:
        beta_est = extract_beta(VDS, np.nanmax(Y_slope_above_search), correct_series_R)
    
    eta_SS = extract_eta_SS(SS_min, T)
    ID0 = compute_ID0(beta_est, eta_SS, VDS, T)

    logID = np.log10(np.clip(ID, 1e-300, None))
    off_slope, off_intercept = _local_linear_fit(VGS, logID, idx_off, npts=npts_fit)
    VTOFF = extrapolate_to_value(VGS, logID, idx_off, np.log10(ID0), npts=npts_fit)
    VGS_limit = VTOFF + VOV_limit if VOV_limit is not None else np.inf
    above_mask_search = above_mask_search & (VGS <= VGS_limit)

    # =======================================================================
    # V_T,on (above-threshold; method depends on correct_series_R)
    # =======================================================================
    Rtot_used, method = None, "raw_ID"
    on_array, on_ylabel = ID, "I_D"

    if not correct_series_R:
        gm_above_raw_search = np.where(above_mask_search & np.isfinite(ID), gm_raw, -np.inf)
        idx_on = int(np.nanargmax(gm_above_raw_search))
        VTON = extrapolate_to_value(VGS, ID, idx_on, 0.0, npts=npts_fit)
        on_array, on_ylabel = ID, "I_D (raw)"

    elif RS is not None and RD is not None:
        Rtot_used = RS + RD
        method = "ID_prime_correction"
        # Lightly smooth ID *before* applying the correction: I_D' is a
        # nonlinear function of I_D (division by a near-unity, ID-dependent
        # factor), so applying it point-by-point to noisy raw data and then
        # differentiating amplifies noise more than smoothing first does.
        ID_smoothed = smooth(ID, window_length=window_length, polyorder=polyorder)
        IDp = correct_ID_series_R(ID_smoothed, VDS, Rtot_used)
        gm_p = compute_gm(VGS, IDp, window_length, polyorder, n_resample)
        gm_p_above_search = np.where(above_mask_search & np.isfinite(IDp), gm_p, -np.inf)
        idx_on = int(np.nanargmax(gm_p_above_search))
        VTON = extrapolate_to_value(VGS, IDp, idx_on, 0.0, npts=npts_fit)
        # recompute beta from the corrected data (more accurate once R is removed)
        beta_est = extract_beta(VDS, np.nanmax(gm_p_above_search))
        on_array, on_ylabel = IDp, "I_D' (R-corrected)"

    else:
        method = "Y_function"
        # The correct point for the Y-based extrapolation is where Y ITSELF
        # is most linear in VGS, i.e. where its local slope peaks -- NOT
        # where dI_D/dVGS (g_m) peaks. Y already divides out most of the g_m
        # curvature, so its own point of maximum slope is generally at a
        # different VGS than g_m's peak, and using g_m's peak here biases
        # the extrapolation point away from where Y is actually straight.
        #
        # IMPORTANT (noise): Y = I_D/sqrt(g_m) is already a derived quantity
        # that involved one differentiation (g_m), so it carries amplified
        # noise relative to raw I_D. Computing a pointwise numerical
        # derivative of Y (e.g. via a Savitzky-Golay filter) would compound
        # that noise the way a second derivative of I_D does. Instead, we
        # scan a WIDE sliding-window OLS slope across Y directly -- this
        # performs only ONE level of differencing (like the final V_T
        # extrapolation fits), just repeated at every candidate index, and
        # the wider window suppresses Y's already-elevated noise floor.
        # yw = y_slope_window
        # if yw is None:
        #     above_idx = np.where(above_mask_search)[0]
        #     span = (above_idx.max() - above_idx.min() + 1) if len(above_idx) > 0 else window_length
        #     # Wide enough to average over a good fraction of the
        #     # above-threshold region -- Y's noise floor is elevated (it
        #     # already involved one differentiation via g_m), so a narrow
        #     # window here re-introduces the very noise sensitivity we're
        #     # trying to avoid.
        #     yw = _odd(max(3 * window_length, span // 3, npts_fit))
        # Y_slope_scan = rolling_ols_slope(VGS, Y, window=yw)
        # Near the array edges, rolling_ols_slope's window necessarily
        # shrinks (fewer supporting points -> higher variance), so those
        # points must not be allowed to win the argmax just by chance; only
        # consider candidates with a fully-supported window.
        y_margin = max(margin, yw // 2)
        above_mask_for_Y = _exclude_edges(above_mask, n, y_margin) & (VGS <= VGS_limit)
        if not np.any(above_mask_for_Y):
            above_mask_for_Y = above_mask_search
        Y_slope_above_search = np.where(above_mask_for_Y, Y_slope_scan, -np.inf)
        idx_on = int(np.nanargmax(Y_slope_above_search))
        if npts_fit_Y is None:
            npts_fit_Y = max(npts_fit, int(0.5 / np.nanmean(np.diff(VGS))))
        npts_fit = npts_fit_Y
        VTON = extrapolate_to_value(VGS, Y, idx_on, 0.0, npts=npts_fit)
        # Refine beta from the Y-line's own local slope at that point:
        # Y ~ sqrt(beta*VDS)*(VGS - VTON - eta_SS*VDS/2)  =>  beta = slope^2/VDS.
        y_slope, _ = _local_linear_fit(VGS, Y, idx_on, npts=npts_fit)
        if np.isfinite(y_slope) and y_slope > 0:
            beta_est = y_slope ** 2 / VDS
        # opportunistic diagnostic: estimate Rtot even though it wasn't needed
        Rtot_used = estimate_series_R(VGS, ID, VDS, beta_est, gm=gm_raw, idx=idx_on)
        on_array, on_ylabel = Y, "Y = I_D/sqrt(g_m)"
        if idx_on >= n - 1 - y_margin - 1:
            import warnings
            warnings.warn(
                "Y-function: the point of maximum slope sits at the edge of "
                "the measured VGS range. This likely means the true "
                "most-linear region extends beyond your sweep (mobility "
                "degradation onset not yet reached) -- treat V_T,on/V_TR "
                "from this method with extra caution for this dataset.",
                stacklevel=2)

    on_slope, on_intercept = _local_linear_fit(VGS, on_array, idx_on, npts=npts_fit)
    VTON -= eta_SS * VDS / 2.0
    VTR = VTON - VTOFF

    if not return_details:
        return VTON, VTOFF, VTR

    details = dict(
        method=method, idx_on=idx_on, idx_off=idx_off,
        gm=gm_raw, SS=SS, Y=Y, SS_min=SS_min, beta=beta_est,
        eta_SS=eta_SS, ID0=ID0, Rtot_used=Rtot_used,
        sub_mask=sub_mask, above_mask=above_mask,
        off_fit=(off_slope, off_intercept), on_fit=(on_slope, on_intercept),
        on_array=on_array, on_ylabel=on_ylabel,
    )
    return VTON, VTOFF, VTR, details


# ----------------------------------------------------------------------------
# Visualization (optional; requires matplotlib)
# ----------------------------------------------------------------------------
def plot_extraction(VGS, ID, VTON, VTOFF, details, title="", figsize=(11, 4.2)):
    """
    Visualize a single V_TR extraction result returned by
    `extract_VTR(..., return_details=True)`:

      * Left panel:  log10(I_D) vs V_GS, with the subthreshold linear fit
                      (the one actually used internally) extrapolated to
                      V_T,off.
      * Right panel: whatever quantity the chosen method used for the
                      above-threshold fit -- raw I_D, the R-corrected I_D',
                      or the Y-function -- vs V_GS, extrapolated to V_T,on.

    Both fit lines are the EXACT local linear regressions used internally
    (captured in details['off_fit'] / details['on_fit']), not an
    approximate reconstruction, so the plot faithfully shows what the
    algorithm did.
    """
    import matplotlib.pyplot as plt

    VGS = np.asarray(VGS, dtype=float)
    ID = np.asarray(ID, dtype=float)
    idx_off, idx_on = details["idx_off"], details["idx_on"]
    off_slope, off_intercept = details["off_fit"]
    on_slope, on_intercept = details["on_fit"]
    on_array, on_ylabel = details["on_array"], details["on_ylabel"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # --- Left: log10(I_D) vs V_GS -> V_T,off -------------------------------
    logID = np.log10(np.clip(ID, 1e-14, None))
    ax1.plot(VGS, logID, ".", ms=3, color="tab:blue", alpha=0.5, label="data")
    fx = np.linspace(min(VGS.min(), VTOFF) - 0.05, max(VGS[idx_off] + 0.3, VTOFF + 0.1), 50)
    ax1.plot(fx, off_slope * fx + off_intercept, "--", color="tab:red", lw=2,
              label="subthreshold fit")
    ax1.axvline(VTOFF, color="tab:red", ls=":", lw=1)
    ax1.axhline(np.log10(details["ID0"]), color="gray", ls=":", lw=1, label="$I_{D0}$")
    ax1.plot(VGS[idx_off], logID[idx_off], "o", color="tab:red", ms=7,
              label="fit point ($SS_{min}$)")
    ax1.set_xlabel("$V_{GS}$ (V)")
    ax1.set_ylabel("$\\log_{10}(I_D)$")
    ax1.set_title(f"Subthreshold fit $\\rightarrow$ $V_{{T,off}}$ = {VTOFF:.3f} V")
    ax1.legend(fontsize=8, loc="lower right")

    # --- Right: on_array vs V_GS -> V_T,on ---------------------------------
    ax2.plot(VGS, on_array, ".", ms=3, color="tab:blue", alpha=0.5, label="data")
    fx2 = np.linspace(min(VGS[idx_on] - 0.3, VTON - 0.1), max(VGS.max(), VTON + 0.1), 50)
    ax2.plot(fx2, on_slope * fx2 + on_intercept, "--", color="tab:green", lw=2,
              label="above-threshold fit")
    ax2.axvline(VTON, color="tab:green", ls=":", lw=1)
    ax2.axhline(0, color="gray", ls=":", lw=1)
    ax2.plot(VGS[idx_on], on_array[idx_on], "o", color="tab:green", ms=7,
              label="fit point (max slope)")
    ax2.set_ylim(bottom=min(on_array.min(), 0.0) * 1.1, top=on_array[-1] * 1.1)
    ax2.set_xlabel("$V_{GS}$ (V)")
    ax2.set_ylabel(on_ylabel)
    ax2.set_title(f"Above-threshold fit $\\rightarrow$ $V_{{T,on}}$ = {VTON:.3f} V")
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Loading measured data from a Keithley/ACS-style .xls export
# ----------------------------------------------------------------------------
def load_transfer_curve_xls(path, sheet_data="Data", sheet_settings="Settings",
                             vgs_col="GateV", id_col="DrainI", forward_only=True):
    """
    Load a linear-mode I_D-V_GS transfer curve from a Keithley/ACS
    (Clarius-style) .xls export, which typically has 'Data', 'Calc', and
    'Settings' sheets.

    If Dual Sweep Mode was enabled (forward + reverse V_GS sweep in the
    same file), V_TR extraction requires a single monotonic sweep, so by
    default (`forward_only=True`) only the FORWARD leg -- V_GS increasing
    up to (and including) its first peak -- is returned; the reverse leg
    is discarded.

    Parameters
    ----------
    path : str
        Path to the .xls file.
    sheet_data, sheet_settings : str
        Sheet names for the raw I-V data and the instrument settings
        (used only to try to recover V_DS automatically).
    vgs_col, id_col : str
        Column names for the gate voltage and drain current in `sheet_data`.
    forward_only : bool
        If True (default), keep only the forward (increasing V_GS) sweep.
        If False, keep only the backward (decreasing V_GS) sweep.

    Returns
    -------
    VGS, ID : ndarray
        Forward-sweep or backward-sweep gate voltage and drain current.
    VDS : float or None
        Drain bias, parsed from the Settings sheet's "Start/Level" row
        (Drain column) if present; None if it could not be determined
        (in which case you should supply it manually).
    """
    import pandas as pd

    df = pd.read_excel(path, sheet_name=sheet_data)
    VGS_full = df[vgs_col].to_numpy(dtype=float)
    ID_full = df[id_col].to_numpy(dtype=float)

    if forward_only:
        d = np.diff(VGS_full)
        turn = np.where(d <= 0)[0]  # first index where VGS stops increasing
        end = int(turn[0]) + 1 if len(turn) > 0 else len(VGS_full)
        VGS, ID = VGS_full[:end], ID_full[:end]
    else:
        d = np.diff(VGS_full)
        turn = np.where(d < 0)[0]  # first index where VGS starts decreasing
        start = int(turn[0]) if len(turn) > 0 else len(VGS_full) - 1
        VGS, ID = VGS_full[start:], ID_full[start:]

    # Best-effort V_DS recovery from the Settings sheet (Keithley ACS format:
    # a "Start/Level" row with Drain/Source/Gate columns; Drain is V_DS here
    # since it's held at a fixed "Voltage Bias" while Gate is swept).
    VDS = None
    try:
        settings = pd.read_excel(path, sheet_name=sheet_settings, header=None)
        row = settings[settings[0] == "Start/Level"]
        if not row.empty:
            VDS = float(row.iloc[0, 1])
    except Exception:
        pass

    sort_idx = np.argsort(VGS)
    VGS, ID = VGS[sort_idx], ID[sort_idx]
    return VGS, ID, VDS

def try_convert_to_float(X): 
    try: 
        return float(X) 
    except (ValueError, TypeError): 
        return float('nan')

def agilent_csv_cleaner(fname):
    full_data_set = []
    with open(fname, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        line_count = 0
        for row in csv_reader:
            # print(line_count, row)
            if row[0] in ['Dimension1', 'Dimension2', 'DataName', 'DataValue']:
                full_data_set.append(row)
            line_count += 1
    print_file(f'Processed {line_count} lines')

    line_count = 0
    data_set = []
    N1 = 0
    N2 = 0
    while line_count < len(full_data_set):
        row = full_data_set[line_count]
        if row[0] == 'Dimension1':
            N1 = int(row[1])
            print_file('Dimension1:', N1)
            line_count += 1
            row = full_data_set[line_count]
            N2 = int(row[1])
            print_file('Dimension2:', N2)
            line_count += 1
            temp_data = [[[str(X) for X in full_data_set[line_count][1:]]] for ii in range(N2)]
            line_count += 1
            for ii in range(N2):
                for elem in range(N1):
                    # print(full_data_set[line_count])
                    try:
                        temp_data[ii].append([try_convert_to_float(X) for X in full_data_set[line_count][1:]])
                        line_count += 1
                    except IndexError:
                        temp_data = None
                        break
            if not temp_data == None:
                data_set += temp_data
    print_file(f'Number of datasets: {len(data_set)}')

    pd_data_set = [pd.DataFrame(X[1:], columns=X[0]) for X in data_set]

    return pd_data_set, N1

def get_sweep(VGS_full, ID_full, direction='forward'):
    """
    Get the forward or backward sweep from the VGS and ID arrays.

    Parameters
    ----------
    VGS_full : array_like
        Gate voltage array.
    ID_full : array_like
        Drain current array.
    direction : str
        'forward' for increasing VGS, 'backward' for decreasing VGS.

    Returns
    -------
    VGS : ndarray
        Gate voltage sweep.
    ID : ndarray
        Drain current sweep.
    """
    VGS_full = np.asarray(VGS_full, dtype=float)
    ID_full = np.asarray(ID_full, dtype=float)

    if direction == 'forward':
        dVGS_full = np.diff(VGS_full)
        turn_idx = np.where(dVGS_full <= 0)[0]
        end_idx = int(turn_idx[0]) + 1 if len(turn_idx) > 0 else len(VGS_full)
        VGS, ID = VGS_full[:end_idx], ID_full[:end_idx]
    elif direction == 'backward':
        dVGS_full = np.diff(VGS_full)
        turn_idx = np.where(dVGS_full < 0)[0]
        start_idx = int(turn_idx[0]) if len(turn_idx) > 0 else len(VGS_full) - 1
        VGS, ID = VGS_full[start_idx:], ID_full[start_idx:]
    else:
        raise ValueError("direction must be 'forward' or 'backward'")

    sort_idx = np.argsort(VGS)
    VGS, ID = VGS[sort_idx], ID[sort_idx]
    return VGS, ID


# ----------------------------------------------------------------------------
# gm/Id, gm/Cgg diagnostics and a derivative-based VTR (measured data)
# ----------------------------------------------------------------------------
# Plot styling constants (dataviz-skill validated: blue/orange categorical
# pair, "#e1e0d9"-family chrome). Shared across all functions in this section
# so callers get a consistent look without re-declaring these.
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_MUTED = "#898781"
TEXT_PRIMARY = "#0b0b0b"
# Sequential single-hue (blue) ramp, light->dark, for multi-frequency CV
# overlays. Steps 250/400/550/700 of the palette's blue ramp (validated:
# --ordinal, light mode -- a 6-step version failed the adjacent-lightness-gap
# check, so cap multi-frequency overlays at 4 traces).
SEQ_BLUE_4 = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
# gm/Id = d ln(Id)/dVg (never gradient(Id)/Id -- the latter amplifies noise at
# low Vg, per TR_related_equations.py's derivative_extrema gotcha). gm/Cgg is
# the fT convention used throughout scratch.MD Sec. 5 (gm/Cgg, no 2*pi).
COLOR_GM_ID = "#2a78d6"   # categorical slot 1 (blue)   -- left axis
COLOR_GM_CGG = "#eb6834"  # categorical slot 2 (orange) -- right axis


def style_axes(ax):
    """Shared light-mode chart chrome: hairline recessive grid, no top/right
    spines, muted ticks/labels."""
    ax.grid(True, color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_MUTED)
    ax.xaxis.label.set_color(TEXT_PRIMARY)
    ax.yaxis.label.set_color(TEXT_PRIMARY)
    ax.title.set_color(TEXT_PRIMARY)


def freq_label(freq_hz):
    """Format a frequency in Hz as e.g. '1 MHz' / '100 kHz' for legends."""
    return f"{freq_hz/1e6:.2g} MHz" if freq_hz >= 1e6 else f"{freq_hz/1e3:.3g} kHz"


def compute_gmid_gmcgg(VGS, ID, Vg_cv=None, Cgg_cv=None, ID_limit=2e-12,
                        window_length=5, polyorder=3):
    """gm, gm/Id = d ln(Id)/dVg, and a "rising" signal for the twin-axis
    speed-vs-efficiency comparison: gm/Cgg if a CV sweep (Vg_cv, Cgg_cv) is
    supplied, else plain gm. Cgg data is often unavailable (most Id-Vg-only
    measurement setups don't have a paired CV sweep), so gm alone is the
    default -- it shares gm/Cgg's rise-then-plateau shape, just without the
    capacitance normalization.

    Points below ID_limit (raw Id at the measurement noise floor -- same
    convention as extract_VTR's ID_limit) are dropped from VGS/ID BEFORE any
    smoothing/differentiation is done, not masked afterward: at the noise
    floor, log(Id) swings wildly (Id can even flip sign before abs()), and if
    those points are left in the Savitzky-Golay window, the filter's boundary
    response contaminates the derivative at points just *inside* the valid
    region too -- masking to NaN only after the fact doesn't undo that. This
    mirrors extract_VTR's own ID_limit handling (there it guards which points
    the SS-minimum search can land on; here the equivalent is keeping the
    noise floor out of the derivative computation itself).

    Vg_cv, Cgg_cv, if given, are a separate (Vg, |Cgg|) CV sweep, interpolated
    onto the (already-trimmed) Id-Vg sweep's VGS grid -- the two measurements
    need not share a Vg grid.

    Returns (gm, gm_over_id, rising, rising_kind, valid), where rising_kind
    is "gm_over_cgg" or "gm" depending on whether CV data was supplied, and
    valid is a boolean mask (over the ORIGINAL VGS) marking the kept points;
    gm/gm_over_id/rising are NaN outside it.
    """
    VGS = np.asarray(VGS, dtype=float)
    ID = np.abs(np.asarray(ID, dtype=float))
    # Largest contiguous run only: an isolated noise spike deep in the
    # off-state can transiently clear ID_limit too, leaving a small
    # disconnected island next to the real turn-on run (see
    # _largest_true_run's docstring) -- keep just the genuine "on" region.
    valid = _largest_true_run(ID >= ID_limit)
    VGS_v, ID_v = VGS[valid], ID[valid]

    gm_v = compute_gm(VGS_v, ID_v, window_length=window_length, polyorder=polyorder)
    gm_over_id_v = smooth_derivative(VGS_v, np.log(np.clip(ID_v, 1e-300, None)), order=1,
                                      window_length=window_length, polyorder=polyorder)

    if Vg_cv is not None and Cgg_cv is not None:
        Cgg_interp = np.interp(VGS_v, Vg_cv, Cgg_cv)
        rising_v = gm_v / Cgg_interp
        rising_kind = "gm_over_cgg"
    else:
        rising_v = gm_v.copy()
        rising_kind = "gm"

    gm = np.full(VGS.shape, np.nan)
    gm_over_id = np.full(VGS.shape, np.nan)
    rising = np.full(VGS.shape, np.nan)
    gm[valid] = gm_v
    gm_over_id[valid] = gm_over_id_v
    rising[valid] = rising_v
    return gm, gm_over_id, rising, rising_kind, valid


_RISING_LABELS = {
    "gm_over_cgg": "$g_m/C_{GG}$  (s$^{-1}$)",
    "gm": "$g_m$  (S)",
}
_RISING_DERIV_LABELS = {
    "gm_over_cgg": "$d(g_m/C_{GG})/dV_G$  (s$^{-1}$V$^{-1}$)",
    "gm": "$d(g_m)/dV_G$  (S/V)",
}


def plot_gmid_gmcgg(device, VGS, ID, savepath, title, Vg_cv=None, Cgg_cv=None,
                     ID_limit=2e-12):
    """Twin-axis plot: gm/Id (left, solid) vs Vg, and gm/Cgg or (if no CV
    data is supplied) plain gm (right, dashed) vs Vg. Axis color/ticks/spines
    match their curve so the twin-axis mapping is unambiguous two ways
    (color + line style), not just one."""
    import matplotlib.pyplot as plt

    gm, gm_over_id, rising, rising_kind, valid = compute_gmid_gmcgg(
        VGS, ID, Vg_cv, Cgg_cv, ID_limit)

    fig, ax_l = plt.subplots(figsize=(6.5, 4.5))
    ax_r = ax_l.twinx()

    ax_l.plot(VGS, gm_over_id, color=COLOR_GM_ID, lw=2, zorder=3)
    ax_r.plot(VGS, rising, color=COLOR_GM_CGG, lw=2, linestyle="--",
              dashes=(5, 3), zorder=3)

    ax_l.set_xlabel("$V_G$ (V)")
    ax_l.set_ylabel("$g_m/I_D$  (V$^{-1}$)   [solid]", color=COLOR_GM_ID)
    ax_r.set_ylabel(f"{_RISING_LABELS[rising_kind]}   [dashed]", color=COLOR_GM_CGG)
    ax_l.set_title(title)
    style_axes(ax_l)
    ax_l.tick_params(axis="y", colors=COLOR_GM_ID)
    ax_l.spines["left"].set_color(COLOR_GM_ID)

    ax_r.grid(False)
    for side in ("top", "left"):
        ax_r.spines[side].set_visible(False)
    ax_r.spines["right"].set_color(COLOR_GM_CGG)
    ax_r.tick_params(axis="y", colors=COLOR_GM_CGG)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print_file(f"Saved {device} gm/Id & {rising_kind} plot to {savepath}")


# Measured-data analog of derivative_extrema() in TR_related_equations.py:
# there, VTR_deriv = Vg*(peak of dfT/dVg) - Vg*(trough of d(gm/Id)/dVg) --
# the Vg-separation between where transport efficiency (gm/Id) is degrading
# fastest and where speed (gm/Cgg, or gm alone if Cgg is unavailable) is
# rising fastest.
#
# gm/Id and gm/Cgg (or gm) are THEMSELVES once-differentiated quantities (gm
# already involved one derivative of Id), so a further pointwise
# Savitzky-Golay derivative here would compound noise the same way a second
# derivative of Id would -- this is exactly the situation rolling_ols_slope
# was written for (see its docstring / the Y-function branch of extract_VTR).
# A windowed OLS slope is used instead of smooth_derivative for this second
# level.
#
# Peak (rising side): centroid of Vg weighted by max(+d(rising)/dVg, 0) over
# the WHOLE search domain. The rising signal only rises and plateaus over the
# measured Vg range rather than falling back down, so weighting by its own
# magnitude would just chase wherever the sweep happens to stop (tested: it
# moved by ~1 V for a Vg range change of ~1 V) -- but its DERIVATIVE really
# does decay back to ~0 once the plateau is reached, so a full-domain
# derivative-weighted centroid is safe and matches extract_VTR closely
# (tested against both example devices here).
#
# Trough (gm/Id side): NOT the same full-domain treatment, even though it
# looks symmetric on paper. d(gm/Id)/dVg's negative lobe does NOT decay back
# to ~0 the way the rising side's positive lobe does -- gm/Id itself keeps
# falling roughly like 1/Vov well past the sharp trough, so its derivative
# stays measurably negative over a long, slowly-decaying tail. A full-domain
# centroid there is dominated by that tail and lands far from the true
# trough (tested: it shifted the Si nFET trough from 0.575 V to 0.69 V --
# past the rising centroid, making VTR_deriv go NEGATIVE, and moved the OSFET
# trough from 0.225 V to 0.31 V). Instead, find the sharp argmin first (as
# the original method did), then take the derivative-weighted centroid in a
# small window AROUND it (trough_half_width points each side, default
# window_length) -- this still averages over several points to suppress
# point-to-point noise, but stays local enough that the long tail can't bias
# it (tested: <=3 mV from the plain-argmin location for both example
# devices, vs. 60-115 mV of bias from the full-domain version).
def extract_VTR_derivative(VGS, ID, Vg_cv=None, Cgg_cv=None, ID_limit=2e-12,
                            window_length=5, polyorder=3, slope_window=None,
                            edge_margin=None, trough_half_width=None):
    VGS = np.asarray(VGS, dtype=float)
    n = len(VGS)
    gm, gm_over_id, rising, rising_kind, valid = compute_gmid_gmcgg(
        VGS, ID, Vg_cv, Cgg_cv, ID_limit, window_length, polyorder)

    sw = slope_window if slope_window is not None else _odd(max(3 * window_length, n // 8))

    # Compute the windowed-OLS slope ONLY over the trimmed (noise-floor
    # removed) range, same rationale as compute_gmid_gmcgg's own trimming:
    # rolling_ols_slope sizes its window in units of consecutive ARRAY
    # entries, so running it on the full NaN-padded array would silently let
    # a window span across the off-state gap as if those were adjacent,
    # contiguous samples, rather than consistently using `sw` real
    # neighboring points on each side.
    VGS_v = VGS[valid]
    d_gmid_v = rolling_ols_slope(VGS_v, gm_over_id[valid], window=sw)
    d_rising_v = rolling_ols_slope(VGS_v, rising[valid], window=sw)
    d_gmid = np.full(n, np.nan)
    d_rising = np.full(n, np.nan)
    d_gmid[valid] = d_gmid_v
    d_rising[valid] = d_rising_v

    # Exclude `margin` points from each end of the TRIMMED run -- not the raw
    # sweep's endpoints (those are off-state and already outside `valid`) --
    # since that's where rolling_ols_slope's own window-shrinking edge
    # effects actually live.
    margin = edge_margin if edge_margin is not None else max(window_length, sw // 2)
    search_mask_v = _exclude_edges(np.ones(len(VGS_v), dtype=bool), len(VGS_v), margin)
    search_mask = np.zeros(n, dtype=bool)
    search_mask[valid] = search_mask_v
    if not np.any(search_mask):
        search_mask = valid

    # Trough: sharp argmin locates the transition, then a small local
    # centroid (half-width in points) around it suppresses point noise
    # without picking up the long negative tail.
    half_w = trough_half_width if trough_half_width is not None else window_length
    idx_min = int(np.nanargmin(np.where(search_mask, d_gmid, np.inf)))
    lo, hi = max(0, idx_min - half_w), min(n, idx_min + half_w + 1)
    w_trough = np.clip(-d_gmid[lo:hi], 0.0, None)
    w_trough = np.nan_to_num(w_trough, nan=0.0)
    Vg_trough = np.trapezoid(VGS[lo:hi] * w_trough, VGS[lo:hi]) / np.trapezoid(w_trough, VGS[lo:hi])

    # Centroid of the rising edge: weighted mean of Vg using the positive
    # part of the rising signal's derivative (over search_mask) as the weight.
    w_peak = np.where(search_mask, np.clip(d_rising, 0.0, None), 0.0)
    w_peak = np.nan_to_num(w_peak, nan=0.0)
    Vg_peak = np.trapezoid(VGS * w_peak, VGS) / np.trapezoid(w_peak, VGS)

    VTR_deriv = Vg_peak - Vg_trough

    details = dict(gm=gm, gm_over_id=gm_over_id, rising=rising, rising_kind=rising_kind,
                    d_gmid=d_gmid, d_rising=d_rising, slope_window=sw,
                    idx_trough=idx_min, trough_window=(lo, hi),
                    Vg_trough=Vg_trough, Vg_peak=Vg_peak)
    return VTR_deriv, details


def plot_derivative_extrema(device, VGS, VTR_deriv, details, savepath, title):
    """Diagnostic plot for extract_VTR_derivative: d(gm/Id)/dVg (left, solid,
    with a small window around its argmin shaded to show what the trough
    centroid is averaging over -- NOT its whole negative lobe, which has a
    long tail that would bias the centroid -- see extract_VTR_derivative's
    comment) and d(gm/Cgg)/dVg or d(gm)/dVg (right, dashed, with its full
    positive lobe shaded to show what the peak centroid is averaging over)."""
    import matplotlib.pyplot as plt

    d_gmid, d_rising = details["d_gmid"], details["d_rising"]
    rising_kind = details["rising_kind"]
    lo, hi = details["trough_window"]

    fig, ax_l = plt.subplots(figsize=(6.5, 4.5))
    ax_r = ax_l.twinx()

    ax_l.plot(VGS, d_gmid, color=COLOR_GM_ID, lw=2, zorder=3)
    ax_l.fill_between(VGS[lo:hi], 0, np.clip(d_gmid[lo:hi], None, 0), color=COLOR_GM_ID,
                       alpha=0.12, zorder=2, label="_nolegend_")
    ax_r.plot(VGS, d_rising, color=COLOR_GM_CGG, lw=2, linestyle="--", dashes=(5, 3), zorder=3)
    ax_r.fill_between(VGS, 0, np.clip(d_rising, 0, None), color=COLOR_GM_CGG, alpha=0.12,
                       zorder=2, label="_nolegend_")

    ax_l.axvline(details["Vg_trough"], color=COLOR_GM_ID, ls=":", lw=1)
    ax_r.axvline(details["Vg_peak"], color=COLOR_GM_CGG, ls=":", lw=1)

    ax_l.set_xlabel("$V_G$ (V)")
    ax_l.set_ylabel("$d(g_m/I_D)/dV_G$  (V$^{-2}$)   [solid]", color=COLOR_GM_ID)
    ax_r.set_ylabel(f"{_RISING_DERIV_LABELS[rising_kind]}   [dashed]", color=COLOR_GM_CGG)
    ax_l.set_title(f"{title}\n"
                    f"$V_{{TR,deriv}}$ = {VTR_deriv:.3f} V   "
                    f"(centroid @ {details['Vg_peak']:.3f} V, trough @ {details['Vg_trough']:.3f} V)")
    style_axes(ax_l)
    ax_l.tick_params(axis="y", colors=COLOR_GM_ID)
    ax_l.spines["left"].set_color(COLOR_GM_ID)
    ax_r.grid(False)
    for side in ("top", "left"):
        ax_r.spines[side].set_visible(False)
    ax_r.spines["right"].set_color(COLOR_GM_CGG)
    ax_r.tick_params(axis="y", colors=COLOR_GM_CGG)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print_file(f"Saved {device} derivative-extrema plot to {savepath}")


def plot_gm_and_gmid(device, VGS, details_or_ID, savepath, title, ID_limit=2e-12,
                      window_length=5, polyorder=3):
    """Diagnostic plot: gm (right, dashed) and gm/Id (left, solid) vs Vg --
    the two RAW curves that feed extract_VTR_derivative's gm/Id side, useful
    when d(gm/Id)/dVg itself looks noisy or multi-lobed (a kink or shoulder
    in gm, or a plateau/wiggle in gm/Id, shows up directly here rather than
    being buried in a second level of differencing).

    `details_or_ID` accepts EITHER the `details` dict already returned by
    extract_VTR_derivative (reuses its gm/gm_over_id, no recomputation) OR a
    raw ID array (computes gm/gm_over_id fresh, e.g. for standalone use)."""
    import matplotlib.pyplot as plt

    if isinstance(details_or_ID, dict):
        gm, gm_over_id = details_or_ID["gm"], details_or_ID["gm_over_id"]
    else:
        gm, gm_over_id, _, _, _ = compute_gmid_gmcgg(
            VGS, details_or_ID, ID_limit=ID_limit,
            window_length=window_length, polyorder=polyorder)

    fig, ax_l = plt.subplots(figsize=(6.5, 4.5))
    ax_r = ax_l.twinx()

    ax_l.plot(VGS, gm_over_id, color=COLOR_GM_ID, lw=2, zorder=3)
    ax_r.plot(VGS, gm, color=COLOR_GM_CGG, lw=2, linestyle="--", dashes=(5, 3), zorder=3)

    ax_l.set_xlabel("$V_G$ (V)")
    ax_l.set_ylabel("$g_m/I_D$  (V$^{-1}$)   [solid]", color=COLOR_GM_ID)
    ax_r.set_ylabel("$g_m$  (S)   [dashed]", color=COLOR_GM_CGG)
    ax_l.set_title(title)
    style_axes(ax_l)
    ax_l.tick_params(axis="y", colors=COLOR_GM_ID)
    ax_l.spines["left"].set_color(COLOR_GM_ID)
    ax_r.grid(False)
    for side in ("top", "left"):
        ax_r.spines[side].set_visible(False)
    ax_r.spines["right"].set_color(COLOR_GM_CGG)
    ax_r.tick_params(axis="y", colors=COLOR_GM_CGG)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    plt.close(fig)
    print_file(f"Saved {device} gm & gm/Id plot to {savepath}")


# ----------------------------------------------------------------------------
# Demo: run the extraction on a real measured transfer curve
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    XLS_PATH = "C:\\Users\\koust\\OneDrive - Stanford\\Autumn_2024_25\\Discussions\\TR_extraction_paper_v1\\TR_different_IGZO_ChiHsin_v0\\IGZO_UCSD\\L[2]_W[10]_D[07]\\IdVg_lin_0_3_0p05_L[2]_W[10]_D[07].xls"

    VGS, ID, VDS = load_transfer_curve_xls(XLS_PATH, forward_only=True)
    if VDS is None:
        VDS = 0.05  # fallback if it couldn't be parsed from the Settings sheet
    T = 300.0

    print(f"Loaded {len(VGS)} forward-sweep points from {XLS_PATH.split('/')[-1]}")
    print(f"VGS: {VGS.min():.3f} to {VGS.max():.3f} V   VDS = {VDS:.4f} V\n")

    common_kwargs = dict(off_frac=1e-3, on_frac=0.2, window_length=11, npts_fit=7)

    for label, kwargs in [
        ("No R correction (raw ID)",       dict(correct_series_R=False)),
        ("Y-function (RS/RD unknown)",     dict(correct_series_R=True)),
    ]:
        VTON, VTOFF, VTR, det = extract_VTR(VGS, ID, VDS, T=T,
                                             return_details=True, **common_kwargs, **kwargs)
        print(f"{label:35s}: VTON={VTON:.3f} V  VTOFF={VTOFF:.3f} V  "
              f"VTR={VTR:.3f} V  (method={det['method']}, "
              f"R_tot_est={det['Rtot_used']})")

        try:
            import matplotlib.pyplot as plt
            fig = plot_extraction(VGS, ID, VTON, VTOFF, det, title=label)
            fname = f"vtr_extraction_{det['method']}.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"    -> saved plot to {fname}")
        except ImportError:
            pass