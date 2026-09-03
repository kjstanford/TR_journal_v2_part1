"""Implementation of the TR (transition-region) drain-current equations.

Reference: scratch/TR_related_equations.pdf

    Q_free(phi) = C2D * phiT * log(1 + exp(phi/phiT))
    Q_tail(phi) = Cgch * Vtr * 2F1(1, phiT/phi_tail; 1 + phiT/phi_tail; -exp(-phi/phiT))
                = Cgch * Vtr * F_tail(phi)

    Cgch * [Vg - Vt - n*phi(Vy) - n*Vy] = Q_free(phi(Vy)) + Q_tail(phi(Vy))

  => psi(Vy) = phi(Vy) + Vy = [Vg - Vt - Q_free/Cgch - Vtr*F_tail] / n

    G(Vg, Vd, Vs)      = int_{Vs}^{Vd} Q_free(phi(Vy)) dVy
    Id(Vg, Vd, Vs)     = mu_eff * (W/L) * G(Vg, Vd, Vs)
    tau_charge(Vg, Vd) = int_0^{0.9Vd} dV / G(Vg, Vd, V)   # source sweeps
    tau_leak(Vg, Vd)   = int_{0.9Vd}^{Vd} dV / G(Vg, V, 0) # drain sweeps
    FoM = tau_leak(Vhold, Vdd) / tau_charge(Vboost, Vdd)

Note the code's G/G_curve keep their original (Vd, Vs, Vg) argument order,
while tau_charge/tau_leak follow the PDF and take Vg first.

Unit convention (self-consistent, no extra factors needed):
    voltages   [V]
    capacitance per area [F/cm^2]
    charge per area      [C/cm^2]
    G                    [C/cm^2 * V]
    mobility             [cm^2/(V.s)]  ->  Id in [A]
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import hyp2f1

K_B = 1.380649e-23  # J/K
Q_E = 1.602176634e-19  # C

# numpy >= 2.0 renamed trapz -> trapezoid; keep working on both
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass
class TRParams:
    """Device / model parameters for the TR equations."""

    C2D: float = 2.0e-5      # 2-D density-of-states capacitance [F/cm^2]
    Cgch: float = 3.45e-6    # gate-to-channel capacitance [F/cm^2]  (~1 nm EOT)
    n: float = 1.10          # body factor [-]
    Vt: float = 0.30         # threshold voltage [V]
    Vtr: float = 0.30        # tail-charge voltage scale [V] (Q_tail -> Cgch*Vtr)
    phi_tail: float = 0.045  # band-tail characteristic voltage [V] (Urbach energy/q)
    T: float = 300.0         # temperature [K]

    @property
    def phiT(self) -> float:
        """Thermal voltage kT/q [V]."""
        return K_B * self.T / Q_E

    @property
    def a(self) -> float:
        """Hypergeometric parameter a = phiT / phi_tail."""
        return self.phiT / self.phi_tail


# --------------------------------------------------------------------------- #
# Charge models
# --------------------------------------------------------------------------- #
def q_free(phi, p: TRParams):
    """Free (band) charge density Q_free(phi) [C/cm^2].

    Uses logaddexp so that phi/phiT >> 1 does not overflow: for large phi this
    tends to C2D*phi, for phi << 0 it tends to C2D*phiT*exp(phi/phiT).
    """
    phi = np.asarray(phi, dtype=float)
    return p.C2D * p.phiT * np.logaddexp(0.0, phi / p.phiT)


def _f_tail_asymptotic(logx, a, n_terms=6):
    """Large-x expansion of 2F1(1, a; 1+a; -x), parameterised by log(x).

        2F1 = a*x^-a*pi/sin(pi*a) - a * sum_{k>=0} (-1)^k / ((1+k-a) * x^(k+1))

    Valid for non-integer a > 0 by analytic continuation.  Taking log(x) as the
    argument keeps every power finite (they underflow to 0) for x beyond the
    float range, which is exactly where the expansion is most accurate.
    """
    logx = np.asarray(logx, dtype=float)
    # Integer a is a removable pole of this form: the pi/sin(pi*a) prefactor and
    # the k = a - 1 series term both blow up and cancel (the true function has a
    # log there).  Nudge off it rather than dividing by sin(pi*a) = 0.  Only
    # reachable when phi_tail is an exact submultiple of phiT.
    if abs(a - round(a)) < 1e-9 and round(a) >= 1:
        a = round(a) + 1e-9
    lead = a * np.exp(-a * logx) * np.pi / np.sin(np.pi * a)
    xinv = np.exp(-logx)
    series = np.zeros_like(logx)
    for k in range(n_terms):
        series += ((-1.0) ** k) / (1.0 + k - a) * xinv ** (k + 1)
    return lead - a * series


def f_tail(phi, p: TRParams):
    """F_tail(phi) = 2F1(1, a; 1+a; -exp(-phi/phiT)), a = phiT/phi_tail.

    Monotonically increasing in phi, F_tail(+inf) = 1, F_tail(-inf) = 0, with
    the deep-subthreshold limit F_tail ~ exp(phi/phi_tail) (the Urbach tail).
    """
    phi = np.asarray(phi, dtype=float)
    a = p.a
    scalar = phi.ndim == 0
    phi = np.atleast_1d(phi)

    # log(x) with x = exp(-phi/phiT)
    u = -phi / p.phiT
    out = np.empty_like(phi)

    direct = u < np.log(1e8)
    if np.any(direct):
        out[direct] = hyp2f1(1.0, a, 1.0 + a, -np.exp(u[direct]))
    if np.any(~direct):
        # x itself would overflow, so the expansion works from log(x).
        out[~direct] = _f_tail_asymptotic(u[~direct], a)

    out = np.clip(out, 0.0, 1.0)
    return float(out[0]) if scalar else out


def q_tail(phi, p: TRParams):
    """Tail / trapped charge density Q_tail(phi) [C/cm^2]."""
    return p.Cgch * p.Vtr * f_tail(phi, p)


# --------------------------------------------------------------------------- #
# Surface-potential solver
# --------------------------------------------------------------------------- #
def _H(phi, p: TRParams):
    """Strictly increasing function with n*phi + Qfree/Cgch + Vtr*F_tail = B."""
    return p.n * phi + q_free(phi, p) / p.Cgch + p.Vtr * f_tail(phi, p)


def solve_phi_scalar(Vy: float, Vg: float, p: TRParams, xtol: float = 1e-14) -> float:
    """Solve the charge-balance equation for phi at a single channel point Vy."""
    B = Vg - p.Vt - p.n * Vy

    # H(phi) >= n*phi and H is strictly increasing, so B/n (or 0) brackets above.
    hi = max(B / p.n, 0.0)
    while _H(hi, p) < B:  # numerical safety net
        hi += 1.0

    lo = min(B / p.n, 0.0) - 1.0
    for _ in range(200):
        if _H(lo, p) <= B:
            break
        lo -= max(1.0, abs(lo))
    else:  # pragma: no cover
        raise RuntimeError(f"could not bracket phi at Vy={Vy}, Vg={Vg}")

    return brentq(lambda ph: _H(ph, p) - B, lo, hi, xtol=xtol, rtol=1e-15)


def solve_phi(Vy, Vg: float, p: TRParams):
    """Vectorised surface-potential solve, phi(Vy) for an array of Vy."""
    Vy = np.asarray(Vy, dtype=float)
    flat = np.array([solve_phi_scalar(float(v), Vg, p) for v in Vy.ravel()])
    return flat.reshape(Vy.shape)


def psi(Vy, Vg: float, p: TRParams):
    """Quasi-Fermi-referenced potential psi(Vy) = phi(Vy) + Vy."""
    return solve_phi(Vy, Vg, p) + np.asarray(Vy, dtype=float)


# --------------------------------------------------------------------------- #
# G, Id, tau
# --------------------------------------------------------------------------- #
def G_curve(Vs_array, Vd: float, Vg: float, p: TRParams, npts: int = 201):
    """G(Vd, Vs) for every Vs in `Vs_array`, at fixed Vd and Vg.

    Builds one fine Q_free(Vy) profile on [min(Vs, 0), Vd] and integrates it
    cumulatively, so the whole curve costs a single pass of root solves:

        G(Vd, Vs) = I(Vd) - I(Vs),   I(V) = int_{V0}^{V} Q_free(phi(Vy)) dVy
    """
    Vs_array = np.atleast_1d(np.asarray(Vs_array, dtype=float))
    lo = min(float(Vs_array.min()), Vd)
    hi = max(float(Vs_array.max()), Vd)
    if hi - lo < 1e-15:
        return np.zeros_like(Vs_array)

    grid = np.linspace(lo, hi, npts)
    qf = q_free(solve_phi(grid, Vg, p), p)

    # cumulative trapezoid on the fine grid, then interpolate onto Vs/Vd
    dI = 0.5 * (qf[1:] + qf[:-1]) * np.diff(grid)
    I = np.concatenate(([0.0], np.cumsum(dI)))
    return np.interp(Vd, grid, I) - np.interp(Vs_array, grid, I)


def G(Vd: float, Vs: float, Vg: float, p: TRParams, npts: int = 201) -> float:
    """Scalar G(Vd, Vs) [C/cm^2 * V]."""
    return float(G_curve(Vs, Vd, Vg, p, npts=npts)[0])


def drain_current(Vd: float, Vs: float, Vg: float, p: TRParams,
                  mu_eff: float = 100.0, W_over_L: float = 1.0) -> float:
    """Id = mu_eff * (W/L) * G(Vd, Vs)  [A], with mu_eff in cm^2/(V.s)."""
    return mu_eff * W_over_L * G(Vd, Vs, Vg, p)


def gm_over_id(Vg_array, Vd: float, Vs: float, p: TRParams,
               npts: int = 201):
    """gm/Id vs Vg at fixed Vd, Vs.  Returns (gm_over_id [1/V], G).

        gm/Id = (dId/dVg)/Id = (dG/dVg)/G = d ln(G) / dVg

    Id = mu_eff*(W/L)*G, so mu_eff and W/L cancel exactly - gm/Id is a pure
    property of the model, independent of mobility and geometry.  That is what
    makes it the standard way to compare transport efficiency across devices.

    The derivative is a central difference on ln G, which is the accurate way
    to do it: deep in subthreshold ln G is very nearly linear in Vg, so the
    truncation error nearly vanishes exactly where G itself spans decades.

    The ideal limit is 1/(n*phiT); band-tail charge degrades it.
    """
    Vg_array = np.asarray(Vg_array, dtype=float)
    g = np.array([G_curve(np.array([Vs]), Vd, vg, p, npts=npts)[0]
                  for vg in Vg_array])
    return np.gradient(np.log(g), Vg_array), g


def _G_and_QG(Vg: float, Vd: float, Vs: float, p: TRParams, npts: int = 201):
    """One solve pass returning (G, Q_G) at a single Vg.

    Q_G is the channel-averaged TOTAL gate charge per area [C/cm^2].  The gate
    sees the charge distributed over the channel, so it is the length average

        Q_G = (1/L) * int_0^L (Q_free + Q_tail) dy

    and the y <-> V map from `channel_profile` (dy/dV = mu*W*Q_free/Id, with
    L = mu*W*G/Id) turns that into a pure V integral with every prefactor
    cancelling:

        Q_G = (1/G) * int_Vs^Vd (Q_free + Q_tail) * Q_free dV

    i.e. a Q_free-weighted average of the total charge, not a plain average -
    regions holding more free charge occupy more of the channel length.
    """
    grid = np.linspace(Vs, Vd, npts)
    phi = solve_phi(grid, Vg, p)
    qf = q_free(phi, p)
    qt = q_tail(phi, p)
    g = float(_trapz(qf, grid))
    return g, float(_trapz((qf + qt) * qf, grid)) / g


def f_t(Vg_array, Vd: float, Vs: float, p: TRParams, mu_eff: float = 100.0,
        W_um: float = 1.0, L_um: float = 1.0, L_ov_um: float = 1.0,
        npts: int = 201):
    """fT = gm / Cgg  [s^-1], with Cgg the total gate CAPACITANCE in farads.

    Returns (fT, Cgg, gm, Id):

        gm    = mu_eff * (W/L) * dG/dVg                    [A/V]
        Id    = mu_eff * (W/L) * G                         [A]
        C_int = W * L * d(Q_free + Q_tail)/dVg             [F]  intrinsic
        C_par = Cgch * L_ov * W                            [F]  parasitic
        Cgg   = C_int + C_par
        fT    = gm / Cgg

    W_um, L_um, L_ov_um in microns.  C_par is the gate-to-source/drain overlap
    capacitance: a bias-independent floor that does NOT vanish with the
    channel, so it is what stops fT from levelling off in subthreshold (where
    C_int -> 0 while gm -> 0 too).  L_ov_um is a single overlap length as
    written; double it to account for both the source and drain sides.

    W no longer cancels from fT once C_par is included - it multiplies gm and
    C_par identically but C_int scales as W*L, so the intrinsic/parasitic
    balance, and hence fT, now depends on L_ov/L rather than on W.

    Note this is literally gm/Cgg as defined; the conventional transit
    frequency is gm/(2*pi*Cgg), a factor 2*pi smaller.
    """
    Vg_array = np.asarray(Vg_array, dtype=float)
    gq = [_G_and_QG(vg, Vd, Vs, p, npts=npts) for vg in Vg_array]
    g = np.array([a for a, _ in gq])
    QG = np.array([b for _, b in gq])

    W_cm, L_cm, L_ov_cm = W_um * 1e-4, L_um * 1e-4, L_ov_um * 1e-4
    gm = mu_eff * (W_cm / L_cm) * np.gradient(g, Vg_array)      # A/V
    Id = mu_eff * (W_cm / L_cm) * g                             # A
    C_int = W_cm * L_cm * np.gradient(QG, Vg_array)             # F
    C_par = p.Cgch * L_ov_cm * W_cm                             # F, constant
    Cgg = C_int + C_par
    return gm / Cgg, Cgg, gm, Id


def _refine_extremum(x, y, i):
    """Sub-grid location of an extremum at index i by parabolic interpolation.

    np.argmin/argmax only resolve the extremum to the grid pitch (here ~6 mV),
    which is coarse next to the differences we are reporting.  Fitting a
    parabola through the three samples around the peak recovers the vertex to
    well below the pitch, which is what makes the reported values stable
    against changing n_vg.  Assumes a uniform grid.
    """
    if i <= 0 or i >= len(x) - 1:
        return float(x[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2.0 * y1 + y2
    if den == 0.0:
        return float(x[i])
    return float(x[i] + 0.5 * (x[1] - x[0]) * (y0 - y2) / den)


def derivative_extrema(p: TRParams = None, Vd: float = 0.05, Vs: float = 0.0,
                       Vg_min: float = -0.25, Vg_max: float = 1.25,
                       n_vg: int = 241, npts: int = 201, mu_eff: float = 100.0,
                       W_um: float = 1.0, L_um: float = 1.0,
                       L_ov_um: float = 1.0, cases=None, verbose: bool = True):
    """Vg at the extrema of d(gm/Id)/dVg and dfT/dVg, and their difference.

    For each case returns a dict with

        Vg_gm_id : Vg at the MINIMUM of d(gm/Id)/dVg  (steepest roll-off of
                   transport efficiency as the device leaves subthreshold)
        Vg_fT    : Vg at the MAXIMUM of dfT/dVg       (steepest rise of speed)
        diff     : Vg_fT - Vg_gm_id

    The difference is the bias-axis gap between "efficiency is degrading
    fastest" and "speed is improving fastest" - a compact read on how much Vg
    headroom separates the analog sweet spot from the speed sweet spot.

    Both are second derivatives of solved quantities, so they are the noisiest
    things in this file; `_refine_extremum` plus n_vg = 241 gives values stable
    to <=0.2 mV against doubling the grid.

    Unlike gm/Id, Vg_fT is NOT geometry-free: it is set by where C_int
    overtakes C_par, so it moves with L_ov/L (mu_eff, however, cancels).
    """
    p = p or TRParams()
    cases = cases if cases is not None else vary_cases(p, "Vtr", 0.0,
                                                       tex="V_\\mathrm{tr}")
    Vg = np.linspace(Vg_min, Vg_max, n_vg)

    rows = []
    for lab, pc, _style in cases:
        ft, _Cgg, _gm, Id = f_t(Vg, Vd, Vs, pc, mu_eff=mu_eff, W_um=W_um,
                                L_um=L_um, L_ov_um=L_ov_um, npts=npts)
        # gm/Id as d ln(Id)/dVg, NOT as gradient(Id)/Id.  The two agree
        # analytically but not numerically: Id spans decades here, so a central
        # difference on Id itself carries an O((k*h)^2/6) relative error that
        # grows with the local slope k, biasing gm/Id upward at the low-Vg end
        # and dragging the reported minimum onto the sweep boundary.  ln(Id) is
        # near-linear in subthreshold, so the log form is nearly exact there.
        d_gm_id = np.gradient(np.gradient(np.log(Id), Vg), Vg)
        d_ft = np.gradient(ft, Vg)
        a = _refine_extremum(Vg, d_gm_id, int(np.argmin(d_gm_id)))
        b = _refine_extremum(Vg, d_ft, int(np.argmax(d_ft)))
        rows.append({"label": lab, "Vg_gm_id": a, "Vg_fT": b, "diff": b - a})

    if verbose:
        plain = {"$V_\\mathrm{tr}$": "Vtr", "$\\phi_\\mathrm{tail}$": "phi_tail"}
        print(f"\nextrema of dY/dVg   (Vd = {Vd:g} V, Vs = {Vs:g} V, "
              f"L = {L_um:g} um, L_ov = {L_ov_um:g} um, W = {W_um:g} um)")
        print(f"{'case':<34}{'Vg* gm/Id':>11}{'Vg* fT':>10}{'diff (V)':>11}")
        for r in rows:
            lab = r["label"]
            for tex, txt in plain.items():
                lab = lab.replace(tex, txt)
            print(f"{lab:<34}{r['Vg_gm_id']:>11.4f}{r['Vg_fT']:>10.4f}"
                  f"{r['diff']:>11.4f}")
    return rows


# Upper limit of the Vs integration / sweep, as a fraction of Vd.  G(Vd, Vs)
# vanishes linearly as Vs -> Vd, so tau's integrand diverges like 1/(Vd - Vs);
# stopping at 0.9*Vd keeps the integral finite.
VS_FRAC = 0.9


def tau_charge(Vg: float, Vd: float, p: TRParams, npts: int = 201,
               vs_frac: float = VS_FRAC) -> float:
    """tau_charge(Vg, Vd) = int_0^{vs_frac*Vd} dV / G(Vg, Vd, V)   [cm^2/C].

    The *source* sweeps while the drain is held at Vd.  Units:
    [V] / [C/cm^2 * V] = cm^2/C.  Since Id = mu_eff*(W/L)*G,

        tau = mu_eff * (W/L) * int dV/Id = mu_eff * (W/L) * R_eff,

    so tau is an effective on-resistance scaled by mu_eff*(W/L); see `delay`.

    (Note the argument order here follows the PDF, Vg first; the older `G` /
    `G_curve` helpers take Vd, Vs, Vg.)
    """
    V = np.linspace(0.0, vs_frac * Vd, npts)
    g = G_curve(V, Vd, Vg, p, npts=npts)
    if not np.all(g > 0):
        return np.inf  # Q_free underflowed to 0: too far off to resolve
    return float(_trapz(1.0 / g, V))


def tau_leak(Vg: float, Vd: float, p: TRParams, npts: int = 201,
             vs_frac: float = VS_FRAC) -> float:
    """tau_leak(Vg, Vd) = int_{vs_frac*Vd}^{Vd} dV / G(Vg, V, 0)   [cm^2/C].

    Note this is NOT tau_charge over a different interval: here the *drain*
    sweeps from vs_frac*Vd up to Vd while the source stays at 0, so the
    integrand is G(Vg, V, 0) = int_0^V Q_free dVy - the full-channel integral
    to a moving upper limit, which stays finite and O(G(Vg, Vd, 0)) rather than
    collapsing to zero.

    phi(Vy) depends only on Vy and Vg, never on the drain bias, so one solve
    pass over [0, Vd] serves every V in the range: G(Vg, V, 0) is just the
    running integral of that single Q_free profile.
    """
    grid = np.linspace(0.0, Vd, npts)
    qf = q_free(solve_phi(grid, Vg, p), p)
    dI = 0.5 * (qf[1:] + qf[:-1]) * np.diff(grid)
    I = np.concatenate(([0.0], np.cumsum(dI)))  # I(V) = G(Vg, V, 0)

    V = np.linspace(vs_frac * Vd, Vd, npts)
    g = np.interp(V, grid, I)
    if not np.all(g > 0):
        return np.inf  # Q_free underflowed to 0: too far off to resolve
    return float(_trapz(1.0 / g, V))


def fom(Vboost: float, Vhold: float, Vdd: float, p: TRParams,
        npts: int = 201) -> float:
    """FoM = tau_leak(Vhold, Vdd) / tau_charge(Vboost, Vdd)   [dimensionless].

    Charging is evaluated at the boosted gate, leakage at the hold gate, both
    against the same supply Vdd.  Larger is better: a large FoM means the node
    leaks away slowly relative to how fast it can be charged.
    """
    return (tau_leak(Vhold, Vdd, p, npts=npts)
            / tau_charge(Vboost, Vdd, p, npts=npts))


def tau_curve(Vs_array, Vd: float, Vg: float, p: TRParams, npts: int = 201):
    """Running form of tau: tau(Vd, Vs) = int_0^{Vs} dVs' / G(Vd, Vs')  [cm^2/C].

    Same integrand as `tau`, evaluated cumulatively so it can be plotted
    against Vs.  Its value at the upper limit Vs = VS_FRAC*Vd is exactly the
    tau(Vd) of the PDF, so the curve shows how much each part of the channel
    contributes to the total.  Requires Vs_array to start at 0 and be sorted.
    """
    Vs_array = np.atleast_1d(np.asarray(Vs_array, dtype=float))
    g = G_curve(Vs_array, Vd, Vg, p, npts=npts)
    integrand = 1.0 / g
    dI = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(Vs_array)
    return np.concatenate(([0.0], np.cumsum(dI)))


def channel_profile(Vd: float, Vg: float, p: TRParams, Vs: float = 0.0,
                    npts: int = 201):
    """Map the quasi-Fermi potential onto position along the channel.

    Current continuity gives Id = mu_eff*W*Q_free(V)*dV/dy, so integrating from
    the source to the point whose quasi-Fermi potential is V,

        Id * y = mu_eff * W * G(V, Vs)     and     Id * L = mu_eff * W * G(Vd, Vs)

    which divides out every prefactor:

        y/L = G(V, Vs) / G(Vd, Vs) = int_Vs^V Q_free dV' / int_Vs^Vd Q_free dV'

    Returns (y_over_L, Vy, phi), each of length npts, with y_over_L running
    0 -> 1.  Note Q_free(phi(Vy)) depends on Vy only through the charge-balance
    solve at fixed Vg, so it is the same integrand used by `G_curve`.
    """
    if Vd <= Vs:
        raise ValueError(f"need Vd > Vs, got Vd={Vd}, Vs={Vs}")

    Vy = np.linspace(Vs, Vd, npts)
    phi = solve_phi(Vy, Vg, p)
    qf = q_free(phi, p)

    dI = 0.5 * (qf[1:] + qf[:-1]) * np.diff(Vy)
    I = np.concatenate(([0.0], np.cumsum(dI)))
    return I / I[-1], Vy, phi


def delay(Vd: float, Vg: float, p: TRParams, C_load: float = 1e-15,
          mu_eff: float = 100.0, W_over_L: float = 1.0, **kw) -> float:
    """Switching delay t = C_load * tau / (mu_eff * (W/L))  [s].

    C_load in F, mu_eff in cm^2/(V.s); tau/(mu_eff*W_over_L) is R_eff in ohms.
    """
    return C_load * tau_charge(Vg, Vd, p, **kw) / (mu_eff * W_over_L)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
# Sequential single-hue ramps, light -> dark (magnitude encoding for Vg / Vd).
BLUE_RAMP = ["#8CBFEC", "#4A8CD4", "#255FA0", "#143F6E"]
AMBER_RAMP = ["#EFB65E", "#D08D24", "#9C5F0F", "#6B4008"]

# Two-series categorical pair (validated: normal dE 31.5, worst-CVD dE 25.9).
# Here they distinguish the two *scalings* of one quantity, not two quantities:
# LOG_COLOR is the log (left) rendering, LIN_COLOR the linear (right) one.
LOG_COLOR = "#255FA0"
LIN_COLOR = "#C9801A"

# Same validated pair reused where the two series are quantities, not scalings
# (phi and Vy are both volts, so that plot needs no second axis at all).
PHI_COLOR = "#255FA0"
VY_COLOR = "#C9801A"

# Categorical case colours for plots where the case is the ONLY dimension, so
# colour is free to carry it (validated: worst-CVD dE 16.2, normal dE >= 25.9).
# Used redundantly with the CASE_STYLES dashes, never alone.
CASE_COLORS = ["#255FA0", "#C9801A", "#8E4EC6", "#1B8A6B"]

INK = "#1F2933"
INK_MUTED = "#7B8794"
GRID = "#E4E7EB"


# Line styles carrying the *case* in the comparison plots, in assignment order.
SOLID_STYLE = {"linestyle": "-"}
DASHED_STYLE = {"linestyle": "--", "dashes": (5, 3)}
DOTTED_STYLE = {"linestyle": ":", "dashes": (1, 1.8)}
DASHDOT_STYLE = {"linestyle": "-.", "dashes": (6, 2, 1, 2)}
CASE_STYLES = [SOLID_STYLE, DASHED_STYLE, DOTTED_STYLE, DASHDOT_STYLE]


def make_cases(p: TRParams, specs):
    """Build a `cases` list from [(label, {field: value, ...}), ...].

    Styles are assigned in CASE_STYLES order, so the first spec is solid, the
    second dashed, the third dotted.  Use this when the cases vary more than
    one field at a time; `vary_cases` is the one-field shorthand.

        make_cases(p, [("$V_\\\\mathrm{tr}$ = 0", {"Vtr": 0.0}),
                       ("thin tail", {"Vtr": 0.3, "phi_tail": 0.045})])
    """
    from dataclasses import replace

    return [(lab, replace(p, **overrides), CASE_STYLES[i % len(CASE_STYLES)])
            for i, (lab, overrides) in enumerate(specs)]


def vary_cases(p: TRParams, field: str, alt, tex: str = None, unit: str = "V"):
    """Two-case spec for the comparison plots: p as-is vs p with one field changed.

    Returns [(label, params, style), ...] with the baseline solid and the
    variant dashed - the shape `plot_G_tau_compare` and `plot_G_vs_Vg` expect
    for their `cases` argument.

        vary_cases(p, "Vtr", 0.0)
        vary_cases(p, "phi_tail", 0.09, tex="\\\\phi_\\\\mathrm{tail}")
    """
    from dataclasses import replace

    name = tex or f"\\mathrm{{{field}}}"
    suffix = f" {unit}" if unit else ""

    def lab(v):
        return f"${name}$ = {v:g}{suffix}" if v else f"${name}$ = 0"

    return [
        (lab(getattr(p, field)), p, SOLID_STYLE),
        (lab(alt), replace(p, **{field: alt}), DASHED_STYLE),
    ]


def _proxy_handles(series, cases):
    """Legend proxies for the two encodings: color = series, dashes = case.

    `series` is what color encodes: the log/linear scale pair by default, or
    the plotted quantities where the chart has no second axis.
    """
    from matplotlib.lines import Line2D

    series = series or [("log scale (left)", LOG_COLOR),
                        ("linear scale (right)", LIN_COLOR)]
    return ([Line2D([], [], color=c, lw=2.0, label=lab) for lab, c in series],
            [Line2D([], [], color=INK_MUTED, lw=2.0, label=lab, **style)
             for lab, _, style in cases])


def _two_part_legend(fig, cases, y=0.005, series=None):
    """Two side-by-side legends: color = scale, line style = case.

    With N cases the flat product (scale x case) would need 2N entries that all
    repeat the same two facts.  Splitting keeps it at 2 + N and matches how the
    chart actually encodes things: color says which axis, dashes say which case.

    Call this *after* fig.tight_layout(rect=...) has reserved bottom margin -
    the legends sit in figure coordinates and will otherwise overlap the axes.
    """
    from matplotlib.lines import Line2D

    scale_h, case_h = _proxy_handles(series, cases)

    fig.legend(handles=scale_h, frameon=False, fontsize=9, labelcolor=INK,
               loc="lower center", ncol=1, bbox_to_anchor=(0.30, y))
    fig.legend(handles=case_h, frameon=False, fontsize=9, labelcolor=INK,
               loc="lower center", ncol=1, bbox_to_anchor=(0.68, y))


def _label(ax, x, y, text, color, anchor="left"):
    """Direct-label a curve at one of its endpoints."""
    dx, ha = (4, "left") if anchor == "left" else (-4, "right")
    ax.annotate(text, xy=(x, y), xytext=(dx, 5), textcoords="offset points",
                color=color, fontsize=8.5, fontweight="bold",
                ha=ha, va="bottom", zorder=4)


def _style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3)


def plot_G_vs_Vs(p: TRParams = None, Vd: float = 0.6, Vg: float = 0.8,
                 Vg_list=None, Vd_list=None, npts: int = 201,
                 savepath=None, show: bool = False):
    """Plot G(Vd, Vs) vs Vs.

    Panel (a): fixed Vd, family of Vg.
    Panel (b): fixed Vg, family of Vd.
    """
    import matplotlib.pyplot as plt

    p = p or TRParams()
    Vg_list = list(Vg_list) if Vg_list is not None else [0.4, 0.7, 1.0, 1.3]
    Vd_list = list(Vd_list) if Vd_list is not None else [0.2, 0.4, 0.6, 0.8]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    fig.patch.set_facecolor("white")

    # ---- panel (a): family of Vg at fixed Vd --------------------------------
    Vs = np.linspace(0.0, VS_FRAC * Vd, npts)
    for color, vg in zip(BLUE_RAMP, Vg_list):
        g = G_curve(Vs, Vd, vg, p)
        ax_a.plot(Vs, g, color=color, linewidth=2.0, zorder=3,
                  label=f"$V_\\mathrm{{g}}$ = {vg:.2f} V")
        # curves fan out at Vs = 0, so label there
        _label(ax_a, Vs[0], g[0], f"{vg:.2f} V", color, anchor="left")
    _style_axes(ax_a, "$V_\\mathrm{s}$  (V)",
                "$\\mathcal{G}(V_\\mathrm{d}, V_\\mathrm{s})$  (C/cm$^2\\cdot$V)",
                f"(a)  $V_\\mathrm{{d}}$ = {Vd:.2f} V, sweeping $V_\\mathrm{{g}}$")
    ax_a.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")

    # ---- panel (b): family of Vd at fixed Vg --------------------------------
    # darkest (largest Vd) first: in saturation the curves coincide, and drawing
    # the shorter/lighter ones last keeps every series visible where it exists.
    for color, vd in zip(reversed(AMBER_RAMP), reversed(Vd_list)):
        vs = np.linspace(0.0, VS_FRAC * vd, npts)
        g = G_curve(vs, vd, Vg, p)
        ax_b.plot(vs, g, color=color, linewidth=2.0, zorder=3,
                  label=f"$V_\\mathrm{{d}}$ = {vd:.2f} V")
        # curves coincide at Vs = 0 once saturated; they terminate at
        # Vs = VS_FRAC*Vd, which is where they are distinguishable.
        _label(ax_b, vs[-1], g[-1], f"{vd:.2f} V", color, anchor="right")
    _style_axes(ax_b, "$V_\\mathrm{s}$  (V)",
                "$\\mathcal{G}(V_\\mathrm{d}, V_\\mathrm{s})$  (C/cm$^2\\cdot$V)",
                f"(b)  $V_\\mathrm{{g}}$ = {Vg:.2f} V, sweeping $V_\\mathrm{{d}}$")
    h, lab = ax_b.get_legend_handles_labels()  # restore ascending-Vd order
    ax_b.legend(h[::-1], lab[::-1], frameon=False, fontsize=8.5,
                labelcolor=INK, loc="upper right")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, facecolor=fig.get_facecolor())
        print(f"saved -> {savepath}")
    if show:
        plt.show()
    return fig


def plot_G_tau_vs_Vs(p: TRParams = None, Vd: float = 0.6, Vg: float = None,
                     npts: int = 201, savepath=None, show: bool = False):
    """Two tiles vs Vs at a single bias, each on a log (left) and linear (right) axis.

    (a)  G(Vd, Vs)                                       [C/cm^2 * V]
    (b)  tau(Vd, Vs) = int_0^{Vs} dVs'/G(Vd, Vs')        [cm^2/C]

    Vs runs from 0 to VS_FRAC*Vd, so the endpoint of tile (b) is the tau(Vd) of
    the PDF.  Vg defaults to Vd (diode-connected).

    Each tile draws ONE quantity twice - log on the left axis, linear on the
    right - which is the usual device-transfer-curve convention.  It is not a
    two-measure dual axis: both curves are the same data, so the axis tint,
    the dashed linear stroke and the direct labels only have to tell the reader
    which *scale* they are reading, never which quantity.
    """
    import matplotlib.pyplot as plt

    p = p or TRParams()
    Vg = Vd if Vg is None else Vg

    Vs = np.linspace(0.0, VS_FRAC * Vd, npts)
    g = G_curve(Vs, Vd, Vg, p)
    t = tau_curve(Vs, Vd, Vg, p)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0], g, "(a)", "$\\mathcal{G}(V_\\mathrm{d}, V_\\mathrm{s})$",
         "C/cm$^2\\cdot$V", "$\\mathcal{G}$"),
        (axes[1], t, "(b)", "$\\tau(V_\\mathrm{d}, V_\\mathrm{s})$",
         "cm$^2$/C", "$\\tau$"),
    ]

    handles = []
    for ax_log, y, tag, sym, unit, short in panels:
        ax_lin = ax_log.twinx()

        # log rendering (left).  tau starts at exactly 0, which a log axis
        # cannot show, so mask it; the window is then set from the smallest
        # positive value so that no part of either curve is ever clipped.
        # (For tau that lower decade is grid-dependent - it is the first
        # trapezoid of the cumulative integral, not a physical floor.)
        y_log = np.ma.masked_less_equal(y, 0.0)
        ax_log.semilogy(Vs, y_log, color=LOG_COLOR, linewidth=2.0, zorder=3,
                        label="log scale (left)")
        top = float(np.max(y))
        ax_log.set_ylim(float(y_log.min()) / 2.0, top * 2.0)

        # linear rendering (right)
        ax_lin.plot(Vs, y, color=LIN_COLOR, linewidth=2.0, linestyle="--",
                    dashes=(5, 3), zorder=3, label="linear scale (right)")
        ax_lin.set_ylim(0.0, top * 1.08)

        _style_axes(ax_log, "$V_\\mathrm{s}$  (V)", f"{sym}  ({unit})",
                    f"{tag}  {sym}")
        ax_log.yaxis.label.set_color(LOG_COLOR)
        ax_log.tick_params(axis="y", colors=LOG_COLOR)

        ax_lin.set_ylabel(f"{sym}  ({unit})", color=LIN_COLOR, fontsize=10)
        ax_lin.tick_params(axis="y", colors=LIN_COLOR, labelsize=9, length=3)
        ax_lin.grid(False)
        for side in ("top", "left"):
            ax_lin.spines[side].set_visible(False)
        ax_lin.spines["right"].set_color(GRID)
        ax_lin.spines["bottom"].set_color(GRID)

        # direct labels so the scale is identifiable without the legend
        k = int(0.55 * (npts - 1))
        ax_log.annotate(f"{short}  log", xy=(Vs[k], y[k]), xytext=(0, -18),
                        textcoords="offset points", color=LOG_COLOR,
                        fontsize=9, fontweight="bold", ha="center")
        m = int(0.80 * (npts - 1))
        ax_lin.annotate(f"{short}  linear", xy=(Vs[m], y[m]), xytext=(-6, 8),
                        textcoords="offset points", color=LIN_COLOR,
                        fontsize=9, fontweight="bold", ha="right")

        if not handles:  # one shared legend, taken from the first tile
            handles = [ax_log.get_lines()[0], ax_lin.get_lines()[0]]

    fig.suptitle(f"$V_\\mathrm{{d}}$ = $V_\\mathrm{{g}}$ = {Vd:.2f} V"
                 f"   \u2014   $V_\\mathrm{{s}}$ swept to "
                 f"{VS_FRAC:g}$V_\\mathrm{{d}}$ = {VS_FRAC*Vd:.3f} V,"
                 f"   $\\tau(V_\\mathrm{{d}})$ = {t[-1]:.3e} cm$^2$/C",
                 color=INK, fontsize=10.5, y=1.0)

    fig.legend(handles, [hh.get_label() for hh in handles], frameon=False,
               fontsize=9, labelcolor=INK, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        print(f"saved -> {savepath}")
    if show:
        plt.show()
    return fig


def plot_G_tau_compare(p: TRParams = None, Vd: float = 1.0,
                       Vg: float = None, npts: int = 201, cases=None,
                       savepath=None, show: bool = False):
    """The plot_G_tau_vs_Vs pair with two parameter cases overlaid.

    Three tiles vs Vs - (a) G, (b) 1/G and (c) tau - each still carrying a log
    (left) and a linear (right) axis, with both cases drawn on top of each
    other: the first case solid, the second dashed.

    `cases` is a [(label, TRParams, style_kwargs), ...] pair, most easily built
    with `vary_cases`; it defaults to Vtr = p.Vtr vs Vtr = 0, i.e. Q_tail
    switched off entirely.

    Encoding note: in `plot_G_tau_vs_Vs` the dashed stroke marks the linear
    rendering, but here dashes are spoken for by the case.  So the scale is
    carried by color alone - blue = log/left, amber = linear/right, each
    matching its own tinted axis - and the line style carries the case.
    """
    import matplotlib.pyplot as plt

    p = p or TRParams()
    Vg = Vd if Vg is None else Vg
    Vs = np.linspace(0.0, VS_FRAC * Vd, npts)

    cases = cases if cases is not None else vary_cases(p, "Vtr", 0.0,
                                                       tex="V_\\mathrm{tr}")
    data = []
    for _, pc, _ in cases:
        g = G_curve(Vs, Vd, Vg, pc)
        # 1/G is tau's integrand, so column (b) is literally what column (c)
        # accumulates - G never reaches 0 on [0, VS_FRAC*Vd], so it is finite.
        data.append((g, 1.0 / g, tau_curve(Vs, Vd, Vg, pc)))

    cols = [("$\\mathcal{G}(V_\\mathrm{d}, V_\\mathrm{s})$", "C/cm$^2\\cdot$V",
             "$\\mathcal{G}$", "a"),
            ("$1/\\mathcal{G}(V_\\mathrm{d}, V_\\mathrm{s})$",
             "cm$^2$/(C$\\cdot$V)", "$1/\\mathcal{G}$", "b"),
            ("$\\tau(V_\\mathrm{d}, V_\\mathrm{s})$", "cm$^2$/C",
             "$\\tau$", "c")]

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.8))
    fig.patch.set_facecolor("white")

    for col, (sym, unit, short, tag) in enumerate(cols):
        ax_log = axes[col]
        ax_lin = ax_log.twinx()

        for (case_lab, _, style), d in zip(cases, data):
            y = d[col]
            # tau starts at exactly 0, which a log axis cannot show
            y_log = np.ma.masked_less_equal(y, 0.0)
            ax_log.semilogy(Vs, y_log, color=LOG_COLOR, linewidth=2.0,
                            zorder=3, label=f"log (left), {case_lab}", **style)
            ax_lin.plot(Vs, y, color=LIN_COLOR, linewidth=2.0, zorder=3,
                        label=f"linear (right), {case_lab}", **style)

        # limits span both cases, so neither is clipped or auto-hidden
        ys = [d[col] for d in data]
        top = max(float(np.max(y)) for y in ys)
        bot = min(float(np.min(y[y > 0])) for y in ys)
        ax_log.set_ylim(bot / 2.0, top * 2.0)
        ax_lin.set_ylim(0.0, top * 1.08)

        _style_axes(ax_log, "$V_\\mathrm{s}$  (V)", f"{sym}  ({unit})",
                    f"({tag})  {sym}")
        ax_log.yaxis.label.set_color(LOG_COLOR)
        ax_log.tick_params(axis="y", colors=LOG_COLOR)

        ax_lin.set_ylabel(f"{sym}  ({unit})", color=LIN_COLOR, fontsize=10)
        ax_lin.tick_params(axis="y", colors=LIN_COLOR, labelsize=9, length=3)
        ax_lin.grid(False)
        for side in ("top", "left"):
            ax_lin.spines[side].set_visible(False)
        ax_lin.spines["right"].set_color(GRID)
        ax_lin.spines["bottom"].set_color(GRID)

        # label the scale on the solid (Vtr = p.Vtr) curves only - the two
        # cases run too close together to direct-label without collisions,
        # so the case is left to the line style and the legend.
        # anchor on the topmost case at that x so the label clears the others
        k = int(0.55 * (npts - 1))
        i_top = int(np.argmax([d[col][k] for d in data]))
        ax_log.annotate(f"{short}  log", xy=(Vs[k], data[i_top][col][k]),
                        xytext=(0, 12), textcoords="offset points",
                        color=LOG_COLOR, fontsize=9, fontweight="bold",
                        ha="center")
        m = int(0.80 * (npts - 1))
        j_top = int(np.argmax([d[col][m] for d in data]))
        ax_lin.annotate(f"{short}  linear", xy=(Vs[m], data[j_top][col][m]),
                        xytext=(-6, 8), textcoords="offset points",
                        color=LIN_COLOR, fontsize=9, fontweight="bold",
                        ha="right")

    taus = "    ".join(f"{d[2][-1]:.3e} ({lab})"
                       for (lab, _, _), d in zip(cases, data))
    fig.suptitle(f"$V_\\mathrm{{d}}$ = $V_\\mathrm{{g}}$ = {Vd:.2f} V,   "
                 f"$V_\\mathrm{{s}}$ swept to {VS_FRAC:g}$V_\\mathrm{{d}}$"
                 f" = {VS_FRAC * Vd:.3f} V"
                 f"\n$\\tau(V_\\mathrm{{d}})$ [cm$^2$/C]:   {taus}",
                 color=INK, fontsize=10.5, y=1.02)

    fig.tight_layout(rect=(0, 0.17, 1, 1))
    _two_part_legend(fig, cases)
    if savepath:
        fig.savefig(savepath, dpi=200, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        print(f"saved -> {savepath}")
    if show:
        plt.show()
    return fig


def plot_G_vs_Vg(p: TRParams = None, Vd_list=(1.0, 0.05), Vs: float = 0.0,
                 Vg_min: float = -0.25, Vg_max: float = 1.25, n_vg: int = 41,
                 npts: int = 101, cases=None, savepath=None,
                 show: bool = False):
    """G(Vd, Vs) vs Vg, one tile per Vd, with two parameter cases overlaid.

    Defaults give two tiles at Vs = 0 with Vg swept 0 -> 1.0 V:
        (a) Vd = 1.00 V   saturation
        (b) Vd = 0.05 V   linear / triode
    Since Id = mu_eff*(W/L)*G, each tile is an Id-Vg transfer curve up to that
    constant.

    `cases` is a [(label, TRParams, style_kwargs), ...] pair, most easily built
    with `vary_cases`; it defaults to Vtr = p.Vtr vs Vtr = 0.  The first case
    is drawn solid, the second dashed.

    Encoding follows `plot_G_tau_compare`, not `plot_G_tau_vs_Vs`: with the
    dashes spoken for by the case, the scale is carried by color alone -
    blue = log/left, amber = linear/right, each matching its own tinted axis.

    Tiles are autoscaled independently.  The comparison being made *within* a
    tile is the case, and both cases there do share limits; across tiles G
    differs by more than an order of magnitude simply because the integration
    window [Vs, Vd] is ~20x narrower at Vd = 0.05 V, so a shared scale would
    flatten the low-Vd tile without saying anything useful.

    Cost is len(Vd_list) * 2 * n_vg * npts brentq calls.
    """
    import matplotlib.pyplot as plt

    p = p or TRParams()
    Vd_list = list(Vd_list)
    Vg = np.linspace(Vg_min, Vg_max, n_vg)

    cases = cases if cases is not None else vary_cases(p, "Vtr", 0.0,
                                                       tex="V_\\mathrm{tr}")

    fig, axes = plt.subplots(1, len(Vd_list),
                             figsize=(6.9 * len(Vd_list), 5.0), squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor("white")

    for tile, (ax_log, Vd) in enumerate(zip(axes, Vd_list)):
        ax_lin = ax_log.twinx()
        data = [np.array([G_curve(np.array([Vs]), Vd, vg, pc, npts=npts)[0]
                          for vg in Vg]) for _, pc, _ in cases]

        for (case_lab, _, style), g in zip(cases, data):
            g_log = np.ma.masked_less_equal(g, 0.0)
            ax_log.semilogy(Vg, g_log, color=LOG_COLOR, linewidth=2.0,
                            zorder=3, label=f"log (left), {case_lab}", **style)
            ax_lin.plot(Vg, g, color=LIN_COLOR, linewidth=2.0, zorder=3,
                        label=f"linear (right), {case_lab}", **style)

        # limits span both Vtr cases, so neither is clipped
        top = max(float(np.max(g)) for g in data)
        bot = min(float(np.min(g[g > 0])) for g in data)
        ax_log.set_ylim(bot / 2.0, top * 2.0)
        ax_lin.set_ylim(0.0, top * 1.08)

        _style_axes(ax_log, "$V_\\mathrm{g}$  (V)",
                    "$\\mathcal{G}(V_\\mathrm{d}, V_\\mathrm{s})$  "
                    "(C/cm$^2\\cdot$V)",
                    f"({'ab'[tile]})  $V_\\mathrm{{d}}$ = {Vd:.2f} V, "
                    f"$V_\\mathrm{{s}}$ = {Vs:.2f} V")
        ax_log.yaxis.label.set_color(LOG_COLOR)
        ax_log.tick_params(axis="y", colors=LOG_COLOR)

        ax_lin.set_ylabel("$\\mathcal{G}(V_\\mathrm{d}, V_\\mathrm{s})$  "
                          "(C/cm$^2\\cdot$V)", color=LIN_COLOR, fontsize=10)
        ax_lin.tick_params(axis="y", colors=LIN_COLOR, labelsize=9, length=3)
        ax_lin.grid(False)
        for side in ("top", "left"):
            ax_lin.spines[side].set_visible(False)
        ax_lin.spines["right"].set_color(GRID)
        ax_lin.spines["bottom"].set_color(GRID)

        # anchor on the topmost case at that x so the label clears the others
        k = int(0.45 * (n_vg - 1))
        i_top = int(np.argmax([g[k] for g in data]))
        ax_log.annotate("$\\mathcal{G}$  log", xy=(Vg[k], data[i_top][k]),
                        xytext=(0, 12), textcoords="offset points",
                        color=LOG_COLOR, fontsize=9, fontweight="bold",
                        ha="center")
        m = int(0.86 * (n_vg - 1))
        j_top = int(np.argmax([g[m] for g in data]))
        ax_lin.annotate("$\\mathcal{G}$  linear", xy=(Vg[m], data[j_top][m]),
                        xytext=(-6, 6), textcoords="offset points",
                        color=LIN_COLOR, fontsize=9, fontweight="bold",
                        ha="right")

    fig.tight_layout(rect=(0, 0.16, 1, 1))
    _two_part_legend(fig, cases)
    if savepath:
        fig.savefig(savepath, dpi=200, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        print(f"saved -> {savepath}")
    if show:
        plt.show()
    return fig


def plot_channel_profile(p: TRParams = None, Vd: float = 1.0, Vg: float = 1.0,
                         Vs: float = 0.0, L: float = None, npts: int = 201,
                         cases=None, savepath=None, show: bool = False):
    """phi(Vy) and Vy plotted against position y along the channel.

    Position comes from `channel_profile`; see its docstring for the current-
    continuity argument that turns the quasi-Fermi potential into y/L.

    phi and Vy are both voltages, so this is a single linear axis with two
    series - no log/linear pair and no twin axis.

    `cases` is the usual [(label, TRParams, style_kwargs), ...] list, from
    `make_cases` or `vary_cases`; leave it None for a single-case plot of `p`.
    Color always carries the quantity (blue phi, amber Vy) and, when cases are
    given, line style carries the case - the same split used elsewhere.

    psi = phi + Vy is drawn as a light reference in the single-case plot only;
    with several cases the extra curves cost more than they explain.

    Pass L (in um) to label the x axis in microns instead of y/L; the shape is
    identical either way, since y/L is what the model actually determines.
    """
    import matplotlib.pyplot as plt

    p = p or TRParams()
    single = cases is None
    cases = cases if cases is not None else [("", p, SOLID_STYLE)]
    xlabel = "$y$  ($\\mu$m)" if L else "$y/L$"

    fig, ax = plt.subplots(figsize=(7.6, 4.8) if single else (8.2, 5.2))
    fig.patch.set_facecolor("white")

    first = None
    for lab, pc, style in cases:
        y, Vy, phi = channel_profile(Vd, Vg, pc, Vs=Vs, npts=npts)
        x = y * L if L else y
        ax.plot(x, phi, color=PHI_COLOR, linewidth=2.0, zorder=3,
                label="$\\varphi(V_y)$   surface potential", **style)
        ax.plot(x, Vy, color=VY_COLOR, linewidth=2.0, zorder=3,
                label="$V_y$   quasi-Fermi potential", **style)
        if single:
            ax.plot(x, phi + Vy, color=INK_MUTED, linewidth=1.4,
                    linestyle=":", dashes=(1, 1.8), zorder=2,
                    label="$\\psi = \\varphi + V_y$")
        if first is None:
            first = (x, Vy, phi)

    _style_axes(ax, xlabel, "potential  (V)",
                f"Channel profile   |   $V_\\mathrm{{d}}$ = {Vd:.2f} V, "
                f"$V_\\mathrm{{g}}$ = {Vg:.2f} V, "
                f"$V_\\mathrm{{s}}$ = {Vs:.2f} V")

    x0, Vy0, phi0 = first
    if single:
        # curves are furthest apart at the drain end
        _label(ax, x0[-1], phi0[-1], "$\\varphi$", PHI_COLOR, anchor="right")
        _label(ax, x0[-1], Vy0[-1], "$V_y$", VY_COLOR, anchor="right")
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
        fig.tight_layout()
    else:
        # mid-channel, where the two quantities are well separated and the
        # cases have not yet piled up against the drain-end asymptote
        k = int(0.55 * (npts - 1))
        _label(ax, x0[k], phi0[k], "$\\varphi$", PHI_COLOR, anchor="right")
        _label(ax, x0[k], Vy0[k], "$V_y$", VY_COLOR, anchor="right")
        # the upper-left of this plot is empty, so both legends fit inside the
        # axes - no reserved figure margin and no gap below the x label
        series_h, case_h = _proxy_handles(
            [("$\\varphi(V_y)$   surface potential", PHI_COLOR),
             ("$V_y$   quasi-Fermi potential", VY_COLOR)], cases)
        leg = ax.legend(handles=series_h, frameon=False, fontsize=9,
                        labelcolor=INK, loc="upper left")
        ax.add_artist(leg)
        ax.legend(handles=case_h, frameon=False, fontsize=9, labelcolor=INK,
                  loc="upper left", bbox_to_anchor=(0.0, 0.84))
        fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        print(f"saved -> {savepath}")
    if show:
        plt.show()
    return fig


# Axes a FoM map can sweep: TRParams fields, plus the three bias knobs that are
# arguments to `fom` rather than device parameters.
_BIAS_KEYS = ("Vboost", "Vhold", "Vdd")

_AXIS_LABEL = {
    "Vt": "$V_t$  (V)",
    "Vtr": "$V_\\mathrm{tr}$  (V)",
    "phi_tail": "$\\phi_\\mathrm{tail}$  (V)",
    "n": "$n$",
    "C2D": "$C_\\mathrm{2D}$  (F/cm$^2$)",
    "Cgch": "$C_\\mathrm{gch}$  (F/cm$^2$)",
    "T": "$T$  (K)",
    "Vboost": "$V_\\mathrm{boost}$  (V)",
    "Vhold": "$V_\\mathrm{hold}$  (V)",
    "Vdd": "$V_\\mathrm{dd}$  (V)",
}


def _axis_label(name):
    return _AXIS_LABEL.get(name, name)


def fom_map(p: TRParams = None, Vboost: float = 1.0, Vhold: float = 0.0,
            Vdd: float = 1.0, x=("Vt", (0.0, 0.6)), y=("Vtr", (0.0, 0.6)),
            nx: int = 21, ny: int = 21, npts: int = 101, verbose=True):
    """FoM over a 2-D sweep of any two knobs.  Returns (xv, yv, F) with F (ny, nx).

    `x` and `y` are (name, (lo, hi)) where name is either a TRParams field
    (Vt, Vtr, phi_tail, n, ...) or one of Vboost / Vhold / Vdd.

    Note mu_eff is deliberately NOT available as an axis: tau is defined as
    mu_eff*(W/L)*R_eff, so mu_eff cancels exactly out of the tau_leak/tau_charge
    ratio and a FoM map over it would be flat by construction.

    npts = 101 keeps a 21x21 grid to ~1 min and is accurate to ~0.3% on FoM.
    """
    from dataclasses import replace

    p = p or TRParams()
    (xname, xr), (yname, yr) = x, y
    for nm in (xname, yname):
        if nm not in _BIAS_KEYS and nm not in TRParams.__dataclass_fields__:
            raise ValueError(f"unknown axis {nm!r}")

    xv = np.linspace(*xr, nx)
    yv = np.linspace(*yr, ny)
    F = np.empty((ny, nx))
    for i, yval in enumerate(yv):
        for j, xval in enumerate(xv):
            bias = {"Vboost": Vboost, "Vhold": Vhold, "Vdd": Vdd}
            over = {}
            for nm, val in ((xname, xval), (yname, yval)):
                (bias if nm in _BIAS_KEYS else over)[nm] = float(val)
            F[i, j] = fom(bias["Vboost"], bias["Vhold"], bias["Vdd"],
                          replace(p, **over), npts=npts)
        if verbose:
            print(f"  fom_map row {i + 1}/{ny}  ({yname} = {yval:.3f})",
                  flush=True)
    return xv, yv, F


# Sequential single-hue ramp for the scalar field: light -> dark, one hue.
_SEQ_BLUE_STEPS = ["#F2F7FC", "#C3DCF3", "#8CBFEC", "#4A8CD4", "#255FA0",
                   "#123A63"]


def plot_fom_map(p: TRParams = None, Vboost: float = 1.0, Vhold: float = 0.0,
                 Vdd: float = 1.0, x=("Vt", (0.0, 0.6)),
                 y=("Vtr", (0.0, 0.6)), n_levels: int = 12, savepath=None,
                 show: bool = False, **kw):
    """2-D colour map of FoM = tau_leak(Vhold, Vdd) / tau_charge(Vboost, Vdd).

    `x` / `y` pick the swept knobs, see `fom_map`; everything else is held at
    the value in `p` (or the Vboost/Vhold/Vdd arguments) and named in the title.

    The ramp is a single hue light -> dark (magnitude encoding), quantised into
    filled contour bands that can be matched against the colourbar, with
    labelled lines for exact reading.

    Levels are geometric and the norm is logarithmic.  FoM is a ratio of two
    exponentially-varying times, so depending on the axes it can span several
    decades (Vboost) or only a few x with a heavily skewed spatial distribution
    (Vt) - log bands handle both, linear bands handle neither.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    p = p or TRParams()
    xname, yname = x[0], y[0]
    xv, yv, F = fom_map(p, Vboost=Vboost, Vhold=Vhold, Vdd=Vdd, x=x, y=y, **kw)
    cmap = LinearSegmentedColormap.from_list("tr_seq_blue", _SEQ_BLUE_STEPS)

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    fig.patch.set_facecolor("white")

    levels = np.geomspace(F.min(), F.max(), n_levels + 1)
    cf = ax.contourf(xv, yv, F, levels=levels, cmap=cmap,
                     norm=LogNorm(F.min(), F.max()))
    cs = ax.contour(xv, yv, F, levels=levels[1::2], colors="#5A6B7C",
                    linewidths=0.8, alpha=0.85)
    ax.clabel(cs, fmt="%.3g", fontsize=8, colors="#33414F")

    # everything held fixed, named so the map is self-describing
    held = {"Vboost": Vboost, "Vhold": Vhold, "Vdd": Vdd,
            "Vt": p.Vt, "Vtr": p.Vtr, "phi_tail": p.phi_tail}
    fixed = "  ".join(f"{_AXIS_LABEL[k].split('  ')[0]} = {v:g} V"
                      for k, v in held.items() if k not in (xname, yname))

    _style_axes(ax, _axis_label(xname), _axis_label(yname),
                "FoM = $\\tau_\\mathrm{leak}(V_\\mathrm{hold}, V_\\mathrm{dd})"
                " / \\tau_\\mathrm{charge}(V_\\mathrm{boost}, V_\\mathrm{dd})$"
                f"\nfixed:  {fixed}")
    ax.grid(False)  # contour bands carry the reading; a grid on top is noise

    # one tick per band is unreadable past ~12 bands; thin them instead
    ticks = levels if n_levels <= 12 else levels[::2]
    cb = fig.colorbar(cf, ax=ax, pad=0.02, format="%.3g", ticks=ticks)
    cb.set_label("FoM  (dimensionless, higher = better)", color=INK, fontsize=9)
    cb.ax.tick_params(colors=INK_MUTED, labelsize=8.5)
    cb.outline.set_visible(False)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        print(f"saved -> {savepath}")
    if show:
        plt.show()
    return fig


def plot_gm_over_id(p: TRParams = None, Vd: float = 0.05, Vs: float = 0.0,
                    Vg_min: float = -0.25, Vg_max: float = 1.25,
                    n_vg: int = 241, npts: int = 201, cases=None,
                    savepath=None, show: bool = False):
    """gm/Id and its Vg-derivative, one curve per case.

    (a) gm/Id = d ln(G)/dVg           [1/V]
    (b) d(gm/Id)/dVg = d2 ln(G)/dVg2  [1/V^2]

    Defaults to the linear-regime bias Vd = 0.05 V, Vs = 0.  See `gm_over_id`:
    mu_eff and W/L cancel, so these compare the cases on transport efficiency
    alone.  Tile (b) is a second derivative of ln G; n_vg = 241 keeps it
    converged (the peak moves <1.5% between 121 and 241 points) and its
    point-to-point jitter under ~1% of the signal.

    Encoding differs from the other comparison plots on purpose.  There is only
    one quantity per tile, so colour is free to carry the case instead of the
    scale; the CASE_STYLES dashes are kept as a redundant second encoding.
    """
    import matplotlib.pyplot as plt

    p = p or TRParams()
    cases = cases if cases is not None else vary_cases(p, "Vtr", 0.0,
                                                       tex="V_\\mathrm{tr}")
    Vg = np.linspace(Vg_min, Vg_max, n_vg)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.6, 5.0))
    fig.patch.set_facecolor("white")

    ideal = 1.0 / (p.n * p.phiT)
    ax_a.axhline(ideal, color=INK_MUTED, linewidth=1.3, linestyle=":",
                 dashes=(1, 1.8), zorder=2,
                 label=f"$1/(n\\phi_T)$ = {ideal:.1f} V$^{{-1}}$  (ideal limit)")
    ax_b.axhline(0.0, color=INK_MUTED, linewidth=1.0, zorder=2)

    handles = []
    for (lab, pc, style), color in zip(cases, CASE_COLORS):
        gmid, _ = gm_over_id(Vg, Vd, Vs, pc, npts=npts)
        (ln,) = ax_a.plot(Vg, gmid, color=color, linewidth=2.0, zorder=3,
                          label=lab, **style)
        ax_b.plot(Vg, np.gradient(gmid, Vg), color=color, linewidth=2.0,
                  zorder=3, label=lab, **style)
        handles.append(ln)

    _style_axes(ax_a, "$V_\\mathrm{g}$  (V)", "$g_m/I_d$  (V$^{-1}$)",
                "(a)  $g_m/I_d$")
    ax_a.set_ylim(0, ideal * 1.12)
    _style_axes(ax_b, "$V_\\mathrm{g}$  (V)",
                "$d(g_m/I_d)/dV_\\mathrm{g}$  (V$^{-2}$)",
                "(b)  $d(g_m/I_d)/dV_\\mathrm{g}$")

    fig.suptitle(f"$V_\\mathrm{{d}}$ = {Vd:.2f} V,  "
                 f"$V_\\mathrm{{s}}$ = {Vs:.2f} V", color=INK,
                 fontsize=10.5, y=1.0)

    handles = [ax_a.lines[0]] + handles  # ideal-limit line first
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.legend(handles, [h.get_label() for h in handles], frameon=False,
               fontsize=9, labelcolor=INK, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.0))

    if savepath:
        fig.savefig(savepath, dpi=200, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        print(f"saved -> {savepath}")
    if show:
        plt.show()
    return fig


def plot_f_t(p: TRParams = None, Vd: float = 0.05, Vs: float = 0.0,
             Vg_min: float = -0.25, Vg_max: float = 1.25, n_vg: int = 241,
             npts: int = 201, mu_eff: float = 100.0, W_um: float = 1.0,
             L_um: float = 1.0, L_ov_um: float = 1.0, cases=None,
             savepath=None, show: bool = False):
    """gm, Cgg and fT vs Vg, one curve per case.  See `f_t`.

    (a) gm  = mu_eff*(W/L)*dG/dVg                        [uS]
    (b) Cgg = W*L*d(Q_free+Q_tail)/dVg + Cgch*L_ov*W     [fF]
    (c) fT  = gm/Cgg on the left axis, overlaid against gm/Id on the right

    Tile (c) is the analog design trade-off: speed against transport
    efficiency, both against Vg.  gm/Id is mobility- and geometry-free while
    fT is not, so they cannot share a scale - hence the twin axis, which is
    otherwise worth avoiding.  Each axis names its own line style so no reader
    has to guess which curve belongs to which scale.

    Encoding differs from tiles (a)/(b) in one respect: here line style has to
    carry the QUANTITY (solid fT, dashed gm/Id), so it cannot also carry the
    case.  Colour therefore carries the case throughout this figure and the
    CASE_STYLES dashes are not used - the three case colours pass CVD
    validation on their own (worst-case dE 16.2).

    All axes are LINEAR.  These quantities span decades over this Vg range, so
    everything below roughly Vg = 0.3 V sits on the baseline.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    p = p or TRParams()
    cases = cases if cases is not None else vary_cases(p, "Vtr", 0.0,
                                                       tex="V_\\mathrm{tr}")
    Vg = np.linspace(Vg_min, Vg_max, n_vg)

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))
    fig.patch.set_facecolor("white")
    ax_r = axes[2].twinx()

    for (lab, pc, _style), color in zip(cases, CASE_COLORS):
        ft, Cgg, gm, Id = f_t(Vg, Vd, Vs, pc, mu_eff=mu_eff, W_um=W_um,
                              L_um=L_um, L_ov_um=L_ov_um, npts=npts)
        axes[0].plot(Vg, gm * 1e6, color=color, linewidth=2.0, zorder=3)
        axes[1].plot(Vg, Cgg * 1e15, color=color, linewidth=2.0, zorder=3)
        axes[2].plot(Vg, ft, color=color, linewidth=2.0, zorder=3)
        # gm/Id straight from the same pass, no extra solve
        ax_r.plot(Vg, gm / Id, color=color, linewidth=2.0, linestyle="--",
                  dashes=(5, 3), zorder=3)

    for ax, (ylab, tag) in zip(axes, [
            ("$g_m$  ($\\mu$S)", "(a)  $g_m$"),
            ("$C_\\mathrm{gg}$  (fF)", "(b)  $C_\\mathrm{gg}$"),
            ("$f_T = g_m/C_\\mathrm{gg}$  (s$^{-1}$)   [solid]",
             "(c)  $f_T$  vs  $g_m/I_d$")]):
        _style_axes(ax, "$V_\\mathrm{g}$  (V)", ylab, tag)
        ax.set_ylim(bottom=0)

    ax_r.set_ylabel("$g_m/I_d$  (V$^{-1}$)   [dashed]", color=INK, fontsize=10)
    ax_r.tick_params(axis="y", colors=INK_MUTED, labelsize=9, length=3)
    ax_r.set_ylim(bottom=0)
    ax_r.grid(False)
    for side in ("top", "left"):
        ax_r.spines[side].set_visible(False)
    ax_r.spines["right"].set_color(GRID)

    fig.suptitle(f"$V_\\mathrm{{d}}$ = {Vd:.2f} V,  "
                 f"$V_\\mathrm{{s}}$ = {Vs:.2f} V,  "
                 f"$\\mu_\\mathrm{{eff}}$ = {mu_eff:g} cm$^2$/V$\\cdot$s,  "
                 f"$W$ = {W_um:g} $\\mu$m,  $L$ = {L_um:g} $\\mu$m,  "
                 f"$L_\\mathrm{{ov}}$ = {L_ov_um:g} $\\mu$m",
                 color=INK, fontsize=10.5, y=1.0)

    case_h = [Line2D([], [], color=c, lw=2.0, label=lab)
              for (lab, _, _), c in zip(cases, CASE_COLORS)]
    quant_h = [Line2D([], [], color=INK_MUTED, lw=2.0,
                      label="$f_T$  (left)"),
               Line2D([], [], color=INK_MUTED, lw=2.0, linestyle="--",
                      dashes=(5, 3), label="$g_m/I_d$  (right, tile c)")]
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.legend(handles=case_h, frameon=False, fontsize=9, labelcolor=INK,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.055))
    fig.legend(handles=quant_h, frameon=False, fontsize=9, labelcolor=INK,
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))

    if savepath:
        fig.savefig(savepath, dpi=200, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        print(f"saved -> {savepath}")
    if show:
        plt.show()
    return fig


# --------------------------------------------------------------------------- #
def _self_check(p: TRParams):
    """Sanity checks on the limiting behaviour of the model pieces."""
    # F_tail limits
    assert abs(f_tail(5.0, p) - 1.0) < 1e-6, f_tail(5.0, p)
    assert f_tail(-5.0, p) < 1e-3, f_tail(-5.0, p)
    # asymptotic branch must join the hyp2f1 branch smoothly
    lhs = hyp2f1(1.0, p.a, 1.0 + p.a, -1e8)
    rhs = float(_f_tail_asymptotic(np.log(1e8), p.a))
    assert abs(lhs - rhs) / lhs < 1e-6, (lhs, rhs)
    # deep-subthreshold limit F_tail ~ exp(phi/phi_tail) (Urbach tail)
    r = f_tail(-0.6, p) / f_tail(-0.7, p)
    assert abs(r / np.exp(0.1 / p.phi_tail) - 1.0) < 1e-3, r
    # charge balance is actually satisfied at the solved phi
    for Vy in (0.0, 0.25, 0.6):
        ph = solve_phi_scalar(Vy, 0.8, p)
        res = (p.Cgch * (0.8 - p.Vt - p.n * ph - p.n * Vy)
               - q_free(ph, p) - q_tail(ph, p))
        assert abs(res) < 1e-14, (Vy, ph, res)
    # G endpoints and monotonicity
    assert abs(G(0.6, 0.6, 0.8, p)) < 1e-12
    gs = G_curve(np.linspace(0, 0.6, 21), 0.6, 0.8, p)
    assert np.all(np.diff(gs) < 0)
    # The running tau must reproduce the closed-form tau(Vd) at Vs = 0.9*Vd.
    # Both sides must use the SAME grid: tau() integrates on its own
    # linspace(0, VS_FRAC*Vd, npts), so feeding tau_curve a different number of
    # points compares two discretisations rather than checking the identity,
    # and the residual is then just quadrature error (~1e-3 at npts=201).
    N = 201
    vs = np.linspace(0.0, VS_FRAC * 0.6, N)
    tc = tau_curve(vs, 0.6, 0.6, p, npts=N)
    assert tc[0] == 0.0 and np.all(np.diff(tc) > 0)
    ref = tau_charge(0.6, 0.6, p, npts=N)
    assert abs(tc[-1] / ref - 1.0) < 1e-12, (tc[-1], ref)
    # tau_leak sweeps the DRAIN over [0.9Vd, Vd] with Vs = 0, so its integrand
    # G(Vg, V, 0) stays close to the full-channel G(Vg, Vd, 0) instead of
    # collapsing.  Cross-check against the rectangle rule over that narrow
    # window, and against G_curve evaluated the long way at both ends.
    tl = tau_leak(1.0, 1.0, p, npts=401)
    g_lo = G(VS_FRAC * 1.0, 0.0, 1.0, p, npts=401)
    g_hi = G(1.0, 0.0, 1.0, p, npts=401)
    assert g_hi > g_lo > 0, (g_lo, g_hi)
    assert (1 - VS_FRAC) / g_hi < tl < (1 - VS_FRAC) / g_lo, (tl, g_lo, g_hi)
    # FoM is the ratio of the two, but charge is at Vboost and leak at Vhold
    fv = fom(1.0, 0.0, 1.0, p, npts=401)
    ratio = (tau_leak(0.0, 1.0, p, npts=401)
             / tau_charge(1.0, 1.0, p, npts=401))
    assert abs(fv / ratio - 1) < 1e-12, (fv, ratio)
    # channel position map: monotonic 0 -> 1, endpoints pinned to Vs and Vd
    yy, vy, ph = channel_profile(1.0, 1.0, p, Vs=0.0, npts=201)
    assert yy[0] == 0.0 and abs(yy[-1] - 1.0) < 1e-12
    assert np.all(np.diff(yy) > 0), "y(V) must increase along the channel"
    assert vy[0] == 0.0 and abs(vy[-1] - 1.0) < 1e-12
    assert np.all(np.diff(ph) < 0), "phi must fall from source to drain"
    print("self-check: OK")


def main():
    from dataclasses import replace

    p = TRParams()
    _self_check(p)

    # Vd, Vg = 0.6, 0.8
    # print(f"\nphiT = {p.phiT*1e3:.2f} mV,  a = phiT/phi_tail = {p.a:.3f},  "
    #       f"C2D/Cgch = {p.C2D/p.Cgch:.1f}")
    # print(f"\nVg = {Vg} V, Vd = {Vd} V")
    # print(f"{'Vs [V]':>8} {'phi(Vs) [V]':>12} {'Qfree [C/cm2]':>15} "
    #       f"{'G [C.V/cm2]':>14} {'Id [A] (mu=100, W/L=1)':>24}")
    # for vs in np.linspace(0.0, VS_FRAC * Vd, 7):
    #     ph = solve_phi_scalar(vs, Vg, p)
    #     print(f"{vs:8.3f} {ph:12.5f} {float(q_free(ph, p)):15.4e} "
    #           f"{G(Vd, vs, Vg, p):14.4e} {drain_current(Vd, vs, Vg, p):24.4e}")

    Vb = 1.0  # diode-connected bias for the two-tile figure
    Vs = np.linspace(0.0, VS_FRAC * Vb, 401)
    gc = G_curve(Vs, Vb, Vb, p)
    tc = tau_curve(Vs, Vb, Vb, p)
    print(f"\ndiode-connected Vd = Vg = {Vb} V, Vs up to {VS_FRAC:g}*Vd")
    print(f"{'Vs [V]':>8} {'G [C.V/cm2]':>14} {'tau(Vd,Vs) [cm2/C]':>20}")
    for k in np.linspace(0, len(Vs) - 1, 7).astype(int):
        print(f"{Vs[k]:8.3f} {gc[k]:14.4e} {tc[k]:20.4e}")
    # the running integral must land on the closed-form tau(Vd)
    print(f"tau_curve endpoint {tc[-1]:.5e} vs "
          f"tau_charge() {tau_charge(Vb, Vb, p, npts=len(Vs)):.5e} cm^2/C")

    d = Path(__file__).parent
    # # plot_G_vs_Vs(p, Vd=Vd, Vg=Vg, savepath=d / "TR_G_vs_Vs.png")
    # plot_G_tau_vs_Vs(p, Vd=Vb, savepath=d / "TR_G_tau_vs_Vs.png")
    # plot_G_tau_compare(p, Vd=Vb,
    #                    savepath=d / "TR_G_tau_vtr_compare.png")
    # plot_G_vs_Vg(p, Vd_list=(1.0, 0.05), Vs=0.0, Vg_min=0.0, Vg_max=1.0,
    #              savepath=d / "TR_G_vs_Vg.png")

    # # same two comparisons, but varying the band-tail width instead of Vtr
    # pt = vary_cases(p, "phi_tail", 0.09, tex="\\phi_\\mathrm{tail}")
    # plot_G_tau_compare(p, Vd=Vb, cases=pt,
    #                    savepath=d / "TR_G_tau_phitail_compare.png")
    # plot_G_vs_Vg(p, Vd_list=(1.0, 0.05), Vs=0.0, Vg_min=0.0, Vg_max=1.0,
    #              cases=pt, savepath=d / "TR_G_vs_Vg_phitail.png")

    # all three together: no tail charge, and Vtr on with either tail width.
    # phi_tail is irrelevant when Vtr = 0, so that case needs no tail label.
    three = make_cases(p, [
        ("$V_\\mathrm{tr}$ = 0", {"Vtr": 0.0}),
        ("$V_\\mathrm{tr}$ = 0.3 V, $\\phi_\\mathrm{tail}$ = 0.045 V",
         {"Vtr": 0.3, "phi_tail": 0.045}),
        ("$V_\\mathrm{tr}$ = 0.3 V, $\\phi_\\mathrm{tail}$ = 0.09 V",
         {"Vtr": 0.3, "phi_tail": 0.09}),
    ])
    # plot_G_tau_compare(p, Vd=Vb, Vg=Vb+0.3, cases=three,
    #                    savepath=d / "TR_G_tau_3case_compare.png")
    # plot_G_vs_Vg(p, Vd_list=(1.0, 0.05), Vs=0.0, Vg_min=-0.25, Vg_max=1.25,
    #              cases=three, savepath=d / "TR_G_vs_Vg_3case.png")

    # plot_channel_profile(p, Vd=1.0, Vg=1.0, Vs=0.0, cases=three,
    #                      savepath=d / "TR_channel_profile_3cases.png")
    # plot_channel_profile(p, Vd=0.05, Vg=1.0, Vs=0.0, cases=three,
    #                      savepath=d / "TR_channel_profile_3case.png")

    print(f"\ntau_charge(Vg=1.0, Vd=1.0) = {tau_charge(1.0, 1.0, p):.4e} cm^2/C")
    print(f"tau_leak  (Vg=0.0, Vd=1.0) = {tau_leak(0.0, 1.0, p):.4e} cm^2/C")
    print(f"FoM(Vboost=1.0, Vhold=0.0, Vdd=1.0) = {fom(1.0, 0.0, 1.0, p):.4f}")
    plot_gm_over_id(p, Vd=1.0, Vs=0.0, cases=three,
                    savepath=d / "TR_gm_over_id.png")
    plot_f_t(p, Vd=0.05, Vs=0.0, mu_eff=100.0, W_um=1.0, L_um=1.0,
             L_ov_um=1.0, cases=three, savepath=d / "TR_fT.png")
    derivative_extrema(p, Vd=0.05, Vs=0.0, mu_eff=100.0, W_um=1.0, L_um=1.0,
                       L_ov_um=1.0, cases=three)

    # plot_fom_map(p, Vboost=1.5, Vhold=0.0, Vdd=1.0,
    #              savepath=d / "TR_fom_map.png")
    # # Vtr vs Vboost at fixed Vt: FoM is mu_eff-independent by construction
    # # (mu cancels out of the tau ratio), so Vboost takes that axis instead.
    # plot_fom_map(replace(p, Vt=0.3), Vhold=0.0, Vdd=1.0,
    #              x=("Vboost", (1.0, 1.6)), y=("Vtr", (0.0, 0.6)),
    #              savepath=d / "TR_fom_map_Vtr_Vboost.png")
    # # ~10 decades of FoM across 0.6 V of hold bias, hence the extra levels.
    # plot_fom_map(replace(p, Vt=0.3), Vboost=1.5, Vdd=1.0,
    #              x=("Vhold", (-0.6, 0.0)), y=("Vtr", (0.0, 0.6)),
    #              n_levels=20, savepath=d / "TR_fom_map_Vtr_Vhold.png")
    # # phi_tail >= 0.03 V keeps a = phiT/phi_tail below 1: a tail narrower than
    # # kT is not a physical Urbach tail, and integer a is a pole of the 2F1 form
    # # (guarded in _f_tail_asymptotic, but no reason to sweep through it).
    # plot_fom_map(replace(p, Vt=0.3), Vboost=1.5, Vhold=0.0, Vdd=1.0,
    #              x=("phi_tail", (0.03, 0.12)), y=("Vtr", (0.0, 0.6)),
    #              savepath=d / "TR_fom_map_Vtr_phitail.png")


if __name__ == "__main__":
    main()
