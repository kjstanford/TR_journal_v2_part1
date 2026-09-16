"""OO implementation of the OSFET band-tail model, Id only.

Sources:
  * scratch/TR_related_equations.pdf  -- the TR charge/current formulation
  * References/195702_1_5.0233899.pdf -- Kang, Toprasertpong, Oka, Mori,
    Takenaka, Takagi, J. Appl. Phys. 136, 195702 (2024), which supplies the
    mobile/localized split of the band-tail DOS.
  * References/An_All-Region_BSIM_Thin-Film_Transistor_Model_for_Display_and
    _BEOL_3-D_Integration_Applications.pdf -- Pahwa, Salahuddin, Hu, IEEE
    TED 71, 2024, which supplies the floating tail density (its N_trap, Eq. 3)
    and the power-law temperature dependence of the tail width (its Eq. 28).

DOS model (energies referred to the band edge E0, expressed as voltages,
eps = (E - E0)/q, so eps < 0 is inside the tail):

    band states      D(eps) = D2D                                 eps > 0
    whole tail       D(eps) = D_tail0*exp(eps/phi_tail)           eps < 0

The tail amplitude is set by the FLOATING areal tail-state density

    Ntail = D_tail0 * phi_tail          [cm^-2]

i.e. the tail DOS integrated over all energy.  D_tail0 = D_tail(E0) is then
Ntail/phi_tail, so it is Ntail -- not the band-edge DOS -- that is held fixed.
Setting Ntail = C2D*phi_tail/q recovers the Kang-paper convention in which the
tail is pinned to the band-edge DOS (D_tail0 = D2D).

The tail is split at eps = -phi_ref into MOBILE states (shallow, carry
current) and LOCALIZED states (deep, screen the gate but do not conduct):

    D_mobile(eps) = D2D*exp(eps/phi_tail)                  -phi_ref < eps < 0
    D_mobile(eps) = D2D*exp(-phi_ref/phi_tail)
                         * exp((eps + phi_ref)/phi_b)           eps < -phi_ref
    D_local(eps)  = D(eps) - D_mobile(eps)                      eps < -phi_ref

with phi_b < phi_tail, so the mobile DOS decays faster than the whole tail
below the boundary.  Occupancy is full Fermi-Dirac throughout.

Charge densities (C/cm^2), with C2D = q^2*D2D, Ctail0 = q*D_tail0 = q*Ntail/
phi_tail (the tail analogue of C2D), and phi = (EF - E0)/q:

    Qfree(phi)   = C2D*phiT*log(1 + exp(phi/phiT))
    Qtail(phi)   = q*Ntail*F(phi_tail, phi)
    Qmob(phi)    = Ctail0*[ phi_tail*F(phi_tail, phi)
                            + exp(-phi_ref/phi_tail)
                              * (phi_b*F(phi_b, phi + phi_ref)
                                 - phi_tail*F(phi_tail, phi + phi_ref)) ]
    Qloc(phi)    = Qtail(phi) - Qmob(phi)

    F(w, x) = 2F1(1, phiT/w; 1 + phiT/w; -exp(-x/phiT))

F(w, x) is exactly the Fermi-Dirac integral over an exponential tail of width
w whose amplitude is 1 at its upper edge, divided by w; it rises monotonically
from 0 to 1.  `mobile_tail` switches the split on; when it is False the ENTIRE
exponential tail is given zero mobility and only Qfree conducts.

`tail_temp_dep` switches on the tail-width temperature dependence of Eq. (28)
of the BSIM TFT paper, T_trap(T) = T_trap0*(T/T0)**T_trapt, which in the
voltage units used here reads

    phi_tail(T) = phi_tail * (T/T0)**T_trapt

with `phi_tail` the value at the reference temperature T0.  Two normalizations
of the narrowing tail are available, selected by `tail_fix_Dtail`:

  tail_fix_Dtail = False (default)   Ntail is held constant, so
      D_tail0 = Ntail/phi_tail(T) moves inversely -- a tail that narrows gets
      correspondingly taller, conserving the total number of tail states.  The
      tail then still holds a fixed charge q*Ntail however cold it gets, so it
      fills up once the gate has swept q*Ntail/Cgch past Vt and its
      capacitance then collapses.

  tail_fix_Dtail = True              D_tail0 is held constant at its T0 value
      Ntail/phi_tail, so the effective density Ntail(T) = D_tail0*phi_tail(T)
      shrinks with the width -- tail states are removed rather than compressed,
      and Ctail0 no longer grows at low T.

Only the product D_tail0*phi_tail(T) enters the charge, so the two differ
solely in how much tail charge survives at low T; they coincide at T = T0.

Charge balance, which implicitly defines phi(Vy):

    Cgch*[Vg - Vt - n*phi(Vy) - n*Vy] = Qfree(phi(Vy)) + Qtail(phi(Vy))

Note the gate sees the TOTAL tail charge whether or not it conducts, so the
balance equation -- and hence phi(Vy) -- does not depend on `mobile_tail`.
Only the current integrand does:

    Qcond(phi)     = Qfree(phi) + Qmob(phi)      (Qmob = 0 if not mobile_tail)
    G(Vg, Vd, Vs)  = int_{Vs}^{Vd} Qcond(phi(Vy)) dVy
    Id(Vg, Vd, Vs) = mu_eff * (W/L) * G(Vg, Vd, Vs)

Unit convention (self-consistent, no extra factors needed):
    voltages             [V]
    capacitance per area [F/cm^2]
    charge per area      [C/cm^2]
    G                    [C/cm^2 * V]
    mobility             [cm^2/(V.s)]  ->  Id in [A]
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import hyp2f1

K_B = 1.380649e-23  # J/K
Q_E = 1.602176634e-19  # C

# Ntail that pins the tail to the band-edge DOS (D_tail0 = D2D) at the default
# C2D and phi_tail, i.e. the Kang-paper convention.  Used as the Ntail default
# so that the shipped parameter set is self-consistent.
NTAIL_PINNED = 2.0e-5 * 0.045 / Q_E  # 5.617e12 cm^-2


@dataclass
class OSFETBandTailParams:
    """Device / model parameters for the OSFET band-tail Id model.

    The tail amplitude is set by `Ntail`, the total areal density of tail
    states, which floats independently of the band-edge DOS `C2D` (the N_trap
    of the BSIM TFT paper).  Its default `NTAIL_PINNED` happens to pin the tail
    to the band-edge DOS, reproducing the Kang-paper convention.

    `phi_ref` and `phi_b` default to the Kang paper's fitted ratios (its
    Fig. 12: |Eref| ~ 0.5*Wt, Wb ~ 0.37*Wt) applied to `phi_tail`.  They are
    inactive unless `mobile_tail` is True.

    `T_trapt` defaults to 0.511, the a-IGZO value fitted in Fig. 10 of the BSIM
    TFT paper.  It, `T0` and `tail_fix_Dtail` are inactive unless
    `tail_temp_dep` is True.
    """

    C2D: float = 2.0e-5       # band-edge DOS capacitance q^2*D2D [F/cm^2]
    Cgch: float = 3.45e-6     # gate-to-channel capacitance [F/cm^2]
    n: float = 1.10           # body factor [-]
    Vt: float = 0.30          # threshold voltage [V]
    Ntail: float = NTAIL_PINNED  # total areal tail-state density [cm^-2]
    phi_tail: float = 0.045   # whole-tail width Wt/q at T0 [V]
    phi_ref: float = 0.0225   # mobile/localized boundary |Eref|/q [V]
    phi_b: float = 0.0165     # mobile-tail width Wb/q [V], must be < phi_tail
    mobile_tail: bool = False  # False -> whole tail has zero mobility
    tail_temp_dep: bool = False  # True -> phi_tail follows Eq. (28)
    tail_fix_Dtail: bool = False  # True -> hold D_tail0 fixed, not Ntail
    T0: float = 300.0         # reference temperature for Eq. (28) [K]
    T_trapt: float = 0.511    # power-law exponent of Eq. (28) [-]
    T: float = 300.0          # temperature [K]
    mu_eff: float = 100.0     # effective mobility [cm^2/(V.s)]
    W_over_L: float = 1.0     # channel width-to-length ratio [-]

    def __post_init__(self):
        if self.T <= 0.0:
            raise ValueError(f"T must be > 0, got {self.T}")
        if self.tail_temp_dep and self.T0 <= 0.0:
            raise ValueError(f"T0 must be > 0, got {self.T0}")
        if self.phi_ref < 0.0:
            raise ValueError(f"phi_ref must be >= 0, got {self.phi_ref}")
        if self.Ntail < 0.0:
            raise ValueError(f"Ntail must be >= 0, got {self.Ntail}")
        # Without a temperature-dependent width the two normalizations are
        # algebraically identical (phi_tail_eff == phi_tail), so accepting the
        # flag on its own would silently do nothing.
        if self.tail_fix_Dtail and not self.tail_temp_dep:
            raise ValueError(
                "tail_fix_Dtail only has meaning together with tail_temp_dep: "
                "with a fixed tail width, holding D_tail0 fixed and holding "
                "Ntail fixed are the same thing.")
        # The two switches are alternative explanations of the same observable
        # (the cryogenic SS floor) and are not physically composable: Eq. (28)
        # rescales the tail width with T while the mobile/localized split
        # assumes a mobility edge fixed in energy, so combining them makes
        # phi_ref and phi_b drift with temperature for no stated reason.
        if self.mobile_tail and self.tail_temp_dep:
            raise ValueError(
                "mobile_tail and tail_temp_dep are mutually exclusive: the "
                "mobility edge (phi_ref, phi_b) is defined at a fixed energy, "
                "so it is not meaningful to also rescale the tail width with "
                "temperature via Eq. (28).  Enable exactly one.")
        # phi_b > phi_tail would make the mobile DOS exceed the whole tail
        # below the boundary, i.e. a negative localized density.  (phi_tail_eff
        # == phi_tail whenever mobile_tail is set, since the switches are
        # mutually exclusive.)
        if self.mobile_tail and self.phi_b > self.phi_tail:
            raise ValueError(
                f"need phi_b <= phi_tail, got phi_b={self.phi_b}, "
                f"phi_tail={self.phi_tail}")

    @property
    def phiT(self) -> float:
        """Thermal voltage kT/q [V]."""
        return K_B * self.T / Q_E

    @property
    def phi_tail_eff(self) -> float:
        """Tail width at the device temperature [V].

        Equals `phi_tail` unless `tail_temp_dep` is set, in which case it
        follows Eq. (28) of the BSIM TFT paper,
        phi_tail(T) = phi_tail * (T/T0)**T_trapt.
        """
        if not self.tail_temp_dep:
            return self.phi_tail
        return self.phi_tail * (self.T / self.T0) ** self.T_trapt

    @property
    def Ntail_eff(self) -> float:
        """Areal tail-state density at the device temperature [cm^-2].

        Equals `Ntail` unless `tail_fix_Dtail` is set, in which case the
        band-edge tail DOS is what is held fixed and the density follows the
        shrinking width, Ntail(T) = (Ntail/phi_tail)*phi_tail_eff.
        """
        if not self.tail_fix_Dtail:
            return self.Ntail
        return self.Ntail * self.phi_tail_eff / self.phi_tail

    @property
    def Ctail0(self) -> float:
        """Band-edge tail-DOS capacitance q*D_tail(E0) [F/cm^2].

        Equals q*Ntail_eff/phi_tail_eff.  Directly comparable to `C2D`, and
        equal to it when the tail is pinned to the band-edge DOS.  At fixed
        Ntail it varies inversely with `phi_tail_eff`, so a tail narrowed by
        `tail_temp_dep` gets taller; under `tail_fix_Dtail` it is constant.
        """
        return Q_E * self.Ntail_eff / self.phi_tail_eff


class OSFETBandTailModel:
    """Band-tail drain-current model built on `OSFETBandTailParams`."""

    def __init__(self, params: OSFETBandTailParams = None):
        self.p = params or OSFETBandTailParams()

    # ----------------------------------------------------------------- #
    # Tail occupancy factor
    # ----------------------------------------------------------------- #
    def _hyp_asymptotic(self, logx, a: float, n_terms: int = 6):
        """Large-x expansion of 2F1(1, a; 1+a; -x), parameterised by log(x).

            2F1 = a*x^-a*pi/sin(pi*a) - a * sum_{k>=0} (-1)^k / ((1+k-a) * x^(k+1))

        Taking log(x) keeps every power finite (they underflow to 0) for x
        beyond the float range, which is exactly where the expansion is most
        accurate.
        """
        logx = np.asarray(logx, dtype=float)
        # Integer a is a removable pole of this form (the pi/sin(pi*a)
        # prefactor and the k = a - 1 series term blow up and cancel); nudge
        # off it rather than dividing by sin(pi*a) = 0.
        if abs(a - round(a)) < 1e-9 and round(a) >= 1:
            a = round(a) + 1e-9
        lead = a * np.exp(-a * logx) * np.pi / np.sin(np.pi * a)
        xinv = np.exp(-logx)
        series = np.zeros_like(logx)
        for k in range(n_terms):
            series += ((-1.0) ** k) / (1.0 + k - a) * xinv ** (k + 1)
        return lead - a * series

    def _F(self, w: float, x):
        """F(w, x) = 2F1(1, phiT/w; 1 + phiT/w; -exp(-x/phiT)).

        The Fermi-Dirac integral over an exponential tail of width `w` and unit
        amplitude at its upper edge, divided by `w`.  Monotonically increasing
        in x, F(-inf) = 0, F(+inf) = 1, with the deep-subthreshold limit
        F -> exp(x/w).
        """
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)

        a = self.p.phiT / w
        u = -x / self.p.phiT  # log of the hypergeometric argument magnitude
        out = np.empty_like(x)

        direct = u < np.log(1e8)
        if np.any(direct):
            out[direct] = hyp2f1(1.0, a, 1.0 + a, -np.exp(u[direct]))
        if np.any(~direct):
            out[~direct] = self._hyp_asymptotic(u[~direct], a)

        out = np.clip(out, 0.0, 1.0)
        return float(out[0]) if scalar else out

    def f_tail(self, phi):
        """Whole-tail occupancy factor F(phi_tail_eff, phi), in [0, 1]."""
        return self._F(self.p.phi_tail_eff, phi)

    # ----------------------------------------------------------------- #
    # Charge models
    # ----------------------------------------------------------------- #
    def q_free(self, phi):
        """Free (band-state) charge density Qfree(phi) [C/cm^2].

        Uses logaddexp so phi/phiT >> 1 does not overflow: for large phi this
        tends to C2D*phi, for phi << 0 it tends to C2D*phiT*exp(phi/phiT).
        """
        p = self.p
        phi = np.asarray(phi, dtype=float)
        return p.C2D * p.phiT * np.logaddexp(0.0, phi / p.phiT)

    def q_tail(self, phi):
        """Total band-tail charge density Qtail(phi) = q*Ntail_eff*F [C/cm^2].

        Screens the gate regardless of whether the states conduct, so this is
        what enters the charge-balance equation.  Saturates at q*Ntail_eff once
        the whole tail is filled, independently of `phi_tail_eff`.
        """
        p = self.p
        return Q_E * p.Ntail_eff * self._F(p.phi_tail_eff, phi)

    def q_tail_mobile(self, phi):
        """Mobile (current-carrying) part of the tail charge [C/cm^2].

        Zero when `mobile_tail` is False, i.e. the whole tail is immobile.
        """
        p = self.p
        phi = np.asarray(phi, dtype=float)
        if not p.mobile_tail:
            return np.zeros_like(phi)
        pt = p.phi_tail_eff
        shifted = phi + p.phi_ref
        return p.Ctail0 * (
            pt * self._F(pt, phi)
            + np.exp(-p.phi_ref / pt)
            * (p.phi_b * self._F(p.phi_b, shifted)
               - pt * self._F(pt, shifted)))

    def q_tail_localized(self, phi):
        """Localized (non-conducting) part of the tail charge [C/cm^2]."""
        return self.q_tail(phi) - self.q_tail_mobile(phi)

    def q_cond(self, phi):
        """Conducting charge density: band states plus mobile tail [C/cm^2]."""
        return self.q_free(phi) + self.q_tail_mobile(phi)

    # ----------------------------------------------------------------- #
    # Surface-potential solver
    # ----------------------------------------------------------------- #
    def _H(self, phi):
        """Strictly increasing LHS of n*phi + (Qfree + Qtail)/Cgch = B."""
        p = self.p
        return p.n * phi + (self.q_free(phi) + self.q_tail(phi)) / p.Cgch

    def solve_phi_scalar(self, Vy: float, Vg: float, xtol: float = 1e-14) -> float:
        """Solve the charge-balance equation for phi at a single channel point Vy."""
        p = self.p
        B = Vg - p.Vt - p.n * Vy

        hi = max(B / p.n, 0.0)
        while self._H(hi) < B:  # numerical safety net
            hi += 1.0

        lo = min(B / p.n, 0.0) - 1.0
        for _ in range(200):
            if self._H(lo) <= B:
                break
            lo -= max(1.0, abs(lo))
        else:  # pragma: no cover
            raise RuntimeError(f"could not bracket phi at Vy={Vy}, Vg={Vg}")

        return brentq(lambda ph: self._H(ph) - B, lo, hi, xtol=xtol, rtol=1e-15)

    def solve_phi(self, Vy, Vg: float):
        """Vectorised surface-potential solve, phi(Vy) for an array of Vy."""
        Vy = np.asarray(Vy, dtype=float)
        flat = np.array([self.solve_phi_scalar(float(v), Vg) for v in Vy.ravel()])
        return flat.reshape(Vy.shape)

    def psi(self, Vy, Vg: float):
        """Quasi-Fermi-referenced potential psi(Vy) = phi(Vy) + Vy."""
        return self.solve_phi(Vy, Vg) + np.asarray(Vy, dtype=float)

    # ----------------------------------------------------------------- #
    # G, Id
    # ----------------------------------------------------------------- #
    def _channel_grid(self, lo: float, hi: float, npts: int, ratio: float = 1.05):
        """Integration grid for the channel integral, refined near `lo`.

        In deep subthreshold the charge-balance solve gives dphi/dVy = -1, so
        the integrand falls off as exp(-(Vy - lo)/phiT) and lives in a sliver of
        width ~phiT at the `lo` end -- 0.35 mV at 4 K, against a uniform pitch
        of (hi - lo)/npts.  A uniform grid then integrates a spike it never
        samples and over-estimates G by orders of magnitude at low T.

        The grid is the union of
          * a uniform grid of `npts` points, which resolves the smooth
            above-threshold variation over the full span, and
          * a geometric cluster near `lo` whose first step is phiT/10 and whose
            pitch grows by `ratio`, which resolves the subthreshold spike.

        phiT is the smallest scale the integrand can vary on (the free charge
        always carries it, and any mobile tail is wider), so resolving it is
        sufficient in both regimes.  The cluster costs only ~log(span/phiT)
        extra points.

        Residual error is dominated by the geometric region around
        Vy - lo ~ 3*phiT, where the local pitch is (ratio - 1) of the position
        and the integrand is still O(1) fraction of its peak; it therefore
        scales as (ratio - 1)^2.  Measured at Vd = 1 V, Vg = 0.1 V, npts = 201
        against a converged reference:

            ratio   1.20    1.15    1.10    1.05    1.03
            err     0.55%   0.33%   0.16%   0.045%  0.020%
            points  258     275     309     412     549

        (The uniform grid this replaced was +569% at 4 K.)  Raise `ratio` to
        trade accuracy for speed on large sweeps.
        """
        uniform = np.linspace(lo, hi, npts)
        span = hi - lo
        first = 0.1 * self.p.phiT
        if span <= first * (npts - 1):
            return uniform  # uniform pitch already finer than the cluster
        n_geo = int(np.ceil(np.log(span / first) / np.log(ratio))) + 1
        cluster = lo + np.geomspace(first, span, n_geo)
        # np.unique sorts and de-duplicates the union
        return np.unique(np.concatenate((uniform, cluster)))

    def G(self, Vg: float, Vd: float, Vs: float, npts: int = 201) -> float:
        """G(Vg, Vd, Vs) = int_{Vs}^{Vd} Qcond(phi(Vy)) dVy   [C/cm^2 * V].

        `npts` sets the uniform part of the grid; extra points are added near
        the high-charge end automatically (see `_channel_grid`).
        """
        if Vd == Vs:
            return 0.0
        lo, hi = (Vs, Vd) if Vd > Vs else (Vd, Vs)
        grid = self._channel_grid(lo, hi, npts)
        integral = float(np.trapezoid(self.q_cond(self.solve_phi(grid, Vg)), grid))
        return integral if Vd > Vs else -integral

    def Id(self, Vg: float, Vd: float, Vs: float, npts: int = 201) -> float:
        """Id(Vg, Vd, Vs) = mu_eff * (W/L) * G(Vg, Vd, Vs)   [A]."""
        p = self.p
        return p.mu_eff * p.W_over_L * self.G(Vg, Vd, Vs, npts=npts)


if __name__ == "__main__":
    from dataclasses import replace

    base = OSFETBandTailParams()
    print(f"Ntail = {base.Ntail:.4e} cm^-2   (pinned value {NTAIL_PINNED:.4e})")
    # mobile_tail and tail_temp_dep are mutually exclusive, so only these three
    # configurations exist.
    for label, kw in (
            ("neither           ", {}),
            ("mobile_tail       ", {"mobile_tail": True}),
            ("Eq28, fixed Ntail ", {"tail_temp_dep": True}),
            ("Eq28, fixed Dtail ", {"tail_temp_dep": True,
                                    "tail_fix_Dtail": True})):
        for T in (300.0, 50.0):
            p = replace(base, T=T, **kw)
            m = OSFETBandTailModel(p)
            print(f"{label}  T={T:>5.0f} K  "
                  f"phi_tail_eff={p.phi_tail_eff*1e3:>6.2f} mV  "
                  f"Ntail_eff={p.Ntail_eff:.3e}  "
                  f"Ctail0={p.Ctail0:.3e}  "
                  f"Id(0.8, 0.5, 0) = {m.Id(0.8, 0.5, 0.0):.4e} A")
