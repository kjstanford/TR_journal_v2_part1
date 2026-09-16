# OSFET Band-Tail Model (v0) — Drain Current

Implementation: [osfet_band_tail_model_v0.py](osfet_band_tail_model_v0.py)

Sources:

- [../scratch/TR_related_equations.pdf](../scratch/TR_related_equations.pdf) —
  the TR charge/current formulation.
- [References/195702_1_5.0233899.pdf](References/195702_1_5.0233899.pdf) —
  Kang, Toprasertpong, Oka, Mori, Takenaka & Takagi, *J. Appl. Phys.* **136**,
  195702 (2024), which supplies the mobile/localized split of the tail DOS.
- [References/An_All-Region_BSIM_Thin-Film_Transistor_Model...pdf](References/An_All-Region_BSIM_Thin-Film_Transistor_Model_for_Display_and_BEOL_3-D_Integration_Applications.pdf)
  — Pahwa, Salahuddin & Hu, *IEEE TED* **71** (2024), which supplies the
  floating tail density (its `N_trap`, Eq. 3) and the power-law temperature
  dependence of the tail width (its Eq. 28).

## Physical picture

Channel charge has three components:

- **Band (free) charge** — carriers in extended states above the band edge,
  with a constant 2-D density of states. Always conducts.
- **Mobile tail charge** — carriers in the *shallow* part of the exponential
  band-tail (Urbach) distribution, above the mobility edge. Conducts.
- **Localized tail charge** — carriers in the *deep* part of the tail, below
  the mobility edge. Screens the gate field but carries no current.

The split between the last two is the contribution of the reference paper.
Its significance is that the localized charge only *degrades* the subthreshold
swing by a screening factor (∝ T, so it still vanishes as T → 0), whereas the
mobile tail charge gives the conducting charge an `exp(φ/W_b)` dependence and
therefore a **temperature-independent SS floor** — the saturation universally
observed in cryogenic measurements.

## Density of states

Energies are referred to the band edge `E0` and expressed as voltages,
`ε = (E − E0)/q`, so `ε < 0` is inside the tail. With `D2D` the 2-D DOS:

```
band states    D(ε) = D2D                                        ε > 0
whole tail     D(ε) = D_tail0·exp(ε/phi_tail)                    ε < 0
```

The tail is split at `ε = −phi_ref` into mobile and localized parts:

```
D_mobile(ε) = D_tail0·exp(ε/phi_tail)                    −phi_ref < ε < 0
D_mobile(ε) = D_tail0·exp(−phi_ref/phi_tail)·exp((ε+phi_ref)/phi_b)  ε < −phi_ref
D_local(ε)  = D(ε) − D_mobile(ε)                                 ε < −phi_ref
```

with `phi_b < phi_tail`, so the mobile DOS decays faster than the whole tail
below the boundary. Occupancy is full Fermi–Dirac throughout (not Boltzmann).

### Tail amplitude: `Ntail`

The tail amplitude is set by the **floating** total areal tail-state density

```
Ntail = D_tail0 · phi_tail          [cm⁻²]
```

i.e. the tail DOS integrated over all energy. `D_tail0 = D_tail(E0)` is then
`Ntail/phi_tail`, so it is **`Ntail`, not the band-edge DOS, that is held
fixed**. This is the `N_trap` of the BSIM TFT paper (its Eq. 3), which likewise
treats the trap prefactor as independent of the band DOS `Nc`.

Setting `Ntail = C2D·phi_tail/q` recovers the Kang-paper convention in which the
tail is pinned to the band-edge DOS (`D_tail0 = D2D`); the module exposes that
value as `NTAIL_PINNED` and uses it as the `Ntail` default, so the shipped
parameter set is self-consistent. The module-level convenience

```
Ctail0 = q·D_tail0 = q·Ntail/phi_tail_eff      [F/cm²]
```

is the tail analogue of `C2D` and equals it exactly at the pinned default.

When the tail width becomes temperature dependent, which of `Ntail` and
`D_tail0` is the constant becomes a real choice; see
[Tail-width temperature dependence](#tail-width-temperature-dependence). The
quantity actually used by the charge expressions is the property `Ntail_eff`,
which equals `Ntail` in every case except `tail_fix_Dtail`.

## Charge densities

With `C2D = q²·D2D`, `phiT = kT/q`, and `phi = (E_F − E0)/q`:

```
Qfree(phi) = C2D·phiT·log(1 + exp(phi/phiT))
Qtail(phi) = q·Ntail_eff·F(phi_tail, phi)
Qmob(phi)  = Ctail0·[ phi_tail·F(phi_tail, phi)
                      + exp(−phi_ref/phi_tail)
                        ·( phi_b·F(phi_b, phi+phi_ref)
                           − phi_tail·F(phi_tail, phi+phi_ref) ) ]
Qloc(phi)  = Qtail(phi) − Qmob(phi)
```

Note `Qtail` saturates at `q·Ntail_eff` once the tail is filled, independently
of `phi_tail`. With `Ntail` held fixed this saturation level is a constant
`q·Ntail` at every temperature (verified in code to machine precision across
T = 4–600 K); under `tail_fix_Dtail` it shrinks with the width.

where

```
F(w, x) = 2F1(1, phiT/w; 1 + phiT/w; −exp(−x/phiT))
```

`2F1` is the Gauss hypergeometric function. `F(w, x)` is *exactly* the
Fermi–Dirac integral over an exponential tail of width `w` and unit amplitude
at its upper edge, divided by `w` (verified numerically against direct
quadrature to ~1e-12 relative error). It rises monotonically from 0 to 1, with
the deep-subthreshold limit `F → exp(x/w)`.

Limiting cases of `Qmob`, both verified in code:

| condition | result |
|---|---|
| `mobile_tail = False` | `Qmob = 0` — whole tail immobile |
| `phi_ref = 0` | `Qmob = Ctail0·phi_b·F(phi_b, phi)` |
| `phi_b = phi_tail` | `Qmob = Qtail` — whole tail mobile |

## Tail-width temperature dependence

`tail_temp_dep` switches on Eq. (28) of the BSIM TFT paper,
`T_trap(T) = T_trap0·(T/T0)^T_trapt`, which in the voltage units used here is

```
phi_tail_eff(T) = phi_tail · (T/T0)**T_trapt
```

with `phi_tail` the value at the reference temperature `T0`. When the switch is
off, `phi_tail_eff == phi_tail` at all temperatures.

Every charge expression uses `phi_tail_eff`, so this propagates automatically.
Because the tail charge enters the charge balance, this switch — unlike
`mobile_tail` — *does* change the electrostatics and hence `phi(Vy)`.

### Which amplitude is held fixed: `tail_fix_Dtail`

Only the product `D_tail0·phi_tail_eff` enters the charge, so once the width
moves with temperature one of the two factors has to be declared constant. The
flag `tail_fix_Dtail` selects which, and is only meaningful together with
`tail_temp_dep` (setting it alone raises `ValueError`, since with a fixed width
the two choices are algebraically identical). The two agree at `T = T0`.

| | `tail_fix_Dtail = False` (default) | `tail_fix_Dtail = True` |
|---|---|---|
| held fixed | `Ntail` — total tail states | `D_tail0` — band-edge tail DOS |
| `Ntail_eff(T)` | `Ntail` | `Ntail·phi_tail_eff/phi_tail` |
| `Ctail0(T)` | `q·Ntail/phi_tail_eff` — **grows** as tail narrows | `q·Ntail/phi_tail` — **constant** |
| picture | states are *compressed* into a narrower window | states are *removed* with the window |
| tail charge `q·Ntail_eff/Cgch` | 0.261 V at all T | 0.261 V at 300 K → 0.029 V at 4 K |

The consequences at low temperature are opposite, and neither is a numerical
artifact:

- **fixed `Ntail`** — screening *strengthens* as T falls (`Ctail0` reaches
  52·`Cgch` at 4 K), which flattens SS. But the tail still holds only the fixed
  charge `q·Ntail`, so it **fills up** once the gate has swept
  `q·Ntail/Cgch` = 0.261 V past `Vt`; `C_q` then collapses and SS dips before
  the free-carrier capacitance takes over. This produces a visible kink in
  `Id`–`Vg` whose position tracks `Vt + q·Ntail/Cgch` to 1–2 % (verified over
  `Ntail` scaled 0.5×–4×).
- **fixed `D_tail0`** — screening is T-independent in amplitude while the tail
  loses states, so SS collapses ∝ T *faster* than with no switch at all, and
  the exhaustion kink moves down to just above `Vt`.

### `mobile_tail` and `tail_temp_dep` are mutually exclusive

Setting both `mobile_tail` and `tail_temp_dep` raises `ValueError`. They are
alternative explanations of the *same* observable — the cryogenic SS floor —
and are not physically composable: the mobility edge (`phi_ref`, `phi_b`) is
defined at a fixed energy, so rescaling the tail width with temperature would
make the edge drift through the tail for no stated physical reason. Enable
exactly one.

## Charge balance

The gate sees the **total** channel charge, conducting or not:

```
Cgch·[Vg − Vt − n·phi(Vy) − n·Vy] = Qfree(phi(Vy)) + Qtail(phi(Vy))
```

`n` is the body/back-gate coupling factor, `Vt` the threshold voltage. In the
fixed-point form the code solves:

```
n·phi + [Qfree(phi) + Qtail(phi)]/Cgch = Vg − Vt − n·Vy
```

The left side is strictly increasing in `phi`, so the root is unique and is
found with bracketing plus Brent's method.

**Consequence of the pinning:** because the balance uses the total tail charge,
`phi(Vy)` does *not* depend on the `mobile_tail` switch. Flipping the switch
changes only the current integrand, never the electrostatics.

## Drain current

```
Qcond(phi)     = Qfree(phi) + Qmob(phi)
G(Vg, Vd, Vs)  = ∫[Vs → Vd] Qcond(phi(Vy)) dVy
Id(Vg, Vd, Vs) = mu_eff · (W/L) · G(Vg, Vd, Vs)
```

This is a Pao–Sah-style channel integral, valid into saturation. (The
reference paper instead uses the single-point linear-region form
`Id = (W/L)·q·ns·mu·Vds`, valid only as `Vds → 0`; this reduces to it as
`Vd → Vs`.) Sign convention: `G` and `Id` are negative if `Vd < Vs`.

Mobility is constant, and the mobile tail states share `mu_eff` with the band
states — the same assumption the paper makes, and the one it flags as its main
weakness. Giving the tail states a separate, lower mobility is the natural next
refinement.

Only `Id` is implemented. The `tau_charge`, `tau_leak`, `fT`, `gm/Id` and FoM
quantities defined in the TR PDF are not part of this module.

## Code structure

- **`OSFETBandTailParams`** — `dataclass` of device/model parameters:
  `C2D`, `Cgch`, `n`, `Vt`, `Ntail`, `phi_tail`, `phi_ref`, `phi_b`,
  `mobile_tail`, `tail_temp_dep`, `tail_fix_Dtail`, `T0`, `T_trapt`, `T`,
  `mu_eff`, `W_over_L`.
  Derived properties: `phiT`, `phi_tail_eff` (Eq. 28), `Ntail_eff`, `Ctail0`.
  Validates `T > 0`, `T0 > 0`, `phi_ref >= 0`, `Ntail >= 0`,
  `tail_fix_Dtail` implies `tail_temp_dep`, `mobile_tail` xor
  `tail_temp_dep`, and
  `phi_b <= phi_tail_eff` (otherwise the localized density would go negative —
  checked against the *effective* width, which `tail_temp_dep` can shrink).

- **`OSFETBandTailModel`** — wraps a params instance:
  - `q_free`, `q_tail`, `q_tail_mobile`, `q_tail_localized`, `q_cond`
  - `f_tail(phi)` — whole-tail occupancy `F(phi_tail, phi)`
  - `solve_phi_scalar` / `solve_phi` — surface-potential solve
  - `psi` — quasi-Fermi-referenced potential
  - `G`, `Id`

### Parameter correspondence with the paper

| this module | source | paper symbol |
|---|---|---|
| `C2D` | Kang | `q²·D2D = q²·g_v·m*/(πħ²)` |
| `Ntail` | BSIM TFT | `N_trap` — tail prefactor (areal here, volumetric there) |
| `phi_tail` | Kang / BSIM TFT | `Wt/q` = `V_trap = kT_trap/q` — whole-tail width |
| `phi_ref` | Kang | `|Eref|/q` — mobile/localized boundary |
| `phi_b` | Kang | `Wb/q` — mobile-tail width |
| `T0`, `T_trapt` | BSIM TFT | `T0`, `T_trapt` of Eq. (28) |
| `tail_fix_Dtail` | — | no paper counterpart; neither source states which amplitude is conserved |

Defaults for `phi_ref` and `phi_b` follow the Kang paper's fitted ratios (its
Fig. 12: `|Eref| ≈ 0.5·Wt`, `Wb ≈ 0.37·Wt`) applied to `phi_tail`. Its absolute
Si values are `Wt` = 1–11 meV, `Wb` = 0.4–4.3 meV, `|Eref|` = 0.6–4.2 meV over
`Nsub` = 1e15–2e18 cm⁻³; the 45 meV default `phi_tail` here is an
amorphous-oxide-like Urbach energy, not a Si value. `T_trapt` defaults to
0.511, the a-IGZO value fitted in Fig. 10 of the BSIM TFT paper.

### Behavior of the switches

SS (mV/dec) at `Id` = 1e-11 A, `Vd` = 50 mV, default parameters, over the four
permitted configurations (generated by `plot_ss_vs_T.py`, which also plots
`Id`–`Vg` and SS–`Id`):

| T (K) | `phi_tail_eff` | Boltzmann | neither switch | `mobile_tail` | Eq. 28, fixed `Ntail` | Eq. 28, fixed `D_tail0` |
|---|---|---|---|---|---|---|
| 300 | 45.00 mV | 59.53 | 66.04 | 65.74 | 66.04 | 66.04 |
| 150 | 31.58 mV | 29.77 | 40.30 | 43.64 | 36.02 | 35.06 |
| 60 | 19.77 mV | 11.91 | 34.42 | 43.95 | 25.57 | 19.02 |
| 25 | 12.64 mV | 4.96 | 23.98 | 44.02 | 27.28 | 12.47 |
| 10 | 7.91 mV | 1.98 | 12.12 | 44.03 | 28.62 | 7.57 |
| 4 | 4.96 mV | 0.79 | 5.28 | 44.04 | 25.85 | 4.03 |

Reading the columns:

- **neither switch** — SS collapses ∝ T with no floor. Realizes only the Kang
  paper's Eq. (11), `SS → (kT/q)·ln10·(n + Ctail/Cgch)`.
- **`mobile_tail`** — the only configuration with a *genuine* floor: flat at
  44.0 mV/dec from 4 K to 150 K. Verified to track
  `n·(phi_b/q)·ln10` to within 1 % for `phi_b >= 16.5` mV, i.e. the Kang
  paper's Eq. (12) with the body factor. (At smaller `phi_b` the ratio rises
  as the localized-tail screening term stops being negligible.)
- **Eq. 28, fixed `Ntail`** — looks floored at ~26–29 mV/dec, but **this is
  not a floor**. Probed at a fixed current it merely happens to sit mid-slope:
  at 4 K the same device runs from 0.87 mV/dec (= `n·ln10·kT/q`, the asymptote
  once the tail capacitance is exhausted) through 26 mV/dec at 1e-11 A to
  49 mV/dec near threshold. The apparent flatness in T, and the shallow
  minimum near 60 K, are artifacts of the fixed-`Id` probe.
- **Eq. 28, fixed `D_tail0`** — SS collapses ∝ T and ends up *below* the
  no-switch column, because the tail loses states as it narrows and so screens
  less.

Two cautions for attributing a measured SS floor to a mobility edge. First,
only `mobile_tail` gives a real floor here — but a fixed-current SS-vs-T plot
cannot tell it apart from fixed-`Ntail` Eq. 28 without looking at the `Id`
dependence, which is why `plot_ss_vs_T.py` plots SS–`Id` alongside. Second, the
`Ntail`/`D_tail0` choice is not settled by either source paper, and it changes
the low-temperature trend qualitatively (compare the last two columns), so it
should be stated explicitly rather than inherited from a default.

## Numerical notes

- `q_free` uses `np.logaddexp`, so it does not overflow for `phi/phiT >> 1`.
- `_F` evaluates `2F1` via `scipy.special.hyp2f1` when the argument is in a
  safe range, and switches to a large-argument asymptotic series
  (parameterized by `log(x)` so every term stays finite) when direct
  evaluation would overflow. This matters deep in subthreshold at low
  `phi_tail` / low temperature.
- `solve_phi_scalar` brackets the root by walking outward from `B/n` before
  calling `brentq`.
- **Non-uniform channel grid.** In deep subthreshold the charge-balance solve
  gives `dphi/dVy = −1`, so the integrand falls as `exp(−(Vy−Vs)/phiT)` and
  lives in a sliver of width ~`phiT` at the source end — 0.35 mV at 4 K,
  against a uniform pitch of 5 mV at `Vd` = 1 V. A uniform grid integrates a
  spike it never samples, over-estimating `Id` by **+569% at 4 K** and +25% at
  30 K. `_channel_grid` therefore uses the union of
  - a uniform grid of `npts` points, resolving the smooth above-threshold
    variation over the full span, and
  - a geometric cluster near the high-charge end, first step `phiT/10`, pitch
    growing by `ratio` (default 1.05).

  `phiT` is the smallest scale the integrand can vary on (the free charge
  always carries it; any mobile tail is wider), so resolving it suffices in
  both regimes. Residual error is dominated by the geometric region around
  `Vy − Vs ≈ 3·phiT` and scales as `(ratio−1)²`. Measured at `Vd` = 1 V,
  `Vg` = 0.1 V, `npts` = 201:

  | `ratio` | 1.20 | 1.15 | 1.10 | 1.05 | 1.03 |
  |---|---|---|---|---|---|
  | error | 0.55% | 0.33% | 0.16% | 0.045% | 0.020% |
  | points | 258 | 275 | 309 | 412 | 549 |

  At the 1.05 default, error is ≤0.05% from 4 K to 300 K, above and below
  threshold, with either switch. `npts` controls only the uniform part; the
  refinement is added automatically, and results are stable against `npts`
  (3.2003e-244 at 4 K for `npts` = 51…401). Raise `ratio` to trade accuracy
  for speed on large sweeps.
- With the tail immobile below ~10 K, `Id` underflows to exactly 0.0 deep in
  subthreshold (the true value is ~1e-250), so log-scale plots must mask
  non-positive values.
- Requires numpy ≥ 2.0 (`np.trapezoid`). Run with
  `/opt/shared_python_science/bin/python3`.
