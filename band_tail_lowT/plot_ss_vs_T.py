"""Id-Vg, SS-Id and SS(Iref)-T plots for the three band-tail configurations.

Three mutually exclusive configurations of `OSFETBandTailParams` are compared
at Vd = 50 mV (linear region):

    all immobile          mobile_tail=False, tail_temp_dep=False
                          the whole tail screens but does not conduct
    partially immobile    mobile_tail=True
                          tail above -phi_ref conducts, deeper states do not
    Eq.28, fixed Ntail    tail_temp_dep=True
                          width follows Eq. (28), tail states conserved
    Eq.28, fixed Dtail    tail_temp_dep=True, tail_fix_Dtail=True
                          width follows Eq. (28), band-edge tail DOS held fixed

Sweep results are cached in figs/ss_sweeps.npz; delete it to recompute.
"""

import os
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from osfet_band_tail_model_v0 import OSFETBandTailParams, OSFETBandTailModel

VD = 0.05          # drain bias [V]
VS = 0.0
IREF = 1e-11       # reference current for the SS(T) plot [A]
VG_TOP = 0.8       # top of the Vg sweep [V], well above Vt = 0.3
I_FLOOR = 1e-13    # current at which to start the Vg sweep [A]
NVG = 140
NPTS_G = 101       # uniform part of the channel grid (clustered part is added)

TEMPS = np.array([4.0, 6.0, 10.0, 15.0, 25.0, 40.0,
                  60.0, 100.0, 150.0, 200.0, 250.0, 300.0])

CASES = (("all immobile",       {}),
         ("partially immobile", {"mobile_tail": True}),
         ("Eq.28, fixed Ntail", {"tail_temp_dep": True}),
         ("Eq.28, fixed Dtail", {"tail_temp_dep": True,
                                 "tail_fix_Dtail": True}))

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "figs", "ss_sweeps.npz")


def find_vg_at(model, target, vg_hi):
    """Bisect for the Vg giving Id = `target`, searching below `vg_hi`.

    Id underflows to exactly 0 deep in subthreshold at low T, so the lower
    bracket is found by stepping down until Id is below target (or zero) rather
    than by assuming a fixed window.
    """
    lo = vg_hi
    for _ in range(80):
        lo -= 0.25
        idl = model.Id(lo, VD, VS, npts=NPTS_G)
        if idl < target:      # includes the underflowed idl == 0 case
            break
    else:
        return lo
    hi = vg_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if model.Id(mid, VD, VS, npts=NPTS_G) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    return 0.5 * (lo + hi)


def sweep(params):
    """Id(Vg) on a per-temperature Vg window, plus SS(Vg) = dVg/dlog10(Id)."""
    model = OSFETBandTailModel(params)
    vg_lo = find_vg_at(model, I_FLOOR, VG_TOP)
    vg = np.linspace(vg_lo, VG_TOP, NVG)
    idv = np.array([model.Id(float(v), VD, VS, npts=NPTS_G) for v in vg])

    # Mask the deep-subthreshold points where exp(phi/phiT) underflows to 0.
    ok = np.isfinite(idv) & (idv > 0.0)
    ss = np.full_like(idv, np.nan)
    if ok.sum() > 2:
        ss[ok] = np.gradient(vg[ok], np.log10(idv[ok]))
    return vg, idv, ss


def ss_at_iref(idv, ss):
    """SS interpolated at Id = IREF, in mV/dec; NaN if IREF is out of range."""
    ok = np.isfinite(idv) & (idv > 0.0) & np.isfinite(ss)
    if ok.sum() < 2:
        return np.nan
    lg, s = np.log10(idv[ok]), ss[ok]
    if not (lg[0] <= np.log10(IREF) <= lg[-1]):
        return np.nan
    return float(np.interp(np.log10(IREF), lg, s)) * 1e3


def compute():
    data = {}
    base = OSFETBandTailParams()
    for ci, (label, kw) in enumerate(CASES):
        ssref = []
        for ti, T in enumerate(TEMPS):
            p = replace(base, T=float(T), **kw)
            vg, idv, ss = sweep(p)
            data[f"vg_{ci}_{ti}"] = vg
            data[f"id_{ci}_{ti}"] = idv
            data[f"ss_{ci}_{ti}"] = ss
            ssref.append(ss_at_iref(idv, ss))
            print(f"{label:<20s} T={T:>5.1f} K  Vg in [{vg[0]:+.3f}, {vg[-1]:+.3f}]  "
                  f"Id(top)={idv[-1]:.3e} A  SS(Iref)={ssref[-1]:.3f} mV/dec",
                  flush=True)
        data[f"ssref_{ci}"] = np.array(ssref)
    np.savez(CACHE, **data)
    return data


def load():
    if os.path.exists(CACHE):
        print(f"using cached sweeps in {CACHE}")
        return dict(np.load(CACHE))
    return compute()


def main():
    data = load()
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / (len(TEMPS) - 1)) for i in range(len(TEMPS))]

    fig, axes = plt.subplots(len(CASES), 3, figsize=(15.5, 4.33 * len(CASES)))

    for ci, (label, _) in enumerate(CASES):
        ax_iv, ax_ss, ax_T = axes[ci]

        for ti, T in enumerate(TEMPS):
            vg = data[f"vg_{ci}_{ti}"]
            idv = data[f"id_{ci}_{ti}"]
            ss = data[f"ss_{ci}_{ti}"]
            ok = np.isfinite(idv) & (idv > 0.0)
            ax_iv.semilogy(vg[ok], idv[ok], color=colors[ti], lw=1.3,
                           label=f"{T:g} K")
            okss = ok & np.isfinite(ss)
            ax_ss.semilogx(idv[okss], ss[okss] * 1e3, color=colors[ti], lw=1.3,
                           label=f"{T:g} K")

        ax_iv.set_xlabel(r"$V_g$ [V]")
        ax_iv.set_ylabel(r"$I_d$ [A]")
        ax_iv.set_title(f"{label}\n" r"$I_d$ vs $V_g$, $V_d$ = 50 mV")
        ax_iv.set_ylim(I_FLOOR * 0.5, None)
        ax_iv.grid(True, which="both", alpha=0.25)
        ax_iv.legend(fontsize=6.5, ncol=2, loc="lower right")

        ax_ss.axvline(IREF, color="k", ls=":", lw=1.0)
        ax_ss.set_xlabel(r"$I_d$ [A]")
        ax_ss.set_ylabel("SS [mV/dec]")
        ax_ss.set_title(f"{label}\nSS vs $I_d$")
        ax_ss.set_xlim(I_FLOOR, None)
        ax_ss.set_ylim(0, 700)
        ax_ss.grid(True, which="both", alpha=0.25)
        ax_ss.legend(fontsize=6.5, ncol=2, loc="upper left")

        ssref = data[f"ssref_{ci}"]
        ax_T.plot(TEMPS, ssref, "o-", color="C0", lw=1.5, ms=4,
                  label=f"model, $I_d$ = {IREF:g} A")
        # Boltzmann limit (kT/q)ln10, the floor a trap-free device would show.
        ax_T.plot(TEMPS, 0.0861733 * TEMPS * np.log(10.0), "k--", lw=1.2,
                  label=r"Boltzmann $(kT/q)\ln 10$")
        ax_T.set_xlabel("T [K]")
        ax_T.set_ylabel(f"SS at $I_d$ = {IREF:g} A [mV/dec]")
        ax_T.set_title(f"{label}\nSS vs T")
        ax_T.set_xlim(0, TEMPS[-1] * 1.03)
        ax_T.set_ylim(0, None)
        ax_T.grid(True, alpha=0.25)
        ax_T.legend(fontsize=7.5, loc="upper left")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, "figs", f"ss_vs_T.{ext}")
        fig.savefig(out, dpi=170 if ext == "png" else None)
        print(f"wrote {out}")

    print("\nSS at Id = %g A  [mV/dec]" % IREF)
    hdr = "  T[K] " + "".join(f"{lab:>21s}" for lab, _ in CASES)
    print(hdr)
    for ti, T in enumerate(TEMPS):
        row = f"{T:>6.1f} " + "".join(
            f"{data[f'ssref_{ci}'][ti]:>21.3f}" for ci in range(len(CASES)))
        print(row)


if __name__ == "__main__":
    main()
