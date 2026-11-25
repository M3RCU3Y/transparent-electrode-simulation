# opaque front contact comparison (report section 4.9)
# compares silver films, thick/heavily doped ito, and bare metal grids
# outputs ../data/opaque_ag.csv and ../data/numbers_opaque.tex

import os

import numpy as np
from numpy import trapezoid

import physics as ph
import electrodes as el
import tmm
import device as dv

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
LAMW = np.arange(300.0, 2501.0, 5.0)     # spectrum grid nm
GLASS = 1.52
NUM = {}                                 # latex macros


def macro(name, value, fmt="{:.2f}"):
    NUM[name] = fmt.format(value)


def save(name, header, cols):
    np.savetxt(os.path.join(OUT, name), np.column_stack(cols), delimiter=",",
               header=",".join(header), comments="", fmt="%.6g")


def T_sol(lam, T, lo=300, hi=1200):
    """solar photon weighted average of T between lo and hi nm"""
    m = (lam >= lo) & (lam <= hi)
    F = ph.photon_flux(lam[m], ph.am15g(lam[m]))
    return trapezoid(F * T[m], lam[m]) / trapezoid(F, lam[m])


T_GLASS = tmm.film_on_thick_substrate([], [], GLASS, LAMW)[1]    # bare glass baseline


def device_eta(optics_shj, optics_psk, Rs):
    """evaluate both devices for one contact and clamp eta at 0"""
    shj = dv.shj_evaluate(*optics_shj, Rs)
    psk = dv.psk_evaluate(*optics_psk, Rs)
    return max(shj["eta"], 0.0), max(psk["eta"], 0.0), shj, psk


# 1. silver films from 7 nm to 120 nm
def silver_films():
    thicknesses = np.array([7, 8, 9, 10, 12, 14, 16, 18, 20,
                            25, 30, 35, 40, 50, 60, 80, 100, 120], float)
    rows = []
    for t in thicknesses:
        n_wide = np.sqrt(ph.ag_eps(ph.eV_from_nm(LAMW), t))     # glass spectrum
        R, T, A = tmm.film_on_thick_substrate([n_wide], [t], GLASS, LAMW)
        n_dev = np.sqrt(ph.ag_eps(ph.eV_from_nm(dv.LAM), t))    # device grid
        Rs = ph.ag_resistivity(t) / (t * 1e-9)
        eta_shj, eta_psk, _, _ = device_eta(dv.shj_optics_film([n_dev], [t]),
                                            dv.psk_optics_film([n_dev], [t]), Rs)
        rows.append([t, Rs, T_sol(LAMW, T / T_GLASS) * 100, T_sol(LAMW, R) * 100,
                     T_sol(LAMW, A) * 100, eta_shj, eta_psk])
    rows = np.array(rows)
    save("opaque_ag.csv", ["t", "Rs", "Tsol", "Rsol", "Asol", "etaSHJ", "etaPSK"], rows.T)

    # save key thicknesses for report text
    for t, key in [(7, "Seven"), (10, "Ten"), (20, "Twenty"), (50, "Fifty"), (100, "Hundred")]:
        r = rows[thicknesses == t][0]
        macro(f"agRs{key}", r[1], "{:.2f}" if r[1] < 1 else "{:.1f}")
        macro(f"agT{key}", r[2], "{:.1f}" if r[2] >= 1 else "{:.2f}")
        macro(f"agR{key}", r[3], "{:.0f}")
        macro(f"agEtaSHJ{key}", r[5], "{:.1f}")
        macro(f"agEtaPSK{key}", r[6], "{:.1f}")

    best_shj, best_psk = np.argmax(rows[:, 5]), np.argmax(rows[:, 6])
    macro("agBestTSHJ", rows[best_shj, 0], "{:.0f}")
    macro("agBestEtaSHJ", rows[best_shj, 5], "{:.1f}")
    macro("agBestTPSK", rows[best_psk, 0], "{:.0f}")
    macro("agBestEtaPSK", rows[best_psk, 6], "{:.1f}")
    return rows


# 2. thick and heavily doped ito
def thick_ito():
    """1 um and 3 um ito films"""
    d = el.TCOS["ITO"]
    rho = el.oxide_resistivity(d["n"], d["mu"])
    optics, _, _ = dv.tco_family("ITO")
    n_wide = el.oxide_index(LAMW, d["host"], d["n"], d["mu"])
    for t, key in [(1000, "OneUm"), (3000, "ThreeUm")]:
        R, T, A = tmm.film_on_thick_substrate([n_wide], [t], GLASS, LAMW)
        eta_shj, eta_psk, _, _ = device_eta(optics("SHJ", t), optics("PSK", t), rho / (t * 1e-9))
        macro(f"itoT{key}", T_sol(LAMW, T / T_GLASS) * 100, "{:.0f}")
        macro(f"itoRs{key}", rho / (t * 1e-9), "{:.1f}")
        macro(f"itoEtaSHJ{key}", eta_shj, "{:.1f}")
        macro(f"itoEtaPSK{key}", eta_psk, "{:.1f}")


def doped_ito():
    """carrier density needed to push plasma edge to 1107 nm (Si) or 750 nm (vis)"""
    hp = el.HOSTS["In2O3"]

    def plasma_wavelength(n):
        return ph.screened_plasma_wavelength(n, ph.effective_mass(n, hp["m0"], hp["C"]), hp["eps_vis"])

    def density_for(lam_target):
        # log bisection since plasma wavelength drops as n increases
        lo, hi = 1e20, 1e23
        for _ in range(80):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if plasma_wavelength(mid) > lam_target else (lo, mid)
        return np.sqrt(lo * hi)

    macro("nPlasmaSi", density_for(1107) / 1e21, "{:.1f}")
    macro("nPlasmaVis", density_for(750) / 1e21, "{:.1f}")
    macro("nPlasmaVisFactor", density_for(750) / el.TCOS["ITO"]["n"], "{:.0f}")


# 3. bare metal grid with no transparent electrode
def bare_grid():
    """use 1e12 ohm/sq to represent no tco layer
    in rear junction shj the wafer still conducts laterally"""
    Tin_ar = dv.shj_optics_ar_only()[1]
    g_rear = dv.shj_evaluate(Tin_ar, 0 * Tin_ar, 1e12, junction="rear")
    g_front = dv.shj_evaluate(Tin_ar, 0 * Tin_ar, 1e12, junction="front")
    macro("gridRearEta", g_rear["eta"], "{:.2f}")
    macro("gridRearS", g_rear["s"] * 1e3, "{:.2f}")
    macro("gridRearShade", g_rear["shade"] * 100, "{:.1f}")
    macro("gridRearRs", g_rear["Rs_eff"], "{:.0f}")
    macro("gridFrontEta", max(g_front["eta"], 0.0), "{:.1f}")
    g_module = dv.psk_evaluate(dv.psk_optics_bare(), 0 * dv.LAM, 1e12)
    macro("gridPSKEta", max(g_module["eta"], 0.0), "{:.1f}")


def write_numbers():
    with open(os.path.join(OUT, "numbers_opaque.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by opaque.py\n")
        for k in sorted(NUM):
            f.write(f"\\newcommand{{\\{k}}}{{{NUM[k]}}}\n")


if __name__ == "__main__":
    silver = silver_films()
    thick_ito()
    doped_ito()
    bare_grid()
    write_numbers()
    for k in sorted(NUM):
        print(f"{k:22s} {NUM[k]}")
    print(np.array2string(silver, precision=2, suppress_small=True, max_line_width=150))
