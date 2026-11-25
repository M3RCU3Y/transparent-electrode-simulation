# main script that runs all calculations for the report
# writes csv files and numbers.tex / numbers.json into ../data/
# run with python run_all.py (add quick flag for a short monte carlo test)

import json
import os
import sys
import time

import numpy as np
from numpy import trapezoid

import physics as ph
import electrodes as el
import tmm
import device as dv
import sq

QUICK = "--quick" in sys.argv
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)
NUM = {}                                  # latex macros
SCALES = []                               # loss breakdown scale factors
RNG = np.random.default_rng(20261105)     # fixed seed for reproducibility


def save(name, header, cols, fmt="%.6g"):
    arr = np.column_stack(cols)
    np.savetxt(os.path.join(OUT, name), arr, delimiter=",", header=",".join(header),
               comments="", fmt=fmt)


def macro(name, value, fmt="{:.2f}"):
    NUM[name] = fmt.format(value) if not isinstance(value, str) else value


def progress(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


LAMW = np.arange(300.0, 2501.0, 5.0)          # wide wavelength grid nm
GLASS = 1.52                                  # glass index


def T_sol(lam, T, lo=300, hi=1200):
    """solar photon weighted average of T from lo to hi nm"""
    m = (lam >= lo) & (lam <= hi)
    F = ph.photon_flux(lam[m], ph.am15g(lam[m]))
    return trapezoid(F * T[m], lam[m]) / trapezoid(F, lam[m])


def glass_only(lam):
    R, T, A = tmm.film_on_thick_substrate([], [], GLASS, lam)
    return T


# 1. material properties: permittivity, band gap and mobility
def part_materials():
    progress("dielectric functions")
    for nm, d in el.TCOS.items():
        hp = el.HOSTS[d["host"]]
        eps = el.oxide_eps(LAMW, d["host"], d["n"], d["mu"])
        N = np.sqrt(eps)
        alpha = 4 * np.pi * N.imag / (LAMW * 1e-7)
        save(f"dielectric_{nm}.csv", ["lam", "e1", "e2", "n", "k", "alpha"],
             [LAMW, eps.real, eps.imag, N.real, N.imag, alpha])
        mstar = ph.effective_mass(d["n"], hp["m0"], hp["C"])
        macro(f"mstar{nm}", mstar)
        macro(f"lamp{nm}", ph.screened_plasma_wavelength(d["n"], mstar, hp["eps_vis"]), "{:.0f}")
        macro(f"Egopt{nm}", ph.optical_gap(d["n"], hp))
        macro(f"EF{nm}", ph.fermi_energy(d["n"], hp["m0"], hp["C"]))
        macro(f"gamma{nm}", ph.drude_gamma(d["mu"], mstar) * 1000, "{:.0f}")
        macro(f"rho{nm}", el.oxide_resistivity(d["n"], d["mu"]) * 100 * 1e4)   # 1e-4 ohm cm
        macro(f"mulim{nm}", ph.mu_total(d["n"], hp), "{:.0f}")
        macro(f"muii{nm}", ph.mu_ionised_impurity(d["n"], hp), "{:.0f}")
        i550 = np.argmin(abs(LAMW - 550))
        macro(f"nvis{nm}", N.real[i550])

    progress("optical gap and mobility vs n")
    n_grid = np.logspace(19, np.log10(2.5e21), 120)
    for host, hp in el.HOSTS.items():
        EF = ph.fermi_energy(n_grid, hp["m0"], hp["C"])
        EFpar = ph.fermi_energy(n_grid, hp["m0"], 0.0)
        if hp["mv"] == np.inf:
            val = 0
        else:
            val = (ph.hbar * ph.fermi_wavevector(n_grid)) ** 2 / (2 * hp["mv"] * ph.me) / ph.q
        B = ph.bgn(n_grid, hp["eps_s"], hp["m0"], hp["C"])
        Eg = ph.optical_gap(n_grid, hp)
        mstar = ph.effective_mass(n_grid, hp["m0"], hp["C"])
        mii = ph.mu_ionised_impurity(n_grid, hp)
        mL = np.full_like(n_grid, hp["mu_L"])
        # grain boundary cases for 30 nm grains with 1e13 and 3e13 1/cm2 traps
        mgb1, phib1 = ph.mu_grain_boundary(n_grid, hp, 1e13, 30.0)
        mgb2, phib2 = ph.mu_grain_boundary(n_grid, hp, 3e13, 30.0)
        mt = ph.mu_total(n_grid, hp)
        mt1 = 1 / (1 / mt + 1 / mgb1)
        mt2 = 1 / (1 / mt + 1 / mgb2)
        rho_lim = 1 / (n_grid * 1e6 * ph.q * mt * 1e-4) * 100     # ohm cm
        lp = ph.screened_plasma_wavelength(n_grid, mstar, hp["eps_vis"])
        save(f"bands_{host}.csv",
             ["n", "EF", "EFpar", "BM", "BMpar", "BGN", "Eg", "mstar", "lamp"],
             [n_grid, EF, EFpar, hp["Eg0"] + EF + val, hp["Eg0"] + EFpar + val, B, Eg, mstar, lp])
        save(f"mobility_{host}.csv",
             ["n", "mu_ii", "mu_L", "mu_gb1", "mu_gb2", "mu_lim",
              "mu_tot1", "mu_tot2", "rho_lim", "phib1", "phib2"],
             [n_grid, mii, mL, np.minimum(mgb1, 1e5), np.minimum(mgb2, 1e5), mt,
              mt1, mt2, rho_lim, phib1, phib2])
    # marker points for the four oxide films
    rows = []
    for nm, d in el.TCOS.items():
        rows.append([d["n"], d["mu"], el.oxide_resistivity(d["n"], d["mu"]) * 100])
    save("films.csv", ["n", "mu", "rho"], np.array(rows).T)
    for nm, d in el.TCOS.items():
        save(f"film_{nm}.csv", ["n", "mu", "rho"],
             [[d["n"]], [d["mu"]], [el.oxide_resistivity(d["n"], d["mu"]) * 100]])


# 2. films on glass: spectra, haacke figure of merit, T vs Rs map
def tco_on_glass(nm, t, lam=LAMW, d=None):
    d = d or el.TCOS[nm]
    N = el.oxide_index(lam, d["host"], d["n"], d["mu"])
    return tmm.film_on_thick_substrate([N], [t], GLASS, lam)


def part_films():
    progress("spectra at Rs = 10 ohm/sq")
    Tg = glass_only(LAMW)
    cols, head = [LAMW, Tg], ["lam", "Tglass"]
    for nm, d in el.TCOS.items():
        rho = el.oxide_resistivity(d["n"], d["mu"])
        t = rho / 10 * 1e9                     # thickness for 10 ohm/sq in nm
        R, T, A = tco_on_glass(nm, t)
        cols += [T, R, A]
        head += [f"T_{nm}", f"R_{nm}", f"A_{nm}"]
        macro(f"tTen{nm}", t, "{:.0f}")
        macro(f"TsolTen{nm}", T_sol(LAMW, T / Tg) * 100, "{:.1f}")
        macro(f"AsolTen{nm}", T_sol(LAMW, A) * 100, "{:.1f}")
        macro(f"TvisTen{nm}", T_sol(LAMW, T / Tg, 400, 800) * 100, "{:.1f}")
        macro(f"TnirTen{nm}", T_sol(LAMW, T / Tg, 900, 1200) * 100, "{:.1f}")
    save("spectra_rs10.csv", head, cols)

    progress("thickness dependence and Haacke figure of merit")
    ts = np.concatenate([np.arange(10, 100, 5.0),
                         np.arange(100, 2001, 20.0),
                         np.arange(2100, 6001, 100.0)])
    for nm, d in el.TCOS.items():
        rho = el.oxide_resistivity(d["n"], d["mu"])
        N = el.oxide_index(LAMW, d["host"], d["n"], d["mu"])
        Tg = glass_only(LAMW)
        Ts, Tv, As, Rs = [], [], [], []
        for t in ts:
            R, T, A = tmm.film_on_thick_substrate([N], [t], GLASS, LAMW)
            Ts.append(T_sol(LAMW, T / Tg))
            Tv.append(T_sol(LAMW, T / Tg, 400, 800))
            As.append(T_sol(LAMW, A))
            Rs.append(rho / (t * 1e-9))
        Ts, Tv, As, Rs = map(np.array, (Ts, Tv, As, Rs))
        phi = Ts**10 / Rs
        phiA = (1 - As) ** 10 / Rs            # absorption only haacke proxy
        save(f"thick_{nm}.csv", ["t", "Rs", "Tsol", "Tvis", "Asol", "phiH", "phiHA"],
             [ts, Rs, Ts, Tv, As, phi * 1e3, phiA * 1e3])
        j = np.argmax(phi)
        macro(f"toptTrans{nm}", ts[j], "{:.0f}")
        macro(f"phimaxTrans{nm}", phi[j] * 1e3, "{:.1f}")
        i = np.argmax(phiA)
        # beer lambert estimate t* = 1/(10 alpha) using alpha from 50 nm film
        k = np.argmin(abs(ts - 50))
        alpha_eff = -np.log(1 - As[k]) / (ts[k] * 1e-7)       # 1/cm
        sigma = 1 / (rho * 100)                                # S/cm
        t_an = 1 / (10 * alpha_eff) * 1e7
        phi_an = sigma / (10 * np.e * alpha_eff)
        macro(f"topt{nm}", ts[i], "{:.0f}")
        macro(f"phimax{nm}", phiA[i] * 1e3, "{:.1f}")
        macro(f"tan{nm}", t_an, "{:.0f}")
        macro(f"phian{nm}", phi_an * 1e3, "{:.1f}")
        macro(f"alphaeff{nm}", alpha_eff, "{:.0f}")
        macro(f"sigovalpha{nm}", sigma / alpha_eff, "{:.2f}")

    progress("T-Rs landscape")
    Tg = glass_only(LAMW)
    for nm, d in el.TCOS.items():
        rho = el.oxide_resistivity(d["n"], d["mu"])
        N = el.oxide_index(LAMW, d["host"], d["n"], d["mu"])
        tt = np.geomspace(15, 3000, 70)
        Tl = [T_sol(LAMW, tmm.film_on_thick_substrate([N], [t], GLASS, LAMW)[1] / Tg) for t in tt]
        save(f"land_{nm}.csv", ["Rs", "T"], [rho / (tt * 1e-9), np.array(Tl)])
    # bare silver and ZnO/Ag/ZnO with 40 nm ZnO, silver 6 to 25 nm
    Nz = el.oxide_index(LAMW, "ZnO", 0.0, 1.0)
    ag = np.arange(6, 25.1, 1.0)
    T1, T2, R1 = [], [], []
    for a in ag:
        Nag = np.sqrt(ph.ag_eps(ph.eV_from_nm(LAMW), a))
        T1.append(T_sol(LAMW, tmm.film_on_thick_substrate([Nag], [a], GLASS, LAMW)[1] / Tg))
        T_dmd = tmm.film_on_thick_substrate([Nz, Nag, Nz], [40, a, 40], GLASS, LAMW)[1]
        T2.append(T_sol(LAMW, T_dmd / Tg))
        R1.append(ph.ag_resistivity(a) / (a * 1e-9))
    save("land_Ag.csv", ["Rs", "T", "tAg"], [R1, T1, ag])
    save("land_DMD.csv", ["Rs", "T", "tAg"], [R1, T2, ag])
    # graphene 1 to 8 layers
    Rs1, n2d = ph.graphene_sheet_resistance()
    Ng = np.sqrt(ph.graphene_eps(ph.eV_from_nm(LAMW)))
    Tgr = []
    for layers in range(1, 9):
        T_gr = tmm.film_on_thick_substrate([Ng] * layers, [ph.GRAPHENE_D * 1e9] * layers, GLASS, LAMW)[1]
        Tgr.append(T_sol(LAMW, T_gr / Tg))
    save("land_GRA.csv", ["Rs", "T", "N"], [Rs1 / np.arange(1, 9), Tgr, np.arange(1, 9)])
    macro("RsGraOne", Rs1, "{:.0f}")
    macro("nGra", n2d / 1e13, "{:.1f}")
    for nm, d in el.NETWORKS.items():
        R = np.logspace(0, 3.7, 80)
        save(f"land_{nm}.csv", ["Rs", "T"], [R, el.network_T(R, d["ratio"])])
    # constant haacke contour lines
    R = np.logspace(0, 3.7, 60)
    cols, head = [R], ["Rs"]
    for phi in [1e-4, 1e-3, 1e-2, 3e-2]:
        cols.append(np.clip((phi * R) ** 0.1, 0, 1))
        head.append(f"phi{phi:g}")
    save("haacke_iso.csv", head, cols)

    progress("ZnO/Ag/ZnO map: best layer thicknesses for visible transmittance")
    Tg = glass_only(LAMW)
    ds = np.arange(10, 91, 5.0)
    ags = np.arange(6, 18.1, 1.0)
    rows = []
    best = None
    for a in ags:
        Nag = np.sqrt(ph.ag_eps(ph.eV_from_nm(LAMW), a))
        for dd in ds:
            T = tmm.film_on_thick_substrate([Nz, Nag, Nz], [dd, a, dd], GLASS, LAMW)[1] / Tg
            tv = T_sol(LAMW, T, 400, 800)
            tsol = T_sol(LAMW, T)
            rows.append([a, dd, tv, tsol])
            if best is None or tv > best[2]:
                best = (a, dd, tv, tsol, T)
    rows = np.array(rows)
    save("dmd_map.csv", ["tAg", "d", "Tvis", "Tsol"], rows.T)
    macro("dmdAg", best[0], "{:.0f}")
    macro("dmdD", best[1], "{:.0f}")
    macro("dmdTvis", best[2] * 100, "{:.1f}")
    macro("dmdTsol", best[3] * 100, "{:.1f}")
    macro("dmdRs", ph.ag_resistivity(best[0]) / (best[0] * 1e-9), "{:.1f}")
    t_ag, t_zno = best[0], best[1]
    n_ag = np.sqrt(ph.ag_eps(ph.eV_from_nm(LAMW), t_ag))
    R, T, A = tmm.film_on_thick_substrate([Nz, n_ag, Nz], [t_zno, t_ag, t_zno], GLASS, LAMW)
    # matched sheet resistance ito for comparison
    Rs_dmd = ph.ag_resistivity(t_ag) / (t_ag * 1e-9)
    t_ito = el.oxide_resistivity(el.TCOS["ITO"]["n"], el.TCOS["ITO"]["mu"]) / Rs_dmd * 1e9
    Ri, Ti, Ai = tco_on_glass("ITO", t_ito)
    save("dmd_spectrum.csv", ["lam", "T", "R", "A", "T_ITO", "A_ITO"], [LAMW, T, R, A, Ti, Ai])
    macro("dmdItoT", t_ito, "{:.0f}")


# 3. design map of absorption at 10 ohm/sq over (n, mu)
def part_designmap():
    progress("design map")
    lam = np.arange(300.0, 2001.0, 10.0)
    Tg = glass_only(lam)
    ns = np.logspace(np.log10(5e19), np.log10(2e21), 36)
    mus = np.logspace(np.log10(8), np.log10(250), 36)
    Rs_target = 10.0
    rows = []
    for mu in mus:
        for n in ns:
            rho = el.oxide_resistivity(n, mu)
            t = rho / Rs_target * 1e9
            if t > 5000:                       # skip if over 5 um thick
                rows.append([n, mu, t, np.nan, np.nan])
                continue
            N = el.oxide_index(lam, "In2O3", n, mu)
            R, T, A = tmm.film_on_thick_substrate([N], [t], GLASS, lam)
            rows.append([n, mu, t, T_sol(lam, A) * 100, T_sol(lam, T / Tg) * 100])
    rows = np.array(rows)
    # clip absorption at 30% for colourmap
    rows[:, 3] = np.where(np.isnan(rows[:, 3]), 30.0, np.minimum(rows[:, 3], 30.0))
    np.savetxt(os.path.join(OUT, "designmap.csv"), rows, delimiter=",",
               header="n,mu,t,Asol,Tsol", comments="", fmt="%.5g")
    hp = el.HOSTS["In2O3"]
    n_grid = np.logspace(np.log10(5e19), np.log10(2e21), 80)
    # intrinsic mobility limit and 60% realistic bound
    save("designmap_limit.csv", ["n", "mu_lim", "mu_real"],
         [n_grid, ph.mu_total(n_grid, hp), 0.6 * ph.mu_total(n_grid, hp)])
    # constant thickness lines mu = 1/(n q Rs t)
    cols, head = [n_grid], ["n"]
    for t in [100, 300, 1000]:
        cols.append(1 / (n_grid * 1e6 * ph.q * Rs_target * t * 1e-9) * 1e4)
        head.append(f"t{t}")
    save("designmap_iso.csv", head, cols)

    progress("optimum carrier density (Gordon criterion with mobility limit)")
    lam = np.arange(300.0, 1201.0, 5.0)
    Tg = glass_only(lam)
    ts = np.geomspace(20, 40000, 70)
    for host in ["In2O3", "ZnO"]:
        hp = el.HOSTS[host]
        n_grid = np.logspace(np.log10(5e19), np.log10(2e21), 28)
        out = []
        for n in n_grid:
            res = []
            for frac in (1.0, 0.6):
                mu = frac * ph.mu_total(n, hp)
                N = el.oxide_index(lam, host, n, mu)
                rho = el.oxide_resistivity(n, mu)
                best = 0
                for t in ts:
                    R, T, A = tmm.film_on_thick_substrate([N], [t], GLASS, lam)
                    phi = (1 - T_sol(lam, A)) ** 10 / (rho / (t * 1e-9))
                    best = max(best, phi)
                res.append(best * 1e3)
            mstar = ph.effective_mass(n, hp["m0"], hp["C"])
            mu = ph.mu_total(n, hp)
            # gordon metric scales as (mu m*)^2
            gordon = (mu * 1e-4) ** 2 * (mstar * ph.me) ** 2
            out.append([n, res[0], res[1], gordon])
        out = np.array(out)
        out[:, 3] = out[:, 3] / out[:, 3].max()
        save(f"nopt_{host}.csv", ["n", "phi_lim", "phi_real", "gordon"], out.T)
        i = np.argmax(out[:, 1])
        macro("nopt" + {"In2O3": "InO", "ZnO": "ZnO"}[host], out[i, 0] / 1e20, "{:.1f}")


# 4. device simulations
def part_grid():
    """grid and stripe loss curves for figs 4.5 and 4.7"""
    progress("grid / module loss curves")
    p = dv.SHJ
    J0 = dv.J0_of(p)
    s = np.linspace(0.4e-3, 5e-3, 150)       # finger spacing m
    Jgen = 400.0                             # 40 mA/cm2 in A/m2
    cols, head = [s * 1e3], ["s"]
    shade = p["w_f"] / s + p["busbar_shade"]
    Voc, Vmp, Jmp, Pmp = dv.diode_mpp_vec(Jgen * (1 - shade), J0, p["n_id"])
    pfin = p["rho_f"] * Jmp * p["L_half"] ** 2 * s / (3 * Vmp * p["w_f"] * p["h_f"])
    cols += [shade * 100, pfin * 100]
    head += ["shade", "finger"]
    for Rs in [20, 50, 100, 200]:
        plat = Rs * Jmp * s**2 / (12 * Vmp)
        tot = 1 - (1 - shade) * (1 - plat - pfin)
        cols += [plat * 100, tot * 100]
        head += [f"lat{Rs}", f"tot{Rs}"]
    save("grid_losses.csv", head, cols)
    Rss = np.logspace(0.5, 3, 40)
    s_best, s_cube_root = [], []
    for Rs in Rss:
        plat = Rs * Jmp * s**2 / (12 * Vmp)
        tot = 1 - (1 - shade) * (1 - plat - pfin)
        s_best.append(s[np.argmin(tot)] * 1e3)
        # analytic cube root spacing estimate
        s_cube_root.append((6 * Vmp.mean() * p["w_f"] / (Rs * Jmp.mean())) ** (1 / 3) * 1e3)
    save("sopt.csv", ["Rs", "s_num", "s_an"], [Rss, s_best, s_cube_root])

    psk = dv.PSK
    J0 = dv.J0_of(psk)
    Voc, Vmp, Jmp, Pmp = dv.diode_mpp(230.0, J0, psk["n_id"])   # 23 mA/cm2
    W = np.linspace(1e-3, 20e-3, 200)        # stripe width m
    cols, head = [W * 1e3], ["W"]
    dead = 1 - W / (W + psk["dead"])
    cols.append(dead * 100)
    head.append("dead")
    for Rs in [5, 10, 20, 50]:
        plat = Rs * Jmp * W**2 / (3 * Vmp)
        tot = 1 - (1 - dead) * (1 - plat)
        cols += [plat * 100, tot * 100]
        head += [f"lat{Rs}", f"tot{Rs}"]
    save("module_losses.csv", head, cols)
    Rss = np.logspace(0, 3, 40)
    W_best, lo = [], []
    for Rs in Rss:
        plat = Rs * Jmp * W**2 / (3 * Vmp)
        tot = 1 - (1 - dead) * (1 - plat)
        W_best.append(W[np.argmin(tot)] * 1e3)
        lo.append(tot.min() * 100)
    save("wopt.csv", ["Rs", "W", "loss"], [Rss, W_best, lo])


def summarise(arch, b, ref):
    """split efficiency loss into parasitic absorption, reflection, shading/dead area, resistance
    scaled so the 4 parts sum to the total deficit"""
    eta_i = ref["eta"]
    J_i = ref["Jgen"] / 10
    J_in = b["J_in"]
    opt_par = eta_i * b["J_par"] / J_i
    opt_ref = eta_i * max(J_i - J_in - b["J_par"], 0) / J_i
    if arch == "SHJ":
        sh = eta_i * (b["shade"] - ref["shade"])
        res = eta_i * (b["p_lat"] + b["p_fin"] - ref["p_lat"] - ref["p_fin"])
    else:
        sh = eta_i * (ref["gff"] - b["gff"])
        res = eta_i * (b["p_lat"] - ref["p_lat"])
    comp = np.array([opt_par, opt_ref, sh, res])
    deficit = eta_i - b["eta"]
    scale = deficit / comp.sum() if comp.sum() > 0 else 1.0
    SCALES.append(scale)
    return list(comp * scale) + [scale]


def part_devices():
    """optimise all 9 electrodes in shj and psk"""
    progress("device optimisation (baseline)")
    results = {}
    for arch in ["SHJ", "PSK"]:
        ref = dv.ideal_reference(arch)
        macro(f"etaIdeal{arch}", ref["eta"])
        macro(f"JIdeal{arch}", ref["Jgen"] / 10)
        res = {}
        curves = {}
        for nm in el.TCOS:
            b, cur = dv.optimise(arch, "tco", dv.tco_family(nm), keep_curve=True)
            res[nm] = b
            curves[nm] = cur
            if arch == "SHJ":
                br, curr = dv.optimise(arch, "tco", dv.tco_family(nm), junction="rear", keep_curve=True)
                res[nm + "rear"] = br
                curves[nm + "rear"] = curr
        b = dv.optimise(arch, "dmd", dv.dmd_family(), grid=dv.dmd_grid())
        res["DMD"] = b
        res["GRA"] = dv.optimise(arch, "graphene", dv.graphene_family())
        for nm in el.NETWORKS:
            res[nm] = dv.optimise(arch, "network", dv.network_family(nm))
        results[arch] = res
        # eta vs thickness curves for the 4 oxides
        cols, head = [dv.T_GRID], ["t"]
        for nm in el.TCOS:
            cols += [[c["eta"] for c in curves[nm]], [c["J_in"] for c in curves[nm]],
                     [c["J_par"] for c in curves[nm]], [c["Rs"] for c in curves[nm]]]
            head += [f"eta_{nm}", f"Jin_{nm}", f"Jpar_{nm}", f"Rs_{nm}"]
            if arch == "SHJ":
                cols.append([c["eta"] for c in curves[nm + "rear"]])
                head.append(f"etarear_{nm}")
        save(f"eta_vs_t_{arch}.csv", head, cols)
        # summary table and loss breakdown
        rows = []
        for i, nm in enumerate(el.ORDER):
            b = res[nm]
            L = summarise(arch, b, ref)
            rows.append([i, b["eta"], ref["eta"] - b["eta"]] + L)
            macro(f"eta{arch}{nm}", b["eta"])
            macro(f"Rs{arch}{nm}", b["Rs"], "{:.0f}")
            macro(f"Jin{arch}{nm}", b["J_in"])
            macro(f"Jpar{arch}{nm}", b["J_par"])
            if nm in el.TCOS:
                macro(f"t{arch}{nm}", b["x"], "{:.0f}")
            if arch == "SHJ":
                macro(f"s{arch}{nm}", b["s"] * 1e3)
            else:
                macro(f"W{arch}{nm}", b["W"] * 1e3, "{:.1f}")
        rows = np.array(rows)
        save(f"loss_{arch}.csv",
             ["i", "eta", "deficit", "parasitic", "reflection", "shading", "resistive", "scale"],
             rows.T, fmt="%.4f")
        if arch == "SHJ":
            for nm in el.TCOS:
                macro(f"etaSHJrear{nm}", res[nm + "rear"]["eta"])
                macro(f"sSHJrear{nm}", res[nm + "rear"]["s"] * 1e3)
                macro(f"tSHJrear{nm}", res[nm + "rear"]["x"], "{:.0f}")
        dm = res["DMD"]["x"]
        macro(f"dmd{arch}", f"{dm[0]:.0f}/{dm[1]:.0f}/{dm[2]:.0f}")
        macro(f"N{arch}GRA", res["GRA"]["x"], "{:.0f}")
    if "SHJ" in results:
        macro("arSHJ", dv.shj_optics_ar_only()[2], "{:.0f}")
    macro("idealPSKd", dv.psk_optics_ideal()[2], "{:.0f}")
    return results


def part_nscan():
    """scan carrier density for In2O3 in both devices at full and 60% mobility limit"""
    progress("carrier-density scan of In2O3-based electrode in both devices")
    hp = el.HOSTS["In2O3"]
    n_grid = np.logspace(np.log10(5e19), np.log10(2e21), 22)
    rows = []
    for n in n_grid:
        r = [n]
        for frac in (1.0, 0.6):
            mu = frac * ph.mu_total(n, hp)
            ov = dict(host="In2O3", n=n, mu=mu)
            for arch in ("SHJ", "PSK"):
                b = dv.optimise(arch, "tco", dv.tco_family("ITO", ov))
                r += [b["eta"], b["x"], b["Rs"]]
        rows.append(r)
    rows = np.array(rows)
    save("nscan.csv",
         ["n", "SHJ_lim", "tSHJ_lim", "RsSHJ_lim", "PSK_lim", "tPSK_lim", "RsPSK_lim",
          "SHJ_real", "tSHJ_real", "RsSHJ_real", "PSK_real", "tPSK_real", "RsPSK_real"],
         rows.T)
    for col, name in [(7, "SHJ"), (10, "PSK")]:
        i = np.argmax(rows[:, col])
        macro(f"nbest{name}", rows[i, 0] / 1e20, "{:.1f}")
        macro(f"etanbest{name}", rows[i, col])
        macro(f"tnbest{name}", rows[i, col + 1], "{:.0f}")
        macro(f"Rsnbest{name}", rows[i, col + 2], "{:.0f}")


# 5. monte carlo sensitivity runs
def lognorm(mu, s):
    return mu * np.exp(RNG.normal(0, s))


def sample_family(nm):
    """draw random input parameters for electrode nm (table 3.4)"""
    if nm in el.TCOS:
        d = el.TCOS[nm]
        ov = dict(n=lognorm(d["n"], 0.15), mu=lognorm(d["mu"], 0.20),
                  m0_scale=RNG.uniform(0.93, 1.07), eps_scale=RNG.uniform(0.97, 1.03),
                  Eg_shift=RNG.normal(0, 0.05))
        return "tco", dv.tco_family(nm, ov), None
    if nm == "DMD":
        A = RNG.uniform(0.6, 1.6)
        return "dmd", dv.dmd_family(A_size=A), A
    if nm == "GRA":
        return "graphene", dv.graphene_family(EF=RNG.uniform(0.30, 0.55),
                                              mu=np.exp(RNG.uniform(np.log(1000), np.log(5000))),
                                              beta=RNG.uniform(0.7, 1.0)), None
    d = el.NETWORKS[nm]
    return "network", dv.network_family(nm, RNG.uniform(d["lo"], d["hi"])), None


def part_montecarlo(Nmc):
    """run Nmc random input trials per device and track rank probabilities"""
    progress(f"Monte Carlo, N = {Nmc}")
    # slightly coarser grids to keep runtime manageable
    T_MC = np.concatenate([np.arange(20, 200, 10.0), np.arange(200, 1001, 40.0)])
    dmd_coarse = [(d1, a, d2)
                  for d1 in (30, 40, 50, 60)
                  for a in (7, 8, 10, 12)
                  for d2 in (20, 40, 60, 80)]
    for arch in ["SHJ", "PSK"]:
        E = np.zeros((Nmc, len(el.ORDER)))
        for i in range(Nmc):
            for j, nm in enumerate(el.ORDER):
                kind, fam, extra = sample_family(nm)
                if kind == "tco":
                    b = dv.optimise(arch, kind, fam, grid=T_MC)
                elif kind == "dmd":
                    tmin = RNG.uniform(6, 9)           # min continuous silver thickness
                    g = [x for x in dmd_coarse if x[1] >= tmin] or dmd_coarse
                    b = dv.optimise(arch, kind, fam, grid=g)
                else:
                    b = dv.optimise(arch, kind, fam)
                E[i, j] = b["eta"]
            if (i + 1) % 25 == 0:
                progress(f"  {arch}: {i + 1}/{Nmc}")
        np.savetxt(os.path.join(OUT, f"mc_raw_{arch}.csv"), E, delimiter=",",
                   header=",".join(el.ORDER), comments="", fmt="%.4f")
        q = np.percentile(E, [5, 25, 50, 75, 95], axis=0)
        save(f"mc_{arch}.csv", ["i", "p5", "p25", "p50", "p75", "p95", "mean"],
             [np.arange(len(el.ORDER)), q[0], q[1], q[2], q[3], q[4], E.mean(0)], fmt="%.4f")
        ranks = np.argsort(np.argsort(-E, axis=1), axis=1)     # 0 = first place
        P = np.zeros((len(el.ORDER), len(el.ORDER)))
        for j in range(len(el.ORDER)):
            for r in range(len(el.ORDER)):
                P[j, r] = np.mean(ranks[:, j] == r)
        rows = [[j, r, P[j, r]] for j in range(len(el.ORDER)) for r in range(len(el.ORDER))]
        np.savetxt(os.path.join(OUT, f"rank_{arch}.csv"), np.array(rows), delimiter=",",
                   header="electrode,rank,p", comments="", fmt="%.4f")
        for j, nm in enumerate(el.ORDER):
            macro(f"mc{arch}{nm}", q[2, j])        # median
            macro(f"mclo{arch}{nm}", q[0, j])      # 5th percentile
            macro(f"mchi{arch}{nm}", q[4, j])      # 95th percentile
            macro(f"pfirst{arch}{nm}", P[j, 0] * 100, "{:.0f}")
    macro("Nmc", Nmc, "{:d}")


# 6. validation checks and spectrum test
def part_validation():
    """benchmark checks against known results (report table 3.5)"""
    progress("validation")
    lam = np.arange(300.0, 1201.0, 5.0)
    # quarter wave AR check and lossless energy conservation R + T = 1
    n1 = np.sqrt(1.52)
    d = 550 / (4 * n1)
    R, T = tmm.unpolarised([1.0, n1, 1.52], [np.inf, d, np.inf], 0.0, lam)[:2]
    i = np.argmin(abs(lam - 550))
    macro("valQWR", R[i], "{:.1e}")
    errs = []
    for th in np.linspace(0, 1.3, 7):
        for pol in "sp":
            o = tmm.coh_tmm(pol, [1.0, 2.0 + 0j, 1.45, 3.5], [np.inf, 80, 120, np.inf], th, lam)
            errs.append(np.max(abs(o["R"] + o["T"] - 1)))
    macro("valEnergy", max(errs), "{:.0e}")
    # graphene absorption should be close to pi * alpha (~2.29%)
    Ng = np.sqrt(ph.graphene_eps(ph.eV_from_nm(lam), EF=0.0001, mu_cm2=3000))
    R, T, A = tmm.unpolarised([1.0, Ng, 1.0], [np.inf, ph.GRAPHENE_D * 1e9, np.inf], 0.0, lam)
    Nd = np.sqrt(ph.graphene_eps(ph.eV_from_nm(lam), EF=0.45, mu_cm2=3000))
    Rd, Td, Ad = tmm.unpolarised([1.0, Nd, 1.0], [np.inf, ph.GRAPHENE_D * 1e9, np.inf], 0.0, lam)
    alpha_fs = ph.q**2 / (4 * np.pi * ph.eps0 * ph.hbar * ph.c0)
    pi_alpha = np.full_like(lam, np.pi * alpha_fs * 100)
    save("val_graphene.csv", ["lam", "A0", "Adoped", "pialpha"], [lam, A * 100, Ad * 100, pi_alpha])
    macro("valGraA", A[np.argmin(abs(lam - 550))] * 100, "{:.2f}")
    macro("pialpha", np.pi * alpha_fs * 100, "{:.2f}")
    # bulk silver reflectance and resistivity
    Nag = np.sqrt(ph.ag_eps(ph.eV_from_nm(lam)))
    Rag = np.abs((1 - Nag) / (1 + Nag)) ** 2
    macro("valAgR", Rag[np.argmin(abs(lam - 550))] * 100, "{:.1f}")
    macro("valAgRho", ph.ag_resistivity(1e6) * 1e8, "{:.2f}")   # 1 mm approx bulk
    # shockley queisser check on the solar spectrum
    lam_fine = np.linspace(280, 4000, 15000)
    G = ph.am15g(lam_fine)
    save("spectrum.csv", ["lam", "G"], [lam_fine[::10], G[::10]])
    r = sq.sq_efficiency([1.12, 1.34, 1.6], lam_fine, G)    # Si, SQ peak, perovskite
    macro("JscSi", r[0][1])
    macro("JscPsk", r[2][1])
    macro("SQmax", r[1][3], "{:.1f}")
    macro("SQsi", r[0][3], "{:.1f}")
    Egs = np.linspace(0.7, 2.6, 60)
    rr = np.array(sq.sq_efficiency(Egs, lam_fine, G))
    save("sq.csv", ["Eg", "Jsc", "Vmp", "eta"], rr.T)
    # ZnO mobility limit check against ellmer 2001
    macro("valMuZnO", ph.mu_total(5e20, el.HOSTS["ZnO"]), "{:.0f}")
    macro("valMuITO", ph.mu_total(1e21, el.HOSTS["In2O3"]), "{:.0f}")


def part_spectrum_sensitivity():
    """rerun baseline with a 5778 K blackbody to check if ranking shifts"""
    progress("spectral sensitivity (pure 5778 K black body)")
    lam = dv.LAM
    saved_flux = dv.FLUX.copy()
    bb = ph.planck_toa(lam, 5778.0)
    lam_norm = np.linspace(280, 4000, 5000)
    bb *= 1000 / trapezoid(ph.planck_toa(lam_norm, 5778.0), lam_norm)   # scale to 1000 W/m2
    dv.FLUX = ph.photon_flux(lam, bb)
    dv._AR_CACHE.clear()
    out = []
    for arch in ["SHJ", "PSK"]:
        etas = []
        for nm in el.ORDER:
            if nm in el.TCOS:
                b = dv.optimise(arch, "tco", dv.tco_family(nm))
            elif nm == "DMD":
                b = dv.optimise(arch, "dmd", dv.dmd_family(), grid=dv.dmd_grid())
            elif nm == "GRA":
                b = dv.optimise(arch, "graphene", dv.graphene_family())
            else:
                b = dv.optimise(arch, "network", dv.network_family(nm))
            etas.append(b["eta"])
        out.append(etas)
    dv.FLUX = saved_flux           # restore am1.5g flux
    dv._AR_CACHE.clear()
    return np.array(out)


def write_numbers():
    with open(os.path.join(OUT, "numbers.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by run_all.py\n")
        for k, v in sorted(NUM.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    with open(os.path.join(OUT, "numbers.json"), "w", encoding="utf-8") as f:
        json.dump(NUM, f, indent=1)


if __name__ == "__main__":
    t0 = time.time()
    macro("specname", dv.SPEC_NAME)
    part_validation()
    part_materials()
    part_films()
    part_designmap()
    part_grid()
    res = part_devices()
    macro("lossScaleMin", min(SCALES))
    macro("lossScaleMax", max(SCALES))
    part_nscan()
    sens = part_spectrum_sensitivity()
    base = np.array([[res[a][nm]["eta"] for nm in el.ORDER] for a in ["SHJ", "PSK"]])
    # check rank agreement and max shift under blackbody spectrum
    for k, a in enumerate(["SHJ", "PSK"]):
        rank_base = np.argsort(np.argsort(-base[k]))
        rank_bb = np.argsort(np.argsort(-sens[k]))
        macro(f"sensRankSame{a}", int(np.sum(rank_base == rank_bb)), "{:d}")
        macro(f"sensMaxShift{a}", np.max(abs((sens[k] - sens[k].mean()) - (base[k] - base[k].mean()))))
    save("sensitivity.csv", ["i", "SHJbase", "SHJbb", "PSKbase", "PSKbb"],
         [np.arange(len(el.ORDER)), base[0], sens[0], base[1], sens[1]], fmt="%.4f")
    write_numbers()                 # save before monte carlo
    part_montecarlo(6 if QUICK else 300)
    write_numbers()
    progress(f"done in {time.time() - t0:.0f} s")
