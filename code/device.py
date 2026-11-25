# device simulation for shj silicon cell and perovskite thin film module
# each electrode family returns (optics(arch, x), Rs(x)) where x is thickness, layers, or Rs

import numpy as np
from numpy import trapezoid
import physics as ph
import electrodes as el
import tmm

LAM = np.arange(300.0, 1201.0, 4.0)               # device wavelength grid nm
SPEC, SPEC_NAME = ph.load_spectrum(LAM)
FLUX = ph.photon_flux(LAM, SPEC)                  # photons/m2/s/nm

# shj cell parameters (report table 3.3)
SHJ = dict(
    Voc_ref=0.740,       # reference Voc V
    Jsc_ref=40.0,        # reference Jsc mA/cm2 to set J0
    n_id=1.0,            # ideality factor
    r_other=0.25,        # other series resistance ohm cm2
    w_f=35e-6,           # finger width m
    h_f=15e-6,           # finger height m
    rho_f=5e-8,          # silver paste resistivity ohm m
    L_half=7.8e-3,       # half finger length m
    busbar_shade=0.015,  # busbar shading fraction
    W_um=150.0,          # wafer thickness um
    rho_wafer=1.5,       # wafer resistivity ohm cm
)
# perovskite module parameters
PSK = dict(
    Voc_ref=1.15,        # V
    Jsc_ref=24.0,        # mA/cm2
    n_id=1.3,
    r_other=1.0,         # ohm cm2
    dead=150e-6,         # P1 to P3 scribe dead width m
    glass_n=1.52,
)
AR_INDEX = 2.0      # ideal AR index for non film electrodes in shj
VT = ph.kB * 298.15 / ph.q    # thermal voltage at 25 C


# diode functions
def diode_mpp(Jsc, J0, n_id):
    """find ideal diode mpp by scanning voltage, currents in A/m2"""
    nVt = n_id * VT
    Voc = nVt * np.log(Jsc / J0 + 1)
    V = np.linspace(0, Voc, 3000)
    J = Jsc - J0 * np.expm1(V / nVt)
    P = V * J
    i = np.argmax(P)
    return Voc, V[i], J[i], P[i]


def diode_mpp_vec(Jsc, J0, n_id, nV=1500):
    """vectorised diode mpp for an array of Jsc values"""
    Jsc = np.atleast_1d(Jsc)
    nVt = n_id * VT
    Voc = nVt * np.log(Jsc / J0 + 1)
    x = np.linspace(0, 1, nV)[None, :]
    V = Voc[:, None] * x
    P = V * (Jsc[:, None] - J0 * np.expm1(V / nVt))
    i = np.argmax(P, axis=1)
    r = np.arange(len(Jsc))
    Vmp = V[r, i]
    return Voc, Vmp, P[r, i] / Vmp, P[r, i]


def J0_of(arch):
    """saturation current J0 in A/m2 matching reference Voc and Jsc"""
    return arch["Jsc_ref"] * 10 / np.exp(arch["Voc_ref"] / (arch["n_id"] * VT))


# optics helpers
def jph(T_eff):
    """photocurrent density in A/m2"""
    return ph.q * trapezoid(FLUX * T_eff, LAM)


# precompute absorber data on device grid
SI_LL = el.si_lambertian_absorption(LAM, SHJ["W_um"])
SI_N = el.si_index(LAM)[0]
PSK_ABS = el.perovskite_absorption(LAM)
PSK_N = np.full_like(LAM, el.PEROVSKITE["n"], dtype=complex)


def shj_optics_film(n_films, d_films):
    Tin, Apar, R = tmm.textured_front_into_absorber(n_films, d_films, SI_N, LAM)
    return Tin, Apar


def psk_optics_film(n_films, d_films):
    Tin, Apar, R = tmm.superstrate_into_absorber(n_films, d_films, PSK["glass_n"], PSK_N, LAM)
    return Tin, Apar


_AR_CACHE = {}


def shj_optics_ar_only():
    """best lossless AR layer thickness for shj cell"""
    if "shj" not in _AR_CACHE:
        best = None
        for d in np.arange(50, 111, 2.0):
            Tin, A = shj_optics_film([np.full_like(LAM, AR_INDEX, dtype=complex)], [d])
            j = jph(Tin * SI_LL)
            if best is None or j > best[0]:
                best = (j, Tin, d)
        _AR_CACHE["shj"] = best
    return _AR_CACHE["shj"]


def psk_optics_bare():
    """perovskite under bare glass with no front electrode"""
    if "psk" not in _AR_CACHE:
        Tin, A = psk_optics_film([], [])
        _AR_CACHE["psk"] = Tin
    return _AR_CACHE["psk"]


def psk_optics_ideal():
    """lossless index matching layer n = sqrt(n_glass * n_abs) at best thickness"""
    if "psk_ideal" not in _AR_CACHE:
        nm = np.sqrt(PSK["glass_n"] * el.PEROVSKITE["n"])
        best = None
        for d in np.arange(20, 400, 2.0):
            Tin, A = psk_optics_film([np.full_like(LAM, nm, dtype=complex)], [d])
            j = jph(Tin * PSK_ABS)
            if best is None or j > best[0]:
                best = (j, Tin, d)
        _AR_CACHE["psk_ideal"] = best
    return _AR_CACHE["psk_ideal"]


# device loss models
def shj_evaluate(Tin, Apar, Rs_tco, junction="front", ideal=False):
    """evaluate shj efficiency and optimise finger spacing s"""
    p = SHJ
    J0 = J0_of(p)
    Jgen = jph(Tin * SI_LL)
    if junction == "rear":
        # wafer helps carry lateral current in rear junction cell
        Rs_w = p["rho_wafer"] * 1e-2 / (p["W_um"] * 1e-6)       # ohm/sq
        Rs_eff = 1 / (1 / Rs_tco + 1 / Rs_w)
    else:
        Rs_eff = Rs_tco
    s = np.linspace(0.4e-3, 6e-3, 225)                 # finger spacings m
    shade = p["w_f"] / s + p["busbar_shade"]
    Jsc = Jgen * (1 - shade)
    Voc, Vmp, Jmp, Pmp = diode_mpp_vec(Jsc, J0, p["n_id"])
    # fractional losses
    p_lat = Rs_eff * Jmp * s**2 / (12 * Vmp)           # lateral tco loss
    p_fin = p["rho_f"] * Jmp * p["L_half"] ** 2 * s / (3 * Vmp * p["w_f"] * p["h_f"])   # finger loss
    p_oth = p["r_other"] * 1e-4 * Jmp / Vmp            # rest of series resistance
    P = Pmp * (1 - p_lat - p_fin - p_oth)
    i = int(np.argmax(P))
    best = dict(P=P[i], s=s[i], shade=shade[i], Jsc=Jsc[i], Voc=Voc[i], Pmp0=Pmp[i],
                p_lat=p_lat[i], p_fin=p_fin[i], p_oth=p_oth[i], Jgen=Jgen)
    best["eta"] = best["P"] / 1000 * 100
    best["Rs_eff"] = Rs_eff
    return best


def psk_evaluate(Tin, Apar, Rs):
    """evaluate perovskite module efficiency and optimise stripe width W"""
    p = PSK
    J0 = J0_of(p)
    Jgen = jph(Tin * PSK_ABS)
    Voc, Vmp, Jmp, Pmp = diode_mpp(Jgen, J0, p["n_id"])
    p_oth = p["r_other"] * 1e-4 * Jmp / Vmp
    W = np.linspace(1e-3, 25e-3, 481)                  # stripe widths m
    p_lat = Rs * Jmp * W**2 / (3 * Vmp)                # full stripe lateral loss
    gff = W / (W + p["dead"])                          # geometric fill factor
    P = Pmp * gff * (1 - p_lat - p_oth)
    i = int(np.argmax(P))
    best = dict(P=P[i], W=W[i], gff=gff[i], p_lat=p_lat[i], p_oth=p_oth,
                Jsc=Jgen, Voc=Voc, Pmp0=Pmp, Jgen=Jgen)
    best["eta"] = best["P"] / 1000 * 100
    return best


# electrode families
def tco_family(name, overrides=None):
    """returns (optics, Rs, N) for a tco with optional monte carlo overrides"""
    d = dict(el.TCOS[name]); d.update(overrides or {})
    kw = {k: d[k] for k in ("m0_scale", "eps_scale", "Eg_shift") if k in d}
    N = el.oxide_index(LAM, d["host"], d["n"], d["mu"], **kw)
    rho = el.oxide_resistivity(d["n"], d["mu"])

    def optics(arch, t):
        f = shj_optics_film if arch == "SHJ" else psk_optics_film
        return f([N], [t])

    def Rs(t):
        return rho / (t * 1e-9)
    return optics, Rs, N


def dmd_family(A_size=1.0, tAg_min=7.0, d_diel=None):
    """ZnO/Ag/ZnO multilayer with x = (d1, dAg, d2)"""
    Nz = el.oxide_index(LAM, "ZnO", 0.0, 1.0)

    def optics(arch, x):
        d1, dAg, d2 = x
        Nag = np.sqrt(ph.ag_eps(ph.eV_from_nm(LAM), dAg, A_size))
        f = shj_optics_film if arch == "SHJ" else psk_optics_film
        # layers ordered from incident side
        return f([Nz, Nag, Nz], [d1, dAg, d2])

    def Rs(x):
        return ph.ag_resistivity(x[1], A_size) / (x[1] * 1e-9)
    return optics, Rs


def graphene_family(EF=0.45, mu=3000.0, beta=1.0, arch_ar=True):
    Ng = np.sqrt(ph.graphene_eps(ph.eV_from_nm(LAM), EF, mu))
    Rs1, n2d = ph.graphene_sheet_resistance(EF, mu)

    def optics(arch, N):
        N = int(N)
        if arch == "SHJ":
            # ideal AR cap on top of graphene layers
            films = [np.full_like(LAM, AR_INDEX, dtype=complex)] + [Ng] * N
            dl = [shj_optics_ar_only()[2]] + [ph.GRAPHENE_D * 1e9] * N
            return shj_optics_film(films, dl)
        return psk_optics_film([Ng] * N, [ph.GRAPHENE_D * 1e9] * N)

    def Rs(N):
        return Rs1 / N**beta
    return optics, Rs


def network_family(name, ratio=None):
    r = ratio if ratio is not None else el.NETWORKS[name]["ratio"]

    def optics(arch, Rs):
        Te = el.network_T(Rs, r)
        if arch == "SHJ":
            Tin = shj_optics_ar_only()[1] * Te
        else:
            Tin = psk_optics_bare() * Te
        return Tin, (1 - Te) * np.ones_like(LAM)

    def Rsf(Rs):
        return Rs
    return optics, Rsf


# optimisers
T_GRID = np.concatenate([np.arange(20, 200, 5.0), np.arange(200, 1001, 20.0)])
RS_GRID = np.logspace(0, 3.6, 73)


def optimise(arch, kind, fam, junction="front", grid=None, keep_curve=False):
    """sweep design variable and pick best efficiency"""
    optics, Rs = fam[0], fam[1]
    if kind == "tco":
        xs = grid if grid is not None else T_GRID
    elif kind == "graphene":
        xs = np.arange(1, 9)
    elif kind == "network":
        xs = RS_GRID
    elif kind == "dmd":
        xs = grid
    curve = []
    best = None
    for x in xs:
        Tin, Apar = optics(arch, x)
        R = Rs(x)
        res = shj_evaluate(Tin, Apar, R, junction) if arch == "SHJ" else psk_evaluate(Tin, Apar, R)
        res["x"] = x
        res["Rs"] = R
        absorb = SI_LL if arch == "SHJ" else PSK_ABS
        res["J_par"] = jph(Apar * absorb) / 10      # parasitic loss mA/cm2
        res["J_in"] = jph(Tin * absorb) / 10
        if keep_curve:
            curve.append(res)
        if best is None or res["eta"] > best["eta"]:
            best = res
    return (best, curve) if keep_curve else best


def dmd_grid(coarse=False):
    if coarse:
        d = np.arange(20, 81, 10.0); a = np.arange(7, 17, 2.0)
    else:
        d = np.arange(10, 91, 5.0); a = np.arange(7, 18.1, 1.0)
    return [(d1, ag, d2) for d1 in d for ag in a for d2 in d]


def ideal_reference(arch):
    """lossless zero resistance reference electrode"""
    if arch == "SHJ":
        Tin = shj_optics_ar_only()[1]
        return shj_evaluate(Tin, 0 * Tin, 1e-6)
    Tin = psk_optics_ideal()[1]
    return psk_evaluate(Tin, 0 * Tin, 1e-6)
