# material inputs for the 9 electrodes and 2 absorbers
# values taken from literature (report tables 3.1 and 3.2)

from functools import lru_cache

import numpy as np

import physics as ph

# host oxide parameters
# Eg0 undoped gap eV, m0 band edge mass, C non parabolicity
# eps_s static permittivity, eps_vis visible permittivity, mu_L lattice mobility, mv valence mass
HOSTS = {
    "In2O3": dict(Eg0=3.60, m0=0.18, C=0.50, eps_s=9.0, eps_vis=3.90, mu_L=250.0,
                  mv=np.inf, TL_dE0=1.2, TL_C=1.6),
    "ZnO":   dict(Eg0=3.37, m0=0.24, C=0.29, eps_s=8.75, eps_vis=3.75, mu_L=200.0,
                  mv=1.8, TL_dE0=1.0, TL_C=1.4),
    "SnO2":  dict(Eg0=3.60, m0=0.25, C=0.30, eps_s=11.5, eps_vis=3.85, mu_L=250.0,
                  mv=np.inf, TL_dE0=1.2, TL_C=1.6),
}

# four oxide films, n in 1/cm3 and mu in cm2/Vs
TCOS = {
    "ITO":  dict(host="In2O3", n=1.0e21, mu=40.0, label="ITO (In$_2$O$_3$:Sn)"),
    "IOH":  dict(host="In2O3", n=1.5e20, mu=100.0, label="IO:H (In$_2$O$_3$:H)"),
    "AZO":  dict(host="ZnO",   n=5.0e20, mu=25.0, label="AZO (ZnO:Al)"),
    "FTO":  dict(host="SnO2",  n=4.0e20, mu=25.0, label="FTO (SnO$_2$:F)"),
}

# network electrodes described by dc to optical conductivity ratio, lo and hi for monte carlo
NETWORKS = {
    "AgNW":   dict(ratio=350.0, lo=150.0, hi=500.0, label="Ag nanowires"),
    "PEDOT":  dict(ratio=35.0, lo=20.0, hi=60.0, label="PEDOT:PSS"),
    "CNT":    dict(ratio=13.0, lo=5.0, hi=25.0, label="CNT network"),
}

# consistent ordering for tables and plots
ORDER = ["ITO", "IOH", "AZO", "FTO", "DMD", "GRA", "AgNW", "PEDOT", "CNT"]
LABELS = {"ITO": "ITO", "IOH": "IO:H", "AZO": "AZO", "FTO": "FTO",
          "DMD": "ZnO/Ag/ZnO", "GRA": "Graphene", "AgNW": "Ag NW",
          "PEDOT": "PEDOT:PSS", "CNT": "CNT"}


# cache the slow kramers kronig background for monte carlo runs
# pass lam as bytes since numpy arrays are not hashable
@lru_cache(maxsize=4096)
def _tl_background(host, Eg, lam_key, eps_vis, dE0, Cb):
    lam = np.frombuffer(lam_key)
    hp = dict(HOSTS[host])
    hp["TL_dE0"] = dE0
    hp["TL_C"] = Cb
    A, E0, Cbb, off = ph.calibrate_tl(hp, Eg, eps_vis)
    return ph.tl_eps(ph.eV_from_nm(lam), A, E0, Cbb, Eg, off)


def oxide_eps(lam, host, n_cm3, mu, m0_scale=1.0, eps_scale=1.0, Eg_shift=0.0):
    """complex permittivity of a doped oxide (TL background + drude)"""
    hp = dict(HOSTS[host])
    hp["m0"] *= m0_scale
    Eg = float(ph.optical_gap(n_cm3, hp)) + Eg_shift if n_cm3 > 0 else hp["Eg0"] + Eg_shift
    Eg = round(Eg, 3)                      # round slightly so cache hits more often
    lam = np.ascontiguousarray(lam, float)
    background = _tl_background(host, Eg, lam.tobytes(), round(hp["eps_vis"] * eps_scale, 4),
                                hp["TL_dE0"], hp["TL_C"])
    if n_cm3 <= 0:
        return background
    mstar = ph.effective_mass(n_cm3, hp["m0"], hp["C"])
    return background + ph.drude_eps(ph.eV_from_nm(lam), n_cm3, mu, mstar)


def oxide_index(lam, host, n_cm3, mu, **kw):
    return np.sqrt(oxide_eps(lam, host, n_cm3, mu, **kw))


def oxide_resistivity(n_cm3, mu):
    """resistivity in ohm m"""
    return 1 / (n_cm3 * 1e6 * ph.q * mu * 1e-4)


def network_T(Rs, ratio):
    """tinkham transmittance relation for network electrodes"""
    return (1 + ph.Z0 / (2 * Rs) / ratio) ** -2


# silicon optical data roughly from green 2008
_SI_LAM = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850,
                    900, 950, 1000, 1050, 1100, 1150, 1200], float)
_SI_N = np.array([5.00, 5.48, 5.57, 4.67, 4.30, 4.08, 3.94, 3.85, 3.78, 3.73,
                  3.69, 3.66, 3.62, 3.60, 3.57, 3.55, 3.54, 3.53, 3.52])
_SI_ALPHA = np.array([1.73e6, 1.04e6, 9.52e4, 2.55e4, 1.11e4, 6.39e3, 4.14e3,
                      2.81e3, 1.90e3, 1.30e3, 8.50e2, 5.35e2, 3.06e2, 1.57e2,
                      6.40e1, 1.63e1, 3.50, 3.2e-1, 2.2e-2])   # 1/cm


def si_index(lam):
    """silicon complex refractive index and alpha in 1/cm"""
    n = np.interp(lam, _SI_LAM, _SI_N)
    alpha = np.exp(np.interp(lam, _SI_LAM, np.log(_SI_ALPHA)))   # log interpolation
    k = alpha * 1e2 * lam * 1e-9 / (4 * np.pi)
    return n + 1j * k, alpha


def si_lambertian_absorption(lam, W_um=150.0):
    """lambertian light trapping absorption in silicon wafer of thickness W_um"""
    n, alpha = si_index(lam)
    x = 4 * np.real(n) ** 2 * alpha * W_um * 1e-4
    return x / (1 + x)


PEROVSKITE = dict(n=2.45, Eg=1.60)


def perovskite_absorption(lam):
    """smooth step absorption edge for perovskite"""
    lam_gap = ph.HC_EVNM / PEROVSKITE["Eg"]
    return 1 / (1 + np.exp((lam - lam_gap) / 6.0))
