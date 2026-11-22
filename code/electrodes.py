"""
electrodes.py -- the materials: the nine electrodes and the two absorbers.

The film parameters are representative values from the literature (report,
Tables 3.1 and 3.2). They are inputs to the model, not measurements made in
this project.
"""
from functools import lru_cache

import numpy as np

import physics as ph

# --------------------------------------------------------------------------- host oxides
# Eg0     undoped band gap (eV)            m0, C   effective mass and non-parabolicity
# eps_s   static permittivity              eps_vis visible permittivity of the undoped oxide
# mu_L    lattice-limited mobility         mv      valence-band mass (inf = flat band)
# TL_dE0, TL_C  shape of the Tauc-Lorentz background
HOSTS = {
    "In2O3": dict(Eg0=3.60, m0=0.18, C=0.50, eps_s=9.0, eps_vis=3.90, mu_L=250.0,
                  mv=np.inf, TL_dE0=1.2, TL_C=1.6),
    "ZnO":   dict(Eg0=3.37, m0=0.24, C=0.29, eps_s=8.75, eps_vis=3.75, mu_L=200.0,
                  mv=1.8, TL_dE0=1.0, TL_C=1.4),
    "SnO2":  dict(Eg0=3.60, m0=0.25, C=0.30, eps_s=11.5, eps_vis=3.85, mu_L=250.0,
                  mv=np.inf, TL_dE0=1.2, TL_C=1.6),
}

# --------------------------------------------------------------------------- the four oxide films
# n in cm^-3, mu in cm^2/Vs: typical good films from the literature.
TCOS = {
    "ITO":  dict(host="In2O3", n=1.0e21, mu=40.0, label="ITO (In$_2$O$_3$:Sn)"),
    "IOH":  dict(host="In2O3", n=1.5e20, mu=100.0, label="IO:H (In$_2$O$_3$:H)"),
    "AZO":  dict(host="ZnO",   n=5.0e20, mu=25.0, label="AZO (ZnO:Al)"),
    "FTO":  dict(host="SnO2",  n=4.0e20, mu=25.0, label="FTO (SnO$_2$:F)"),
}

# --------------------------------------------------------------------------- network electrodes
# Nanowires, PEDOT:PSS and nanotubes are described by one number, the ratio of
# their DC to optical conductivity. lo and hi are the range used in the Monte Carlo.
NETWORKS = {
    "AgNW":   dict(ratio=350.0, lo=150.0, hi=500.0, label="Ag nanowires"),
    "PEDOT":  dict(ratio=35.0, lo=20.0, hi=60.0, label="PEDOT:PSS"),
    "CNT":    dict(ratio=13.0, lo=5.0, hi=25.0, label="CNT network"),
}

# the order used in every table and figure
ORDER = ["ITO", "IOH", "AZO", "FTO", "DMD", "GRA", "AgNW", "PEDOT", "CNT"]
LABELS = {"ITO": "ITO", "IOH": "IO:H", "AZO": "AZO", "FTO": "FTO",
          "DMD": "ZnO/Ag/ZnO", "GRA": "Graphene", "AgNW": "Ag NW",
          "PEDOT": "PEDOT:PSS", "CNT": "CNT"}


# --------------------------------------------------------------------------- oxide optics
# The Kramers-Kronig background is slow, and the same one is needed thousands
# of times in the Monte Carlo, so it is cached. A NumPy array cannot be a cache
# key, so the wavelength grid is passed in as bytes.
@lru_cache(maxsize=4096)
def _tl_background(host, Eg, lam_key, eps_vis, dE0, Cb):
    lam = np.frombuffer(lam_key)
    hp = dict(HOSTS[host])
    hp["TL_dE0"] = dE0
    hp["TL_C"] = Cb
    A, E0, Cbb, off = ph.calibrate_tl(hp, Eg, eps_vis)
    return ph.tl_eps(ph.eV_from_nm(lam), A, E0, Cbb, Eg, off)


def oxide_eps(lam, host, n_cm3, mu, m0_scale=1.0, eps_scale=1.0, Eg_shift=0.0):
    """Complex permittivity of a doped oxide: the interband background, with a
    gap that depends on n, plus the Drude free electrons.

    m0_scale, eps_scale and Eg_shift nudge the host parameters; they are only
    used by the Monte Carlo test."""
    hp = dict(HOSTS[host])
    hp["m0"] *= m0_scale
    Eg = float(ph.optical_gap(n_cm3, hp)) + Eg_shift if n_cm3 > 0 else hp["Eg0"] + Eg_shift
    Eg = round(Eg, 3)                      # rounding lets nearby gaps share a cache entry
    lam = np.ascontiguousarray(lam, float)
    background = _tl_background(host, Eg, lam.tobytes(), round(hp["eps_vis"] * eps_scale, 4),
                                hp["TL_dE0"], hp["TL_C"])
    if n_cm3 <= 0:
        return background
    mstar = ph.effective_mass(n_cm3, hp["m0"], hp["C"])
    return background + ph.drude_eps(ph.eV_from_nm(lam), n_cm3, mu, mstar)


def oxide_index(lam, host, n_cm3, mu, **kw):
    """Complex refractive index n + ik of a doped oxide."""
    return np.sqrt(oxide_eps(lam, host, n_cm3, mu, **kw))


def oxide_resistivity(n_cm3, mu):
    """Resistivity (ohm m) from rho = 1 / (n e mu)."""
    return 1 / (n_cm3 * 1e6 * ph.q * mu * 1e-4)


def network_T(Rs, ratio):
    """Transmittance of a network electrode at sheet resistance Rs, from the
    thin-sheet (Tinkham) relation T = (1 + Z0 / (2 Rs ratio))^-2."""
    return (1 + ph.Z0 / (2 * Rs) / ratio) ** -2


# --------------------------------------------------------------------------- absorbers
# Silicon: refractive index and absorption coefficient, tabulated roughly after Green (2008).
_SI_LAM = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850,
                    900, 950, 1000, 1050, 1100, 1150, 1200], float)
_SI_N = np.array([5.00, 5.48, 5.57, 4.67, 4.30, 4.08, 3.94, 3.85, 3.78, 3.73,
                  3.69, 3.66, 3.62, 3.60, 3.57, 3.55, 3.54, 3.53, 3.52])
_SI_ALPHA = np.array([1.73e6, 1.04e6, 9.52e4, 2.55e4, 1.11e4, 6.39e3, 4.14e3,
                      2.81e3, 1.90e3, 1.30e3, 8.50e2, 5.35e2, 3.06e2, 1.57e2,
                      6.40e1, 1.63e1, 3.50, 3.2e-1, 2.2e-2])   # cm^-1


def si_index(lam):
    """Complex index of silicon and its absorption coefficient (cm^-1)."""
    n = np.interp(lam, _SI_LAM, _SI_N)
    alpha = np.exp(np.interp(lam, _SI_LAM, np.log(_SI_ALPHA)))   # interpolate on a log scale
    k = alpha * 1e2 * lam * 1e-9 / (4 * np.pi)
    return n + 1j * k, alpha


def si_lambertian_absorption(lam, W_um=150.0):
    """Fraction of light absorbed in a textured wafer W_um thick. Light
    trapping lengthens the path by about 4 n^2 (the Lambertian limit)."""
    n, alpha = si_index(lam)
    x = 4 * np.real(n) ** 2 * alpha * W_um * 1e-4
    return x / (1 + x)


PEROVSKITE = dict(n=2.45, Eg=1.60)


def perovskite_absorption(lam):
    """Perovskite absorption: close to 1 below the band-edge wavelength, then
    a smooth fall-off over a few nm."""
    lam_gap = ph.HC_EVNM / PEROVSKITE["Eg"]
    return 1 / (1 + np.exp((lam - lam_gap) / 6.0))
