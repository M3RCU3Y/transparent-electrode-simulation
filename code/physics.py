"""
physics.py -- the physical models behind the transparent-electrode comparison.

Brandon Deonarine, PHYS 3153 (UWI St. Augustine), 2026.

What is in here, roughly in the order the report uses it:

  * physical constants
  * the free electrons of a heavily doped oxide (degenerate, non-parabolic band)
  * the optical band gap: Burstein-Moss shift minus band-gap renormalisation
  * the interband background permittivity (Tauc-Lorentz, real part by Kramers-Kronig)
  * the Drude model of the free electrons
  * mobility limits: ionised impurities, lattice, grain boundaries
  * optical models for silver and for graphene
  * the AM1.5G solar spectrum

Units: energies in eV, wavelengths in nm, carrier densities in cm^-3 and
mobilities in cm^2/Vs (the units papers use). Everything else is SI.
"""
import os

import numpy as np
from numpy import trapezoid

# --------------------------------------------------------------------------- constants
q = 1.602176634e-19        # elementary charge, C
me = 9.1093837015e-31      # electron mass, kg
hbar = 1.054571817e-34     # J s
h = 6.62607015e-34         # J s
c0 = 2.99792458e8          # speed of light, m/s
eps0 = 8.8541878128e-12    # F/m
kB = 1.380649e-23          # J/K
Z0 = 376.730313668         # impedance of free space, ohm
TEMP = 300.0               # K
kT_eV = kB * TEMP / q
HC_EVNM = 1239.841984      # hc in eV nm, so E(eV) = HC_EVNM / lambda(nm)


def eV_from_nm(lam_nm):
    """Photon energy (eV) for a wavelength (nm)."""
    return HC_EVNM / np.asarray(lam_nm, dtype=float)


# --------------------------------------------------------------------------- free electrons
# In these oxides the conduction band is not a perfect parabola. A common way
# to describe that is   E (1 + C E) = hbar^2 k^2 / (2 m0*),
# where m0* is the effective mass at the band bottom and C the non-parabolicity.

def fermi_wavevector(n_cm3):
    """Fermi wavevector k_F (1/m) of a degenerate electron gas."""
    n = np.asarray(n_cm3, dtype=float) * 1e6
    return (3 * np.pi**2 * n) ** (1 / 3)


def fermi_energy(n_cm3, m0, C):
    """Fermi energy above the conduction-band minimum (eV)."""
    kF = fermi_wavevector(n_cm3)
    Ek = (hbar * kF) ** 2 / (2 * m0 * me) / q    # what a parabolic band would give, eV
    if C == 0:
        return Ek
    # solve E (1 + C E) = Ek for E
    return (-1 + np.sqrt(1 + 4 * C * Ek)) / (2 * C)


def effective_mass(n_cm3, m0, C):
    """Effective mass at the Fermi level, in units of the electron mass.
    It grows as the band fills, which is why heavily doped ITO is 'heavier'."""
    return m0 * (1 + 2 * C * fermi_energy(n_cm3, m0, C))


# --------------------------------------------------------------------------- optical gap
def bgn(n_cm3, eps_s, m0, C):
    """Band-gap renormalisation (eV), the narrowing of the gap caused by the
    electrons interacting with each other (exchange) and with the donor ions.
    Follows the form used by Hamberg and Granqvist."""
    n = np.asarray(n_cm3, float) * 1e6
    eps = eps0 * eps_s
    kF = fermi_wavevector(n_cm3)
    mstar = effective_mass(n_cm3, m0, C) * me
    EF_par = (hbar * kF) ** 2 / (2 * mstar)                  # J
    kTF = np.sqrt(3 * n * q**2 / (2 * eps * EF_par))         # Thomas-Fermi screening wavevector
    dE_ee = (q**2 * kF / (2 * np.pi**2 * eps)
             + q**2 * kTF / (8 * np.pi * eps) * (1 - 4 / np.pi * np.arctan(kF / kTF)))
    aB = 4 * np.pi * eps * hbar**2 / (mstar * q**2)          # effective Bohr radius
    dE_ei = n * q**2 / (eps * aB * kTF**3)
    return (dE_ee + dE_ei) / q


def optical_gap(n_cm3, mat):
    """Optical gap (eV): the undoped gap, pushed up by the filled states
    (Burstein-Moss), pulled down a little by renormalisation."""
    EF = fermi_energy(n_cm3, mat["m0"], mat["C"])
    kF = fermi_wavevector(n_cm3)
    valence = 0.0
    if mat.get("mv", np.inf) != np.inf:
        # a curved valence band adds its own share of the shift (ZnO)
        valence = (hbar * kF) ** 2 / (2 * mat["mv"] * me) / q
    return mat["Eg0"] + EF + valence - bgn(n_cm3, mat["eps_s"], mat["m0"], mat["C"])


# --------------------------------------------------------------------------- interband background
def tl_eps2(E, A, E0, Cb, Eg):
    """Imaginary permittivity of the Tauc-Lorentz model: zero below the gap Eg,
    a broadened absorption peak (centre E0, width Cb) above it."""
    E = np.asarray(E, float)
    out = np.zeros_like(E)
    above = E > Eg
    Ea = E[above]
    out[above] = A * E0 * Cb * (Ea - Eg) ** 2 / ((Ea**2 - E0**2) ** 2 + Cb**2 * Ea**2) / Ea
    return out


# integration grid for Kramers-Kronig: fine up to 12 eV, coarser out to 80 eV
_XI = np.concatenate([np.linspace(0.0, 12.0, 9001)[1:], np.linspace(12.0, 80.0, 2001)[1:]])


def tl_eps1(E, A, E0, Cb, Eg, eps_off):
    """Real permittivity from tl_eps2 by a numerical Kramers-Kronig integral.
    The singular point is handled by subtracting it out and adding back its
    principal value analytically."""
    E = np.atleast_1d(np.asarray(E, float))
    xi = np.concatenate([[Eg], _XI[_XI > Eg]])
    e2xi = tl_eps2(xi, A, E0, Cb, Eg)
    a, b = xi[0], xi[-1]
    out = np.empty_like(E)
    for i, Ei in enumerate(E):
        e2E = tl_eps2(np.array([Ei]), A, E0, Cb, Eg)[0]
        num = xi * e2xi - Ei * e2E
        den = xi**2 - Ei**2
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(np.abs(den) > 1e-12, num / den, 0.0)
        # principal value of  Ei*e2E / (xi^2 - Ei^2)  over [a, b], done by hand
        if e2E != 0.0:
            pv = (1 / (2 * Ei)) * np.log(abs((b - Ei) * (a + Ei) / ((b + Ei) * (a - Ei))))
        else:
            pv = 0.0
        out[i] = eps_off + (2 / np.pi) * (trapezoid(f, xi) + Ei * e2E * pv)
    return out


def tl_eps(E, A, E0, Cb, Eg, eps_off):
    """Complex Tauc-Lorentz permittivity."""
    return tl_eps1(E, A, E0, Cb, Eg, eps_off) + 1j * tl_eps2(E, A, E0, Cb, Eg)


TL_OFFSET = 2.0   # eps_1 at very high energy


def calibrate_tl(mat, Eg, target_eps_vis):
    """Pick the Tauc-Lorentz amplitude A so the background gives the known
    visible permittivity of the undoped oxide, eps_1(2.25 eV) = target_eps_vis
    (a refractive index of about 1.9-2.0). eps_1 is linear in A, so a single
    evaluation is enough."""
    E0, Cb = Eg + mat["TL_dE0"], mat["TL_C"]
    e1_unit = tl_eps1(np.array([2.25]), 1.0, E0, Cb, Eg, 0.0)[0]
    A = (target_eps_vis - TL_OFFSET) / e1_unit
    return A, E0, Cb, TL_OFFSET


# --------------------------------------------------------------------------- Drude model
def plasma_energy(n_cm3, mstar):
    """Unscreened plasma energy hbar*omega_p (eV)."""
    n = np.asarray(n_cm3, float) * 1e6
    return hbar * np.sqrt(n * q**2 / (eps0 * mstar * me)) / q


def drude_gamma(mu_cm2, mstar):
    """Damping hbar*Gamma (eV), from the mobility: tau = m* mu / e."""
    tau = mstar * me * (mu_cm2 * 1e-4) / q
    return hbar / tau / q


def drude_eps(E, n_cm3, mu_cm2, mstar):
    """Free-electron contribution to the permittivity at photon energy E (eV)."""
    Ep = plasma_energy(n_cm3, mstar)
    G = drude_gamma(mu_cm2, mstar)
    E = np.asarray(E, float)
    return -Ep**2 / (E**2 + 1j * G * E)


def screened_plasma_wavelength(n_cm3, mstar, eps_inf):
    """Wavelength (nm) of the plasma edge, where the film starts to reflect."""
    return HC_EVNM / (plasma_energy(n_cm3, mstar) / np.sqrt(eps_inf))


# --------------------------------------------------------------------------- mobility limits
def mu_ionised_impurity(n_cm3, mat, Z=1.0, Ni_over_n=1.0):
    """Mobility limit set by scattering from the donor ions (cm^2/Vs).
    Brooks-Herring-Dingle for a degenerate gas, with the non-parabolic
    correction in the form given by Pisarkiewicz et al. and Ellmer."""
    n = np.asarray(n_cm3, float) * 1e6
    eps = eps0 * mat["eps_s"]
    mstar = effective_mass(n_cm3, mat["m0"], mat["C"])
    m = mstar * me
    xi_d = (3 * np.pi**2) ** (1 / 3) * eps * h**2 * n ** (1 / 3) / (m * q**2)
    xi_np = 1 - mat["m0"] / mstar
    F = ((1 + 4 * xi_np / xi_d * (1 - xi_np / 8)) * np.log(1 + xi_d)
         - xi_d / (1 + xi_d) - 2 * xi_np * (1 - 5 * xi_np / 16))
    mu = 3 * eps**2 * h**3 / (Z**2 * m**2 * q**3) * (1 / Ni_over_n) / F
    return mu * 1e4


def mu_grain_boundary(n_cm3, mat, Nt_cm2, L_nm):
    """Grain-boundary mobility (cm^2/Vs) from Seto's model: electrons must
    hop a barrier at each boundary between crystal grains of size L.
    Returns (mobility, barrier height in eV). Valid while n*L > Nt."""
    n = np.asarray(n_cm3, float) * 1e6
    Nt = Nt_cm2 * 1e4
    L = L_nm * 1e-9
    eps = eps0 * mat["eps_s"]
    m = effective_mass(n_cm3, mat["m0"], mat["C"]) * me
    phi_b = q**2 * Nt**2 / (8 * eps * n) / q            # barrier height, eV
    mu0 = q * L / np.sqrt(2 * np.pi * m * kB * TEMP)    # m^2/Vs
    return mu0 * np.exp(-phi_b / kT_eV) * 1e4, phi_b


def mu_total(n_cm3, mat, Nt_cm2=None, L_nm=None):
    """Combined mobility limit (Matthiessen's rule: 1/mu = sum of 1/mu_i)."""
    inv = 1 / mu_ionised_impurity(n_cm3, mat) + 1 / mat["mu_L"]
    if Nt_cm2 is not None:
        inv = inv + 1 / mu_grain_boundary(n_cm3, mat, Nt_cm2, L_nm)[0]
    return 1 / inv


# --------------------------------------------------------------------------- silver
# Drude plus a constant background. In thin films the electrons also scatter
# off the surfaces, so the damping grows roughly as hbar * v_F / thickness.
AG = dict(eps_inf=3.7, Ep=8.9, G_bulk=0.021, vF=1.39e6)


def ag_gamma(t_nm, A_size=1.0):
    """Damping (eV) of a silver film t_nm thick."""
    return AG["G_bulk"] + A_size * hbar * AG["vF"] / (t_nm * 1e-9) / q


def ag_eps(E, t_nm=np.inf, A_size=1.0):
    """Permittivity of a silver film (bulk silver when t_nm is infinite)."""
    G = AG["G_bulk"] if np.isinf(t_nm) else ag_gamma(t_nm, A_size)
    E = np.asarray(E, float)
    return AG["eps_inf"] - AG["Ep"] ** 2 / (E**2 + 1j * G * E)


def ag_resistivity(t_nm, A_size=1.0):
    """DC resistivity (ohm m), from the same damping as the optics."""
    G = ag_gamma(t_nm, A_size) * q / hbar        # 1/s
    wp = AG["Ep"] * q / hbar
    return G / (eps0 * wp**2)


# --------------------------------------------------------------------------- graphene
GRAPHENE_D = 0.335e-9    # layer spacing, used as the optical thickness of one layer (m)
VF_G = 1.0e6             # Fermi velocity of graphene (m/s)


def graphene_sigma(E, EF=0.45, mu_cm2=3000.0):
    """Sheet conductivity (S) of doped graphene at photon energy E (eV),
    from the Kubo formula: a Drude-like intraband part plus the interband
    part that gives the famous 2.3 % absorption per layer."""
    E = np.asarray(E, float)
    w = E * q / hbar
    tau = (mu_cm2 * 1e-4) * EF * q / (q * VF_G**2)
    D = q**2 * EF * q / (np.pi * hbar**2)
    s_intra = 1j * D / (w + 1j / tau)
    s0 = q**2 / (4 * hbar)
    # smoothed step: transitions are blocked below 2 EF
    step = 0.5 + np.arctan((E - 2 * EF) / (2 * kT_eV)) / np.pi
    log_term = np.log((E + 2 * EF) ** 2 / ((E - 2 * EF) ** 2 + (2 * kT_eV) ** 2))
    s_inter = s0 * (step - 1j / (2 * np.pi) * log_term)
    return s_intra + s_inter


def graphene_eps(E, EF=0.45, mu_cm2=3000.0):
    """Equivalent permittivity of one graphene layer GRAPHENE_D thick."""
    E = np.asarray(E, float)
    w = E * q / hbar
    return 1.0 + 1j * graphene_sigma(E, EF, mu_cm2) / (eps0 * w * GRAPHENE_D)


def graphene_sheet_resistance(EF=0.45, mu_cm2=3000.0):
    """Sheet resistance (ohm/sq) of one layer, and its carrier density (cm^-2)."""
    n2d = (EF * q / (hbar * VF_G)) ** 2 / np.pi
    return 1 / (n2d * q * mu_cm2 * 1e-4), n2d * 1e-4


# --------------------------------------------------------------------------- solar spectrum
def planck_toa(lam_nm, T=5778.0):
    """Black-body sunlight above the atmosphere (W m^-2 nm^-1)."""
    lam = lam_nm * 1e-9
    B = 2 * np.pi * h * c0**2 / lam**5 / np.expm1(h * c0 / (lam * kB * T))
    return B * (6.957e8 / 1.495978707e11) ** 2 * 1e-9    # (solar radius / 1 AU)^2


def _gaussian_band(lam, centre, width, depth):
    """One molecular absorption band of the atmosphere (optical depth)."""
    return depth * np.exp(-0.5 * ((lam - centre) / width) ** 2)


def _clear_sky(lam, airmass, f_diffuse, T_eff):
    """Unnormalised clear-sky irradiance: black-body sunlight minus Rayleigh
    and aerosol scattering, ozone, and the main O2/H2O/CO2 absorption bands."""
    um = lam / 1000
    E0 = planck_toa(lam, T_eff)
    tau_rayleigh = 0.008735 * um ** -4.08
    tau_aerosol = 0.084 * (um / 0.5) ** -1.14
    a_ozone = 10 * np.exp(-(lam - 300) / 9.0) + 0.12 * np.exp(-0.5 * ((lam - 600) / 60) ** 2)
    tau_ozone = a_ozone * 0.34
    tau_gas = (_gaussian_band(lam, 720, 10, 0.08) + _gaussian_band(lam, 762, 3, 1.0)
               + _gaussian_band(lam, 820, 12, 0.15) + _gaussian_band(lam, 940, 25, 1.2)
               + _gaussian_band(lam, 1130, 30, 1.5) + _gaussian_band(lam, 1380, 45, 6.0)
               + _gaussian_band(lam, 1870, 60, 8.0) + _gaussian_band(lam, 2700, 120, 10.0))
    absorbed = np.exp(-airmass * tau_ozone - tau_gas)
    direct = np.exp(-airmass * (tau_rayleigh + tau_aerosol))
    diffuse = f_diffuse * (1 - np.exp(-airmass * (tau_rayleigh + 0.8 * tau_aerosol)))
    return E0 * absorbed * (direct + diffuse)


def am15g_approx(lam_nm, airmass=1.5, f_diffuse=0.30, T_eff=5600.0):
    """A simple model of the AM1.5 global spectrum, scaled to 1000 W/m^2 over
    280-4000 nm (W m^-2 nm^-1). Only used if the ASTM table is missing."""
    lam = np.asarray(lam_nm, float)
    G = _clear_sky(lam, airmass, f_diffuse, T_eff)
    lam_norm = np.linspace(280, 4000, 7441)
    G = G * (1000.0 / trapezoid(_clear_sky(lam_norm, airmass, f_diffuse, T_eff), lam_norm))
    G[lam < 280] = 0.0
    return G


G173_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spectrum_data", "ASTMG173.csv")
_G173 = None


def _g173_table():
    """The ASTM G173-03 reference spectrum (global tilt column), loaded once,
    with its running integral for bin averaging."""
    global _G173
    if _G173 is None:
        d = np.loadtxt(G173_PATH, delimiter=",", skiprows=2)
        lam, G = d[:, 0], d[:, 2]
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (G[1:] + G[:-1]) * np.diff(lam))])
        _G173 = (lam, G, cum)
    return _G173


def g173_available():
    return os.path.exists(G173_PATH)


def am15g(lam_nm):
    """AM1.5 global spectrum on any wavelength grid (W m^-2 nm^-1).

    Each grid point gets the average of the ASTM table over its own bin, so
    the total power is right even on a coarse grid. Falls back to the
    simple model above if the table is not there."""
    lam_nm = np.asarray(lam_nm, float)
    if not g173_available():
        return am15g_approx(lam_nm)
    lt, G, cum = _g173_table()
    if lam_nm.size < 2:
        return np.interp(lam_nm, lt, G, left=0, right=0)
    edges = np.concatenate([[lam_nm[0] - 0.5 * (lam_nm[1] - lam_nm[0])],
                            0.5 * (lam_nm[1:] + lam_nm[:-1]),
                            [lam_nm[-1] + 0.5 * (lam_nm[-1] - lam_nm[-2])]])
    C = np.interp(edges, lt, cum, left=0.0, right=cum[-1])
    return np.diff(C) / np.diff(edges)


def load_spectrum(lam_nm):
    """Return (spectrum, name of its source)."""
    if g173_available():
        return am15g(lam_nm), "ASTM G173-03"
    return am15g_approx(lam_nm), "parametric AM1.5G approximation"


def photon_flux(lam_nm, G):
    """Photons per m^2 per s per nm, from irradiance G (W m^-2 nm^-1)."""
    return G * (lam_nm * 1e-9) / (h * c0)
