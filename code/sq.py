"""
sq.py -- the Shockley-Queisser limit, used to check the solar spectrum.

An ideal single-junction cell absorbs every photon above its band gap and
loses energy only by radiating like a black body. Its best efficiency depends
only on the gap and the spectrum, so getting the published value back (about
33 % at 1.34 eV) shows the spectrum is being handled correctly.
"""
import numpy as np
from numpy import trapezoid

import physics as ph


def sq_efficiency(Eg_list, lam, G, T=298.15):
    """Detailed-balance limit for each band gap in Eg_list (eV).

    lam, G  the spectrum: wavelengths (nm) and irradiance (W m^-2 nm^-1)
    Returns a list of (Eg, Jsc in mA/cm^2, voltage at maximum power, efficiency %)."""
    F = ph.photon_flux(lam, G)
    E = np.linspace(0.3, 6, 6000)            # photon energies for the black-body integral (eV)
    kT = ph.kB * T / ph.q
    out = []
    for Eg in Eg_list:
        absorbed = lam <= ph.HC_EVNM / Eg
        Jsc = ph.q * trapezoid(F[absorbed], lam[absorbed])                   # A/m^2
        # dark current: the cell's own black-body emission above the gap
        Ee = E[E >= Eg]
        bb = 2 * np.pi / (ph.h**3 * ph.c0**2) * (Ee * ph.q) ** 2 * np.exp(-Ee / kT)
        J0 = ph.q * trapezoid(bb, Ee * ph.q)
        V = np.linspace(0, Eg, 4000)
        P = V * (Jsc - J0 * np.expm1(V / kT))
        i = np.argmax(P)
        out.append((Eg, Jsc / 10, V[i], P[i] / 1000 * 100))
    return out


if __name__ == "__main__":
    # quick check with the built-in spectrum model
    lam = np.linspace(280, 4000, 20000)
    G = ph.am15g_approx(lam, f_diffuse=0.55)
    for Eg, J, V, eta in sq_efficiency([1.12, 1.34, 1.5, 1.6], lam, G):
        print(f"Eg = {Eg} eV   Jsc = {J:.2f} mA/cm2   eta = {eta:.2f} %")
