"""
tmm.py -- thin-film optics by the transfer-matrix method.

Each interface and each layer is a 2x2 matrix linking the forward and
backward waves on either side; multiplying them gives the whole stack.
Works at any angle, for s and p polarisation, for all wavelengths at once.

Sign convention: n = n' + ik with k > 0 meaning absorption, fields go as
exp(i(kz - wt)). Follows Born and Wolf, and Byrnes (2016).
"""
import numpy as np


def _cos_theta(n, n0sin):
    """cos(theta) inside a layer of index n (Snell's law), choosing the root
    for a wave that travels forward, or decays, into the layer."""
    ct = np.sqrt(1 - (n0sin / n) ** 2 + 0j)
    flip = (np.imag(n * ct) < 0) | ((np.abs(np.imag(n * ct)) < 1e-12) & (np.real(n * ct) < 0))
    return np.where(flip, -ct, ct)


def coh_tmm(pol, n_list, d_list, th0, lam):
    """Reflection and transmission of a coherent stack of layers.

    pol     "s" or "p"
    n_list  complex index of each layer (number or array over wavelength),
            starting with the medium the light comes from
    d_list  thickness of each layer in nm; the first and last are np.inf
    th0     angle of incidence (rad);  lam  wavelengths (nm)

    Returns a dict with amplitudes r, t and powers R, T (T is the power that
    enters the last medium)."""
    lam = np.atleast_1d(np.asarray(lam, float))
    L = len(n_list)
    N = [np.broadcast_to(np.asarray(nj, complex), lam.shape) for nj in n_list]
    n0sin = N[0] * np.sin(th0)
    ct = [_cos_theta(nj, n0sin) for nj in N]
    kz = [2 * np.pi * N[j] * ct[j] / lam for j in range(L)]

    def rt(i, j):
        """Fresnel coefficients for the interface between layers i and j."""
        ni, nj, ci, cj = N[i], N[j], ct[i], ct[j]
        if pol == "s":
            r = (ni * ci - nj * cj) / (ni * ci + nj * cj)
            t = 2 * ni * ci / (ni * ci + nj * cj)
        else:
            r = (nj * ci - ni * cj) / (nj * ci + ni * cj)
            t = 2 * ni * ci / (nj * ci + ni * cj)
        return r, t

    # the matrix is kept as four arrays (one entry per wavelength) so that
    # every wavelength is handled in one pass
    r01, t01 = rt(0, 1)
    M00 = np.ones_like(lam, complex) / t01
    M01 = r01 / t01
    M10 = r01 / t01
    M11 = np.ones_like(lam, complex) / t01
    for j in range(1, L - 1):
        delta = kz[j] * d_list[j]          # phase picked up crossing layer j
        # a very thick absorbing layer would overflow exp(); past e^-35 it is opaque anyway
        delta = np.where(np.imag(delta) > 35, np.real(delta) + 35j, delta)
        e_m, e_p = np.exp(-1j * delta), np.exp(1j * delta)
        r, t = rt(j, j + 1)
        # propagate through layer j, then cross into layer j+1
        A00, A01, A10, A11 = e_m / t, e_m * r / t, e_p * r / t, e_p / t
        M00, M01, M10, M11 = (M00 * A00 + M01 * A10, M00 * A01 + M01 * A11,
                              M10 * A00 + M11 * A10, M10 * A01 + M11 * A11)
    r = M10 / M00
    t = 1 / M00
    R = np.abs(r) ** 2
    nL, cL, n0, c0 = N[-1], ct[-1], N[0], ct[0]
    if pol == "s":
        T = np.abs(t) ** 2 * np.real(nL * cL) / np.real(n0 * c0)
    else:
        T = np.abs(t) ** 2 * np.real(nL * np.conj(cL)) / np.real(n0 * np.conj(c0))
    return dict(r=r, t=t, R=R, T=T)


def unpolarised(n_list, d_list, th0, lam):
    """R, T and absorbed fraction A for unpolarised light (average of s and p)."""
    s = coh_tmm("s", n_list, d_list, th0, lam)
    p = coh_tmm("p", n_list, d_list, th0, lam)
    R = 0.5 * (s["R"] + p["R"])
    T = 0.5 * (s["T"] + p["T"])
    return R, T, 1 - R - T


def film_on_thick_substrate(n_films, d_films, n_sub, lam, n_in=1.0, n_out=1.0):
    """Coherent film(s) on an incoherent thick, non-absorbing substrate,
    light incident from the film side (e.g. air|TCO|glass|air), normal
    incidence. Returns R, T, A of the whole sample."""
    stack_f = [n_in] + list(n_films) + [n_sub]
    d_f = [np.inf] + list(d_films) + [np.inf]
    Rf, Tf, Af = unpolarised(stack_f, d_f, 0.0, lam)
    # same films seen from the substrate side
    stack_b = [n_sub] + list(n_films)[::-1] + [n_in]
    d_b = [np.inf] + list(d_films)[::-1] + [np.inf]
    Rfb, Tfb, Afb = unpolarised(stack_b, d_b, 0.0, lam)
    Rb = np.abs((n_sub - n_out) / (n_sub + n_out)) ** 2
    Tb = 1 - Rb
    denom = 1 - Rfb * Rb
    T = Tf * Tb / denom
    R = Rf + Tf * Rb * Tfb / denom
    A = 1 - R - T
    return np.real(R), np.real(T), np.real(A)


def superstrate_into_absorber(n_films, d_films, n_glass, n_abs, lam):
    """air | thick glass (incoherent) | films (coherent) | semi-infinite
    absorber, normal incidence.  Returns (T_into_absorber, A_parasitic, R)."""
    Rag = np.abs((1 - n_glass) / (1 + n_glass)) ** 2
    Tag = 1 - Rag
    stack = [n_glass] + list(n_films) + [n_abs]
    d = [np.inf] + list(d_films) + [np.inf]
    Rf, Tf, Af = unpolarised(stack, d, 0.0, lam)
    denom = 1 - Rf * Rag
    Tin = Tag * Tf / denom
    Apar = Tag * Af / denom
    return np.real(Tin), np.real(Apar), np.real(1 - Tin - Apar)


def textured_front_into_absorber(n_films, d_films, n_abs, lam, n_in=1.0,
                                 alpha_deg=54.74):
    """Two-bounce approximation of a random-upright-pyramid front surface:
    light meets a facet at alpha, the specular reflection meets the opposite
    facet at |3 alpha - 180 deg|.  s and p are tracked separately."""
    a1 = np.deg2rad(alpha_deg)
    a2 = np.deg2rad(abs(3 * alpha_deg - 180))
    stack = [n_in] + list(n_films) + [n_abs]
    d = [np.inf] + list(d_films) + [np.inf]
    Tin = 0.0
    Apar = 0.0
    for pol in ("s", "p"):
        o1 = coh_tmm(pol, stack, d, a1, lam)
        o2 = coh_tmm(pol, stack, d, a2, lam)
        A1 = 1 - o1["R"] - o1["T"]
        A2 = 1 - o2["R"] - o2["T"]
        Tin = Tin + 0.5 * (o1["T"] + o1["R"] * o2["T"])
        Apar = Apar + 0.5 * (A1 + o1["R"] * A2)
    Tin, Apar = np.real(Tin), np.real(Apar)
    return Tin, Apar, 1 - Tin - Apar
