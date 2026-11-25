# transfer matrix method for thin film stacks
# sign convention n = n' + ik with k > 0 absorbing, based on born & wolf and byrnes 2016

import numpy as np


def _cos_theta(n, n0sin):
    """cos(theta) inside layer from snell law, picking forward decaying branch"""
    ct = np.sqrt(1 - (n0sin / n) ** 2 + 0j)
    flip = (np.imag(n * ct) < 0) | ((np.abs(np.imag(n * ct)) < 1e-12) & (np.real(n * ct) < 0))
    return np.where(flip, -ct, ct)


def coh_tmm(pol, n_list, d_list, th0, lam):
    """coherent transfer matrix for s or p polarisation
    n_list: complex indices from incident side, d_list: thicknesses in nm (ends are inf)
    th0 in rad, lam in nm"""
    lam = np.atleast_1d(np.asarray(lam, float))
    L = len(n_list)
    N = [np.broadcast_to(np.asarray(nj, complex), lam.shape) for nj in n_list]
    n0sin = N[0] * np.sin(th0)
    ct = [_cos_theta(nj, n0sin) for nj in N]
    kz = [2 * np.pi * N[j] * ct[j] / lam for j in range(L)]

    def rt(i, j):
        # fresnel reflection and transmission at interface i to j
        ni, nj, ci, cj = N[i], N[j], ct[i], ct[j]
        if pol == "s":
            r = (ni * ci - nj * cj) / (ni * ci + nj * cj)
            t = 2 * ni * ci / (ni * ci + nj * cj)
        else:
            r = (nj * ci - ni * cj) / (nj * ci + ni * cj)
            t = 2 * ni * ci / (nj * ci + ni * cj)
        return r, t

    # keep the 2x2 matrix as 4 arrays over wavelength
    r01, t01 = rt(0, 1)
    M00 = np.ones_like(lam, complex) / t01
    M01 = r01 / t01
    M10 = r01 / t01
    M11 = np.ones_like(lam, complex) / t01
    for j in range(1, L - 1):
        delta = kz[j] * d_list[j]          # phase across layer j
        # clamp imaginary phase so exp does not overflow for very thick absorbing films
        delta = np.where(np.imag(delta) > 35, np.real(delta) + 35j, delta)
        e_m, e_p = np.exp(-1j * delta), np.exp(1j * delta)
        r, t = rt(j, j + 1)
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
    """average R, T, A over s and p polarisations"""
    s = coh_tmm("s", n_list, d_list, th0, lam)
    p = coh_tmm("p", n_list, d_list, th0, lam)
    R = 0.5 * (s["R"] + p["R"])
    T = 0.5 * (s["T"] + p["T"])
    return R, T, 1 - R - T


def film_on_thick_substrate(n_films, d_films, n_sub, lam, n_in=1.0, n_out=1.0):
    """coherent film stack on incoherent thick glass substrate at normal incidence"""
    stack_f = [n_in] + list(n_films) + [n_sub]
    d_f = [np.inf] + list(d_films) + [np.inf]
    Rf, Tf, Af = unpolarised(stack_f, d_f, 0.0, lam)
    # reversed stack from substrate side
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
    """air | thick glass | coherent films | absorber at normal incidence
    returns (Tin, Apar, R)"""
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
    """two bounce model for random pyramid textured silicon front surface"""
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
