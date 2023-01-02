import json
import pickle
import sys
from itertools import combinations_with_replacement, product
from typing import Callable, List

import numpy as np
from matplotlib import pyplot as plt
from scipy.misc import derivative
from sympy.utilities import lambdify

from background import get_background, get_background_func
from curved import dotG, eperp2d, epll, magG

location = "/home/gsalinas/GitHub/orbital/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransOrbital as PyT


def get_Hs(back: np.ndarray, params: dict) -> tuple:
    pval = np.array(list(params.values()))
    Hs = np.array([PyT.H(elem, pval) for elem in back[:, 1:]])
    return np.hstack((back[:, 0].reshape(-1,1), Hs.reshape(-1,1)))

def get_H_func(back: np.ndarray, params: dict) -> Callable:
    phix, phiy, phidotx, phidoty = get_background_func(back)
    pval = np.array(list(params.values()))
    return lambda N: PyT.H(np.array((phix(N),phiy(N), phidotx(N), phidoty(N))), pval)

def get_epsilons(back: np.ndarray, params: dict) -> tuple:
    dN = back[1, 0] - back[0, 0]
    Hs = get_Hs(back, params)[:, 1]
    epsilons = -np.gradient(Hs, dN)/Hs
    return np.hstack((back[:, 0].reshape(-1,1), epsilons.reshape(-1,1)))

def get_epsilon_func(back: np.ndarray, params: dict) -> Callable:
    H = get_H_func(back, params)
    return lambda N: -derivative(H, N, dx=1e-6) / H(N)

def get_phi_primes(back: np.ndarray, params: dict) -> tuple:
    nF = PyT.nF()
    Hs = get_Hs(back, params)[:, 1]
    phi_primes = (back[:, nF+1:].T / Hs).T
    return np.hstack((back[:, 0].reshape(-1,1), phi_primes))

def get_phi_prime_func(back: np.ndarray, params: dict) -> Callable:
    _, _, phidotx, phidoty = get_background_func(back)
    H_func = get_H_func(back, params)
    return lambda N: np.array([phidotx(N), phidoty(N)]) / H_func(N)

def get_phi_double_primes(back: np.ndarray, params: dict) -> tuple:
    dN = back[1, 0] - back[0, 0]
    phi_primes = get_phi_primes(back, params)[:, 1:]
    phi_double_primes = np.gradient(phi_primes.T, dN, axis=1).T
    return np.hstack((back[:, 0].reshape(-1,1), phi_double_primes))

def get_phi_double_prime_func(back: np.ndarray, params: dict) -> Callable:
    phiprime = get_phi_prime_func(back, params)
    return lambda N: derivative(phiprime, N, dx=1e-6)

def get_metric_sympy():
    with open("./output/setup/G.txt", "rb") as file:
        G = pickle.load(file)

    return G

def get_metric_fieldsfunc(params: dict) -> Callable:
    nF = PyT.nF()
    G = get_metric_sympy()
    pval = list(params.values())
    params_subs = {'p_'+str(ii): pval[ii] for ii in range(len(pval))}

    return lambdify(['f_'+str(ii) for ii in range(nF)], G.subs(params_subs))

def get_metric_func(back: np.ndarray, params: dict) -> Callable:
    G_fieldsfunc = get_metric_fieldsfunc(params)
    phix, phiy, *_ = get_background_func(back)

    return lambda N: G_fieldsfunc(phix(N), phiy(N))

def get_metrics(back: np.ndarray, params: dict) -> np.ndarray:
    nF = PyT.nF()
    phis = back[:, 1:nF+1]    
    G = get_metric_fieldsfunc(params)
    return np.array([G(phi[0], phi[1]) for phi in phis])

def get_christoffel_fieldfunc(params: dict) -> List[Callable]:
    nF, nP = PyT.nF(), PyT.nP()
    Gamma_sympy = PyTransSetup.fieldmetric(get_metric_sympy(), nF, nP)[1]

    Gamma_func = np.empty((nF, nF, nF)).tolist()
    pval = list(params.values())
    params_subs = {'p_'+str(ii): pval[ii] for ii in range(len(pval))}
    for aa, (bb, cc) in product(range(1, nF+1), combinations_with_replacement(range(1, nF+1), 2)):
        Gamma_func[aa-1][bb-1][cc-1] = Gamma_sympy(-aa, bb, cc).subs(params_subs)
        if bb != cc:
            Gamma_func[aa-1][cc-1][bb-1] = Gamma_func[aa-1][bb-1][cc-1]

    return lambdify(['f_'+str(ii) for ii in range(nF)], Gamma_func)

def get_christoffel_func(back: np.ndarray, params: dict) -> List[Callable]:
    Gamma_fieldsfunc = get_christoffel_fieldfunc(params)
    phix, phiy, *_ = get_background_func(back)

    return lambda N: Gamma_fieldsfunc(phix(N), phiy(N))

def get_christoffels(back: np.ndarray, params: dict) -> np.ndarray:
    nF = PyT.nF()
    phis = back[:, 1:nF+1]
    Gamma_func = get_christoffel_fieldfunc(params)
    Gammas = np.array([Gamma_func(phi[0], phi[1]) for phi in phis])

    return Gammas

def get_etas(back: np.ndarray, params: dict) -> np.ndarray:
    nF = PyT.nF()
    Gammas = get_christoffels(back, params)
    phi_primes = get_phi_primes(back, params)[:, 1:]
    etas = get_phi_double_primes(back, params)[:, 1:]
    for ii in range(len(back[:, 0])):
        for aa in range(nF):
            etas[ii, aa] += sum([Gammas[ii, aa, bb, cc] * phi_primes[ii, bb] * phi_primes[ii, cc]
                                for bb, cc in product(range(nF), repeat=2)])

    return np.hstack((back[:, 0].reshape(-1, 1), etas))

def get_eta_func(back: np.ndarray, params: dict) -> Callable:
    nF = PyT.nF()
    Gammas = get_christoffel_func(back, params)
    phi_prime = get_phi_prime_func(back, params)

    return lambda N: get_phi_double_prime_func(back, params)(N) + phi_prime(N) @ Gammas(N) @ phi_prime(N)

def get_kin_basis(back: np.ndarray, params: dict) -> np.ndarray:
    nF = PyT.nF()
    Gs = get_metrics(back, params)
    etas = get_etas(back, params)[:, 1:]
    eplls = np.array([epll(Gs[ii], back[ii, nF+1:]) for ii in range(len(back[:, 0]))])
    eperps = np.array([eperp2d(Gs[ii], back[ii, nF+1:], etas[ii]) for ii in range(len(back[:, 0]))])
    return np.hstack((back[:, 0].reshape(-1,1), eplls, eperps))

def get_kin_basis_func(back: np.ndarray, params: dict) -> Callable:
    _, _, phidotx, phidoty = get_background_func(back)
    G = get_metric_func(back, params)
    eta = get_eta_func(back, params)

    return lambda N: epll(G(N), np.array([phidotx(N), phidoty(N)])), \
        lambda N: eperp2d(G(N), np.array([phidotx(N), phidoty(N)]), eta(N))

def get_eta_parallel_perp(back: np.ndarray, params: dict) -> np.ndarray:
    nF = PyT.nF()
    Gs = get_metrics(back, params)
    etas = get_etas(back, params)[:, 1:]
    eplls, eperps = get_kin_basis(back, params)[:, 1:nF+1], get_kin_basis(back, params)[:, nF+1:]
    etaplls = np.array([dotG(Gs[ii], etas[ii], eplls[ii]) for ii in range(len(back[:, 0]))])
    etaperps = np.array([dotG(Gs[ii], etas[ii], eperps[ii]) for ii in range(len(back[:, 0]))])

    return np.hstack((back[:, 0].reshape(-1,1), etaplls.reshape(-1,1), etaperps.reshape(-1,1)))

def get_eta_parallel_perp_func(back: np.ndarray, params: dict) -> np.ndarray:
    G = get_metric_func(back, params)
    eta = get_eta_func(back, params)
    eplls, eperps = get_kin_basis_func(back, params)

    return lambda N: dotG(G(N), eta(N), eplls(N)), lambda N: dotG(G(N), eta(N), eperps(N))

if __name__ == '__main__':
    nF, nP = PyT.nF(), PyT.nP()
    with open("./output/setup/params.json", "r") as file:
        params = json.loads(file.readline())
    phi0 = np.array([20., 1.])
    phidot0 = np.array([0., 1.])
    initial = np.concatenate((phi0, phidot0))
    Nrange = (0, 200, 10_000)
    back = get_background(initial, params, Nrange)
    Nini, Nend = back[0, 0], back[-1, 0]

    Hs = get_Hs(back, params)
    H_func = get_H_func(back, params)

    num_points = 500
    Nplot = np.linspace(Nini, Nend, num_points)
    plt.plot(Nplot, [H_func(_) for _ in Nplot], c="k", linewidth=1, linestyle='--')
    plt.scatter(Hs[:, 0], Hs[:, 1], c="b", s=2)
    plt.title('Hubble parameter')
    plt.xlabel(r'$N$', fontsize=16)
    plt.ylabel(r'$H$', fontsize=16)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/background/Hs.png")
    plt.clf()

    epsilons = get_epsilons(back, params)
    epsilon_func = get_epsilon_func(back, params)
    plt.plot(Nplot, [epsilon_func(_) for _ in Nplot], c="k", linewidth=1)
    plt.scatter(epsilons[:, 0], epsilons[:, 1], c="b", s=2)
    plt.title('Epsilon parameter')
    plt.xlabel(r'$N$', fontsize=16)
    plt.ylabel(r'$\epsilon$', fontsize=16)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/background/epsilons.png")
    plt.clf()

    phiprimes = get_phi_primes(back, params)
    phidoubleprimes = get_phi_double_primes(back, params)

    print(phiprimes[len(phiprimes)//2, 1:])
    print(get_phi_prime_func(back, params)(phiprimes[len(phiprimes)//2, 0]))

    print(phidoubleprimes[len(phidoubleprimes)//2, 1:])
    print(get_phi_double_prime_func(back, params)(phidoubleprimes[len(phidoubleprimes)//2, 0]))

    print(get_metrics(back, params)[len(back[:, 0])//2])
    print(get_metric_func(back, params)(back[len(back[:, 0])//2, 0]))

    print(get_christoffels(back, params)[len(back[:, 0])//2])
    print(get_christoffel_func(back, params)(back[len(back[:, 0])//2, 0]))

    print(get_etas(back, params)[len(back[:, 0])//2, 1:])
    print(get_eta_func(back, params)(back[len(back[:, 0])//2, 0]))

    print(get_kin_basis(back, params)[len(back[:, 0])//2, 1:])
    print(get_kin_basis_func(back, params)[0](back[len(back[:, 0])//2, 0]))
    print(get_kin_basis_func(back, params)[1](back[len(back[:, 0])//2, 0]))

    etaskin = get_eta_parallel_perp(back, params)
    etaplls, etaperps = etaskin[:, 1], etaskin[:, 2]
    etapll_func, etaperp_func = get_eta_parallel_perp_func(back, params)

    print(etaplls[len(etaplls)//2])
    print(etapll_func(back[len(back[:, 0])//2, 0]))

    print(etaperps[len(etaperps)//2])
    print(etaperp_func(back[len(back[:, 0])//2, 0]))

    plt.plot(Nplot, np.abs([etapll_func(_) for _ in Nplot]), c="k", linewidth=1)
    plt.scatter(back[:, 0], np.abs(etaplls), c="b", s=2)
    plt.title('Eta parallel')
    plt.xlabel(r'$N$', fontsize=16)
    plt.ylabel(r'$\vert \eta_\parallel \vert$', fontsize=16)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/background/etaplls.png")
    plt.clf()

    plt.plot(Nplot, [etaperp_func(_) for _ in Nplot], c="k", linewidth=1)
    plt.scatter(back[:, 0], etaperps, c="b", s=2)
    plt.title('Eta perpendicular')
    plt.xlabel(r'$N$', fontsize=16)
    plt.ylabel(r'$\eta_\perp$', fontsize=16)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/background/etaperps.png")
    plt.clf()

    omegas = etaperps / np.sqrt(2*epsilons[:, 1])
    plt.plot(back[:, 0], omegas, c="k", linewidth=2)
    plt.title('Turn rate')
    plt.xlabel(r'$N$', fontsize=16)
    plt.ylabel(r'$\omega$', fontsize=16)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/background/omegas.png")
    plt.clf()
