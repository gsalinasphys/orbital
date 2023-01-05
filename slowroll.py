import json
import pickle
import sys
from itertools import combinations_with_replacement, product
from typing import Callable, List

import numdifftools as nd
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from sympy.utilities import lambdify

from background import get_background, get_background_func
from curved import dotG, eperp2d, epll, magG

location = "/home/gsalinas/GitHub/orbital/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransOrbital as PyT


def get_H(phix: Callable, phiy: Callable, phidotx: Callable, phidoty: Callable, pval: np.ndarray) -> Callable:
    return lambda N: PyT.H(np.array((phix(N),phiy(N), phidotx(N), phidoty(N))), pval)

def get_epsilon(H: Callable) -> Callable:
    return lambda N: -nd.Derivative(H)(N) / H(N)

def get_phiprime(phidotx: Callable, phidoty: Callable, H: Callable) -> Callable:
    return lambda N: np.array([phidotx(N), phidoty(N)]) / H(N)

def get_phidoubleprime(phiprime: Callable) -> Callable:
    return nd.Derivative(phiprime)

def get_metric_sympy():
    with open("./output/setup/G.txt", "rb") as file:
        G = pickle.load(file)

    return G

def get_metric_field(pval: np.ndarray) -> Callable:
    G = get_metric_sympy()
    params_subs = {'p_'+str(ii): pval[ii] for ii in range(len(pval))}

    return lambdify(['f_'+str(ii) for ii in range(PyT.nF())], G.subs(params_subs))

def get_metric(G_fieldsfunc: Callable, phix: Callable, phiy: Callable) -> Callable:
    return lambda N: G_fieldsfunc(phix(N), phiy(N))

def get_christoffel_field(pval: np.ndarray) -> List[Callable]:
    nF, nP = PyT.nF(), PyT.nP()
    Gamma_sympy = PyTransSetup.fieldmetric(get_metric_sympy(), nF, nP)[1]

    Gamma_func = np.empty((nF, nF, nF)).tolist()
    params_subs = {'p_'+str(ii): pval[ii] for ii in range(len(pval))}
    for aa, (bb, cc) in product(range(1, nF+1), combinations_with_replacement(range(1, nF+1), 2)):
        Gamma_func[aa-1][bb-1][cc-1] = Gamma_sympy(-aa, bb, cc).subs(params_subs)
        if bb != cc:
            Gamma_func[aa-1][cc-1][bb-1] = Gamma_func[aa-1][bb-1][cc-1]

    return lambdify(['f_'+str(ii) for ii in range(nF)], Gamma_func)

def get_christoffel(Gamma_fieldsfunc: Callable, phix: Callable, phiy: Callable) -> List[Callable]:
    return lambda N: Gamma_fieldsfunc(phix(N), phiy(N))

def get_eta(phidoubleprime: Callable, Gamma: Callable, phiprime: Callable) -> Callable:
    return lambda N: phidoubleprime(N) + phiprime(N) @ Gamma(N) @ phiprime(N)

def get_kinbasis(G: Callable, phidotx: Callable, phidoty: Callable, eta: Callable) -> Callable:
    return lambda N: epll(G(N), np.array([phidotx(N), phidoty(N)])), \
        lambda N: eperp2d(G(N), np.array([phidotx(N), phidoty(N)]), eta(N))

def get_eta_parallelperp(G: Callable, eta: Callable, epll: Callable, eperp: Callable) -> np.ndarray:
    return lambda N: dotG(G(N), eta(N), epll(N)), lambda N: dotG(G(N), eta(N), eperp(N))

def get_mass_matrix(pval: np.ndarray) -> Callable:  # Needs Christoffel symbols here
    return lambda phi: PyT.ddV(phi, pval) / PyT.V(phi, pval) - np.outer(PyT.dV(phi, pval), PyT.dV(phi, pval)) / PyT.V(phi, pval)**2

def _beta(phix: Callable, phiy: Callable, phidotx: Callable, phidoty: Callable,
            epsilon: Callable, epll: Callable, eperp: Callable, etaperp: Callable, pval: np.ndarray) -> Callable:
    return lambda N: -2*epsilon(N) - eperp(N) @ PyT.ddV(np.array([phix(N), phiy(N)]), pval) @ eperp(N) / PyT.V(np.array([phix(N), phiy(N)]), pval) + \
        epll(N) @ PyT.ddV(np.array([phix(N), phiy(N)]), pval) @ epll(N) / PyT.V(np.array([phix(N), phiy(N)]), pval) - \
        2*etaperp(N)**2/(3*epsilon(N)*PyT.H(np.array([phix(N), phiy(N), phidotx(N), phidoty(N)]), pval)**2)


def TSS(beta: Callable, Nexit: float) -> Callable:
    return lambda N: np.exp(quad(beta, Nexit, N))


if __name__ == '__main__':
    nF, nP = PyT.nF(), PyT.nP()
    with open("./output/setup/params.json", "r") as file:
        params = json.loads(file.readline())
    pval = np.array(list(params.values()))
    phi0 = np.array([20., 1.])
    phidot0 = np.array([0., 1.])
    initial = np.concatenate((phi0, phidot0))
    Nrange = (0, 200, 10_000)
    back = get_background(initial, params, Nrange)
    phix, phiy, phidotx, phidoty = get_background_func(back)

    Nini, Nend = back[0, 0], back[-1, 0]
    Nexit = Nend - 55

    H = get_H(phix, phiy, phidotx, phidoty, pval)

    num_points = 500
    Nplot = np.linspace(Nini, Nend, num_points)
    plt.plot(Nplot, [H(_) for _ in Nplot], c="k", linewidth=1)
    plt.title('Hubble parameter')
    plt.xlabel(r'$N$', fontsize=16)
    plt.ylabel(r'$H$', fontsize=16)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/background/Hs.png")
    plt.clf()

    epsilon = get_epsilon(H)
    plt.plot(Nplot, [epsilon(_) for _ in Nplot], c="k", linewidth=1)
    plt.title('Epsilon parameter')
    plt.xlabel(r'$N$', fontsize=16)
    plt.ylabel(r'$\epsilon$', fontsize=16)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/background/epsilons.png")
    plt.clf()

    phiprime = get_phiprime(phidotx, phidoty, H)
    phidoubleprime = get_phidoubleprime(phiprime)
    G_fieldsfunc = get_metric_field(pval)
    G = get_metric(G_fieldsfunc, phix, phiy)
    Gamma_fieldfunc = get_christoffel_field(pval)
    Gamma = get_christoffel(Gamma_fieldfunc, phix, phiy)
    eta = get_eta(phidoubleprime, Gamma, phiprime)
    epll1, eperp1 = get_kinbasis(G, phidotx, phidoty, eta)
    etapll, etaperp = get_eta_parallelperp(G, eta, epll1, eperp1)
    M = get_mass_matrix(pval)

    plt.plot(Nplot, np.abs([etapll(_) for _ in Nplot]), c="k", linewidth=1)
    plt.title('Eta parallel')
    plt.xlabel(r'$N$', fontsize=16)
    plt.ylabel(r'$\vert \eta_\parallel \vert$', fontsize=16)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/background/etaplls.png")
    plt.clf()

    plt.plot(Nplot, [etaperp(_) for _ in Nplot], c="k", linewidth=1)
    plt.title('Eta perpendicular')
    plt.xlabel(r'$N$', fontsize=16)
    plt.ylabel(r'$\eta_\perp$', fontsize=16)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/background/etaperps.png")
    plt.clf()

    # beta = _beta(phix, phiy, phidotx, phidoty, epsilon, epll1, eperp1, etaperp, pval)
    # TSSf = TSS(beta, Nexit)

    # Nafter = np.linspace(Nexit, Nend, 10)
    # plt.plot(Nafter, [TSSf(N)[0] for N in Nafter], c='k')
    # plt.xlim([Nexit, Nafter[-1]])
    # plt.title(r'$T_{SS}$ evolution',fontsize=16);
    # plt.ylabel(r'$T_{SS}$', fontsize=20) 
    # plt.xlabel(r'$N$', fontsize=20)
    # plt.yscale('log')
    # plt.tight_layout()
    # plt.savefig("./output/2pt/TSS.png")
    # plt.clf()