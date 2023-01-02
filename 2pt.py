import json
import sys
from itertools import product

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from background import get_background, get_background_func
from curved import eperp2d, epll
from slowroll import (get_epsilon_func, get_epsilons, get_eta_func, get_etas,
                      get_metric_func, get_metrics)

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi'] = 600

location = "/home/gsalinas/GitHub/orbital/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransOrbital as PyT
import PyTransScripts as PyS


def get_2pt_initial(back: np.ndarray, params: dict, efolds_before: float, NB: float = 8.):
    pval = np.array(list(params.values()))
    
    Ns = back[:, 0]
    Nexit = Ns[-1] - efolds_before

    k = PyS.kexitN(Nexit, back, pval, PyT)
    Nstart, backExitMinus = PyS.ICsBE(NB, k, back, pval, PyT)

    return k, Nstart, backExitMinus

def get_2pts(back: np.ndarray, params: dict, efolds_before: float, NB: float = 8., tol: float = 1e-12) -> tuple:
    nF = PyT.nF()
    pval = np.array(list(params.values()))
    
    k, Nstart, backExitMinus = get_2pt_initial(back, params, efolds_before, NB)

    Ns = back[:, 0]
    Nev = Ns[Ns >= Nstart]
    tols = np.array([tol, tol])
    twoPt = PyT.sigEvolve(Nev, k, backExitMinus, pval, tols, True)

    Nsig = twoPt[:, 0]
    Pzeta = twoPt[:, 1]
    sigma = twoPt[:, 1+1+2*nF:].reshape(len(Nsig), 2*nF, 2*nF)
    Pphi = sigma[:, :nF, :nF]

    k_deformed = k + 0.001*k
    twoPt_deformed = PyT.sigEvolve(Nev, k_deformed, backExitMinus, pval, tols, True)
    Pzeta_deformed = twoPt_deformed[:, 1]
    ns = (np.log(Pzeta_deformed[-1])-np.log(Pzeta[-1])) / (np.log(k_deformed)-np.log(k)) + 4.0

    Pzeta_nodim = Pzeta * k**3 / 2 / np.pi**2

    Pphi_func = lambda N: np.array([np.interp(N, Nsig, Pphi[:, aa, bb]) for aa, bb in product(range(nF), repeat=2)]).reshape(nF, nF)

    return Nsig, Pphi_func, lambda N: np.interp(N, Nsig, Pzeta_nodim), k, ns

def get_PR_PS_CRS(back: np.ndarray, params: dict, efolds_before: float, NB: float = 8., tol: float = 1e-8) -> tuple:
    _, _, phidotx, phidoty = get_background_func(back)
    _, Pphi, _, k, _ = get_2pts(back, params, efolds_before, NB, tol)

    epsilon = get_epsilon_func(back, params)
    eta = get_eta_func(back, params)
    G = get_metric_func(back, params)

    PR = lambda N: epll(G(N), np.array([phidotx(N), phidoty(N)])) @ G(N) @ Pphi(N) @ G(N) @ \
        epll(G(N), np.array([phidotx(N), phidoty(N)])) * k**3 / 4 / np.pi**2 / epsilon(N)
    CRS = lambda N: epll(G(N), np.array([phidotx(N), phidoty(N)])) @ G(N) @ Pphi(N) @ G(N) @ \
        eperp2d(G(N), np.array([phidotx(N), phidoty(N)]), eta(N)) * k**3 / 4 / np.pi**2 / epsilon(N)
    PS = lambda N: eperp2d(G(N), np.array([phidotx(N), phidoty(N)]), eta(N)) @ G(N) @ Pphi(N) @ G(N) @ \
        eperp2d(G(N), np.array([phidotx(N), phidoty(N)]), eta(N)) * k**3 / 4 / np.pi**2 / epsilon(N)

    return PR, CRS, PS


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

    efolds_before = 55.
    Nexit = back[-1, 0] - efolds_before
    Nsig, Pphi, Pzeta, k, ns = get_2pts(back, params, efolds_before)
    print("ns: ", ns)

    num_points = 500
    Nplot = np.linspace(Nsig[0], Nsig[-1], num_points)
    Pphis = np.array([Pphi(_) for _ in Nplot])

    plt.plot(Nplot, Pphis[:, 0, 0], label=r"$P^{11}_\phi$")
    plt.plot(Nplot, np.abs(Pphis[:, 0, 1]), label=r"$\vert P^{12}_\phi \vert$")
    plt.plot(Nplot, Pphis[:, 1, 1], label=r"$P^{22}_\phi$")
    plt.title(r'$P_\phi$ evolution',fontsize=16)
    plt.legend(fontsize=16)
    plt.ylabel(r'Absolute 2pt field correlations', fontsize=20) 
    plt.xlabel(r'$N$', fontsize=20)
    plt.yscale('log')
    plt.axvline(Nexit, c='k', linestyle='--')
    plt.tight_layout()
    plt.savefig("./output/2pt/2pt.png")
    plt.clf()

    PR, CRS, PS = get_PR_PS_CRS(back, params, efolds_before)
    print(f'Power spectrum at the end of inflation: {Pzeta(Nsig[-1]):.3}')

    print(f'Power spectrum at horizon crossing: {Pzeta(Nexit):.3}')
    print(f'Power spectrum at horizon crossing 2: {PR(Nexit):.3}')

    plt.plot(Nplot, [PR(_) for _ in Nplot], c='k')
    plt.plot(Nplot, [Pzeta(_) for _ in Nplot], c='b', linestyle='--')
    plt.axvline(Nexit, c='gray', linestyle='--')
    plt.title(r'$P_R$ evolution',fontsize=16);
    plt.ylabel(r'$P_R$', fontsize=20) 
    plt.xlabel(r'$N$', fontsize=20)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/2pt/PR.png")
    plt.clf()

    plt.plot(Nplot, np.abs([CRS(_) for _ in Nplot]), c='k')
    plt.axvline(Nexit, c='gray', linestyle='--')
    plt.title(r'$C_{RS}$ evolution',fontsize=16);
    plt.ylabel(r'$\vert C_{RS} \vert$', fontsize=20) 
    plt.xlabel(r'$N$', fontsize=20)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/2pt/CRS.png")
    plt.clf()

    plt.plot(Nplot, [PS(_) for _ in Nplot], c='k')
    plt.axvline(Nexit, c='gray', linestyle='--')
    plt.title(r'$P_S$ evolution',fontsize=16);
    plt.ylabel(r'$P_S$', fontsize=20) 
    plt.xlabel(r'$N$', fontsize=20)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("./output/2pt/PS.png")
    plt.clf()

