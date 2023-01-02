import json
import sys
from typing import Callable

import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi'] = 600
location = "/home/gsalinas/GitHub/orbital/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransOrbital as PyT


def get_background(initial: np.ndarray, params: dict, Nrange, tol: float = 1e-12) -> np.ndarray:
    Ns = np.linspace(Nrange[0], Nrange[1], Nrange[2], endpoint=True)
    pval = np.array(list(params.values()))
    tols = np.array([tol, tol])
    back = PyT.backEvolve(Ns, initial, pval, tols, True)
    return back

def get_background_func(back: np.ndarray) -> Callable:
    Ns, phixs, phiys, phidotxs, phidotys = back[:, 0], back[:, 1], back[:, 2], back[:, 3], back[:, 4]

    return lambda N: np.interp(N, Ns, phixs), lambda N: np.interp(N, Ns, phiys), \
        lambda N: np.interp(N, Ns, phidotxs), lambda N: np.interp(N, Ns, phidotys),

if __name__ == '__main__':
    params = {}
    with open("./output/setup/params.json", "w") as file:
        json.dump(params, file)
    phi0 = np.array([20., 1.])
    phidot0 = np.array([0., 1.])
    initial = np.concatenate((phi0, phidot0))
    Nrange = (0, 200, 10_000)
    back = get_background(initial, params, Nrange)
    phix, phiy, phidotx, phidoty = get_background_func(back)

    nF = PyT.nF()
    Ns, phis, phidots = back[:, 0], back[:, 1:nF+1], back[:, nF+1:]

    Nini, Nend = back[0, 0], back[-1, 0]
    print(f'Number of e-folds: {Nend:.3}')
    Nexit = Nend - 55
    iexit = np.argmin(np.abs(Ns - Nexit))

    palette = sns.color_palette("crest", as_cmap=True)
    num_points = 500
    Nplot = np.linspace(Nini, Nend, num_points)
    sns.lineplot(x=phix(Nplot),
                y=phiy(Nplot),
                linewidth=1,
                c='k',
                linestyle='--')
    sns.scatterplot(x=phis.T[0][::len(Ns)//num_points],
                    y=phis.T[1][::len(Ns)//num_points],
                    hue=Ns[::len(Ns)//num_points],
                    s=25,
                    palette=palette)
    plt.scatter(phis[iexit][0], phis[iexit][1], c="k")
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\chi$')
    plt.tight_layout()
    plt.savefig("./output/background/background.png")
    plt.clf()