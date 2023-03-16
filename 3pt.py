import json
import sys
from itertools import combinations_with_replacement

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from background import get_background
from slowroll import get_epsilons

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi'] = 600

location = "/home/gsalinas/GitHub/orbital/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransOrbital as PyT
import PyTransScripts as PyS

nF, nP = PyT.nF(), PyT.nP()
with open("./output/setup/params.json", "r") as file:
    params = json.loads(file.readline())
pval = np.array(list(params.values()))
phi0 = np.array([20., 1.])
phidot0 = np.array([0., 1.])
initial = np.concatenate((phi0, phidot0))
Nrange = (0, 200, 10_000)
back = get_background(initial, params, Nrange)
Ns = back[:, 0]
Nini, Nend = Ns[0], Ns[-1]
epsilon = get_epsilons(back, params)

Nexit = Nend - 55
iexit = np.argmin(np.abs(Ns - Nexit))
k = PyS.kexitN(Nexit, back, pval, PyT) 

alpha = 0.
beta = 1/3.

k1 = k/2 - beta*k/2.
k2 = k/4*(1+alpha+beta)
k3 = k/4*(1-alpha+beta)
kM = np.min(np.array([k1, k2, k3]))

NB = 6.0
Nstart, backExitMinus = PyS.ICsBM(NB, kM, back, pval, PyT)
print(f"3-pt calculation starts at: {Nstart} e-folds")

Nev = Ns[Ns >= Nstart]
back = back[Ns >= Nstart]
epsilon = epsilon[Ns >= Nstart]

tols = np.array([10**-8, 10**-8])
threePt = PyT.alphaEvolve(Nev, k1, k2, k3, backExitMinus, pval, tols, True)
Nalpha = threePt[:, 0]
Babc = threePt[:, 1 + 4 + 2*nF + 6*2*nF*2*nF:]
alpha = np.empty((len(Nalpha), nF, nF, nF))
for ii, jj, kk in combinations_with_replacement(range(nF), 3):
    alpha[:, ii, jj, kk] = Babc[:, ii + 2*nF*jj + 2*nF*2*nF*kk]
Pzetas, B = threePt[:, 1:4], threePt[:, 4]

# plt.plot(Nalpha, np.abs(alpha[:, 0, 0, 0]), label=r"$\alpha^{111}$")
# plt.plot(Nalpha, np.abs(alpha[:, 0, 0, 1]), label=r"$\alpha^{112}$")
# plt.plot(Nalpha, np.abs(alpha[:, 0, 1, 1]), label=r"$\alpha^{122}$")
# plt.plot(Nalpha, np.abs(alpha[:, 1, 1, 1]), label=r"$\alpha^{222}$")
# plt.title(r'$\alpha$ evolution',fontsize=16)
# plt.legend(fontsize=16)
# plt.ylabel(r'Absolute 3pt field correlations', fontsize=20) 
# plt.xlabel(r'$N$', fontsize=20)
# plt.yscale('log')
# plt.axvline(Nexit, c='k', linestyle='--')
# plt.tight_layout()
# plt.savefig("./output/3pt/3pt.png")
# plt.clf()

fNL = 5.0/6.0*B/(Pzetas[:, 1]*Pzetas[:, 2]  + Pzetas[:, 0]*Pzetas[:, 1] + Pzetas[:, 0]*Pzetas[:, 2])
plt.plot(Nalpha, fNL,'r')
plt.title(r'$f_{NL}$ evolution',fontsize=15)
plt.ylabel(r'$f_{NL}$', fontsize=20)
plt.xlabel(r'$N$', fontsize=15)
plt.tight_layout()
plt.savefig("./output/3pt/fNL.png")
plt.clf()

print(f"fNL at the end of inflation: {fNL[-1]}")
