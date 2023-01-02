import pickle

from PyTransport import PyTransSetup
from sympy import Matrix, symarray

nF, nP = 2, 0  # Number of fields and parameters
f, p = symarray('f', nF), symarray('p', nP)

V = f[0]**2 - 2/3*1/f[1]**2
G = Matrix([[f[1]**2, 0], [0, 1]])

with open("./output/setup/G.txt", "wb") as file:
    pickle.dump(G, file)

PyTransSetup.potential(V, nF, nP, True, G)

PyTransSetup.compileName3("Orbital", True)
