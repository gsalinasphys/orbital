import numpy as np


def dotG(G: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    return np.matmul(v1, np.matmul(G, v2))

def magG(G: np.ndarray, v: np.ndarray) -> float:
    return np.sqrt(dotG(G, v, v))

def epll(G: np.ndarray, phidot: np.ndarray) -> np.ndarray:
    return phidot / magG(G, phidot)

def eperp2d(G: np.ndarray, phidot: np.ndarray, eta: np.ndarray) -> np.ndarray:
    epll_vec = epll(G, phidot)
    eperp_notnorm = np.matmul(np.identity(2) - np.outer(epll_vec, np.matmul(G, epll_vec)), eta)
    return eperp_notnorm / magG(G, eperp_notnorm)
