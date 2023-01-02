import json
import sys
from math import cos, pi, sin, sqrt

import matplotlib as mpl
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi'] = 600
location = "/home/gsalinas/GitHub/angular/PyTransport"
sys.path.append(location)

import PyTransSetup

PyTransSetup.pathSet()

import PyTransOrbital as PyT
