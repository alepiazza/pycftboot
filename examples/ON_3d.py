#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from pycftboot import ConformalBlockTable, ConvolvedBlockTable, SDP

dim = 3
k_max = 20
l_max = 15
m_max = 1
n_max = 3

tableG = ConformalBlockTable(dim, k_max, l_max, m_max, n_max, odd_spins=True)
tableFplus = ConvolvedBlockTable(tableG, symmetric=True)
tableFminus = ConvolvedBlockTable(tableG)

N = 10
tables = [tableFminus, tableFplus]
vecS = [[0, 0], [1, 0], [1, 1]]
vecT = [[1, 0], [1 - 2 / N, 0], [-1 + 2 / N, 1]]
vecA = [[1, 0], [-1, 0], [1, 1]]
vector_types = [[vecS, 0, 'singlet'], [vecT, 0, 'symmetric'], [vecA, 1, 'antisymmetric']]

dim_phi = np.linspace(0.515, 0.7, 5)
dim_eps = []

for phi in dim_phi:
    sdp = SDP(phi, tables, vector_types=vector_types, sdpb_mode='docker')
    sdp.sdpb.set_option("procsPerNode", 2)

    lower = 0.5
    upper = 2
    tol = 0.001
    channel = [0, 'singlet']

    result = sdp.bisect(lower, upper, tol, channel, name=f"work/ON_{N}_{phi}")
    dim_eps.append(result)

np.savetxt(f'ON_{N}.txt', np.array(list(zip(dim_phi, dim_eps))))

plt.plot(dim_phi, dim_eps)
plt.savefig(f'ON_{N}.png', dpi=600)
