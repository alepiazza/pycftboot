import numpy as np
import matplotlib.pyplot as plt
import itertools

from pycftboot import ConformalBlockTable, ConvolvedBlockTable, SDP


dim = 3
k_max = 20
l_max = 14
n_max = 4
m_max = 2

table1 = ConformalBlockTable(dim, k_max, l_max, m_max, n_max)
table2 = ConvolvedBlockTable(table1)

n_points = 4

dim_phi = np.linspace(0.5001, 0.7, n_points)
dim_eps = np.linspace(1, 2, n_points)

allowed = []

for dim_phi, dim_eps in itertools.product(dim_phi, dim_eps):
    print(f"Trying ({dim_phi}, {dim_eps})")
    sdp = SDP(dim_phi, table2, sdpb_mode='docker')

    sdp.sdpb.set_option("procsPerNode", 2)

    sdp.add_point(0, dim_eps)
    sdp.set_bound(0, dim)

    result = sdp.iterate(name=f"{dim_phi}_{dim_eps}")

    if result:
        allowed.append((dim_phi, dim_eps))

np.savetxt('ising_gap.txt', allowed)

plt.scatter(*zip(*allowed))
plt.xlabel(r'$\Delta_{\phi}$')
plt.ylabel(r'$\Delta_{\epsilon}$')
plt.savefig('ising_gap.png', dpi=600)
