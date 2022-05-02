from pycftboot import ConformalBlockTable, ConvolvedBlockTable, SDP

import numpy as np
import matplotlib.pyplot as plt


dimensions_phi = np.linspace(0.50001, 0.7, 5)
dimensions_epsilon = []

dim = 3
k_max = 20
l_max = 14
n_max = 4
m_max = 2

table1 = ConformalBlockTable(dim, k_max, l_max, m_max, n_max)
table2 = ConvolvedBlockTable(table1)

for dim_phi in dimensions_phi:
    print("Finding basic bound at external dimension " + str(dim_phi) + "...")

    sdp = SDP(dim_phi, table2, sdpb_mode='singularity')

    sdp.sdpb.set_option("procsPerNode", 2)

    lower = 0.5
    upper = 2
    tol = 0.001
    channel = 0

    result = sdp.bisect(lower, upper, tol, channel, name=f"{dim_phi}")
    print("If crossing symmetry and unitarity hold, the maximum gap we can have for Z2-even scalars is: " + str(result))
    dimensions_epsilon.append(result)

np.savetxt('ising_plot.txt', dimensions_epsilon)

plt.plot(dimensions_phi, dimensions_epsilon)
plt.fill_between(dimensions_phi, dimensions_epsilon, y2=min(dimensions_epsilon), color='blue', alpha=0.1, linewidth=0)
plt.xlabel(r'$\Delta_{\phi}$')
plt.ylabel(r'$\Delta_{\epsilon}$')
plt.savefig('ising_plot.png', dpi=600)
