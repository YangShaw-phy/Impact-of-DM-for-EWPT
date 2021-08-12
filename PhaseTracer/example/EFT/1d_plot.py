import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, axs = plt.subplots(1, 1, figsize=(5, 5))

d = np.loadtxt("scan_result.txt")

axs.plot(d[:,0],d[:,1])

axs.set_ylim(10,1000)

axs.set_xscale('log')
axs.set_yscale('log')

axs.set_xlabel(r"d.o.f.")
axs.set_ylabel(r"$T_C$")

fig.tight_layout()
plt.savefig('dof_Tc.png')


plt.show()
