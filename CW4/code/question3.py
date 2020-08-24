""" question3.py - Script containing code used for Question 3 of CW4.
Tudor Trita Trita
CID: 01199397
MSci Mathematics
M4N10 - Computational Partial Differential Equations.
"""
import matplotlib
matplotlib.rc('figure', figsize=(11, 6))
import matplotlib.pyplot as plt
import numpy as np
import scheme

# Computing Convergence:
# Constants Used Throughout:
q, s, r = 1, 2, 1
N, L = 101, 301
tau = 0.05
gamma = 1.4
NITER = 1500
ITER_INTERVAL = 100

phi_init = np.zeros((N, L), dtype=np.float64)
rho_init = np.zeros((N+2, L+2), dtype=np.float64)
GRID = scheme.Grid(phi_init, q, s, r)

M_array = [0.7, 0.8, 0.9, 1, 1.1]
res_list1_list = []
res_list2_list = []
for M in M_array:
    print(f"M = {M}")
    _, _, _, _, reslist1, reslist2, _ = GRID.check_convergence(
        M, rho_init, iter_interval=ITER_INTERVAL, max_iters=NITER)
    res_list1_list.append(reslist1)
    res_list2_list.append(reslist2)

iterations = np.linspace(0, NITER, int(NITER/ITER_INTERVAL)+1)

plt.figure()
for i, M in enumerate(M_array):
    plt.semilogy(iterations, res_list1_list[i], label=f"M={M}")
plt.legend()
plt.grid()
plt.xlabel("Iterations")
plt.ylabel("Relative Residual")
plt.title(r"Figure 3.01 - Relative Residual for $\phi$ for varying M")
plt.savefig("../figures/fig3.01.png")
plt.show()

plt.figure()
for i, M in enumerate(M_array):
    plt.semilogy(iterations, res_list2_list[i], label=f"M={M}")
plt.legend()
plt.grid()
plt.xlabel("Iterations")
plt.ylabel("Relative Residual")
plt.title(r"Figure 3.02 - Relative Residual for $\rho$ for varying M")
plt.savefig("../figures/fig3.02.png")
plt.show()

NITER = 3000
iterations = np.linspace(0, NITER, int(NITER/ITER_INTERVAL)+1)
M_array = [0.9, 0.92, 0.94, 0.96, 0.98, 1]
res_list1_list = []
res_list2_list = []
for M in M_array:
    print(f"M = {M}")
    _, _, _, _, reslist1, reslist2, _ = GRID.check_convergence(
        M, rho_init, iter_interval=ITER_INTERVAL, max_iters=NITER)
    res_list1_list.append(reslist1)
    res_list2_list.append(reslist2)

plt.figure()
for i, M in enumerate(M_array):
    plt.semilogy(iterations, res_list1_list[i], label=f"M={M}")
plt.legend()
plt.grid()
plt.xlabel("Iterations")
plt.ylabel("Relative Residual")
plt.title(r"Figure 3.03 - Relative Residual for $\phi$ for varying M")
plt.savefig("../figures/fig3.03.png")
plt.show()

plt.figure()
for i, M in enumerate(M_array):
    plt.semilogy(iterations, res_list2_list[i], label=f"M={M}")
plt.legend()
plt.grid()
plt.xlabel("Iterations")
plt.ylabel("Relative Residual")
plt.title(r"Figure 3.04 - Relative Residual for $\rho$ for varying M")
plt.savefig("../figures/fig3.04.png")
plt.show()
