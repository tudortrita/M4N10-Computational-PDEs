""" question4.py - Script containing code used for Question 4 of CW2.
Tudor Trita Trita
CID: 01199397
MSci Mathematics

M4N10 - Computational Partial Differential Equations.
"""
import matplotlib
matplotlib.rcParams['figure.figsize'] = (11.0, 6.0)
import matplotlib.pyplot as plt
import numpy as np

import multigrid


def plot_solution_multigrid(U, q, s, r, N, L, method, fignum, savefig=True, display=True):
    """Helper function to be able to plot MultiGrid solutions Easily."""
    x = np.linspace(-q, s, L)
    y = np.linspace(0, r, N)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(11, 6))
    plt.pcolor(X, Y, U[1:-1, 1:-1][::-1])  # Excluding Ghost Points
    plt.colorbar()
    plt.title(rf"Figure {fignum} - Solution for {method}")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    if savefig:
        plt.savefig(f"../figures/fig{fignum}.png")
    if display:
        plt.show()

# Figure 4.0.1 & 4.0.2 Verification of Implementation:
q, s, r = 1, 2, 2
tau = 0.05
N, L = 129, 129
num_grids = 5
GS_iters_num = 10
phi_init = np.zeros((N, L), dtype=float)
f_init = np.zeros((N, L), dtype=float)

MG = multigrid.MultiGrid(phi_init, f_init, q, s, r, tau)
phi_out, error_val = MG.run_vcycle(num_grids, GS_iters_num, 200, compute_all_errors=False)

u = MG.u_quantity(phi_out)
v = MG.v_quantity(phi_out)

plot_solution_multigrid(phi_out, q, s, r, N, L, "Multi-Grid (5 Grids, (L, N) = (129, 129), 200 V-Cycles)", "4.01")

plt.figure()
x = np.linspace(-q, s, L)
plt.plot(x, u[-1], label=r"$U_{surf}$")
plt.plot(x, v[-1], label=r"$V_{surf}$")
plt.grid()
plt.legend()
plt.title("Figure 4.02 - Multi-Grid (5 Grids, (L, N) = (129, 129), 200 V-Cycles)")
plt.savefig("../figures/fig4.02.png")
plt.show()


# Figure 4.03, 4.04 Convergence Varying keeping (L, N) fixed.
q, s, r = 1, 2, 2
tau = 0.05
GS_iters_num = 10
N1, L1 = 65, 129
N2, L2 = 129, 257
phi_init1 = np.zeros((N1, L1), dtype=float)
f_init1 = np.zeros((N1, L1), dtype=float)
phi_init2 = np.zeros((N2, L2), dtype=float)
f_init2 = np.zeros((N2, L2), dtype=float)

MG1 = multigrid.MultiGrid(phi_init1, f_init1, q, s, r, tau)
MG2 = multigrid.MultiGrid(phi_init2, f_init2, q, s, r, tau)


error_list1 = []
error_list2 = []
for num_grids in [1, 2, 3, 4, 5]:
    print(num_grids)
    _, error_vals1 = MG1.run_vcycle(num_grids, GS_iters_num, 200, compute_all_errors=True)
    _, error_vals2 = MG2.run_vcycle(num_grids, GS_iters_num, 200, compute_all_errors=True)
    error_list1.append(error_vals1)
    error_list2.append(error_vals2)

iters_array = np.arange(0, 210, 10)

# MG1 First
plt.figure()
plt.semilogy(iters_array, error_list1[0], "b", label="No. Grids = 1 (GS)")
plt.semilogy(iters_array, error_list1[1], "k--", label="No. Grids = 2")
plt.semilogy(iters_array, error_list1[2], "g--", label="No. Grids = 3")
plt.semilogy(iters_array, error_list1[3], "r--", label="No. Grids = 4")
plt.semilogy(iters_array, error_list1[4], "y--", label="No. Grids = 5")

plt.xlabel("V-Cycles")
plt.ylabel("Residual Error")
plt.legend()
plt.grid()
plt.title("Figure 4.03 - Multi-Grid | Residual Error (L=65, N=129, GS Smoothing Iterations = 10)")
plt.savefig("../figures/fig4.03.png")
plt.show()

# MG2 First
plt.figure()
plt.semilogy(iters_array, error_list2[0], "b", label="No. Grids = 1 (GS)")
plt.semilogy(iters_array, error_list2[1], "k--", label="No. Grids = 2")
plt.semilogy(iters_array, error_list2[2], "g--", label="No. Grids = 3")
plt.semilogy(iters_array, error_list2[3], "r--", label="No. Grids = 4")
plt.semilogy(iters_array, error_list2[4], "y--", label="No. Grids = 5")

plt.xlabel("V-Cycles")
plt.ylabel("Residual Error")
plt.legend()
plt.grid()
plt.title("Figure 4.04 - Multi-Grid | Residual Error (L=129, N=257, GS Smoothing Iterations = 10)")
plt.savefig("../figures/fig4.04.png")
plt.show()
