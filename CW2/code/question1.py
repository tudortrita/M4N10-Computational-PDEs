""" question1.py - Script containing code used for Question 1 of CW2.
Tudor Trita Trita
CID: 01199397
MSci Mathematics

M4N10 - Computational Partial Differential Equations.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import iteration


def plot_solution(U, G, method, fignum, savefig=True, display=True):
    """ Helper function for plotting solutions easily."""
    x = np.linspace(-G.q, G.s, G.L)
    y = np.linspace(0, G.r, G.N)
    X, Y = np.meshgrid(x, y)

    plt.figure(figsize=(11, 6))
    plt.pcolor(X, Y, U[::-1])
    plt.colorbar()
    plt.title(rf"Figure {fignum} - Solution for {method}")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    if savefig:
        plt.savefig(f"../figures/fig{fignum}.png")
    if display:
        plt.show()


# Figures 1.01, 1.02, 1.03: Plots to verify Implementation
q, s, r = 2, 3, 2
N, L = 251, 251
Niter = 500
omega = 1.8

phi_init = np.zeros((N+2, L+2), dtype=float)
f_init = np.zeros((N+2, L+2), dtype=float)
G1 = iteration.Grid(phi_init, f_init, q, s, r)

phi_jac, _, _, _, _ = G1.run_iteration("Jacobi", Niter)
phi_gs, _, _, _, _ = G1.run_iteration("GS", Niter)
phi_sor, _, _, _, _ = G1.run_iteration("SOR1", Niter, omega=omega)

# Plots of Solution:
plot_solution(phi_jac, G1, f"Jacobi ({Niter} Iterations, N={N}, L={L})", "1.01")
plot_solution(phi_gs, G1, f"Gauss Seidel ({Niter} Iterations, N={N}, L={L})", "1.02")
plot_solution(phi_sor, G1, rf"SOR ({Niter} Iterations, N={N}, L={L}, $\omega = {omega})$", "1.03")

# Figure 1.04: Plots of Errors for all three methods at large number of iterations
q, s, r = 2, 3, 2
N, L = 51, 51
omega = 1.9
phi_init = np.zeros((N+2, L+2), dtype=float)
f_init = np.zeros((N+2, L+2), dtype=float)
G2 = iteration.Grid(phi_init, f_init, q, s, r)

Niters_list = np.logspace(0, 3.2, 25, dtype=int)
res_jac_list = []
res_gs_list = []
res_sor_list = []

for Niter in Niters_list:
    print(Niter)
    _, _, _, res_jac, _ = G2.run_iteration("Jacobi", Niter)
    _, _, _, res_gs, _ = G2.run_iteration("GS", Niter)
    _, _, _, res_sor, _ = G2.run_iteration("SOR1", Niter, omega=omega)
    res_jac_list.append(res_jac)
    res_gs_list.append(res_gs)
    res_sor_list.append(res_sor)

plt.figure(figsize=(11, 6))
plt.semilogy(Niters_list, res_jac_list, "b", label="Jacobi")
plt.semilogy(Niters_list, res_gs_list, "r--", label="GS")
plt.semilogy(Niters_list, res_sor_list, "k--", label=rf"SOR $\omega={omega}$")
plt.grid()
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Relative Residual")
plt.title("Figure 1.04 - Plots of Relative Errors")
plt.savefig("../figures/fig1.04.png")
plt.show()

Figure 1.05, 1.06: Plots of u=U-Surf, v velocity component along y=0
q, s, r = 1, 2, 1
N, L = 121, 241
Niter = 2000
omega = 1.85
x = np.linspace(-q, s, L)
y = np.linspace(0, r, N)
phi_init = np.zeros((N+2, L+2), dtype=float)
f_init = np.zeros((N+2, L+2), dtype=float)
G3 = iteration.Grid(phi_init, f_init, q, s, r)

phi_jac, u_jac, v_jac, res_jac, times_jac = G3.run_iteration("Jacobi", Niter)
phi_gs, u_gs, v_gs, res_gs, times_gs = G3.run_iteration("GS", Niter)
phi_sor, u_sor, v_sor, res_sor, times_sor = G3.run_iteration("SOR1", Niter, omega=omega)

# Figure of U_surf
fig = plt.figure(figsize=(11, 6))
plt.plot(x, u_jac[-1], "b", label=r"$U_{surf}$ Jacobi")
plt.plot(x, u_gs[-1], "r--", label=r"$U_{surf}$ GS")
plt.plot(x, u_sor[-1], "k--", label=r"$U_{surf}$ SOR " + rf"$\omega={omega}$")
plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel(r"$U_{surf}$ = $\frac{\partial \phi}{\partial x}$")
plt.title(r"Figure 1.05 - Plot of $\frac{\partial \phi}{\partial x}$ along " \
          rf"$y=0$ (N={N}, L={L}, {Niter} Iterations)")
plt.savefig("../figures/fig1.05.png")
plt.show()

# Figure of V_surf
fig = plt.figure(figsize=(11, 6))
plt.plot(x, v_jac[-1], "b", label=r"$V_{surf}$ Jacobi")
plt.plot(x, v_gs[-1], "r--", label=r"$V_{surf}$ GS")
plt.plot(x, v_sor[-1], "k--", label=r"$V_{surf}$ SOR " + rf"$\omega={omega}$")
plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel(r"$V_{surf}$ = $\frac{\partial \phi}{\partial y}$")
plt.title(r"Figure 1.06 - Plot of $\frac{\partial \phi}{\partial y}$ along " \
          rf"$y=0$ (N={N}, L={L}, {Niter} Iterations)")
plt.savefig("../figures/fig1.06.png")
plt.show()

Figure 1.07: (u, v) varying across the whole plane for the Iterative Methods.
fig, ax = plt.subplots(3, 2, figsize=(15, 15))
ax[0, 0].pcolor(x, y, u_jac[::-1])
ax[0, 0].set_xlabel("x")
ax[0, 0].set_ylabel("y")
ax[0, 0].set_title(r"$U_{surf}$ Jacobi")

ax[1, 0].pcolor(x, y, u_gs[::-1])
ax[1, 0].set_xlabel("x")
ax[1, 0].set_ylabel("y")
ax[1, 0].set_title(r"$U_{surf}$ GS")

ax[2, 0].pcolor(x, y, u_sor[::-1])
ax[2, 0].set_xlabel("x")
ax[2, 0].set_ylabel("y")
ax[2, 0].set_title(r"$U_{surf}$ SOR $\omega=1.85$")

ax[0, 1].pcolor(x, y, v_jac[::-1])
ax[0, 1].set_xlabel("x")
ax[0, 1].set_ylabel("y")
ax[0, 1].set_title(r"$V_{surf}$ Jacobi")

ax[1, 1].pcolor(x, y, v_gs[::-1])
ax[1, 1].set_xlabel("x")
ax[1, 1].set_ylabel("y")
ax[1, 1].set_title(r"$V_{surf}$ GS")

im = ax[2, 1].pcolor(x, y, v_sor[::-1])
ax[2, 1].set_xlabel("x")
ax[2, 1].set_ylabel("y")
ax[2, 1].set_title(r"$V_{surf}$ SOR $\omega=1.85$")

plt.suptitle("Figure 1.07 - Colour plot of entire (u, v) Perturbation Field Solution",
             fontsize=20)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
plt.colorbar(im, cax=cbar_ax)

plt.savefig("../figures/fig1.07.png")
plt.show()

# Figure 1.08: Computations as required in Guidelines.
loc1 = 82
loc2 = 100  # Corresponding to 0.25
loc3 = 120  # Corresponding to 0.5
loc4 = 140  # Corresponding to 0.75
loc5 = 152  # Corresponding to 0.95

locs = [loc1, loc2, loc3, loc4, loc5]
usurf_jac = np.array([u_jac[-1][i] for i in locs]).round(5)
usurf_gs = np.array([u_gs[-1][i] for i in locs]).round(5)
usurf_sor = np.array([u_sor[-1][i] for i in locs]).round(5)
vsurf_jac = np.array([v_jac[-1][i] for i in locs]).round(5)
vsurf_gs = np.array([v_gs[-1][i] for i in locs]).round(5)
vsurf_sor = np.array([v_sor[-1][i] for i in locs]).round(5)

vals = [0.025, 0.25, 0.5, 0.75, 0.95]
vals_index = [rf"$x = {val}$" for val in vals]
df = pd.DataFrame(np.array([vals_index, usurf_jac, usurf_gs, usurf_sor,
                           vsurf_jac, vsurf_gs, vsurf_sor]).T,
                  index=vals_index,
                  columns=["Values", r"$U_{surf}$ Jacobi", r"$U_{surf}$ GS",
                           r"$U_{surf}$ SOR $\omega=1.1$", r"$V_{surf}$ Jacobi",
                           r"$V_{surf}$ GS", r"$V_{surf}$ SOR $\omega=1.1$"])

# Figure 1.09: Grid Independence
q, s, r = 1, 2, 2
N1, L1 = 40, 80
N2, L2 = N1*2, L1*2
tau = 0.05
Niter1 = 500
Niter2 = 1000

phi_init1 = np.zeros((N1+2, L1+2), dtype=np.float64)
phi_init2 = np.zeros((N2+2, L2+2), dtype=np.float64)
f_init1 = np.zeros((N1+2, L1+2), dtype=np.float64)
f_init2 = np.zeros((N2+2, L2+2), dtype=np.float64)

GRID1 = iteration.Grid(phi_init1, f_init1, q, s, r)
GRID2 = iteration.Grid(phi_init2, f_init2, q, s, r)

phi_jac1, u_jac1, _, _, _ = GRID1.run_iteration("Jacobi", Niter1)
phi_gs1, u_gs1, _, _, _ = GRID1.run_iteration("GS", Niter1)
phi_sor1, u_sor1, _, _, _ = GRID1.run_iteration("SOR1", Niter1, omega=omega)

phi_jac2, u_jac2, _, _, _ = GRID2.run_iteration("Jacobi", Niter2)
phi_gs2, u_gs2, _, _, _ = GRID2.run_iteration("GS", Niter2)
phi_sor2, u_sor2, _, _, _ = GRID2.run_iteration("SOR2", Niter2, omega=omega)


x1 = np.linspace(-q, s, L1)
x2 = np.linspace(-q, s, L2)

fig = plt.figure(figsize=(11, 6))
plt.plot(x1, u_jac1[-1], label=f"Jacobi N={N1}, L={L1} Niter={Niter1}")
plt.plot(x2, u_jac2[-1], label=f"Jacobi N={N2}, L={L2} Niter={Niter2}")
plt.grid()
plt.legend()
plt.xlabel(r"$x$")
plt.ylabel(r"$U_{surf}$ = $\frac{\partial \phi}{\partial x}$")
plt.title(r"Figure 1.09 - $U_{surf}$ = $\frac{\partial \phi}{\partial x}$ for "\
          "Two Grids")
plt.savefig("../figures/fig1.09.png")
plt.show()

fig = plt.figure(figsize=(11, 6))
plt.plot(x1, u_gs1[-1], label=f"GS N={N1}, L={L1} Niter={Niter1}")
plt.plot(x2, u_gs2[-1], label=f"GS N={N2}, L={L2} Niter={Niter2}")
plt.grid()
plt.legend()
plt.xlabel(r"$x$")
plt.ylabel(r"$U_{surf}$ = $\frac{\partial \phi}{\partial x}$")
plt.title(r"Figure 1.10 - $U_{surf}$ = $\frac{\partial \phi}{\partial x}$ for "\
          "Two Grids")
plt.savefig("../figures/fig1.10.png")
plt.show()

fig = plt.figure(figsize=(11, 6))
plt.plot(x1, u_sor1[-1], label=f"SOR N={N1}, L={L1} Niter={Niter1}")
plt.plot(x2, u_sor2[-1], label=f"SOR N={N2}, L={L2} Niter={Niter2}")
plt.grid()
plt.legend()
plt.xlabel(r"$x$")
plt.ylabel(r"$U_{surf}$ = $\frac{\partial \phi}{\partial x}$")
plt.title(r"Figure 1.11 - $U_{surf}$ = $\frac{\partial \phi}{\partial x}$ for "\
          "Two Grids")
plt.savefig("../figures/fig1.11.png")
plt.show()
