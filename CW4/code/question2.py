""" question2.py - Script containing code used for Question 2 of CW4.
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


# Constants Used Throughout:
q, s, r = 1, 2, 1
N, L = 501, 1001
tau = 0.05
gamma = 1.4
NITER = 1e4

phi_init = np.zeros((N, L), dtype=np.float64)
rho_init = np.zeros((N+2, L+2), dtype=np.float64)
GRID = scheme.Grid(phi_init, q, s, r)

M_list = [0.01, 0.2, 0.4, 0.6, 0.8, 0.9]
phi_list = []
rho_list = []
u_list = []
v_list = []

for M in M_list:
    print(f"M = {M}")
    phi, rho, u, v, _, _, _ = GRID.run_iteration(rho_init, NITER, M)
    phi_list.append(phi)
    rho_list.append(rho)
    u_list.append(u)
    v_list.append(v)

# Setting up Grid for Plots:
x = np.linspace(-q, s, L)
y = np.linspace(0, r, N)
X, Y = np.meshgrid(x, y[::-1])

# Figure 2.01, 2.02, 2.03 - Line PLots of U_surf, V_surf, rho_surf
plt.figure()
plt.plot(x, u_list[0][-1], "b", label=r"$U_{surf}$, M=0.01")
plt.plot(x, u_list[1][-1], "r--", label=r"$U_{surf}$, M=0.2")
plt.plot(x, u_list[2][-1], "k--", label=r"$U_{surf}$, M=0.4")
plt.plot(x, u_list[3][-1], "--", label=r"$U_{surf}$, M=0.6")
plt.plot(x, u_list[4][-1], "--", label=r"$U_{surf}$, M=0.8")
plt.plot(x, u_list[5][-1], "--", label=r"$U_{surf}$, M=0.9")
plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel(r"$U_{surf}$")
plt.title(r"Figure 2.01 - Plot of $U_{surf}$ for varying M (N=501, L=1001, Niter=100000)")
plt.savefig("../figures/fig2.01.png")
plt.show()

plt.figure()
plt.plot(x, v_list[0][-1], "b", label=r"$V_{surf}$, M=0.01")
plt.plot(x, v_list[1][-1], "r--", label=r"$V_{surf}$, M=0.2")
plt.plot(x, v_list[2][-1], "k--", label=r"$V_{surf}$, M=0.4")
plt.plot(x, v_list[3][-1], "--", label=r"$V_{surf}$, M=0.6")
plt.plot(x, v_list[4][-1], "--", label=r"$V_{surf}$, M=0.8")
plt.plot(x, v_list[5][-1], "--", label=r"$V_{surf}$, M=0.9")
plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel(r"$V_{surf}$")
plt.title(r"Figure 2.02 - Plot of $V_{surf}$ for varying M (N=501, L=1001, Niter=100000)")
plt.savefig("../figures/fig2.02.png")
plt.show()

plt.figure()
plt.plot(x, rho_list[0][-1], "b", label=r"$\rho_{surf}$, M=0.01")
plt.plot(x, rho_list[1][-1], "r--", label=r"$\rho_{surf}$, M=0.2")
plt.plot(x, rho_list[2][-1], "k--", label=r"$\rho_{surf}$, M=0.4")
plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel(r"$\rho_{surf}$")
plt.title(r"Figure 2.03.1 - Plot of $\rho_{surf}$ for varying M (N=501, L=1001, Niter=100000)")
plt.savefig("../figures/fig2.03.1.png")
plt.show()

plt.figure()
plt.plot(x, rho_list[0][-1], "b", label=r"$\rho_{surf}$, M=0.01")
plt.plot(x, rho_list[1][-1], "r--", label=r"$\rho_{surf}$, M=0.2")
plt.plot(x, rho_list[2][-1], "k--", label=r"$\rho_{surf}$, M=0.4")
plt.plot(x, rho_list[3][-1], "--", label=r"$\rho_{surf}$, M=0.6")
plt.plot(x, rho_list[4][-1], "--", label=r"$\rho_{surf}$, M=0.8")
plt.plot(x, rho_list[5][-1], "--", label=r"$\rho_{surf}$, M=0.9")
plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel(r"$\rho_{surf}$")
plt.title(r"Figure 2.03.2 - Plot of $\rho_{surf}$ for varying M (N=501, L=1001, Niter=100000)")
plt.savefig("../figures/fig2.03.2.png")
plt.show()


# Figures 2.04 to 2.15 - Contour Plots of phi, u, v and rho for entire (x, y) computational domain.
# M = 0.01
plt.figure(figsize=(10, 6))
plt.pcolor(X, Y, phi_list[0])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Figure 2.04 - Colour Plot of $\phi$ for $M=0.01$")
plt.savefig("../figures/fig2.04.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.pcolor(X, Y, u_list[0])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Figure 2.05 - Colour Plot of $u$ for $M=0.01$")
plt.savefig("../figures/fig2.05.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.pcolor(X, Y, v_list[0])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Figure 2.06 - Colour Plot of $v$ for $M=0.01$")
plt.savefig("../figures/fig2.06.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.pcolor(X, Y, rho_list[0])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Figure 2.07 - Colour Plot of $\rho$ for $M=0.01$")
plt.savefig("../figures/fig2.07.png")
plt.show()

# # M = 0.2
plt.figure(figsize=(10, 6))
plt.pcolor(X, Y, phi_list[1])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Figure 2.08 - Colour Plot of $\phi$ for $M=0.2$")
plt.savefig("../figures/fig2.08.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.pcolor(X, Y, u_list[1])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Figure 2.09 - Colour Plot of $u$ for $M=0.2$")
plt.savefig("../figures/fig2.09.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.pcolor(X, Y, v_list[1])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Figure 2.10 - Colour Plot of $v$ for $M=0.2$")
plt.savefig("../figures/fig2.10.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.pcolor(X, Y, rho_list[1])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Figure 2.11 - Colour Plot of $\rho$ for $M=0.2$")
plt.savefig("../figures/fig2.11.png")
plt.show()

# M = 0.4
plt.figure(figsize=(10, 6))
plt.pcolor(X, Y, phi_list[2])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Figure 2.12 - Colour Plot of $\phi$ for $M=0.4$")
plt.savefig("../figures/fig2.12.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.pcolor(X, Y, u_list[2])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Figure 2.13 - Colour Plot of $u$ for $M=0.4$")
plt.savefig("../figures/fig2.13.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.pcolor(X, Y, v_list[2])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Figure 2.14 - Colour Plot of $v$ for $M=0.4$")
plt.savefig("../figures/fig2.14.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.pcolor(X, Y, rho_list[2])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Figure 2.15 - Colour Plot of $\rho$ for $M=0.4$")
plt.savefig("../figures/fig2.15.png")
plt.show()
