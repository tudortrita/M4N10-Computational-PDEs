""" question3.py - Script containing code used for Question 3 of CW2.
Tudor Trita Trita
CID: 01199397
MSci Mathematics

M4N10 - Computational Partial Differential Equations.
"""
import matplotlib.pyplot as plt
import numpy as np

import iteration


# Figure 3.01, 3.02 - U_surf, V_surf at tau=0.05
q, s, r = 1, 2, 1
tau = 0.05
N, L = 101, 201
Niter = 1000
omega = 1.9

phi_init = np.zeros((N+2, L+2), dtype=float)
f_init = np.zeros((N+2, L+2), dtype=float)
G1 = iteration.Grid(phi_init, f_init, q, s, r)

phi_sor1, u_sor1, v_sor1, res_sor1, time_sor1 = G1.run_iteration("SOR1", Niter,
                                                                 tau=tau, omega=omega)
phi_sor2, u_sor2, v_sor2, res_sor2, time_sor2 = G1.run_iteration("SOR2", Niter, tau=tau,
                                                                 omega=omega)

x = np.linspace(-q, s, L)
plt.figure(figsize=(11, 6))
plt.plot(x, u_sor1[-1], "b--", label=r"Normal SOR $(x, y)$")
plt.plot(x, u_sor2[-1], "k--", label=r"Transformed SOR $(x, \eta)$")
plt.xlabel("x")
plt.ylabel(r"$U_{surf}$")
plt.legend()
plt.grid()
plt.title(rf"Figure 3.01 - Comparing SOR Methods at $\tau=0.05$ (L=201, N=101, Niter=1000)")
plt.savefig("../figures/fig3.01.png")
plt.show()

plt.figure(figsize=(11, 6))
plt.plot(x, v_sor1[-1], "b--", label=r"Normal SOR $(x, y)$")
plt.plot(x, v_sor2[-1], "k--", label=r"Transformed SOR $(x, \eta)$")
plt.xlabel("x")
plt.ylabel(r"$V_{surf}$")
plt.legend()
plt.grid()
plt.title(rf"Figure 3.02 - Comparing SOR Methods at $\tau=0.05$ (L=201, N=101, Niter=1000)")
plt.savefig("../figures/fig3.02.png")
plt.show()

# Figure 3.03, 3.04 - U_surf, V_surf at tau=0.4
q, s, r = 1, 2, 1
tau = 0.4
N, L = 101, 201
Niter = 1000
omega = 1.9

phi_init = np.zeros((N+2, L+2), dtype=float)
f_init = np.zeros((N+2, L+2), dtype=float)
G2 = iteration.Grid(phi_init, f_init, q, s, r)

phi_sor1, u_sor1, v_sor1, res_sor1, time_sor1 = G2.run_iteration("SOR1", Niter,
                                                                 tau=tau, omega=omega)
phi_sor2, u_sor2, v_sor2, res_sor2, time_sor2 = G2.run_iteration("SOR2", Niter, tau=tau,
                                                                 omega=omega)

x = np.linspace(-q, s, L)
plt.figure(figsize=(11, 6))
plt.plot(x, u_sor1[-1], "b--", label=r"Normal SOR $(x, y)$")
plt.plot(x, u_sor2[-1], "k--", label=r"Transformed SOR $(x, \eta)$")
plt.xlabel("x")
plt.ylabel(r"$U_{surf}$")
plt.legend()
plt.grid()
plt.title(rf"Figure 3.03 - Comparing SOR Methods at $\tau=0.4$ (L=201, N=101, Niter=1000)")
plt.savefig("../figures/fig3.03.png")
plt.show()

plt.figure(figsize=(11, 6))
plt.plot(x, v_sor1[-1], "b--", label=r"Normal SOR $(x, y)$")
plt.plot(x, v_sor2[-1], "k--", label=r"Transformed SOR $(x, \eta)$")
plt.xlabel("x")
plt.ylabel(r"$V_{surf}$")
plt.legend()
plt.grid()
plt.title(rf"Figure 3.04 - Comparing SOR Methods at $\tau=0.4$ (L=201, N=101, Niter=1000)")
plt.savefig("../figures/fig3.04.png")
plt.show()

# Figure 3.05 - Comparing SOR Methods for varying tau
tau_array = np.linspace(0.001, 0.4, 10)
q, s, r = 1, 2, 1
N, L = 101, 201
Niter = 500
omega = 1.8
phi_init = np.zeros((N, L), dtype=np.float64)
f_init = np.zeros((N, L), dtype=np.float64)
G3 = iteration.Grid(phi_init, f_init, q, s, r)

diff_u_list = []
diff_v_list = []
VERBOSE=True
for i, tau in enumerate(tau_array):
    if VERBOSE:
        print(tau)
    phi_sor1, u_sor1, v_sor1, _, _ = G3.run_iteration("SOR1", Niter, omega=omega, tau=tau)
    phi_sor2, u_sor2, v_sor2, _, _ = G3.run_iteration("SOR2", Niter, omega=omega, tau=tau)

    diff_u = np.sqrt(np.sum(np.abs(u_sor1 - u_sor2)**2))
    diff_v = np.sqrt(np.sum(np.abs(v_sor1 - v_sor2)**2))

    diff_u_list.append(diff_u)
    diff_v_list.append(diff_v)

plt.figure(figsize=(11, 6))
plt.plot(tau_array, diff_u_list, "b", label=r"Difference in $U_{surf}$")
plt.plot(tau_array, diff_v_list, "k--", label=r"Difference in $V_{surf}$")
plt.xlabel(r"$\tau$")
plt.legend()
plt.grid()
plt.title(r"Figure 3.05 - Plot highlighting differences " \
          r"when varying $\tau$ (L=201, N=101, Niter=500)")
plt.savefig("../figures/fig3.05.png")
plt.show()
