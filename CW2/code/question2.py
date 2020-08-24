""" question2.py - Script containing code used for Question 2 of CW2.
Tudor Trita Trita
CID: 01199397
MSci Mathematics

M4N10 - Computational Partial Differential Equations.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import iteration


# Figure 2.01: Plot to find Optimal omega for varying N x L:
q, s, r = 1, 2, 1
N_array = [30, 50, 70, 90]
L_array = [60, 100, 120, 140]
omega_array = (2 - np.logspace(-2, 0, 30))[::-1]
Niter = 2000

errors_master_list = []
for N, L in zip(N_array, L_array):
    phi_init = np.zeros((N+2, L+2), dtype=float)
    f_init = np.zeros((N+2, L+2), dtype=float)
    G = iteration.Grid(phi_init, f_init, q, s, r)
    # Biased towards higher omega since interesting results are there
    errors_list = []
    for omega in omega_array:
        print(N, L, omega)
        _, _, _, errors_sor, _ = G.run_iteration("SOR1", Niter, omega=omega)
        errors_list.append(errors_sor)
    errors_array = np.array(errors_list)
    errors_master_list.append(errors_array)

min_omega0 = omega_array[np.argmin(errors_master_list[0])]
min_error0 = np.min(errors_master_list[0])

min_omega1 = omega_array[np.argmin(errors_master_list[1])]
min_error1 = np.min(errors_master_list[1])

min_omega2 = omega_array[np.argmin(errors_master_list[2])]
min_error2 = np.min(errors_master_list[2])

min_omega3 = omega_array[np.argmin(errors_master_list[3])]
min_error3 = np.min(errors_master_list[3])

plt.figure(figsize=(11, 6))
plt.semilogy(omega_array[:-8], errors_master_list[0][:-8], label="N=30, L=60")
plt.plot(min_omega0, min_error0, "rx", label=rf"$\min \ \omega = {min_omega0:.2f}$")
plt.semilogy(omega_array[:-7], errors_master_list[1][:-7], label="N=50, L=100")
plt.plot(min_omega1, min_error1, "rx", label=rf"$\min \ \omega = {min_omega1:.2f}$")
plt.semilogy(omega_array[:-5], errors_master_list[2][:-5], label="N=70, L=120")
plt.plot(min_omega2, min_error2, "rx", label=rf"$\min \ \omega = {min_omega2:.2f}$")
plt.semilogy(omega_array[:-5], errors_master_list[3][:-5], label="N=90, L=140")
plt.plot(min_omega3, min_error3, "rx", label=rf"$\min \ \omega = {min_omega3:.2f}$")
plt.legend()
plt.grid()
plt.xlabel(r"$\omega$")
plt.ylabel("Relative Error at Iteration 2000")
plt.title("Figure 2.01 - SOR Relative Errors at Iteration 2000")
plt.savefig("../figures/fig2.01.png")
plt.show()

# Figure 2.02 - Theoretical expected rate/curves for same grid-size
N_array = np.linspace(10, 200)
jacobi_spectral_radii = np.cos(np.pi/(N_array))
theor_omegas = 2 / (1 + np.sqrt(1 - jacobi_spectral_radii**2))

plt.figure(figsize=(11, 6))
plt.plot(N_array, theor_omegas)
plt.xlabel("N")
plt.ylabel(r"$\omega$")
plt.title(r"Figure 2.02 - Theoretical Optimal $\omega_{SOR}$ Values for Uniform Grid")
plt.grid()
plt.savefig("../figures/fig2.02.png")
plt.show()

# Theoretical Values with our array
N_array = np.array([30, 50, 70, 90])
L_array = np.array([60, 100, 120, 140])
min_omegas_array = np.array([min_omega0, min_omega1, min_omega2, min_omega3])

jacobi_spectral_radius = 0.5*(np.cos(np.pi/(N_array)) + np.cos(np.pi/(L_array)))
exact_omega = 2/(1 + np.sqrt(1 - jacobi_spectral_radius**2))


data = np.array([["N=30, L=50", "N=50, L=100", "N=70, L=120", "N=90, L=140"],
                 min_omegas_array.round(4),
                 exact_omega.round(4)])

df = pd.DataFrame(data.T, columns=["Grid Size", "Calculated Values", "Theoretical Values"])
