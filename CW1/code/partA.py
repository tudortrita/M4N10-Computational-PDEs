""" partA.py - Script containing code used for computing results and plots in
               Part A of the Project.
Tudor Trita Trita
CID: 01199397
MSci Mathematics

M4N10 - Computational Partial Differential Equations - Project 1
"""
import matplotlib.pyplot as plt
import numpy as np

# Importing schemes used for this question.
from schemes import diffusion_analytic_solution
from schemes import diffusion_explicit_scheme

VERBOSE = False  # Toggle to output more words
DISPLAY = True  # Toggle to display figures by matplotlib
SAVEFIG = False  # Toggle to save figures to files

# Question 3 Code:
t = 0.075
x1 = 0.02
x2 = 0.5

U_exact_one = diffusion_analytic_solution(t, x1)
U_exact_two = diffusion_analytic_solution(t, x2)

# Plot A1 & A2: for N = 51, 101, 201, vary k to see what the error is.
N_array = np.array([51, 101, 201, 401])
k_array = np.logspace(-8, -4, 5, base=10)

error_array_one = np.zeros((2, N_array.size, k_array.size))

print("Starting Question 3 Data and Plots:\n")
for i, N in enumerate(N_array):
    for y, k in enumerate(k_array):
        print(f"N={N}, k={k:.1e}")
        U_FD = diffusion_explicit_scheme(t, k, N)
        U_FD_one = U_FD[int((N-1)*x1)]
        U_FD_two = U_FD[int((N-1)*x2)]

        error_one = np.log10(abs(U_exact_one - U_FD_one))
        error_two = np.log10(abs(U_exact_two - U_FD_two))
        print(f"error_0.02 = {error_one}, error_0.5 = {error_two}")
        error_array_one[0, i, y] = error_one
        error_array_one[1, i, y] = error_two


fig_A1 = plt.figure(figsize=(11, 6))
plt.semilogx(k_array, error_array_one[0, 0, :], "x--", label="N=51")
plt.semilogx(k_array, error_array_one[0, 1, :], ".--", label="N=101")
plt.semilogx(k_array, error_array_one[0, 2, :], "x--", label="N=201")
plt.semilogx(k_array, error_array_one[0, 3, :], ".--", label="N=401")


plt.xlabel(r"$k$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure A1: Error against $k$ for different values of $N$ at $x=0.02$")
if SAVEFIG:
    plt.savefig("fig_A1.png")
if DISPLAY:
    plt.show()

fig_A2 = plt.figure(figsize=(11, 6))
plt.semilogx(k_array, error_array_one[1, 0, :], "x--", label="N=51")
plt.semilogx(k_array, error_array_one[1, 1, :], ".--", label="N=101")
plt.semilogx(k_array, error_array_one[1, 2, :], "x--", label="N=201")
plt.semilogx(k_array, error_array_one[1, 3, :], ".--", label="N=401")

plt.xlabel(r"$k$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure A2: Error against $k$ for different values of $N$ at $x=0.5$")
if SAVEFIG:
    plt.savefig("fig_A2.png")
if DISPLAY:
    plt.show()

# Plot A3: Numerical Stability Restriction on the maximum value of r
# Fix N=101:
N = 101
h_squared = (1/(N - 1))**2
# K values that divide t=0.075 precisely for the plotting:
k_array = np.array([1.00e-06, 2.00e-06, 3.00e-06, 4.00e-06, 5.00e-06, 6.00e-06,
                    8.00e-06, 1.00e-05, 1.20e-05, 1.50e-05, 2.00e-05, 2.40e-05,
                    2.50e-05, 3.00e-05, 4.00e-05, 5.00e-05, 6.00e-05, 7.50e-05,
                    1.00e-04, 1.20e-04, 1.25e-04, 1.50e-04, 2.00e-04])

r_array = k_array / h_squared

error_array_two = np.zeros((2, k_array.size))

for i, k in enumerate(k_array):
        if VERBOSE:
            print(f"N={N}, k={k:.1e}")
        U_FD = diffusion_explicit_scheme(t, k, N)
        U_FD_one = U_FD[int((N-1)*x1)]
        U_FD_two = U_FD[int((N-1)*x2)]

        error_one = np.log10(abs(U_exact_one - U_FD_one))
        error_two = np.log10(abs(U_exact_two - U_FD_two))
        error_array_two[0, i] = error_one
        error_array_two[1, i] = error_two


fig_A3 = plt.figure(figsize=(11, 6))
plt.plot(r_array, error_array_two[0, :], label=r"$x=0.02$")
plt.plot(r_array, error_array_two[1, :], label=r"$x=0.5$")
plt.axvline(x=0.5, color='k', linestyle='--', label=r"$r=0.5$")
plt.xlabel(r"$r$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure A3: Errors against $r$, showing numerical stability " \
          r"around the value $r=0.5$")
if SAVEFIG:
    plt.savefig("fig_A3.png")
if DISPLAY:
    plt.show()


# Plot A4: Numerical Stability Restriction on the maximum value of r fine around 0.5
# Fix N=101:
N = 101
h_squared = (1/(N - 1))**2

# Code to find Suitable k that divide t "exactly"
K_candidates = np.linspace(4e-05, 6e-05, 6553)
C = 0.075 / K_candidates
C2 = np.abs(C - np.round(C))
k_array = K_candidates[C2 < 1e-12]
r_array = k_array / h_squared

error_array_three = np.zeros((2, k_array.size))

for i, k in enumerate(k_array):
        if VERBOSE:
            print(f"N={N}, k={k:.1e}")
        U_FD = diffusion_explicit_scheme(t, k, N)
        U_FD_one = U_FD[int((N-1)*x1)]
        U_FD_two = U_FD[int((N-1)*x2)]

        error_one = np.log10(abs(U_exact_one - U_FD_one))
        error_two = np.log10(abs(U_exact_two - U_FD_two))
        error_array_three[0, i] = error_one
        error_array_three[1, i] = error_two


fig_A4 = plt.figure(figsize=(11, 6))
plt.plot(r_array, error_array_three[0, :], "x--", label=r"$x=0.02$")
plt.plot(r_array, error_array_three[1, :], ".--", label=r"$x=0.5$")
plt.axvline(x=0.5, color='k', linestyle='--', label=r"$r=0.5$")
plt.xlabel(r"$r$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure A4: Errors against $r$, showing numerical stability " \
          r"finely around the value $r=0.5$")
if SAVEFIG:
    plt.savefig("fig_A4.png")
if DISPLAY:
    plt.show()
