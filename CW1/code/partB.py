""" partB.py - Script containing code used for computing results and plots in
               Part B of the Project.
Tudor Trita Trita
CID: 01199397
MSci Mathematics

M4N10 - Computational Partial Differential Equations - Project 1
"""
import time
import matplotlib.pyplot as plt
import numpy as np

# Importing schemes used for this question.
from schemes import diffusion_analytic_solution
from schemes import diffusion_explicit_scheme
from schemes import diffusion_fully_implicit
from schemes import diffusion_crank_nicolson
from schemes import diffusion_fully_implicit_alternative

VERBOSE = False  # Toggle to output more words
DISPLAY = True  # Toggle to display figures by matplotlib
SAVEFIG = False  # Toggle to save figures to files

# Question 2 Code:
t = 0.075
x1 = 0.02
x2 = 0.5

U_exact_one = diffusion_analytic_solution(t, x1)
U_exact_two = diffusion_analytic_solution(t, x2)

# Plot B1 & B2: Errors vs k, h for IMPLICIT SCHEME
N_array = np.array([51, 101, 201, 401])
k_array = np.logspace(-7, -4, 4, base=10)

error_array_one = np.zeros((2, N_array.size, k_array.size))

print("Starting Question 2 Data and Plots:\n")
for i, N in enumerate(N_array):
    for y, k in enumerate(k_array):
        print(f"N={N}, k={k:.1e}")
        U_FD = diffusion_fully_implicit(t, k, N)
        U_FD_one = U_FD[int((N-1)*x1)]
        U_FD_two = U_FD[int((N-1)*x2)]

        error_one = np.log10(abs(U_exact_one - U_FD_one))
        error_two = np.log10(abs(U_exact_two - U_FD_two))
        print(f"error_0.02 = {error_one}, error_0.5 = {error_two}")
        error_array_one[0, i, y] = error_one
        error_array_one[1, i, y] = error_two


fig_B1 = plt.figure(figsize=(11, 6))
plt.semilogx(k_array, error_array_one[0, 0, :], "x--", label="N=51")
plt.semilogx(k_array, error_array_one[0, 1, :], ".--", label="N=101")
plt.semilogx(k_array, error_array_one[0, 2, :], "x--", label="N=201")
plt.semilogx(k_array, error_array_one[0, 3, :], ".--", label="N=401")


plt.xlabel(r"$k$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure B1: (Fully-Implicit) - Error against $k$ for different " \
          r"values of $N$ at $x=0.02$")
if SAVEFIG:
    plt.savefig("fig_B1.png")
if DISPLAY:
    plt.show()

fig_B2 = plt.figure(figsize=(11, 6))
plt.semilogx(k_array, error_array_one[1, 0, :], "x--", label="N=51")
plt.semilogx(k_array, error_array_one[1, 1, :], ".--", label="N=101")
plt.semilogx(k_array, error_array_one[1, 2, :], "x--", label="N=201")
plt.semilogx(k_array, error_array_one[1, 3, :], ".--", label="N=401")

plt.xlabel(r"$k$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure B2: (Fully-Implicit) - Error against $k$ for different " \
          r"values of $N$ at $x=0.5$")
if SAVEFIG:
    plt.savefig("fig_B2.png")
if DISPLAY:
    plt.show()


# Plot B3 & B4: Errors vs k, h for CRANK-NICOLSON SCHEME
N_array = np.array([51, 101, 201, 401])
k_array = np.logspace(-6, -3, 4, base=10)

error_array_two = np.zeros((2, N_array.size, k_array.size))

for i, N in enumerate(N_array):
    for y, k in enumerate(k_array):
        print(f"N={N}, k={k:.1e}")
        U_FD = diffusion_crank_nicolson(t, k, N)
        U_FD_one = U_FD[int((N-1)*x1)]
        U_FD_two = U_FD[int((N-1)*x2)]

        error_one = np.log10(abs(U_exact_one - U_FD_one))
        error_two = np.log10(abs(U_exact_two - U_FD_two))
        print(f"error_0.02 = {error_one}, error_0.5 = {error_two}")
        error_array_two[0, i, y] = error_one
        error_array_two[1, i, y] = error_two


fig_B3 = plt.figure(figsize=(11, 6))
plt.semilogx(k_array, error_array_two[0, 0, :], "x--", label="N=51")
plt.semilogx(k_array, error_array_two[0, 1, :], ".--", label="N=101")
plt.semilogx(k_array, error_array_two[0, 2, :], "x--", label="N=201")
plt.semilogx(k_array, error_array_two[0, 3, :], ".--", label="N=401")


plt.xlabel(r"$k$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure B3: (Crank-Nicolson) - Error against $k$ for different " \
          r"values of $N$ at $x=0.02$")
if SAVEFIG:
    plt.savefig("fig_B3.png")
if DISPLAY:
    plt.show()

fig_B4 = plt.figure(figsize=(11, 6))
plt.semilogx(k_array, error_array_two[1, 0, :], "x--", label="N=51")
plt.semilogx(k_array, error_array_two[1, 1, :], ".--", label="N=101")
plt.semilogx(k_array, error_array_two[1, 2, :], "x--", label="N=201")
plt.semilogx(k_array, error_array_two[1, 3, :], ".--", label="N=401")

plt.xlabel(r"$k$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure B4: (Crank-Nicolson) - Error against $k$ for different " \
          r"values of $N$ at $x=0.5$")
if SAVEFIG:
    plt.savefig("fig_B4.png")
if DISPLAY:
    plt.show()


# Plots B5 Numerical Stability Fully-Implicit and Crank-Nicholson:
N = 101
h_squared = (1/(N - 1))**2
k_array = np.array([1.00e-06, 2.00e-06, 3.00e-06, 4.00e-06, 5.00e-06, 6.00e-06,
                    8.00e-06, 1.00e-05, 1.20e-05, 1.50e-05, 2.00e-05, 2.40e-05,
                    2.50e-05, 3.00e-05, 4.00e-05, 5.00e-05, 6.00e-05, 7.50e-05,
                    1.00e-04, 1.20e-04, 1.25e-04, 1.50e-04, 2.00e-04])

r_array = k_array / h_squared

error_array_three = np.zeros((4, k_array.size))

for i, k in enumerate(k_array):
        if VERBOSE:
            print(f"N={N}, k={k:.1e}")
        U_FD1 = diffusion_fully_implicit(t, k, N)
        U_FD_one1 = U_FD1[int((N-1)*x1)]
        U_FD_two1 = U_FD1[int((N-1)*x2)]
        U_FD2 = diffusion_crank_nicolson(t, k, N)
        U_FD_one2 = U_FD2[int((N-1)*x1)]
        U_FD_two2 = U_FD2[int((N-1)*x2)]

        error_array_three[0, i] = np.log10(abs(U_exact_one - U_FD_one1))
        error_array_three[1, i] = np.log10(abs(U_exact_two - U_FD_two1))
        error_array_three[2, i] = np.log10(abs(U_exact_one - U_FD_one2))
        error_array_three[3, i] = np.log10(abs(U_exact_two - U_FD_two2))

fig_B5 = plt.figure(figsize=(11, 6))
plt.plot(r_array, error_array_three[0, :], label=r"$\theta=1, x=0.02$")
plt.plot(r_array, error_array_three[1, :], label=r"$\theta=1, x=0.5$")
plt.plot(r_array, error_array_three[2, :], label=r"$\theta=0.5, x=0.02$")
plt.plot(r_array, error_array_three[3, :], label=r"$\theta=0.5, x=0.5$")
plt.xlabel(r"$r$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure B5: Numerical Stability of Implicit Schemes")
if SAVEFIG:
    plt.savefig("fig_B5.png")
if DISPLAY:
    plt.show()


# Question 3 Code:
print("Starting Question 3 Data and Plots:\n")

# Figure B6 & B7: Computer Times to reach Performance
N_array = np.array([1001, 1501, 2001])
k_array = np.array([2e-07, 2e-07, 1e-07])
times_array = np.zeros((2, N_array.size))
errors_array = np.zeros((2, N_array.size))

for i, N in enumerate(N_array):
    k = k_array[i]
    print(f"N={N}")
    t1 = time.perf_counter()  # Starting Clock
    U_FD_explicit = diffusion_explicit_scheme(t, k, N)[int((N-1)*x2)]
    t2 = time.perf_counter()  # Ending Clock

    t3 = time.perf_counter()
    U_FD_crank_nicolson = diffusion_crank_nicolson(t, 5e-04, N)[int((N-1)*x2)]
    t4 = time.perf_counter()

    errors_array[0, i] = np.log10(abs(U_exact_two - U_FD_explicit))
    errors_array[1, i] = np.log10(abs(U_exact_two - U_FD_crank_nicolson))

    times_array[0, i] = t2 - t1
    times_array[1, i] = t4 - t3

fig_B6 = plt.figure(figsize=(11, 6))
plt.semilogy(N_array, times_array[0, :], label="Explicit")
plt.semilogy(N_array, times_array[1, :], label="Crank-Nicolson")

plt.xlabel("N")
plt.ylabel("Time")
plt.grid()
plt.legend()
plt.title(r"Figure B6: Computational Time Comparison for Explicit, " \
           "Crank-Nicolson Schemes")
if SAVEFIG:
    plt.savefig("fig_B6.png")
if DISPLAY:
    plt.show()

fig_B7 = plt.figure(figsize=(11, 6))
plt.plot(N_array, times_array[1, :] / times_array[0, :])

plt.xlabel("N")
plt.ylabel("Ratio Time")
plt.grid()
plt.title(r"Figure B7: Computational Time Ratios  Explicit/Crank-Nicolson Schemes")
if SAVEFIG:
    plt.savefig("fig_B7.png")
if DISPLAY:
    plt.show()

# Plot B8: Showing Similar Errors Attained
fig_B8 = plt.figure(figsize=(11, 6))
plt.plot(N_array, errors_array[0, :], label="Explicit")
plt.plot(N_array, errors_array[1, :], label="Crank-Nicolson")
plt.grid()
plt.legend()
plt.xlabel("N")
plt.ylabel(r"$error$")
plt.title("Figure B8: Errors for Crank, Explicit")
if SAVEFIG:
    plt.savefig("fig_B8.png")
if DISPLAY:
    plt.show()


# Question 4 Code:
# Plot B9 & B10: Errors vs k, h for IMPLICIT SCHEME
print("Starting Question 4 Data and Plots:\n")
N_array = np.array([51, 101, 201, 401])
k_array = np.logspace(-6, -3, 4, base=10)

error_array_five = np.zeros((2, N_array.size, k_array.size))

for i, N in enumerate(N_array):
    for y, k in enumerate(k_array):
        print(f"N={N}, k={k:.1e}")
        U_FD = diffusion_fully_implicit_alternative(t, k, N)
        U_FD_one = U_FD[int((N-1)*x1)]
        U_FD_two = U_FD[int((N-1)*x2)]

        error_one = np.log10(abs(U_exact_one - U_FD_one))
        error_two = np.log10(abs(U_exact_two - U_FD_two))
        print(f"error_0.02 = {error_one}, error_0.5 = {error_two}")
        error_array_five[0, i, y] = error_one
        error_array_five[1, i, y] = error_two


fig_B9 = plt.figure(figsize=(11, 6))
plt.semilogx(k_array, error_array_five[0, 0, :], "x--", label="N=51")
plt.semilogx(k_array, error_array_five[0, 1, :], ".--", label="N=101")
plt.semilogx(k_array, error_array_five[0, 2, :], "x--", label="N=201")
plt.semilogx(k_array, error_array_five[0, 3, :], ".--", label="N=401")


plt.xlabel(r"$k$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure B9: (3-Point Time) - Error against $k$ for different " \
          r"values of $N$ at $x=0.02$")
if SAVEFIG:
    plt.savefig("fig_B9.png")
if DISPLAY:
    plt.show()

fig_B10 = plt.figure(figsize=(11, 6))
plt.semilogx(k_array, error_array_five[1, 0, :], "x--", label="N=51")
plt.semilogx(k_array, error_array_five[1, 1, :], ".--", label="N=101")
plt.semilogx(k_array, error_array_five[1, 2, :], "x--", label="N=201")
plt.semilogx(k_array, error_array_five[1, 3, :], ".--", label="N=401")

plt.xlabel(r"$k$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure B10: (3-Point Time) - Error against $k$ for different " \
          r"values of $N$ at $x=0.5$")
if SAVEFIG:
    plt.savefig("fig_B10.png")
if DISPLAY:
    plt.show()
