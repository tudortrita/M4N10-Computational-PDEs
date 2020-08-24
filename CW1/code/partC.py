""" partC.py - Script containing code used for computing results and plots in
               Part C of the Project.
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
from schemes import diffusion_explicit_fourth_order
from schemes import diffusion_fully_implicit_fourth_order

VERBOSE = False  # Toggle to output more words
DISPLAY = True  # Toggle to display figures by matplotlib
SAVEFIG = False  # Toggle to save figures to files


# Question 1 Code:
def fd_coefficients(z, x, m):
    """Fast Numerical Algorithm for Computing Finite Difference Coefficients.

    Parameters
    ----------
    z : location where coefficients to be computed (e.g. 0 for centred FD)
    x : array of coordinates for the grid points (e.g. [-2, -1, 0, 1, 2] for 5-point)
    m : order of the FD coefficients to compute

    Returns
    -------
    coeff : FD coefficients

    Note: Reference to paper containing code & algorithm from which function
    script has been adapted from:
    https://www.colorado.edu/amath/sites/default/files/attached-files/fd_weights.pdf
    """
    n = x.size
    c = np.zeros((m+2, n), dtype=float)
    c[1, 0] = 1
    x1 = np.tile(x, (n, 1))
    A = x1.T - x1
    b = np.cumprod(np.insert(A, 0, np.ones(n, dtype=float), axis=1), axis=1)
    rm = np.cumsum(np.ones((m+2, n-1), dtype=float), axis=0) - 1
    d = np.diag(b).copy()
    d[:n-1] = d[:n-1] / d[1:]

    for i in np.arange(1, n):
        mn = min(i, m+1)
        c[1:mn+2, i] = (d[i-1]*(rm[:mn+1, 0] *c[:mn+1, i-1] - (x[i-1] - z) * c[1:mn+2, i-1]))
        c[1:mn+2, :i] = (((x[i] - z) * c[1:mn+2, :i] - rm[:mn+1, :i] * c[:mn+1, :i]) / (x[i] - x1[:mn+1, :i]))
    return c[1:, :]


x = np.array([-2, -1, 0, 1, 2], dtype=float)
m = 4

central_differences = fd_coefficients(0, x, m)[2, :]
backward_differences = fd_coefficients(-1, x, m)[2, :]
forward_differences = fd_coefficients(1, x, m)[2, :]

np.set_printoptions(4)
print("Central Differences:")
print(central_differences)

print("\nBackward Differences:")
print(backward_differences)

print("\nForward Differences:")
print(forward_differences)

# Getting analytic solution used later:
t = 0.075
x1 = 0.02
x2 = 0.5

U_exact_one = diffusion_analytic_solution(t, x1)
U_exact_two = diffusion_analytic_solution(t, x2)


# Question 2 Code: (FOURTH ORDER EXPLICIT SCHEME)
# Plot C1 & C2: for N = 51, 101, 201, vary k to see what the error is.
N_array = np.array([51, 101, 201, 401])
k_array = np.logspace(-7, -4, 4, base=10)

error_array_one = np.zeros((2, N_array.size, k_array.size))

print("Starting Question 2 Data and Plots:\n")
for i, N in enumerate(N_array):
    for y, k in enumerate(k_array):
        print(f"N={N}, k={k:.1e}")
        U_FD = diffusion_explicit_fourth_order(t, k, N)
        U_FD_one = U_FD[int((N-1)*x1)]
        U_FD_two = U_FD[int((N-1)*x2)]

        error_one = np.log10(abs(U_exact_one - U_FD_one))
        error_two = np.log10(abs(U_exact_two - U_FD_two))
        print(f"error_0.02 = {error_one}, error_0.5 = {error_two}")
        error_array_one[0, i, y] = error_one
        error_array_one[1, i, y] = error_two


fig_C1 = plt.figure(figsize=(11, 6))
plt.semilogx(k_array, error_array_one[0, 0, :], "x--", label="N=51")
plt.semilogx(k_array, error_array_one[0, 1, :], ".--", label="N=101")
plt.semilogx(k_array, error_array_one[0, 2, :], "x--", label="N=201")
plt.semilogx(k_array, error_array_one[0, 3, :], ".--", label="N=401")


plt.xlabel(r"$k$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure C1: (Explicit $4^{th}$ Order) Error against $k$ for " \
          r"different values of $N$ at $x=0.02$")
if SAVEFIG:
    plt.savefig("fig_C1.png")
if DISPLAY:
    plt.show()

fig_C2 = plt.figure(figsize=(11, 6))
plt.semilogx(k_array, error_array_one[1, 0, :], "x--", label="N=51")
plt.semilogx(k_array, error_array_one[1, 1, :], ".--", label="N=101")
plt.semilogx(k_array, error_array_one[1, 2, :], "x--", label="N=201")
plt.semilogx(k_array, error_array_one[1, 3, :], ".--", label="N=401")

plt.xlabel(r"$k$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure C2: (Explicit $4^{th}$ Order) Error against $k$ for " \
          r"different values of $N$ at $x=0.5$")
if SAVEFIG:
    plt.savefig("fig_C2.png")
if DISPLAY:
    plt.show()



# Plot C3: Numerical Stability Restriction on the maximum value of r
# Fix N=101:
N = 101
h_squared = (1/(N - 1))**2
k_array = np.array([1.00e-06, 2.00e-06, 3.00e-06, 4.00e-06, 5.00e-06, 6.00e-06,
                    8.00e-06, 1.00e-05, 1.20e-05, 1.50e-05, 2.00e-05, 2.40e-05,
                    2.50e-05, 3.00e-05, 4.00e-05, 5.00e-05, 6.00e-05, 7.50e-05,
                    1.00e-04, 1.20e-04, 1.25e-04, 1.50e-04, 2.00e-04])

r_array = k_array / h_squared

error_array_two = np.zeros((2, k_array.size))

for i, k in enumerate(k_array):
        if VERBOSE:
            print(f"N={N}, k={k:.1e}")
        U_FD = diffusion_explicit_fourth_order(t, k, N)
        U_FD_one = U_FD[int((N-1)*x1)]
        U_FD_two = U_FD[int((N-1)*x2)]

        error_one = np.log10(abs(U_exact_one - U_FD_one))
        error_two = np.log10(abs(U_exact_two - U_FD_two))
        error_array_two[0, i] = error_one
        error_array_two[1, i] = error_two


# Plot C4: Numerical Stability Restriction on the maximum value of r fine around 0.3
# Fix N=101:
N = 101
h_squared = (1/(N - 1))**2

# Code to find suitable k's that divide 0.075
K_candidates = np.linspace(3e-05, 4e-05, 6553)
C = 0.075 / K_candidates
C2 = np.abs(C - np.round(C))
k_array = K_candidates[C2 < 1e-12]
r_array = k_array / h_squared

error_array_three = np.zeros((2, k_array.size))

for i, k in enumerate(k_array):
        if VERBOSE:
            print(f"N={N}, k={k:.1e}")
        U_FD = diffusion_explicit_fourth_order(t, k, N)
        U_FD_one = U_FD[int((N-1)*x1)]
        U_FD_two = U_FD[int((N-1)*x2)]

        error_one = np.log10(abs(U_exact_one - U_FD_one))
        error_two = np.log10(abs(U_exact_two - U_FD_two))
        error_array_three[0, i] = error_one
        error_array_three[1, i] = error_two


fig_C4 = plt.figure(figsize=(11, 6))
plt.plot(r_array, error_array_three[0, :], "x--", label=r"$x=0.02$")
plt.plot(r_array, error_array_three[1, :], ".--", label=r"$x=0.5$")
plt.axvline(x=0.375, color='k', linestyle='--', label=r"$r=0.375$")
plt.xlabel(r"$r$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure C4: Errors against $r$, showing numerical stability " \
          r"finely around the value $r=0.375$")
if SAVEFIG:
    plt.savefig("fig_C4.png")
if DISPLAY:
    plt.show()

# Question 3 Code: (COMPARING EXPLICIT 2ND AND 4TH ORDER)
N = 21
x_points = np.linspace(0, 1, N)
u_analytic = np.zeros(N)
t = 0.005
k = 1e-06

print("Starting Question 3 Data and Plots:\n")

for i, x in enumerate(x_points[1:-1]):
    u_analytic[i+1] = diffusion_analytic_solution(t, x)

U_explicit_2nd_order = diffusion_explicit_scheme(t, k, N)
U_explicit_4th_order = diffusion_explicit_fourth_order(t, k, N)

fig_C5 = plt.figure(figsize=(11, 6))
plt.plot(x_points, U_explicit_2nd_order, "rx--",
         label=r"Explicit $2^{nd}$ order")
plt.plot(x_points, U_explicit_4th_order, "bx--",
         label=r"Explicit $4^{th}$ order")
plt.plot(x_points, u_analytic, "gx--",
         label=r"Analytic Solution")

plt.legend()
plt.grid()
plt.xlabel(r"$X$")
plt.ylabel(r"$U$")
plt.title(r"Figure C5: Plot of Solution for the Different Schemes ($t=0.005$)")
if SAVEFIG:
    plt.savefig("fig_C5.png")
if DISPLAY:
    plt.show()

error1 = np.log10(np.abs(U_explicit_2nd_order - U_explicit_4th_order))
error2 = np.log10(np.abs(u_analytic - U_explicit_4th_order))
error3 = np.log10(np.abs(u_analytic - U_explicit_2nd_order))

fig_C6 = plt.figure(figsize=(11, 6))
plt.plot(x_points, error1, "rx--",
         label=r"$\log_{10}|u_{2nd \ order} - u_{4th \ order}|$")
plt.plot(x_points, error2, "bx--",
         label=r"$\log_{10}|u_{analytic} - u_{4th \  order}|$")
plt.plot(x_points, error3, "gx--",
         label=r"$\log_{10}|u_{analytic} - u_{2th \ order}|$")
plt.legend()
plt.grid()
plt.xlabel(r"$X$")
plt.ylabel(r"$error$")
plt.title(r"Figure C6: Plot of Error of Solutions ($k=10^{-6}$, $t=0.005$)")
if SAVEFIG:
    plt.savefig("fig_C6.png")
if DISPLAY:
    plt.show()


# Time t=0.075
N = 101
x_points = np.linspace(0, 1, N)
u_analytic = np.zeros(N)
t = 0.075
k = 1e-06

print("Starting Question 3 Data and Plots:\n")

for i, x in enumerate(x_points[1:-1]):
    u_analytic[i+1] = diffusion_analytic_solution(t, x)

U_explicit_2nd_order = diffusion_explicit_scheme(t, k, N)
U_explicit_4th_order = diffusion_explicit_fourth_order(t, k, N)

error1 = np.log10(np.abs(U_explicit_2nd_order - U_explicit_4th_order))
error2 = np.log10(np.abs(u_analytic - U_explicit_4th_order))
error3 = np.log10(np.abs(u_analytic - U_explicit_2nd_order))

fig_C6A = plt.figure(figsize=(11, 6))
plt.plot(x_points, error1, "rx--",
         label=r"$\log_{10}|u_{2nd \ order} - u_{4th \ order}|$")
plt.plot(x_points, error2, "bx--",
         label=r"$\log_{10}|u_{analytic} - u_{4th \  order}|$")
plt.plot(x_points, error3, "gx--",
         label=r"$\log_{10}|u_{analytic} - u_{2th \ order}|$")
plt.legend()
plt.grid()
plt.xlabel(r"$X$")
plt.ylabel(r"$error$")
plt.title(r"Figure C6A: Plot of Error of Solutions ($N=101, k=10^{-6}$, $t=0.075$)")
if SAVEFIG:
    plt.savefig("fig_C6A.png")
if DISPLAY:
    plt.show()



# Question 4 Code: (FOURTH ORDER FULLY-IMPLICIT SCHEME)
print("Starting Question 4 Data and Plots:\n")
# Plot C7 & C8: for N = 51, 101, 201, vary k to see what the error is.
t = 0.075
x1 = 0.02
x2 = 0.5
N_array = np.array([51, 101, 201, 401])
k_array = np.logspace(-7, -3, 5, base=10)

error_array = np.zeros((2, N_array.size, k_array.size))

for i, N in enumerate(N_array):
    for y, k in enumerate(k_array):
        print(f"N={N}, k={k:.1e}")
        U_FD = diffusion_fully_implicit_fourth_order(t, k, N)
        U_FD_one = U_FD[int((N-1)*x1)]
        U_FD_two = U_FD[int((N-1)*x2)]

        error_one = np.log10(abs(U_exact_one - U_FD_one))
        error_two = np.log10(abs(U_exact_two - U_FD_two))
        print(f"error_0.02 = {error_one}, error_0.5 = {error_two}")
        error_array[0, i, y] = error_one
        error_array[1, i, y] = error_two


fig_C7 = plt.figure(figsize=(11, 6))
plt.semilogx(k_array, error_array[0, 0, :], "x--", label="N=51")
plt.semilogx(k_array, error_array[0, 1, :], ".--", label="N=101")
plt.semilogx(k_array, error_array[0, 2, :], "x--", label="N=201")
plt.semilogx(k_array, error_array[0, 3, :], ".--", label="N=401")


plt.xlabel(r"$k$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure C7: (Implicit $4^{th}$ Order) Error against $k$ for " \
          r"different values of $N$ at $x=0.02$, $t=0.075$")
if SAVEFIG:
    plt.savefig("fig_C7.png")
if DISPLAY:
    plt.show()

fig_C8 = plt.figure(figsize=(11, 6))
plt.semilogx(k_array, error_array[1, 0, :], "x--", label="N=51")
plt.semilogx(k_array, error_array[1, 1, :], ".--", label="N=101")
plt.semilogx(k_array, error_array[1, 2, :], "x--", label="N=201")
plt.semilogx(k_array, error_array[1, 3, :], ".--", label="N=401")

plt.xlabel(r"$k$")
plt.ylabel(r"$\log_{10}|u_{numerical} - u_{analytic}|$")
plt.legend()
plt.grid()
plt.title(r"Figure C8: (Implicit $4^{th}$ Order) Error against $k$ for " \
          r"different values of $N$ at $x=0.5$, $t=0.075$")
if SAVEFIG:
    plt.savefig("fig_C8.png")
if DISPLAY:
    plt.show()

# Plot C9, C10: Timings comparing fully-implicit 2nd and 4th order schemes.
t = 0.075
x1 = 0.02
x2 = 0.5
N_array = np.array([101, 201, 401])
k_array = np.array([1e-05, 1e-06, 1e-06])
times_array = np.zeros((2, N_array.size))
errors_array = np.zeros((2, N_array.size))

for i, N in enumerate(N_array):
    k = k_array[i]
    print(f"N={N}")
    t1 = time.perf_counter()  # Starting Clock
    U_FD_2ndorder = diffusion_fully_implicit(t, k, N)[int((N-1)*x2)]
    t2 = time.perf_counter()  # Ending Clock

    t3 = time.perf_counter()
    U_FD_4thorder = diffusion_fully_implicit_fourth_order(t, 5e-07, N)[int((N-1)*x2)]
    t4 = time.perf_counter()

    errors_array[0, i] = np.log10(abs(U_exact_two - U_FD_2ndorder))
    errors_array[1, i] = np.log10(abs(U_exact_two - U_FD_4thorder))

    times_array[0, i] = t2 - t1
    times_array[1, i] = t4 - t3

fig_C9 = plt.figure(figsize=(11, 6))
plt.plot(N_array, times_array[0, :], label=r"Implicit $2^{nd}$ Order")
plt.plot(N_array, times_array[1, :], label=r"Implicit $4^{th}$ Order")

plt.xlabel("N")
plt.ylabel("Time")
plt.grid()
plt.legend()
plt.title(r"Figure C9: Computational Time Comparison for Implicit, " \
          r"$2^{nd}$ and $4^{th}$ Order, ($t=0.075$)")
if SAVEFIG:
    plt.savefig("fig_C9.png")
if DISPLAY:
    plt.show()

# Plot C10: Showing Similar Errors Attained
fig_C10 = plt.figure(figsize=(11, 6))
plt.plot(N_array, errors_array[0, :], label=r"Implicit $2^{nd}$ Order")
plt.plot(N_array, errors_array[1, :], label=r"Implicit $4^{th}$ Order")
plt.grid()
plt.legend()
plt.xlabel("N")
plt.ylabel(r"$error$")
plt.title("Figure C10: Errors for Implicit $2^{nd}$ and $4^{th}$ Order, ($t=0.075$)")
if SAVEFIG:
    plt.savefig("fig_C10.png")
if DISPLAY:
    plt.show()
