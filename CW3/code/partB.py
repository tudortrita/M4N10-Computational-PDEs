""" partB.py - Script containing code used for computing results and plots in
                Part B of the Project.
Tudor Trita Trita
CID: 01199397
MSci Mathematics
M4N10 - Computational Partial Differential Equations - Project 3
"""
import matplotlib
matplotlib.rc('figure', figsize=(11, 6))
import matplotlib.pyplot as plt

import numpy as np

import waves
import importlib
importlib.reload(waves)

# Figure B.01, B.02 - Plots of solution along (x, y=0), (x=0, y) data planes.
N = 501
L = 2
delta = 0.2
h = 1/(N-1)
optim_r = 1/np.sqrt(2)
k = optim_r*h
r = k/h
q = 1

T_list = [0.7, 1.4, 2.1, 2.8, 3.5, 8, 20]
x_plane_list = []
y_plane_list = []
Umat_list = []

WAVE2 = waves.WaveEquation2D(N, L)
_, X, Y, _ = WAVE2.initial_conditions(delta)
for T in T_list:
    _, Umat = WAVE2.numerical_scheme(k, T, delta, q)
    Umat_list.append(Umat)
    x_plane_list.append(Umat[int((N+1)/2), :])
    y_plane_list.append(Umat[:, int((N+1)/2)])

# X-Plane Figure
plt.figure()
for UX, t in zip(x_plane_list, T_list):
    plt.plot(X[int((N+1)/2), :], UX, label=rf"$t={t}$")
plt.grid()
plt.legend()
plt.xlabel("X")
plt.ylabel("U")
plt.title(rf"Figure B.01 - X-Plane at Different Times $N={N}, r={r:.2f}$")
# plt.savefig("../figures/figB.01.png")
plt.show()

# Y-Plane Figure
plt.figure()
for UY, t in zip(y_plane_list, T_list):
    plt.plot(Y[:, int((N+1)/2)], UY, label=rf"$t={t}$")
plt.grid()
plt.legend()
plt.xlabel("Y")
plt.ylabel("U")
plt.title(rf"Figure B.02 - Y-Plane at Different Times $N={N}, r={r:.2f}$")
# plt.savefig("../figures/figB.02.png")
plt.show()


# Figures B.03, B.04, B.05, B.06, B.07, B.08, B.09
for i, T, U in zip([3, 4, 5, 6, 7, 8, 9], T_list, Umat_list):
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, U[::-1])
    plt.colorbar()
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(rf"Figure B.0{i} - Solution at time $t={T}, N={N}$")
    # plt.savefig(f"../figures/figB.0{i}.png")
    plt.show()


# Figure B.10 - Grid Independence
N_list = [101, 201, 301, 401]
L = 2
delta = 0.2
h = 1/(N-1)
optim_r = 1/np.sqrt(2)
k = optim_r*h
r = k/h
q = 1
T = 1.4
x_plane_list = []
y_plane_list = []
Umat_list = []

plt.figure()
for N in N_list:
    WAVE = waves.WaveEquation2D(N, L)
    _, X, Y, _ = WAVE.initial_conditions(delta)
    _, Umat = WAVE.numerical_scheme(k, T, delta, q)
    plt.plot(X[int((N+1)/2), :], Umat[int((N+1)/2), :], label=rf"$N={N}$")

plt.grid()
plt.legend()
plt.xlabel("X")
plt.ylabel("U")
plt.title(rf"Figure B.10 - X-Plane for different Grid Coarseness")
# plt.savefig("../figures/figB.10.png")
plt.show()

# Question 2 Figures:
N = 401
L = 2
delta = 0.2
h = 1/(N-1)
optim_r = 1/np.sqrt(2)
k = optim_r*h
r = k/h

q = 0.2

T_list = [1.5, 3.5, 5, 8]
x_plane_list = []
y_plane_list = []
Umat_list = []

WAVE = waves.WaveEquation2D(N, L)
_, X, Y, _ = WAVE.initial_conditions(delta)
for T in T_list:
    _, Umat = WAVE.numerical_scheme(k, T, delta, q)
    Umat_list.append(Umat)
    x_plane_list.append(Umat[int((N+1)/2), :])
    y_plane_list.append(Umat[:, int((N+1)/2)])


# Figures B.11, B.12, B.13, B.14
for i, T, U in zip([11, 12, 13, 14], T_list, Umat_list):
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, U[::-1])
    plt.colorbar()
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(rf"Figure B.{i} - Solution at time $t={T}, N={N}, q={q}$")
    plt.savefig(f"../figures/figB.{i}.png")
    plt.show()


q = 0.5
T_list = [1.5, 3.5, 5]
x_plane_list = []
y_plane_list = []
Umat_list = []

WAVE = waves.WaveEquation2D(N, L)
_, X, Y, _ = WAVE.initial_conditions(delta)
for T in T_list:
    _, Umat = WAVE.numerical_scheme(k, T, delta, q)
    Umat_list.append(Umat)
    x_plane_list.append(Umat[int((N+1)/2), :])
    y_plane_list.append(Umat[:, int((N+1)/2)])


# Figures B.15, B.16, B.17
for i, T, U in zip([15, 16, 17], T_list, Umat_list):
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, U[::-1])
    plt.colorbar()
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(rf"Figure B.{i} - Solution at time $t={T}, N={N}, q={q}$")
    plt.savefig(f"../figures/figB.{i}.png")
    plt.show()

q = 2
T_list = [1.2, 1.8, 3]
x_plane_list = []
y_plane_list = []
Umat_list = []

WAVE = waves.WaveEquation2D(N, L)
_, X, Y, _ = WAVE.initial_conditions(delta)
for T in T_list:
    _, Umat = WAVE.numerical_scheme(k, T, delta, q)
    Umat_list.append(Umat)
    x_plane_list.append(Umat[int((N+1)/2), :])
    y_plane_list.append(Umat[:, int((N+1)/2)])

# Figures B.18, B.19, B.20
for i, T, U in zip([18, 19, 20], T_list, Umat_list):
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, U[::-1])
    plt.colorbar()
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(rf"Figure B.{i} - Solution at time $t={T}, N={N}, q={q}$")
    plt.savefig(f"../figures/figB.{i}.png")
    plt.show()
