""" partA.py - Script containing code used for computing results and plots in
                Part A of the Project.
Tudor Trita Trita
CID: 01199397
MSci Mathematics
M4N10 - Computational Partial Differential Equations - Project 3
"""
import math

import matplotlib
matplotlib.rc('figure', figsize=(11, 6))
import matplotlib.pyplot as plt

import numpy as np

import waves

# Figure A.00 (Initial Condition)
WAVE0 = waves.WaveEquation1D(1001, 2)
U01, X = WAVE0.initial_conditions(0.5, 1)
U02, X = WAVE0.initial_conditions(0.1, 1)
U03, X = WAVE0.initial_conditions(0.01, 1)

plt.figure()
plt.plot(X, U01, "b", label=r"$\delta=0.5$")
plt.plot(X, U02, "k--", label=r"$\delta=0.1$")
plt.plot(X, U03, "r", label=r"$\delta=0.01$")
plt.legend()
plt.xlabel("X")
plt.ylabel("U")
plt.grid()
plt.title("Figure A.00 - Initial Conditions as in the Question")
plt.savefig("../figures/figA.00.png")
plt.show()


# Figure A.01, A.02, A.03 (Wave satisfying Solid Wall Condition):
T = 4
L = 2
N = 10001
h = 2*L/(N-1)
k = 0.0001
r = k/h
L = 2
times_list = [0.1, 0.8, 2.2]

WAVE = waves.WaveEquation1D(N, L)

delta = 0.5
U1, _ = WAVE.numerical_scheme(k, T, delta, "SOLID_WALL",
                              display_times=times_list)
plt.title(rf"Figure A.01 - Waves (Solid Wall), $r={r}, \delta={delta}, N={N}$")
# plt.savefig("../figures/figA.01.png")
plt.show()

delta = 0.1
U2, _ = WAVE.numerical_scheme(k, T, delta, "SOLID_WALL",
                              display_times=times_list)
plt.title(rf"Figure A.02 - Waves (Solid Wall), $r={r}, \delta={delta}, N={N}$")
# plt.savefig("../figures/figA.02.png")
plt.show()

delta = 0.01
U3, _ = WAVE.numerical_scheme(k, T, delta, "SOLID_WALL",
                              display_times=times_list)
plt.title(rf"Figure A.03 - Waves (Solid Wall), $r={r}, \delta={delta}, N={N}$")
# plt.savefig("../figures/figA.03.png")
plt.show()

# Figures A.04, A.05, A.06 (Wave Satisfying Minimal Reflections)
T = 3
L = 2
N = 10001
h = 2*L/(N-1)
k = 0.0001
r = k/h
L = 2
times_list = [0.1, 0.8, 2.2]

delta = 0.5
U4, _ = WAVE.numerical_scheme(k, T, delta, "MINIMAL_REFLECTIONS",
                              display_times=times_list)
plt.title(rf"Figure A.04 - Waves (Minimal Reflections), $r={r}, \delta={delta}, N={N}$")
# plt.savefig("../figures/figA.04.png")
plt.show()

delta = 0.1
U5, _ = WAVE.numerical_scheme(k, T, delta, "MINIMAL_REFLECTIONS",
                              display_times=times_list)
plt.title(rf"Figure A.05 - Waves (Minimal Reflections), $r={r}, \delta={delta}, N={N}$")
# plt.savefig("../figures/figA.05.png")
plt.show()

delta = 0.01
U6, _ = WAVE.numerical_scheme(k, T, delta, "MINIMAL_REFLECTIONS",
                              display_times=times_list)
plt.title(rf"Figure A.06 - Waves (Minimal Reflections), $r={r}, \delta={delta}, N={N}$")
# plt.savefig("../figures/figA.06.png")
plt.show()

# Figure A.07 (Grid Independence)
L = 2
Nlist = [2001, 4001, 6001]
delta = 0.1
times_list = [0.3, 1.3]

plt.figure()
for N in Nlist:
    h = 2*L/(N-1)
    k = h
    WAVE = waves.WaveEquation1D(N, L)
    for c, T in zip(["-", "--"], times_list):
        U, X = WAVE.numerical_scheme(k, T, delta, "SOLID_WALL", display_final=False)
        plt.plot(X, U, c, label=rf"$N={N}, t={T}$")

plt.xlim((-2, 2))
plt.ylim((-0.2, 1.2))
plt.legend()
plt.grid()
plt.xlabel(r"$X$")
plt.ylabel(r"$U$")
plt.title("Figure A.07 - Waves Showing Grid Independence")
# plt.savefig("../figures/figA.07.png")
plt.show()


# Figure A.08, A.09, A.10 - Showing Dispersion:
T = 1.4
L = 2
N = 10001
h = 2*L/(N-1)
k1 = h
k2 = h/2
k3 = h/4
r1 = k1/h
r2 = k2/h
r3 = k3/h
delta = 0.01

WAVE = waves.WaveEquation1D(N, L)

X = np.linspace(-L, L, N)
U1, _ = WAVE.numerical_scheme(k1, T, delta, "SOLID_WALL", display_final=False)
U2, _ = WAVE.numerical_scheme(k2, T, delta, "SOLID_WALL", display_final=False)
U3, _ = WAVE.numerical_scheme(k3, T, delta, "SOLID_WALL", display_final=False)

plt.figure()
plt.plot(X, U1, "k", label=rf"$r={r1}$")
plt.plot(X, U2, "b--", label=rf"$r={r2}$")
plt.plot(X, U3, "r--", label=rf"$r={r3}$")
plt.grid()
plt.legend()
plt.xlim(-1.45, -1.3)
plt.xlabel("X")
plt.ylabel("U")
plt.title(f"Figure A.08 - Dissipation and Dispersion Effects (N={N})")
# plt.savefig("../figures/figA.08.png")
plt.show()

T = 1.4
L = 2
N = 5001
h = 2*L/(N-1)
k1 = h
k2 = h/2
k3 = h/4
r1 = k1/h
r2 = k2/h
r3 = k3/h
delta = 0.01

WAVE = waves.WaveEquation1D(N, L)

X = np.linspace(-L, L, N)
U1, _ = WAVE.numerical_scheme(k1, T, delta, "SOLID_WALL", display_final=False)
U2, _ = WAVE.numerical_scheme(k2, T, delta, "SOLID_WALL", display_final=False)
U3, _ = WAVE.numerical_scheme(k3, T, delta, "SOLID_WALL", display_final=False)

plt.figure()
plt.plot(X, U1, "k", label=rf"$r={r1}$")
plt.plot(X, U2, "b--", label=rf"$r={r2}$")
plt.plot(X, U3, "r--", label=rf"$r={r3}$")
plt.grid()
plt.legend()
plt.xlim(-1.45, -1.3)
plt.xlabel("X")
plt.ylabel("U")
plt.title(f"Figure A.09 - Dissipation and Dispersion Effects (N={N})")
# plt.savefig("../figures/figA.09.png")
plt.show()

T = 1.4
L = 2
N = 2001
h = 2*L/(N-1)
k1 = h
k2 = h/2
k3 = h/4
r1 = k1/h
r2 = k2/h
r3 = k3/h
delta = 0.01

WAVE = waves.WaveEquation1D(N, L)

X = np.linspace(-L, L, N)
U1, _ = WAVE.numerical_scheme(k1, T, delta, "SOLID_WALL", display_final=False)
U2, _ = WAVE.numerical_scheme(k2, T, delta, "SOLID_WALL", display_final=False)
U3, _ = WAVE.numerical_scheme(k3, T, delta, "SOLID_WALL", display_final=False)

plt.figure()
plt.plot(X, U1, "k", label=rf"$r={r1}$")
plt.plot(X, U2, "b--", label=rf"$r={r2}$")
plt.plot(X, U3, "r--", label=rf"$r={r3}$")
plt.grid()
plt.legend()
plt.xlim(-1.45, -1.3)
plt.xlabel("X")
plt.ylabel("U")
plt.title(f"Figure A.10 - Dissipation and Dispersion Effects (N={N})")
# plt.savefig("../figures/figA.10.png")
plt.show()

T = 1.4
L = 2
N = 1001
h = 2*L/(N-1)
k1 = h
k2 = h/2
k3 = h/4
r1 = k1/h
r2 = k2/h
r3 = k3/h
delta = 0.01

WAVE = waves.WaveEquation1D(N, L)

X = np.linspace(-L, L, N)
U1, _ = WAVE.numerical_scheme(k1, T, delta, "SOLID_WALL", display_final=False)
U2, _ = WAVE.numerical_scheme(k2, T, delta, "SOLID_WALL", display_final=False)
U3, _ = WAVE.numerical_scheme(k3, T, delta, "SOLID_WALL", display_final=False)

plt.figure()
plt.plot(X, U1, "k", label=rf"$r={r1}$")
plt.plot(X, U2, "b--", label=rf"$r={r2}$")
plt.plot(X, U3, "r--", label=rf"$r={r3}$")
plt.grid()
plt.legend()
plt.xlim(-1.45, -1.3)
plt.xlabel("X")
plt.ylabel("U")
plt.title(f"Figure A.12 - Dissipation and Dispersion Effects (N={N})")
# plt.savefig("../figures/figA.12.png")
plt.show()

# Figure A.11 - Showing dissipation/dispersion on small box:
T = 1.4
L = 2
N = 10001
h = 2*L/(N-1)
k1 = h
k2 = h/2
k3 = h/4
r1 = k1/h
r2 = k2/h
r3 = k3/h
delta = 0.01

WAVE = waves.WaveEquation1D(N, L)

X = np.linspace(-L, L, N)
U1, _ = WAVE.numerical_scheme(k1, T, delta, "SOLID_WALL", initial_condition_type=2,
                              display_final=False)
U2, _ = WAVE.numerical_scheme(k2, T, delta, "SOLID_WALL", initial_condition_type=2,
                              display_final=False)
U3, _ = WAVE.numerical_scheme(k3, T, delta, "SOLID_WALL", initial_condition_type=2,
                              display_final=False)

plt.figure()
plt.plot(X, U1, "k", label=rf"$r={r1}$")
plt.plot(X, U2, "b--", label=rf"$r={r2}$")
plt.plot(X, U3, "r--", label=rf"$r={r3}$")
plt.grid()
plt.legend()
plt.xlim(-1.45, -1.3)
plt.xlabel("X")
plt.ylabel("U")
plt.title(f"Figure A.11 - Dissipation and Dispersion Effects on constant Function (N={N})")
# plt.savefig("../figures/figA.11.png")
plt.show()


# Figure A.12 - Showing Accuracy:
n_array = np.arange(1, 30)
T_array = 4*n_array
L = 2
delta_list = [0.01, 0.1, 0.5]
N = 1001
accuracy_array = np.zeros((len(delta_list), len(T_array)))
h = 4/(N-1)
k = h
r = k/h

WAVE = waves.WaveEquation1D(N, 2)

for j, delta in enumerate(delta_list):
    U0, X = WAVE.initial_conditions(delta, 1)
    for i, T in enumerate(T_array):
        U4, _ = WAVE.numerical_scheme(k, T, delta, "SOLID_WALL", display_final=False)
        try:
            error = math.log10(np.max(np.abs(U4 - U0)))
        except:
            error = -16  # Machine epsilon
        accuracy_array[j, i] = error

plt.figure()
plt.plot(T_array, accuracy_array[0], "b", label=r"$\delta=0.01$")
plt.plot(T_array, accuracy_array[1], "k--", label=r"$\delta=0.1$")
plt.plot(T_array, accuracy_array[2], "r", label=r"$\delta=0.5$")
plt.grid()
plt.legend()
plt.xlabel("Times")
plt.ylabel("Error")
plt.title(r"Figure A.13 - Accuracy at different Times $|u_4 - u_0|$")
# plt.savefig("../figures/figA.13.png")
plt.show()
