""" waves.py - Script containing classes that compute the appropriate wave
               equations.
Tudor Trita Trita
CID: 01199397
MSci Mathematics
M4N10 - Computational Partial Differential Equations - Project 3
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse


class WaveEquation1D:
    """Class to compute waves with the parameters in Part A of the CW."""
    def __init__(self, N, L):
        self.N = N
        self.L = L
        self.h = 2*L/(N-1)

    def initial_conditions(self, delta, initial_condition_type):
        """Sets the 1-D Initial Conditions at t=0 cos(pix/2delta)."""
        X = np.linspace(-self.L, self.L, self.N)
        U0 = np.zeros(X.shape, dtype=float)

        if initial_condition_type == 1:
            for i, x in enumerate(X):
                if np.abs(x) <= delta:
                    U0[i] = np.cos(0.5*np.pi*x/delta)
        elif initial_condition_type == 2:
            for i, x in enumerate(X):
                if np.abs(x) <= delta:
                    U0[i] = 1
        return U0, X

    def boundary_conditions(self, A, B, boundary_type):
        """Creates boundary conditions of 2 types: Solid Wall and Minimal
           Reflections at the boundaries.
        """
        if boundary_type == "MINIMAL_REFLECTIONS":
            # Left-moving Wave: BC's at x=-L
            A[0, 0] = 2*(1 - self.r2)/(1 + self.r)
            A[0, 1] = 2*self.r2/(1 + self.r)
            B[0, 0] = (self.r - 1)/(1 + self.r)

            # # Right-moving Wave: BC's at x=+L
            A[-1, -1] = 2*(1 - self.r2)/(1 + self.r)
            A[-1, -2] = 2*self.r2/(1 + self.r)
            B[-1, -1] = (self.r - 1)/(1 + self.r)

        elif boundary_type == "SOLID_WALL":
            A[0, 1] = 2*self.r2
            A[-1, -2] = 2*self.r2
        else:
            raise Exception("Invalid Boundary Type")
        return A, B

    def numerical_scheme(self, k, T, delta, boundary_type, display_times=None,
                         display_final=True, xlims=(-2, 2), ylims=(-0.2, 1.2),
                         initial_condition_type=1):
        """Solves the Wave equation numerically using the Leapfrog scheme."""
        T_range = np.linspace(0, T, int(T/k) + 1)
        self.r = k/self.h
        self.r2 = self.r**2

        U, X = self.initial_conditions(delta, initial_condition_type)
        U1 = U.copy()  # k^th Time-Step
        U2 = U.copy()  # (k-1)^th Time-Step

        A = scipy.sparse.diags((self.r2, 2*(1-self.r2), self.r2),
                               (-1, 0, 1), shape=(self.N, self.N), format="csr")
        B = scipy.sparse.diags((-1,), (0,), shape=(self.N, self.N), format="csr")

        # Apply Initial Boundary Conditions and do first time-step
        A2 = A.copy()
        A2 *= 0.5
        if boundary_type == "MINIMAL_REFLECTIONS":
            # Left-moving Wave: BC's at x=-L
            A2[0, 0] = 1 - self.r2
            A2[0, 1] = self.r2

            # Right-moving Wave: BC's at x=+L
            A2[-1, -1] = 1 - self.r2
            A2[-1, -2] = self.r2

        elif boundary_type == "SOLID_WALL":
            A2[0, 1] = self.r2
            A2[-1, -2] = self.r2

        U = A2 @ U1
        U1 = U.copy()

        # Apply main boundary Conditions:
        A, B = self.boundary_conditions(A, B, boundary_type)

        # Main Loop:
        if (display_times is not None) or display_final:
            fig = plt.figure(figsize=(11, 6))
            ax = plt.axes()

        # Main loop - rest of times:
        for i, t in enumerate(T_range[2:]):
            U = A @ U1 + B @ U2
            U2 = U1.copy()
            U1 = U.copy()

            # Plot selected times if selected:
            try:
                if t in display_times:
                    fig, ax = self.plot_solution(U, t, X, fig=fig, ax=ax,
                                                 xlims=xlims, ylims=ylims)
            except:
                pass

        if (display_times is not None) or display_final:
            fig, ax = self.plot_solution(U, T, X, final=True, fig=fig, ax=ax,
                                         xlims=xlims, ylims=ylims)
        return U, X

    def plot_solution(self, U, t, X, final=False, fig=None, ax=None,
                      xlims=None, ylims=None):
        """Plots the of solution at different times."""
        ax.plot(X, U, label=rf"Solution $t={t}$")

        if final:
            if xlims is not None:
                plt.xlim(xlims)
            if ylims is not None:
                plt.ylim(ylims)
            plt.legend()
            plt.grid()
            plt.xlabel(r"$X$")
            plt.ylabel(r"$U$")
        return fig, ax


class WaveEquation2D:
    """Class to compute waves with the parameters in Part B of the CW."""
    def __init__(self, N, L):
        self.N = N
        self.N2 = self.N**2
        self.L = L
        self.h = 4/(N-1)
        self.h2 = self.h**2

    def initial_conditions(self, delta):
        """Sets the 2-D Initial Conditions at t=0 cos(pi r / 2 delta)."""
        XY = np.linspace(-2, 2, self.N)

        X, Y = np.meshgrid(XY, XY)
        R = np.sqrt(X**2 + Y**2)
        U0 = np.cos((np.pi*R)/(2*delta))

        for i, x in enumerate(XY):
            for j, y in enumerate(XY[::-1]):
                r = np.sqrt(x**2 + y**2)
                if np.abs(r) <= delta:
                    pass
                else:
                    U0[j, i] = 0
        return U0, X, Y, R

    def boundary_conditions(self, U, U1, U2, q):
        """BC's for 2D Wave Equation."""
        r = self.r
        alpha1 = r/(2/np.sqrt(q) + r)
        alpha2 = r/(2 + r)

        # Discretisations in x-directions first:
        for i in range(1, self.N-1):
            # Top Boundary:
            i2 = self.N + i
            U[i] = alpha1*(2/(r*np.sqrt(q))*(2*U1[i] - U2[i])
                           + r/np.sqrt(q)*(U1[i+1] - 2*U1[i] + U1[i-1])
                           + U2[i] - U2[i2] + U[i2])

            # Bottom Boundary:
            i2 = self.N*(self.N-1) + i
            i3 = self.N*(self.N-2) + i
            U[i2] = alpha1*(2/(r*np.sqrt(q))*(2*U1[i2] - U2[i2])
                           + r/np.sqrt(q)*(U1[i2+1] - 2*U1[i2] + U1[i2-1])
                           + U2[i2] - U2[i3] + U[i3])

        # Discretisations in y-directions next:
        for j in range(1, self.N-1):
            jmin1 = self.N*(j-1)
            j0 = self.N*j
            j1 = self.N*(j+1)

            # Left Boundary:
            U[j0] = alpha2*(2/r*(2*U1[j0] - U2[j0])
                           + q*r*(U1[j1] - 2*U1[j0] + U1[jmin1])
                           + U2[j0] - U2[j0+1] + U[j0+1])

            jmin1 = self.N*j - 1
            j0 = self.N*(j+1) - 1
            j1 = self.N*(j+2) - 1

            # Right Boundary:
            U[j0] = alpha2*(2/r*(2*U1[j0] - U2[j0])
                           + q*r*(U1[j1] - 2*U1[j0] + U1[jmin1])
                           + U2[j0] - U2[j0-1] + U[j0-1])

        # Top-Left Ghost Point: (Forward-Difference in X):
        U[0] = alpha1*(2/(r*np.sqrt(q))*(2*U1[0] - U2[0])
                    + r/np.sqrt(q)*(2*U1[0] - 5*U1[1] + 4*U1[2] - U1[3])
                    + U2[0] - U2[self.N] + U[self.N])

        # Bottom-Left Ghost Point: (Backward-Difference in X):
        i0 = self.N*(self.N-1)
        i1 = self.N*(self.N-2)
        U[i0] = alpha1*(2/(r*np.sqrt(q))*(2*U1[i0] - U2[i0])
                    + r/np.sqrt(q)*(2*U1[i0] - 5*U1[i0+1] + 4*U1[i0+2] - U1[i0+3])
                    + U2[i0] - U2[i1] + U[i1])

        # Top-Right Ghost Point: (Forward-Difference in X)
        i0 = self.N - 1
        i1 = 2*self.N - 1
        U[i0] = alpha1*(2/(r*np.sqrt(q))*(2*U1[i0] - U2[i0])
                    + r/np.sqrt(q)*(2*U1[i0] - 5*U1[i0-1] + 4*U1[i0-2] - U1[i0-3])
                    + U2[i0] - U2[i1] + U[i1])

        # Bottom-Right Ghost Point: (Backrward-Difference in X)
        i0 = self.N2 - 1
        i1 = self.N*(self.N-1) - 1
        U[i0] = alpha1*(2/(r*np.sqrt(q))*(2*U1[i0] - U2[i0])
                    + r/np.sqrt(q)*(2*U1[i0] - 5*U1[i0-1] + 4*U1[i0-2] - U1[i0-3])
                    + U2[i0] - U2[i1] + U[i1])
        return U

    def numerical_scheme(self, k, T, delta, q):
        """Solves the Wave equation numerically using the Leapfrog scheme."""
        k2 = k**2
        self.r = k/self.h
        self.r2 = self.r**2

        T_range = np.linspace(0, T, int(T/k) + 1)
        U0MAT, X, Y, R = self.initial_conditions(delta)
        U0VEC = U0MAT.flatten()  # Transforms into a long N*M vector

        main_diagonal = 2*(1 - self.r2*(1 + q))*np.ones(self.N2)
        first_diagonals = self.r2 * np.ones(self.N2 - 1)
        first_diagonals[self.N-1::self.N] = 0
        nth_diagonals = q*self.r2*np.ones(self.N2 - self.N)

        A = scipy.sparse.diags((nth_diagonals, first_diagonals, main_diagonal,
                                first_diagonals, nth_diagonals),
                               (-self.N, -1, 0, 1, self.N),
                               shape=(self.N2, self.N2),
                               format="csr")

        U = U0VEC.copy()
        U1 = U.copy()
        U2 = U.copy()

        # First Time-Step:
        U =  0.5 * (A @ U1)  # Using neutral derivative initial conditions.
        U1 = U.copy()

        for t in T_range[2:]:
            U = A @ U1 - U2
            U = self.boundary_conditions(U, U1, U2, q)  # Apply open B.C's
            U2 = U1.copy()
            U1 = U.copy()

        Umat = U.reshape(U0MAT.shape)
        return U, Umat
