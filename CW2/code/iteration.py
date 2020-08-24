""" iteration.py - Script used for Grid Generation and Iterative Methods.
Tudor Trita Trita
CID: 01199397
MSci Mathematics

M4N10 - Computational Partial Differential Equations.
"""
import time
import numpy as np


class Grid:
    """Class for performing Jacobi, Gauss-Seidel and Successive-Over-Relaxation Iterations."""
    def __init__(self, U_init, f, q, s, r):
        """Contains grid points and methods for solving Poisson's equation.

        Parameters
        ----------
        U_init : np.ndarray
            Initial state for the grid (N, L).
        q : float
            Size of domain to the left of line y = 0.
        s : float
            Size of domain to the right of line y = 0.
        r : float
            Size of domain on top of line x = 0.
        """
        # Initialising class variables:
        self.U_init = U_init
        self.f_init = f
        self.q = q
        self.s = s
        self.r = r
        self.N = U_init.shape[0]  # Size in the x-direction
        self.L = U_init.shape[1]  # Size in the y-direction

        # Precomputing main variables used in methods:
        self.h = (self.q + self.s) / (self.L - 1)  # Step-size in x-direction
        self.k = self.r / (self.N - 1)  # Step-size in y-direction
        self.h2 = self.h**2
        self.k2 = self.k**2
        self.invh = 1/self.h
        self.invk = 1/self.k
        self.invh2 = 1/self.h2
        self.invk2 = 1/self.k2

        # Precomputing extra variables to aid faster computation:
        self.alpha = 2*(self.invh2 + self.invk2)  # Constant on Division
        self.invalpha = 1 / self.alpha
        self.cx = 1 / (self.h2*self.alpha)  # Constant for x-direction
        self.cy = 1 / (self.k2*self.alpha)  # Constant for y-direction

        # Initialising Larger Grid for Ghost Points:
        self.U_init_larger = np.zeros((self.N+2, self.L+2), dtype=float)
        self.U_init_larger[1:-1, 1:-1] = self.U_init.copy()
        self.f_init_larger = np.zeros((self.N+2, self.L+2), dtype=float)
        self.f_init_larger[1:-1, 1:-1] = self.f_init.copy()

    def run_iteration(self, iterative_method, Niter, omega=None,
                      tau=0.05, res_type="Linfinity"):
        """Driver method for launching iterative schemes."""
        t1 = time.perf_counter()  # Timing Iterations

        if iterative_method == "Jacobi":
            U_out = self.jacobi(self.U_init_larger.copy(), self.f_init_larger,
                                tau, Niter)
        elif iterative_method == "GS":
            U_out = self.GS(self.U_init_larger.copy(), self.f_init_larger,
                            tau, Niter)
        elif iterative_method in ["SOR1", "SOR2"]:
            U_out = self.SOR(self.U_init_larger.copy(), self.f_init_larger,
                             omega, tau, Niter, iterative_method)

        # Calculating Time Taken:
        t2 = time.perf_counter()
        time_diff = t2 - t1  # in seconds

        # Calculating Residual Error:
        residual_score = self.compute_residual(U_out, self.f_init_larger,
                                               iterative_method,
                                               tau=tau, omega=omega,
                                               res_type=res_type)

        # Calculate u, v (du_dx, du_dy) on the range:
        u = self.u_quantity(U_out)
        v = self.v_quantity(U_out)
        return U_out[1:-1, 1:-1], u[1:-1, 1:-1], v[1:-1, 1:-1], residual_score, time_diff

    def compute_residual(self, U, f, iterative_method, tau, omega, res_type):
        """Computes the residual by taking one more iteration."""
        if iterative_method == "Jacobi":
            U_after = self.jacobi(U.copy(), f, tau, 1)
        elif iterative_method == "GS":
            U_after = self.GS(U.copy(), f, tau, 1)
        elif iterative_method in ["SOR1", "SOR2"]:
            U_after = self.SOR(U.copy(), f,
                             omega, tau, 1, iterative_method)

        if res_type == "Linfinity":
            residual_score = np.max(np.abs(U_after - U))
        return residual_score

    def boundary_conditions_one(self, U, tau):
        """Applies boundary conditions to the problem in Q1, Q2."""
        # Updating Neutral Neumann Conditions:
        U[0, :] = U[2, :].copy()  # Top Boundary
        U[:, 0] = U[:, 2].copy()  # Left Boundary
        U[:, -1] = U[:, -3].copy()  # Right Boundary

        # Bottom Boundary:
        for i in range(1, self.L+1):
            xval = - self.q + (i - 1) * self.h
            if (xval < 0) or (xval > 1):
                U[-1, i] = U[-3, i].copy()  # Outside Airfoil

            else:
                # On Airfoil:
                dyb_dx = 2*tau*(1 - 2*xval)
                dU_dx = (U[-2, i+1] - U[-2, i-1])*0.5*self.invh
                U[-1, i] = U[-3, i].copy() - 2*self.k*(1 + dU_dx)*dyb_dx
        return U

    def boundary_conditions_two(self, U, tau):
        """Applies boundary conditions to the problem in Q3."""
        # Updating Neutral Neumann Conditions:
        U[0, :] = U[2, :].copy()  # Top Boundary
        U[:, 0] = U[:, 2].copy()  # Left Boundary
        U[:, -1] = U[:, -3].copy()  # Right Boundary

        # Bottom Boundary:
        for i in range(1, self.L+1):
            xval = - self.q + (i - 1) * self.h
            if (xval < 0) or (xval > 1):
                U[-1, i] = U[-3, i].copy()
            else:
                # On Airfoil:
                c0 = (1 - 2*xval)
                alpha2 = 1/(1 + 4*(tau**2)*(c0**2))
                c1 = 4*tau*self.k*c0 * alpha2
                c2 = self.k*self.invh*2*tau*c0 * alpha2
                U[-1, i] = U[-3, i] - c1 - c2*(U[-2, i+1] - U[-2, i-1])
        return U

    def jacobi(self, U, f, tau, Niter):
        """Performs Niter iterations of Jacobi Iteration on the Poisson Equation."""
        # Applying boundary conditions:
        U = self.boundary_conditions_one(U, tau)
        U_iter = U.copy()

        # Jacobi Iteration:
        for iter in range(Niter):
            for j in range(self.N, 0, -1):
                for i in range(1, self.L + 1):
                    U_iter[j, i] = (self.cx*(U[j, i+1] + U[j, i-1])
                                  + self.cy*(U[j+1, i] + U[j-1, i])
                                  - f[j, i]*self.invalpha)

            # Apply boundary conditions:
            U_iter = self.boundary_conditions_one(U_iter, tau)

            # Transfering U_iter to U:
            U = U_iter.copy()
        return U

    def GS(self, U, f, tau, Niter):
        """Performs Niter iterations of GS Iteration on the Poisson Equation."""
        # Apply Boundary Conditions:
        U = self.boundary_conditions_one(U, tau)

        # Gauss-Seidel Iteration:
        for iter in range(Niter):
            for j in range(self.N, 0, -1):
                for i in range(1, self.L + 1):
                    U[j, i] = (self.cx*(U[j, i+1] + U[j, i-1])
                             + self.cy*(U[j+1, i] + U[j-1, i])
                             - f[j, i]*self.invalpha)

            # Apply boundary conditions:
            U = self.boundary_conditions_one(U, tau)
        return U

    def SOR(self, U, f, omega, tau, Niter, iterative_method):
        """Performs Niter iterations of SOR Iteration on the Poisson Equation."""
        if iterative_method == "SOR1":
            # Apply Boundary Conditions:
            U = self.boundary_conditions_one(U, tau)

            # SOR Iteration 1:
            for iter in range(Niter):
                for j in range(self.N, 0, -1):
                    for i in range(1, self.L+1):
                        Utemp = (self.cx*(U[j, i+1] + U[j, i-1])
                               + self.cy*(U[j+1, i] + U[j-1, i])
                               - f[j, i]*self.invalpha)
                        U[j, i] = (1-omega)*U[j, i] + omega*Utemp
                # Apply Boundary Conditions:
                U = self.boundary_conditions_one(U, tau)

        elif iterative_method == "SOR2":
            # Apply Boundary Conditions:
            U = self.boundary_conditions_two(U, tau)

            # SOR Iteration 2:
            for iter in range(Niter):
                for j in range(self.N, 0, -1):
                    for i in range(1, self.L+1):
                        xval = - self.q + (i - 1) * self.h
                        if (xval < 0) or (xval > 1):
                            Utemp = (self.cx*(U[j, i+1] + U[j, i-1])
                                   + self.cy*(U[j+1, i] + U[j-1, i])
                                   - f[j, i]*self.invalpha)
                        else:
                            c0 = tau*(1 - 2*xval)
                            c1 = c0**2
                            c2 = 1 + 4*tau*c1
                            alpha2 = 1/(2*(self.invh2 + self.invk2*c2))
                            Utemp = (self.invh2*(U[j, i+1] + U[j, i-1])
                                   + self.invk2*(U[j+1, i] + U[j-1, i])*c2
                                   + 2*tau*self.invk*(U[j-1, i] - U[j+1, i])
                                   - tau*c0*self.invh*self.invk*(
                                   U[j-1, i+1] - U[j+1, i+1] - U[j-1, i-1] + U[j+1, i-1]
                                   ) - f[j, i])*alpha2
                        U[j, i] = (1-omega)*U[j, i] + omega*Utemp
                # Apply Boundary Conditions:
                U = self.boundary_conditions_two(U, tau)
        return U

    def u_quantity(self, U):
        """Calculates u=du_dx on the whole domain."""
        u = np.zeros((self.N+2, self.L+2))
        invh2 = 0.5*self.invh
        # Centred Finite Differences:
        for j in range(1, self.N+1):
            for i in range(1, self.L+1):
                u[j, i] = invh2*(U[j, i+1] - U[j, i-1])
        return u[1:-1, 1:-1]

    def v_quantity(self, U):
        v = np.zeros((self.N+2, self.L+2))
        invk2 = 0.5*self.invk

        # Centred Finite Differences:
        for j in range(1, self.N):
            for i in range(1, self.L+1):
                v[j, i] = invk2*(U[j-1, i] - U[j+1, i])

        # Forward Finite Difference:
        for i in range(1, self.L+1):
            v[-2, i] = invk2*(-3*U[-2, i] + 4*U[-3, i] - U[-4, i])
        return v[1:-1, 1:-1]
