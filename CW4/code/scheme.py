""" iteration.py - Script used for Grid Generation and Iterative Methods.
Tudor Trita Trita
CID: 01199397
MSci Mathematics
M4N10 - Computational Partial Differential Equations.
"""
global IMPORTERROR
import time
import numpy as np

try:
    from os import chdir
    chdir("code")
except:
    pass

try:
    import mod
    IMPORT_OK = True
except ModuleNotFoundError:
    IMPORT_OK = False
    print("ERROR Importing Fortran Module - check if has been compiled with f2py?")
    exit_bool = input("Press 1 to exit or 2 to proceed running Python code (SLOW!)").strip()
    if exit_bool == "1":
        import sys
        sys.exit()
    elif exit_bool == "2":
        print("Proceeding with Python!")


class Grid:
    def __init__(self, U_init, q, s, r):
        """Contains grid points and methods for solving Poisson's equation."""
        # Initialising class variables:
        self.U_init = U_init
        self.q = q
        self.s = s
        self.r = r
        self.N = U_init.shape[0]  # Size in the x-direction
        self.L = U_init.shape[1]  # Size in the y-direction

        # Precomputing main variables used in methods:
        self.dx = (self.q + self.s) / (self.L - 1)  # Step-size in x-direction
        self.dy = self.r / (self.N - 1)  # Step-size in y-direction
        self.dx2 = self.dx**2
        self.dy2 = self.dy**2
        self.invdx = 1/self.dx
        self.invdy = 1/self.dy
        self.invdx2 = 1/self.dx2
        self.invdy2 = 1/self.dy2

        # Initialising Larger Grid for Ghost Points:
        self.U_init_larger = np.zeros((self.N+2, self.L+2), dtype=float)
        self.U_init_larger[1:-1, 1:-1] = self.U_init.copy()

    def run_iteration(self, rho_init, Niter, M, tau=0.05, gamma=1.4, language="FORTRAN"):
        """Driver method for launching iterative schemes."""
        t1 = time.perf_counter()
        U_out, rho_out, u, v = self.iteration(self.U_init_larger.copy(), rho_init.copy(),
                                       tau, Niter, M, gamma=gamma, language=language)
        t2 = time.perf_counter()
        tdiff = t2 - t1
        res1, res2 = self.compute_residuals(U_out, rho_out, tau, M, gamma=gamma, language=language)
        return (U_out[1:-1, 1:-1], rho_out[1:-1, 1:-1], u[1:-1, 1:-1], v[1:-1, 1:-1],
               res1, res2, tdiff)

    def check_convergence(self, M, rho, tol1=1e-08, tol2=1e-08, tau=0.05, gamma=1.4,
                          iter_interval=10, max_iters=100000,
                          language="FORTRAN"):
        """Checks number of iterations needed until convergence."""
        current_iter = 0
        iters_list = []
        res_list1 = [0.1]
        res_list2 = [0.1]

        U = self.U_init_larger.copy()  # Initial Guess
        for i in range(int(max_iters/iter_interval)):
            U, rho, u, v = self.iteration(U.copy(), rho.copy(), tau,
                                          iter_interval, M, gamma=gamma, language=language)
            current_iter += iter_interval
            iters_list.append(current_iter)

            res1, res2 = self.compute_residuals(U, rho, tau, M, gamma=gamma, language=language)
            res_list1.append(res1)
            res_list2.append(res2)

            if (res1 < tol1) or (res2 < tol2):
                 # Check if meet tolerance or not converging anymore
                break
        return (U[1:-1, 1:-1], rho[1:-1, 1:-1], u[1:-1, 1:-1], v[1:-1, 1:-1],
                res_list1, res_list2, current_iter)

    def compute_residuals(self, U, rho, tau, M, gamma=1.4, language="FORTRAN"):
        """Computes the residual by taking one more iteration."""
        U_after, rho_after, _, _ = self.iteration(U.copy(), rho.copy(), tau, 1, M, gamma=gamma, language=language)
        res1 = np.max(np.abs(U_after - U))
        res2 = np.max(np.abs(rho_after - rho))
        return res1, res2

    def boundary_conditions(self, U, tau):
        """Applies boundary conditions to the problem in Q1, Q2."""
        # Updating Neutral Neumann Conditions:
        U[0, :] = U[2, :].copy()  # Top Boundary
        U[:, 0] = U[:, 2].copy()  # Left Boundary
        U[:, -1] = U[:, -3].copy()  # Right Boundary
        # Bottom Boundary:
        for i in range(1, self.L+1):
            xval = - self.q + (i - 1) * self.dx
            if (xval < 0) or (xval > 1):
                U[-1, i] = U[-3, i].copy()  # Outside Airfoil
            else:
                # On Airfoil:
                dyb_dx = 2*tau*(1 - 2*xval)
                dU_dx = (U[-2, i+1] - U[-2, i-1])*0.5*self.invdx
                U[-1, i] = U[-3, i] - 2*self.dy*(1 + dU_dx)*dyb_dx
        return U

    def u_quantity(self, U):
        """Calculates u=du_dx on the whole domain."""
        u = np.zeros((self.N+2, self.L+2), dtype=float)

        # Centred Finite Differences:
        for j in range(1, self.N+1):
            for i in range(1, self.L+1):
                u[j, i] = 0.5*self.invdx*(U[j, i+1] - U[j, i-1])
        return u

    def v_quantity(self, U):
        """Calculates v=du_dy on the whole domain."""
        v = np.zeros((self.N+2, self.L+2), dtype=float)

        # Centred Finite Differences:
        for j in range(1, self.N):
            for i in range(1, self.L+1):
                v[j, i] = 0.5*self.invdy*(U[j-1, i] - U[j+1, i])

        # Forward Finite Difference:
        for i in range(1, self.L+1):
            v[-2, i] = 0.5*self.invdy*(-3*U[-2, i] + 4*U[-3, i] - U[-4, i])
        return v

    def rho_derivatives(self, r):
        """Calculates derivatives of rho."""
        dr_dx = np.zeros((self.N+2, self.L+2), dtype=float)
        dr_dy = np.zeros((self.N+2, self.L+2), dtype=float)

        # Centred Finite Differences for Interior:
        for j in range(1, self.N+1):
            for i in range(1, self.L+1):
                dr_dx[j, i] = 0.5*self.invdx*(r[j, i+1] - r[j, i-1])
                dr_dy[j, i] = 0.5*self.invdy*(r[j-1, i] - r[j+1, i])

        # Left and Right Boundaries:
        for j in range(1, self.N+1):
            dr_dx[j, 1] = 0.5*self.invdx*(- 3*r[j, 1] + 4*r[j, 2] - r[j, 3])
            dr_dx[j, -2] = 0.5*self.invdx*(3*r[j, -2] - 4*r[j, -3] + r[j, -4])

        # Top and Bottom Boundaries:
        for i in range(1, self.L+1):
            dr_dy[1, i] = 0.5*self.invdy*(3*r[1, i] - 4*r[2, i] + r[3, i])
            dr_dy[-2, i] = 0.5*self.invdy*(- 3*r[-2, i] + 4*r[-3, i] - r[-4, i])
        return dr_dx, dr_dy

    def rho_hat(self, u, v, M, gamma=1.4):
        """Computes rho hat as in the question."""
        return (1 - (0.5*(gamma-1)*M**2*(2*u + u**2 + v**2)))**(1/(gamma-1))

    def iteration(self, U, rho, tau, Niter, M, gamma=1.4, language="FORTRAN"):
        """Performs Niter iterations of GS Iteration on the modified equation."""
        if language=="FORTRAN" and IMPORT_OK==True:
            rho, phi, u, v,  = mod.iteration.scheme(
                    Niter, tau, M, gamma, self.q, self.s, self.r, rho, U)
            return np.array(phi), np.array(rho), np.array(u), np.array(v)
        else:
            rho_hat = 1 + rho
            a = (gamma-1)/2 * M**2
            b = 1/(gamma-1)
            u = self.u_quantity(U)
            v = self.v_quantity(U)

            # Iterations:
            for iter in range(Niter):
                # Compute rho derivatives for iteration:
                dr_dx, dr_dy = self.rho_derivatives(rho_hat)
                alpha = 2*rho_hat*(self.invdx2 + self.invdy2)
                f = dr_dx*(1 + u) + dr_dy*v
                c1 = rho_hat*self.invdx2
                c2 = rho_hat*self.invdy2

                # Gauss-Seidel:
                for j in range(self.N, 0, -1):
                    for i in range(1, self.L + 1):
                        U[j, i] = (c1[j, i]*(U[j, i+1] + U[j, i-1])
                                 + c2[j, i]*(U[j-1, i] + U[j+1, i])
                                 + f[j, i]) / alpha[j, i]

                # Apply boundary conditions:
                U = self.boundary_conditions(U, tau)
                u = self.u_quantity(U)
                v = self.v_quantity(U)

                rho_hat = (1 - a*(2*u + u**2 + v**2))**b
            rho = rho_hat - 1
            return U, rho, u, v
