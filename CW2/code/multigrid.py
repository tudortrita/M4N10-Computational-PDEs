""" multigrid.py - Script containing MultiGrid code using Gauss-Seidel Iteration.
Tudor Trita Trita
CID: 01199397
MSci Mathematics

M4N10 - Computational Partial Differential Equations.
"""
import math
import time

import numpy as np

class MultiGrid:
    """ Class containing code for multigrid operations using Gauss-Seidel Iteration."""
    def __init__(self, U_init, f, q, s, r, tau):
        """
        Parameters
        ----------
        U_init : np.ndarray
            Initial state for the grid.
        q : float
            Size of domain to the left of line y = 0.
        s : float
            Size of domain to the right of line y = 0.
        r : float
            Size of domain on top of line x = 0.
        """
        self.U_init = U_init
        self.f = f
        self.q = q
        self.s = s
        self.r = r
        self.tau = tau
        self.N = U_init.shape[0]  # Size in the x-direction
        self.L = U_init.shape[1]  # Size in the y-direction

        self.hmaster = (q+s)/(self.L-1)
        self.kmaster = r/(self.N-1)

        # Initialising Larger Grid for Ghost Points:
        self.U_init_larger = np.zeros((self.N+2, self.L+2), dtype=float)
        self.U_init_larger[1:-1, 1:-1] = self.U_init.copy()
        self.f_init_larger = np.zeros((self.N+2, self.L+2), dtype=float)
        self.f_init_larger[1:-1, 1:-1] = self.f.copy()

        # Check these are 2^n + 1
        assert np.isclose(math.log(self.N - 1, 2), int(math.log(self.N - 1, 2)))
        assert np.isclose(math.log(self.L - 1, 2), int(math.log(self.L - 1, 2)))

    def run_vcycle(self, num_grids, Niter, num_cycles, compute_all_errors=True):
        """Driver method which will run num_cycles V-Cycles with Niter GS."""
        self.num_grids = num_grids
        self.Niter = Niter
        error_list = []

        # Run First V-Cycle
        Uout = self.multigrid_vcycle(self.U_init_larger.copy(), self.f_init_larger.copy())
        if compute_all_errors:
            error_list.append(self.compute_error(Uout.copy(), self.f_init_larger.copy()))

        for num in range(num_cycles - 1):
            Uout = self.multigrid_vcycle(Uout.copy(), self.f_init_larger.copy())
            if compute_all_errors and (num % 10 == 0):
                error_list.append(self.compute_error(Uout.copy(), self.f_init_larger.copy()))

        if not compute_all_errors:
            error_list.append(self.compute_error(Uout.copy(), self.f_init_larger.copy()))
        return Uout, np.array(error_list)

    def multigrid_vcycle(self, Uin, f):
        """Multigrid V-Cycle."""
        n, m = f.shape

        # Check if we are at the coarsest level or coarsest possible level (N=3)
        if (((n-3) == (self.N-1) / 2**(self.num_grids-1))
         or ((m-3) == (self.L-1) / 2**(self.num_grids-1))
         or ((n-3) == 2) or ((m-3) == 2)):
            Uout = self.GS(Uin, f, self.Niter)
            return Uout

        # Otherwise begin the cycle from fine to coarsest.

        # Start by smoothing input with Niter GS iterations:
        Usmooth = self.GS(Uin, f, self.Niter)

        # Compute the Residuals:
        res = self.residual(Usmooth, f)

        # Restrict the residual to a coarser grid, half the precision
        reshalf = self.restrict(res)

        # Now solve the error equation A(error) = residual on the next grid
        err = self.multigrid_vcycle(np.zeros(reshalf.shape, dtype=float), reshalf)

        # Now interpolate the course error onto finer grid and add to smoothed
        Usmooth += self.interpolate(err)

        # Finally, smooth out any new high-frequency error (post-smoothing)
        Uout = self.GS(Usmooth, f, self.Niter)
        return Uout

    def interpolate(self, Xc):
        """Interpolation Routine - Inverse Full Weighting."""
        r = Xc.shape[0]
        c = Xc.shape[1]
        n = r - 3  # n = 2^k for some k
        l = c - 3  # l = 2^j for some j

        n22 = int(2*(n) + 3)  # 2*n = 2^{k+1}
        l22 = int(2*(l) + 3)  # 2*l = 2^{j+1}

        Xf = np.zeros((n22, l22), dtype=float)  # Grid twice as fine
        Xf[1:n22-1:2, 1:l22-1:2] = Xc[1:r-1, 1:c-1].copy()
        Xf[1:n22-1:2, 2:l22:2] = 0.5*(Xc[1:r-1, 1:c-1] + Xc[1:r-1, 2:c])
        Xf[2:n22:2, 1:l22-1:2] = 0.5*(Xc[1:r-1, 1:c-1] + Xc[2:r, 1:c-1])
        Xf[2:n22:2, 2:l22:2] = 0.25*(Xc[1:r-1, 1:c-1] + Xc[1:r-1, 2:c]
                               + Xc[2:r, 1:c-1] + Xc[2:r, 2:c])
        return Xf

    def restrict(self, Xf):
        """Restriction Routine."""
        r = Xf.shape[0]
        c = Xf.shape[1]
        n = r - 3  # n = 2^k for some k
        l = c - 3  # l = 2^j for some j
        nhalf = int(n / 2)  # n/2 = 2^{k - 1}
        lhalf = int(l / 2)  # l/2 = 2^{j - 1}

        # Coarse grid of (2^{k - 1} + 3, 2^{l - 1} + 3) including ghost points
        Xc = np.zeros((nhalf+3, lhalf+3), dtype=float)

        Xc[1:nhalf+2, 1:lhalf+2] += (0.25 * Xf[1:r-1:2, 1:c-1:2]
                                 + 0.125 * (Xf[1:r-1:2, 0:c-2:2]
                                          + Xf[1:r-1:2, 2:c:2]
                                          + Xf[0:r-2:2, 1:c-1:2]
                                          + Xf[2:r:2, 1:c-1:2])
                                 + 0.0625 * (Xf[2:r:2, 0:c-2:2]
                                           + Xf[2:r:2, 2:c:2]
                                           + Xf[0:r-2:2, 0:c-2:2]
                                           + Xf[0:r-2:2, 2:c:2]))
        return Xc

    def residual(self, U, f):
        """Computes the residual f - Au."""
        # Precomputing main variables used in methods:
        N2 = U.shape[0]
        L2 = U.shape[1]

        N, L = N2-2, L2-2
        h = (self.q + self.s) / (L - 1)  # Step-size in x-direction
        k = self.r / (N - 1)  # Step-size in y-direction
        h2 = h**2
        k2 = k**2
        invh = 1/h
        invk = 1/k
        invh2 = 1/h2
        invk2 = 1/k2

        # Precomputing extra variables to aid faster computation:
        alpha = 2*(invh2 + invk2)  # Constant on Division
        cx = invh2
        cy = invk2

        # Apply Boundary Conditions:
        U = self.boundary_conditions(U)
        res = np.zeros(U.shape, dtype=float)
        # Residual Computation:
        res[1:N+1, 1:L+1] = (f[1:N+1, 1:L+1] + U[1:N+1, 1:L+1]*alpha
                           - cx*(U[1:N+1, 2:L+2] + U[1:N+1, 0:L])
                           - cy*(U[2:N+2, 1:L+1] + U[0:N, 1:L+1]))
        return res

    def boundary_conditions(self, U):
        """Applies boundary conditions to the problem in Q1, Q2."""
        N2 = U.shape[0]
        L2 = U.shape[1]
        N, L = N2-2, L2-2
        h = (self.q + self.s) / (L - 1)  # Step-size in x-direction
        k = self.r / (N - 1)  # Step-size in y-direction
        invh = 1/h
        invk = 1/k

        # Updating Neutral Neumann Conditions:
        U[0, :] = U[2, :].copy()  # Top Boundary
        U[:, 0] = U[:, 2].copy()  # Left Boundary
        U[:, -1] = U[:, -3].copy()  # Right Boundary

        # Bottom Boundary:
        for i in range(1, L+1):
            xval = - self.q + (i - 1) * h
            if (xval < 0) or (xval > 1):
                U[-1, i] = U[-3, i].copy()  # Outside Airfoil
            else:
                # On Airfoil:
                dyb_dx = 2*self.tau*(1 - 2*xval)
                dU_dx = (U[-2, i+1] - U[-2, i-1])*0.5*invh
                U[-1, i] = U[-3, i].copy() - 2*k*(1 + dU_dx)*dyb_dx
        return U

    def GS(self, U, f, Niter):
        """Performs Niter iterations of GS Iteration for the Poisson Equation."""
        # Precomputing main variables used in methods:
        N2 = U.shape[0]
        L2 = U.shape[1]
        N, L = N2-2, L2-2
        h = (self.q + self.s) / (L - 1)  # Step-size in x-direction
        k = self.r / (N - 1)  # Step-size in y-direction
        h2 = h**2
        k2 = k**2
        invh = 1/h
        invk = 1/k
        invh2 = 1/h2
        invk2 = 1/k2

        # Precomputing extra variables to aid faster computation:
        alpha = 2*(invh2 + invk2)  # Constant on Division
        invalpha = 1 / alpha
        cx = 1 / (h2*alpha)  # Constant for x-direction
        cy = 1 / (k2*alpha)  # Constant for y-direction

        # Apply Boundary Conditions:
        U = self.boundary_conditions(U)
        # Gauss-Seidel Iteration:
        for iter in range(Niter):
            for j in range(N, 0, -1):
                for i in range(1, L + 1):
                    U[j, i] = (cx*(U[j, i+1] + U[j, i-1])
                             + cy*(U[j+1, i] + U[j-1, i])
                             - f[j, i]*invalpha)
            # Apply boundary conditions:
            U = self.boundary_conditions(U)
        return U

    def u_quantity(self, U):
        """Calculates u=du_dx on the whole domain."""
        u = np.zeros((self.N+2, self.L+2))
        h = (self.q + self.s) / (self.L - 1)  # Step-size in x-direction
        invh = 1/h
        invh2 = 0.5*invh
        # Centred Finite Differences:
        for j in range(1, self.N+1):
            for i in range(1, self.L+1):
                u[j, i] = invh2*(U[j, i+1] - U[j, i-1])
        return u[1:-1, 1:-1]

    def v_quantity(self, U):
        v = np.zeros((self.N+2, self.L+2))
        k = self.r / (self.N - 1)  # Step-size in y-direction
        invk = 1/k
        invk2 = 0.5*invk

        # Centred Finite Differences:
        for j in range(1, self.N):
            for i in range(1, self.L+1):
                v[j, i] = invk2*(U[j-1, i] - U[j+1, i])

        # Forward Finite Difference:
        for i in range(1, self.L+1):
            v[-2, i] = invk2*(-3*U[-2, i] + 4*U[-3, i] - U[-4, i])
        return v[1:-1, 1:-1]

    def compute_error(self, Uin, f):
        """Computes the error by taking one more GS iteration."""
        Unew = self.multigrid_vcycle(Uin.copy(), f.copy())
        error = np.max(np.abs(Unew - Uin))
        return error
