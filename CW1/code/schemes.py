""" schemes.py - Script containing methods for computing the diffusion equation.
Tudor Trita Trita
CID: 01199397
MSci Mathematics

M4N10 - Computational Partial Differential Equations - Project 1

Functions:
    - diffusion_analytic_solution
    - diffusion_explicit_scheme
    - diffusion_fully_implicit
    - diffusion_crank_nicolson
    - diffusion_fully_implicit_alternative
    - diffusion_explicit_fourth_order
    - diffusion_fully_implicit_fourth_order
"""
import numpy as np

import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

import warnings
warnings.filterwarnings("ignore")


def diffusion_analytic_solution(t, x, n_terms_sum=10000):
    """Computes the Analytic solution at time t for coordinate x for the
       one-dimensional diffusion equation.

    Parameters
    ----------
    t : time at which solution is computed.
    x : spatial coordinate at which solution is computed.
    n_terms_sum : number of terms until truncation of infinite sum.

    Returns
    -------
    u : solution at time t, spatial coordinate x.
    """
    n_space = np.arange(0, n_terms_sum + 1)  # Number of terms till truncation

    # Computing terms inside the sum vectorised
    sin_term = np.sin((2*n_space + 1) * np.pi * x)
    exp_term = np.exp(-(2*n_space + 1)**2 * np.pi**2 * t)
    inverse_term = 1/(2*n_space + 1)

    truncated_sum = np.sum(inverse_term * exp_term * sin_term)
    u = 4 * truncated_sum / np.pi
    return u


def diffusion_explicit_scheme(t, k, N):
    """Computes the finite difference solution for the diffusion equation using
       the usual explicit scheme.

    Parameters
    ----------
    t : time at which solution is computed.
    k : size of temporal marching step.
    N : number of points to compute the scheme at. Note: h depends on N.

    Returns
    -------
    U : Solution computed over the whole interval.
    """
    h = 1 / (N - 1)  # N-1 Number of Intervals
    r = k / (h**2)  # Computing r

    # Checking that time-step divides t, i.e. a valid time-step.
    error_message = "Error 1: Time-step k is not valid!"
    assert np.allclose(t / k, round(t / k)), error_message
    num_time_steps = int(round(t / k))  # Computing number of time-steps

    # Setting Initial Conditions + Boundary Conditions
    U = np.ones(N, dtype=float)  # N number of points
    U[0], U[-1] = (0, 0)  # Setting first and last points equal to 0

    # Constructing tridiagonal matrix. Make use of Scipy sparse matrix for this
    M = scipy.sparse.diags(diagonals=(r, 1 - 2*r, r),  # Value of Diagonals
                           offsets=(-1, 0, 1),  # Position of Diagonals
                           shape=(N-2, N-2),  # Only need interior of matrix
                           format="csr")  # Compressed Sparse Row - fast mults.
    U_FD_interior = U[1:-1]  # Avoids indexing at each marching timestep
    for t in range(num_time_steps):
        U_FD_interior = M @ U_FD_interior  # matrix multiplication of MU
    U[1:-1] = U_FD_interior  # Loading interior back into FD solution
    return U


def diffusion_fully_implicit(t, k, N):
    """Computes the finite difference solution for the diffusion equation using
       the fully implicit scheme. (theta=1)

    Parameters
    ----------
    t : time at which solution is computed.
    k : size of temporal marching step.
    N : number of points to compute the scheme at. Note: h depends on N.

    Returns
    -------
    U : Solution computed over the whole interval.
    """
    h = 1 / (N - 1)  # N-1 Number of Intervals
    r = k / (h**2)  # Computing r

    # Checking that time-step divides t, i.e. a valid time-step.
    error_message = "Error 1: Time-step k is not valid!"
    assert np.allclose(t / k, round(t / k)), error_message
    num_time_steps = int(round(t / k))  # Computing number of time-steps

    # Setting Initial Conditions + Boundary Conditions
    U = np.ones(N, dtype=float)  # N number of points
    U[0], U[-1] = (0, 0)  # Setting first and last points equal to 0

    # Constructing tridiagonal matrix. Make use of Scipy sparse matrix for this.
    A = scipy.sparse.diags(diagonals=(-r, 1 + 2*r, -r),
                           offsets=(-1, 0, 1),
                           shape=(N-2, N-2),
                           format="csr")

    U_FD_interior = U[1:-1]  # Avoids indexing at each marching timestep
    for t in range(num_time_steps):
        U_FD_interior = scipy.sparse.linalg.spsolve(A, U_FD_interior)
    U[1:-1] = U_FD_interior
    return U


def diffusion_crank_nicolson(t, k, N):
    """Computes the finite difference solution for the diffusion equation using
       the semi-explicit scheme (Crank-Nicolson - theta=1/2).

    Parameters
    ----------
    t : time at which solution is computed.
    k : size of temporal marching step.
    N : number of points to compute the scheme at. Note: h depends on N.

    Returns
    -------
    U : Solution computed over the whole interval.
    """
    h = 1 / (N - 1)  # N-1 Number of Intervals
    r = k / (h**2)  # Computing r

    # Checking that time-step divides t, i.e. a valid time-step.
    error_message = "Error 1: Time-step k is not valid!"
    assert np.allclose(t / k, round(t / k)), error_message
    num_time_steps = int(round(t / k))  # Computing number of time-steps

    # Setting Initial Conditions + Boundary Conditions
    U = np.ones(N, dtype=float)  # N number of points
    U[0], U[-1] = (0, 0)  # Setting first and last points equal to 0

    # Constructing tridiagonal matrix. Make use of Scipy sparse matrix for this.
    A = scipy.sparse.diags(diagonals=(-r/2, 1 + r, -r/2),
                           offsets=(-1, 0, 1),
                           shape=(N-2, N-2),
                           format="csr")

    B = scipy.sparse.diags(diagonals=(r/2, 1 - r, r/2),
                           offsets=(-1, 0, 1),
                           shape=(N-2, N-2),
                           format="csr")

    U_FD_interior = U[1:-1].copy()
    for t in range(num_time_steps):
        RHS = B @ U_FD_interior
        U_FD_interior = scipy.sparse.linalg.spsolve(A, RHS)

    U[1:-1] = U_FD_interior
    return U


def diffusion_fully_implicit_alternative(t, k, N):
    """Computes the finite difference solution for the diffusion equation using
       the fully implicit scheme (theta=1) with the alternative time-discretisation
       as shown in Part B Question 4.

    Parameters
    ----------
    t : time at which solution is computed.
    k : size of temporal marching step.
    N : number of points to compute the scheme at. Note: h depends on N.

    Returns
    -------
    U : Solution computed over the whole interval.
    """
    h = 1 / (N - 1)  # N-1 Number of Intervals
    r = k / (h**2)  # Computing r

    # Checking that time-step divides t, i.e. a valid time-step.
    error_message = "Error 1: Time-step k is not valid!"
    assert np.allclose(t / k, round(t / k)), error_message
    num_time_steps = int(round(t / k))  # Computing number of time-steps

    # Setting Initial Conditions + Boundary Conditions
    U = np.ones(N, dtype=float)  # N number of points
    U[0], U[-1] = (0, 0)  # Setting first and last points equal to 0

    # Constructing tridiagonal matrix. Make use of Scipy sparse matrix for this.
    A1 = scipy.sparse.diags(diagonals=(-r, 1 + 2*r, -r),
                            offsets=(-1, 0, 1),
                            shape=(N-2, N-2),
                            format="csr")

    A2 = scipy.sparse.diags(diagonals=(-2*r, 3 + 4*r, -2*r),
                            offsets=(-1, 0, 1),
                            shape=(N-2, N-2),
                            format="csr")

    U_FD_interior = U[1:-1].copy()  # Avoids indexing at each marching timestep
    U_FD_interior_prev = U[1:-1].copy()
    for t in range(num_time_steps):
        if t == 0:
            # Use fully-implicit scheme at first time-step
            U_FD_interior = scipy.sparse.linalg.spsolve(A1, U_FD_interior)
        else:
            # Begin using 3-point alternative scheme
            RHS = 4*U_FD_interior - U_FD_interior_prev
            U_FD_interior_prev = U_FD_interior.copy()  # Keeping for next iteration
            U_FD_interior = scipy.sparse.linalg.spsolve(A2, RHS)

    U[1:-1] = U_FD_interior
    return U


def diffusion_explicit_fourth_order(t, k, N):
    """Computes the finite difference solution for the diffusion equation using
       a 5-point explicit scheme to fourth-order accuracy. Coefficients computed
       using Lagrangian Interpolation.

    Parameters
    ----------
    t : time at which solution is computed.
    k : size of temporal marching step.
    N : number of points to compute the scheme at. Note: h depends on N.

    Returns
    -------
    U : Solution computed over the whole interval.
    """
    h = 1 / (N - 1)  # N-1 Number of Intervals
    r = k / (h**2)  # Computing r

    # Checking that time-step divides t, i.e. a valid time-step.
    error_message = "Error 1: Time-step k is not valid!"
    assert np.allclose(t / k, round(t / k)), error_message
    num_time_steps = int(round(t / k))  # Computing number of time-steps

    # Setting Initial Conditions + Boundary Conditions
    U = np.ones(N, dtype=float)  # N number of points
    U[0], U[-1] = (0, 0)  # Setting first and last points equal to 0

    # Constructing tridiagonal matrix. Make use of Scipy sparse matrix for this
    M = scipy.sparse.diags(diagonals=(-r/12, 4*r/3, 1 - 5*r/2, 4*r/3, -r/12),
                           offsets=(-2, -1, 0, 1, 2),  # Position of Diagonals
                           shape=(N-2, N-2),  # Only need interior of matrix
                           format="csr")  # Compressed Sparse Row - fast mults.
    # Forward & Backward Difference Coefficients
    forward_differences = np.zeros(N-2, dtype=float)
    forward_differences[:4] = [1 - (5*r/3), r/2, r/3, -r/12]
    backward_differences = np.zeros(N-2, dtype=float)
    backward_differences[-4:] = [-r/12, r/3, r/2, 1 - (5*r/3)]

    # Loading these coefficients in first and last rows of M as sparse matrices
    M[0] = scipy.sparse.csr_matrix(forward_differences)
    M[-1] = scipy.sparse.csr_matrix(backward_differences)

    U_FD_interior = U[1:-1]  # Avoids indexing at each marching timestep
    for t in range(num_time_steps):
        U_FD_interior = M @ U_FD_interior  # matrix multiplication of MU
    U[1:-1] = U_FD_interior  # Loading interior back into FD solution
    return U


def diffusion_fully_implicit_fourth_order(t, k, N):
    """Computes the finite difference solution for the diffusion equation using
       a 5-point fully implicit scheme (theta=1) to fourth-order accuracy.
       Coefficients computed using Lagrangian Interpolation.

    Parameters
    ----------
    t : time at which solution is computed.
    k : size of temporal marching step.
    N : number of points to compute the scheme at. Note: h depends on N.

    Returns
    -------
    U : Solution computed over the whole interval.
    """
    h = 1 / (N - 1)  # N-1 Number of Intervals
    r = k / (h**2)  # Computing r

    # Checking that time-step divides t, i.e. a valid time-step.
    error_message = "Error 1: Time-step k is not valid!"
    assert np.allclose(t / k, round(t / k)), error_message
    num_time_steps = int(round(t / k))  # Computing number of time-steps

    # Setting Initial Conditions + Boundary Conditions
    U = np.ones(N, dtype=float)  # N number of points
    U[0], U[-1] = (0, 0)  # Setting first and last points equal to 0

    # Constructing tridiagonal matrix. Make use of Scipy sparse matrix for this
    M = scipy.sparse.diags(diagonals=(r/12, -4*r/3, 1 + 5*r/2, -4*r/3, r/12),
                           offsets=(-2, -1, 0, 1, 2),  # Position of Diagonals
                           shape=(N-2, N-2),  # Only need interior of matrix
                           format="csr")  # Compressed Sparse Row - fast mults.

    # Forward & Backward Difference Coefficients
    forward_differences = np.zeros(N-2, dtype=float)
    forward_differences[:4] = [1 + (5*r/3), -r/2, -r/3, r/12]
    backward_differences = np.zeros(N-2, dtype=float)
    backward_differences[-4:] = [r/12, -r/3, -r/2, 1 + (5*r/3)]

    # Loading these coefficients in first and last rows of M as sparse matrices
    M[0] = scipy.sparse.csr_matrix(forward_differences)
    M[-1] = scipy.sparse.csr_matrix(backward_differences)

    U_FD_interior = U[1:-1].copy()  # Avoids indexing at each marching timestep
    for t in range(num_time_steps):
        U_FD_interior = scipy.sparse.linalg.spsolve(M, U_FD_interior)
    U[1:-1] = U_FD_interior
    return U
