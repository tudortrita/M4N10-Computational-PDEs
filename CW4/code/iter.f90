! iter.f90 - Contains FORTRAN90 translations to Python code inside the file
!            schemes.py. Code to perform iterations as in CW4.
! Tudor Trita Trita
! CID: 01199397
! MSci Mathematics
! M4N10 - Computational Partial Differential Equations.

MODULE iteration
    IMPLICIT none
CONTAINS

SUBROUTINE scheme(Niter, N, L, tau, M, gamma, q, s, r, rho, phi, u, v)
    integer, intent(in) :: Niter, N, L
    real(kind=8), intent(in) :: tau, M, gamma, q, s, r
    real(kind=8), intent(inout), dimension(N+2, L+2) :: rho, phi
    !f2py intent(in, out) :: rho, phi
    real(kind=8), intent(out), dimension(N+2, L+2) :: u, v

    real(kind=8), dimension(N+2, L+2) :: dr_dx, dr_dy, rho_hat
    real(kind=8), dimension(N+2, L+2) :: c1, c2, alpha, f
    real(kind=8) :: dx, dy, dx2, dy2, invdx, invdy, invdx2, invdy2, a, b
    real(kind=8) :: xval, dyb_dx, dU_dx
    integer :: iter, i, j

    ! Precomputing Grid Constants:
    dx = (q+s)/(dfloat(L)-1.d0)
    dy = r/(dfloat(N)-1.d0)
    dx2 = dx*dx
    dy2 = dy*dy
    invdx = 1.d0/dx
    invdy = 1.d0/dy
    invdx2 = 1.d0/dx2
    invdy2 = 1.d0/dy2

    ! Computing rho_hat, rho derivatives and  phi derivatives
    rho_hat = 1.d0 + rho
    a = (gamma - 1.d0)/2.d0 * (M**2.d0)
    b = 1.d0 / (gamma - 1.d0)

    CALL compute_derivs(N, L, dx, dy, phi, u, v)

    ! Iterations:
    DO iter=1, Niter
        ! Check where we are in algorithm:
        IF (mod(iter, 1000) .eq. 0) THEN
            PRINT*, "   - Iteration ", iter, " of ", Niter
        END IF

        ! Compute rho derivatives for main Gauss-Seidel Iteration:
        CALL compute_derivs(N, L, dx, dy, rho_hat, dr_dx, dr_dy)
        alpha = 2.d0 * rho_hat * (invdx2 + invdy2)
        f = dr_dx * (1.d0 + u) + dr_dy * v

        c1 = rho_hat * invdx2
        c2 = rho_hat * invdy2

        ! Gauss Seidel:
        DO j=N+1,2,-1  ! Backwards in the y-direction
            DO i=2,L+1  ! Forwards in the x-direction
                phi(j, i) = (c1(j, i)*(phi(j, i+1) + phi(j, i-1)) &
                           + c2(j, i)*(phi(j+1, i) + phi(j-1, i)) &
                           + f(j, i)) / alpha(j, i)
            END DO
        END DO

        ! Boundary Conditions:
        phi(1, :) = phi(3, :)  ! Top Boundary
        phi(:, 1) = phi(:, 3) ! Left Boundary
        phi(:, L+2) = phi(:, L)  ! Right Boundary
        DO i=2, L+1  ! Bottom Boundary
            xval = - q + (dfloat(i)-2.d0)*dx
            IF ((xval < 0.d0) .OR. (xval > 1.d0)) THEN  ! Outside Airfoil
                phi(N+2, i) = phi(N, i)
            ELSE  ! On Airfoil:
                dyb_dx = 2.d0*tau*(1.d0 - 2.d0*xval)
                dU_dx = (phi(N+1, i+1) - phi(N+1, i-1))*0.5d0*invdx
                phi(N+2, i) = phi(N, i) - 2.d0*dy*(1.d0 + dU_dx)*dyb_dx
            END IF
        END DO

        CALL compute_derivs(N, L, dx, dy, phi, u, v)
        rho_hat = (1.d0 - a*(2.d0*u + u**2.d0 + v**2.d0))**b
    END DO
    rho = rho_hat - 1.d0
END SUBROUTINE scheme


SUBROUTINE compute_derivs(N, L, dx, dy, U, dU_dx, dU_dy)
    integer, intent(in) :: N, L
    real(kind=8), intent(in) :: dx, dy
    real(kind=8), intent(in), dimension(N+2, L+2) :: U
    real(kind=8), intent(out), dimension(N+2, L+2) :: dU_dx, dU_dy

    real(kind=8) :: invdx, invdy
    integer :: i, j

    invdx = 1.d0/dx
    invdy = 1.d0/dy

    ! Calculating interior:
    DO j=2, N+1
        DO i=2, L+1
            dU_dx(j, i) = 0.5d0 * invdx * (U(j, i+1) - U(j, i-1))
            dU_dy(j, i) = 0.5d0 * invdy * (U(j-1, i) - U(j+1, i))
        END DO
    END DO

    ! Left and Right Boundaries:
    DO j=2, N+1
        dU_dx(j, 2) = 0.5d0 * invdx * (-3.d0*U(j, 2) + 4.d0*U(j, 3) - U(j, 4))
        dU_dx(j, L+1) = 0.5d0 * invdx * (3.d0*U(j, L+1) - 4.d0*U(j, L) + U(j, L-1))
    END DO

    ! Top and Bottom Boundaries:
    DO i=2, L+1
        dU_dy(2, i) = 0.5d0 * invdy * (3.d0*U(2, i) - 4.d0*U(3, i) + U(4, i))
        dU_dy(N+1, i) = 0.5d0 * invdy * (-3.d0*U(N+1, i) + 4.d0*U(N, i) - U(N-1, i))
    END DO
END SUBROUTINE compute_derivs
END MODULE iteration
