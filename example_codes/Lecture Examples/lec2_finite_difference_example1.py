import numpy as np
import matplotlib.pyplot as plt


def fd_2nd_order(N, X, a):
    U = np.zeros(N+1)
    h = a/(N)
    U[0] = 1
    U[1] = np.exp(-h)

    for i in range(N-1):
        U[i+2] = -2*h*U[i+1] + U[i]
    return U

N = 12
a = 5
X = np.linspace(0, a, N+1)

X_exact = np.linspace(0, a, 10000)
U_exact = np.exp(-X_exact)


U = fd_2nd_order(N, X, a)


plt.figure(figsize=(13, 8))
plt.plot(X_exact, U_exact)
plt.plot(X, U)
plt.show()
