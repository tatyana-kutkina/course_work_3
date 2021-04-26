import numpy as np
from scipy.integrate import solve_ivp



# Lorenz paramters and initial conditions.
sigma, beta, rho = 10.0, 8/3.0, 28.0
u0, v0, w0 = -8.0, 8.0, 27.0

# Maximum time point and total number of time points.
tmax = 100

def lorenz(t, X, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

def solve(n):
    # Integrate the Lorenz equations.
    soln = solve_ivp(lorenz, (0, tmax), (u0, v0, w0), args=(sigma, beta, rho),
                     dense_output=True)
    # Interpolate solution onto the time grid, t.
    t = np.linspace(0, tmax, n)

    x, y, z = soln.sol(t)
    return [x, y, z], t
