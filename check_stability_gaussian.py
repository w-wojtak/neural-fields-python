import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# delta range
delta = np.arange(0, 10.1, 0.1)

# Parameters for Gaussian kernel
p = [1.5, 1, 0.1]  # [A, sigma, g_i]

# Gaussian kernel function
def w(x, A, sigma, g_i):
    return A * np.exp(-0.5 * (x)**2 / sigma**2) - g_i

# Compute theta
theta = []
for k in delta:
    theta_i, _ = quad(w, 0, k, args=(p[0], p[1], p[2]))
    theta.append(theta_i)

# Plot results
plt.figure()
plt.plot(delta, theta, '-k', linewidth=2)
plt.xlabel('$\Delta$')
plt.ylabel('$W(\Delta)$')

# Check bump existence for a particular value of theta
theta_value = 1
plt.plot(delta, np.ones(len(delta)) * theta_value, '--k', linewidth=1)
plt.show()
