import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# delta range
delta = np.arange(0, 10.1, 0.1)

# Parameters for Oscillatory kernel
p = [1, 0.3, 0.9]  # [A, b, alpha]

# Oscillatory kernel function
def oscillatory_kernel(x, A, b, alpha):
    return A * (np.exp(-b * np.abs(x)) * ((b * np.sin(np.abs(alpha * x))) + np.cos(alpha * x)))

# Compute theta for Oscillatory
theta_oscillatory = []
for k in delta:
    theta_i, _ = quad(oscillatory_kernel, 0, k, args=(p[0], p[1], p[2]))
    theta_oscillatory.append(theta_i)

# Plot results for Oscillatory
plt.figure()
plt.plot(delta, theta_oscillatory, '-k', linewidth=2)
plt.xlabel('$\Delta$')
plt.ylabel('$W(\Delta)$ - Oscillatory')

# Check bump existence for a particular value of theta
theta_value = 0.9
plt.plot(delta, np.ones(len(delta)) * theta_value, '--k', linewidth=1)
plt.show()