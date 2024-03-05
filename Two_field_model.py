"""
This code reproduces Fig.8 (a) from the article
'A dynamic neural field model of continuous input integration'
by W. Wojtak et al.

For details see the article.

(c) Weronika Wojtak, March 2024
"""

import numpy as np
import matplotlib.pyplot as plt

# Spatial coordinates
L = 4 * np.pi
N = 2**11
dx = 2 * L / N
xDim = np.linspace(-L, L - dx, N)

# Temporal coordinates
dt = 0.01
tspan = np.arange(0, 50, dt)
M = len(tspan)

# Functions
def kernel(x, A_ex, s_ex, A_in, s_in, g_i):
    return A_ex * np.exp(-0.5 * x ** 2 / s_ex ** 2) - A_in * np.exp(-0.5 * x ** 2 / s_in ** 2) - g_i

def gauss(x, mu, sigma):
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)

def sigmoid(x, beta, theta):
    return 1 / (1 + np.exp(-beta * (x - theta)))

# Parameters
beta = 1000
A_ex = 2
s_ex = 1.25
A_in = 1
s_in = 2.5
g_i = 0.1
theta = 0.5
tau = 1

# Initial data
K = 0.0
u_field = -theta * np.ones(N)
v_field = K - u_field
history_u = np.zeros((M, N))
history_v = np.zeros((M, N))

# Connectivity function
w = kernel(xDim, A_ex, s_ex, A_in, s_in, g_i)
wHat = np.fft.fft(w)

# Input
A_I = 1
sigma_I = 1
Input = np.zeros((M, N))
Input_pattern = A_I * gauss(xDim, 0, sigma_I)
num_rows = int(2/dt) - int(1/dt)  # Calculate the correct number of rows for tiling
Input[int(1/dt):int(2/dt), :] = np.tile(Input_pattern, (num_rows, 1))


# Main loop
for i in range(M):
    f = sigmoid(u_field, beta, theta)
    convolution = dx * np.fft.ifftshift(np.real(np.fft.ifft(np.fft.fft(f) * wHat)))
    u_field += dt / tau * (-u_field + convolution + v_field + Input[i, :])
    v_field += dt / tau * (-v_field - convolution + u_field)
    history_u[i, :] = u_field
    history_v[i, :] = v_field

# Plot results
plt.plot(xDim, u_field, 'k', linewidth=3)
plt.plot(xDim, v_field, '--k', linewidth=3)
plt.plot(xDim, theta * np.ones(N), ':k', linewidth=2)
plt.xlabel('x')
plt.ylabel('u(x), v(x)')
plt.xlim([-L, L])
plt.title('t = 50')
plt.show()
