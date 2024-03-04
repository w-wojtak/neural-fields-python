"""
This code uses a forward Euler method to simulate the Amari model
with two inputs.

The kernel w(x) is a Gaussian function. This example illustrates
a decision process, since only the stronger of two inputs triggers
the evolution of a bump.

(c) Weronika Wojtak, March 2024
"""

import numpy as np
import matplotlib.pyplot as plt

# Spatial discretization
L = 15
dx = 0.05
xDim = np.arange(-L, L + dx, dx)  # Include L in the range
N = len(xDim)

# Temporal discretization
T = 10
dt = 0.01
tDim = np.arange(0, T + dt, dt)  # Include T in the range
M = len(tDim)


# Utils
def sigmoid(x, beta, theta):
    return 1 / (1 + np.exp(-beta * (x - theta)))


def gauss(x, mu, sigma):
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


def w_lat(x, A, sigma, g_i):
    return A * np.exp(-0.5 * x ** 2 / sigma ** 2) - g_i


# Parameters
theta = 0.2
beta = 1000
tau = 1

# Set kernel
p = [2, 0.75, 0.5]  # A, sigma, g_i
w = w_lat(xDim, *p)
w_hat = np.fft.fft(w)

# Initial data
u_field = -theta * np.ones(N)

# Inputs
A_I1 = 1
A_I2 = 0.9
sigma_I = 1
distance = 6
Input = np.zeros((M, N))
I_S = A_I1 * gauss(xDim - distance, 0, sigma_I) + A_I2 * gauss(xDim + distance, 0, sigma_I)
Input[int(1 / dt):int(2 / dt), :] = np.tile(I_S, (int(1 / dt), 1))

# Main loop
for i in range(M):
    f = sigmoid(u_field, beta, theta)
    f_hat = np.fft.fft(f)
    convolution = dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * w_hat)))
    u_field += dt / tau * (-u_field + convolution + Input[i, :])

    if i % 10 == 0:
        plt.plot(xDim, u_field, linewidth=2)
        plt.plot(xDim, Input[i, :], linewidth=2)
        plt.plot(xDim, theta * np.ones(N), '--k', linewidth=1)
        plt.xlim([-L, L])
        plt.ylim([-2, 2])
        plt.legend(['u(x,t)', 'I(x,t)'])
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.pause(0.1)
        plt.clf()

    if i % 50 == 0:
        print(f"{i * dt}")

# Plot results
plt.figure()
plt.plot(xDim, u_field, 'k', linewidth=3)
plt.plot(xDim, theta * np.ones(N), ':k', linewidth=3)
plt.xlabel('x')
plt.ylabel('u(x)')
plt.xlim([-L, L])
plt.show()
