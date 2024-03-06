"""
This code uses a forward Euler method to simulate the two field model
with input in two spatial dimensions.

The kernel is a mexican hat function.

For details see 'A dynamic neural field model of continuous input
integration' by W. Wojtak et al.

(c) Weronika Wojtak, March 2024
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Spatial discretization
L = 20
N = 2 ** 9
h = 2 * L / N
x = np.linspace(-L, L - h, N)
X, Y = np.meshgrid(x, x)

# Temporal discretization
T = 20
dt = 0.01
tspan = np.arange(0, T, dt)
M = len(tspan)


# Utils
def sigmoid(x, beta, theta):
    return 1 / (1 + np.exp(-beta * (x - theta)))


def gauss(x, mu, sigma):
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)


def w_mex(x, A_ex, s_ex, A_in, s_in, g_i):
    return A_ex * np.exp(-0.5 * x ** 2 / s_ex ** 2) - A_in * np.exp(-0.5 * x ** 2 / s_in ** 2) - g_i


# Parameters
p = [3, 1, 1.2, 1.6, 0.2]  # A_ex, s_ex, A_in, s_in, g_i
theta = 0
beta = 1000
tau = 1

# Initial data
K = -0.5
u0 = -0.5 * np.ones((N, N))
v0 = K - u0
u = u0
v = v0

# Set kernel
W2d = w_mex(np.sqrt(X ** 2 + Y ** 2), *p)
wHat = np.fft.fft2(W2d)

# Input
I_0 = np.zeros((N, N))
A_I = 3
sigma_I = 1
I_S = A_I * gauss(np.sqrt(X ** 2 + Y ** 2), 0, sigma_I)
t_start = 1
t_stop = 5

# For plotting
plot_activity = 0

if plot_activity == 1:
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

# Main loop
for i in range(M):

    if t_start / dt < i < t_stop / dt:
        Input = I_S
    else:
        Input = I_0

    f = sigmoid(u, beta, theta)
    fHat = np.fft.fft2(f)
    convolution = (2 * L / N) ** 2 * np.fft.ifftshift(np.real(np.fft.ifft2(fHat * wHat)))

    u += dt / tau * (-u + v + convolution + Input)
    v += dt / tau * (-v + u - convolution)

    if plot_activity == 1 and i % 100 == 0:
        ax.clear()
        ax.plot_surface(X, Y, u, cmap='jet', shade=True)
        ax.set_title(f't = {i * dt}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x,y)')
        plt.draw()
        plt.pause(0.1)

    if i % 100 == 0:
        print(f't = {i * dt}')

plt.ioff()  # Turn off interactive mode
plt.show()

# Plot results
plt.figure(figsize=(7, 6))
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, u, cmap='jet', shade=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
plt.show()
