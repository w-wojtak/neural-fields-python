import numpy as np
from src.plotting import *
from src.utils import *
import time
import matplotlib.pyplot as plt

# Specify the file name you used for saving the array
file_name_field = "sequence_memory.npy"
file_name_pars = "field_parameters.npy"

# Load the array from the file
saved_activity = np.load(file_name_field)

# For oscillatory kernel: a, b, alpha.
kernel_pars = [1.5, 0.5, 0.8]

# For gaussian kernel: a_ex, s_ex, w_in.
kernel_pars_gauss = [1.5, 0.9, 0.0]

# Field parameters
field_pars = np.load(file_name_pars)
x_lim, t_lim, dx, dt, theta = field_pars

# # Remaining parameters
h_d_init = -6.245
tau_h_dec = 20
h_wm = -1.0
c_wm = 6  # strength of inhibitory connections from u_wm to u_dec

# Integrate the model
x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)

history_dec = np.zeros([len(t), len(x)])
history_wm = np.zeros([len(t), len(x)])

memory = saved_activity[-1, :]
u_dec = memory + h_d_init

u_wm = h_wm * np.ones(np.shape(x))

h_d = h_d_init

# kernels and ffts
w_hat = np.fft.fft(kernel_osc(x, *kernel_pars))
w_lat = np.fft.fft(kernel_gauss(x, *kernel_pars_gauss))


input_position = [-60, -30, 0, 30, 60]
indices_closest = [np.argmin(np.abs(x - target)) for target in input_position]

u_dec_prev = []

print_evolution = False
print_final = False

count = 1

for i in range(0, len(t)):
    f_dec = np.heaviside(u_dec - theta, 1)
    f_hat_dec = np.fft.fft(f_dec)
    f_wm = np.heaviside(u_wm - 0.5, 1)
    f_hat_wm = np.fft.fft(f_wm)

    conv = dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat_dec * w_lat)))
    conv_wm = dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat_wm * w_hat)))

    h_d = h_d + dt / tau_h_dec  # threshold adaptation

    u_dec = u_dec + dt * (-u_dec + conv + memory + h_d - c_wm * f_wm * u_wm)
    u_wm = u_wm + dt * (-u_wm + conv_wm + h_wm + f_dec * u_dec)

    history_dec[i, :] = u_dec
    history_wm[i, :] = u_wm

    # Check for theta crossings
    for k in indices_closest:
        if u_dec[k] > theta and (i == 0 or u_dec_prev[k] < theta):
            print(f"recalled time {count}: {t[i]:.1f}")
            count += 1

    # Store the current u_dec for the next iteration
    u_dec_prev = u_dec.copy()

# Plotting

if print_evolution:

    upper_lim_y = max([history_wm.max(), history_dec.max()])
    lower_lim_y = min([history_wm.min(), history_dec.min()])

    plt.ion()
    figure, ax = plt.subplots(figsize=(6, 4))
    ax.set_ylim(lower_lim_y, upper_lim_y)
    ax.set_xlim(-x_lim, x_lim)
    plt.xlabel('x')

    line1, = ax.plot(x, history_dec[0, :], label='dec')
    line2, = ax.plot(x, history_wm[0, :], label='wm')

    ax.legend()

    for i in range(history_dec.shape[0]):
        if i % 5 == 0:
            line1.set_xdata(x)
            line1.set_ydata(history_dec[i, :])

            line2.set_xdata(x)
            line2.set_ydata(history_wm[i, :])

            # draw updated values
            figure.canvas.draw()
            figure.canvas.flush_events()
            time.sleep(0.1)


if print_final:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot history_dec in the first subplot
    ax1.plot(x, history_dec[-1, :])
    ax1.set_xlim(-x_lim, x_lim)
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x)')
    ax1.set_title('History Decision')

    # Plot history_wm in the second subplot
    ax2.plot(x, history_wm[-1, :])
    ax2.set_xlim(-x_lim, x_lim)
    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x)')
    ax2.set_title('History WM')

    plt.tight_layout()

    plt.show()
