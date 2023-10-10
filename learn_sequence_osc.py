from src.plotting import *
from src.utils import *

# Kernel parameters
kernel_type = 2  # 0: Gaussian, 1: Mex-hat, 2: Oscillatory.

# For oscillatory kernel: a, b, alpha.
kernel_pars = [1, 0.7, 0.9]

# Field parameters
x_lim, t_lim = 80, 100  # Limits for space and time. Space is set as [-x_lim, x_lim], time as [0, t_lim].
dx, dt = 0.05, 0.1  # Spatial and temporal discretization.
theta = 1  # Threshold for the activation function.
field_pars = [x_lim, t_lim, dx, dt, theta]

# Input parameters
input_flag = True  # Flag indicating if inputs are present.
input_shape = [3, 1.5]  # parameters of gaussian inputs: amplitude, sigma.
input_position = [-60, -30, 0, 30, 60]  # input_position, input_onset_time and input_duration must have the same length.
input_onset_time = [2, 7, 12, 16, 20]
input_duration = [1, 1, 1, 1, 1]

input_pars = [input_shape, input_position, input_onset_time, input_duration]

# Initial condition - gaussian (if input_flag = False).
ic_shape = [0, 2.5, 0.5]  # position, amplitude, sigma

# Integrate the model
field_activity, inputs = simulate_amari(field_pars, kernel_type, kernel_pars, input_flag, input_pars, ic_shape)

# Plotting

# Plot of u(x,t) at time t=end.
fig = plot_final_state_1d(field_activity, field_pars)

# Flat space-time image of the field activity.
# fig = plot_space_time_flat(field_activity, field_pars)

fig.savefig('figures/sequence_osc_final.png')