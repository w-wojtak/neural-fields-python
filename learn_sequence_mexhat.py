from src.plotting import *
from src.utils import *

# Kernel parameters
kernel_type = 1  # 0: Gaussian, 1: Mex-hat, 2: Oscillatory.

# For mex-hat kernel: a_ex, s_ex, a_in, s_in, w_in.
kernel_pars = [3, 1.0, 1.5, 1.5, 0.075]

# Field parameters
x_lim, t_lim = 80, 150  # Limits for space and time. Space is set as [-x_lim, x_lim], time as [0, t_lim].
dx, dt = 0.05, 0.1  # Spatial and temporal discretization.
theta = 0.4  # Threshold for the activation function.
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
plot_final_state_1d(field_activity, field_pars)

# Flat space-time image of the field activity.
# plot_space_time_flat(field_activity, field_pars)