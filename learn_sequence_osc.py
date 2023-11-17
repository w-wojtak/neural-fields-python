from src.plotting import *
from src.utils import *

# For oscillatory kernel: a, b, alpha.
kernel_pars = [1, 0.7, 0.9]

# Field parameters
x_lim, t_lim = 80, 100  # Limits for space and time. Space is set as [-x_lim, x_lim], time as [0, t_lim].
dx, dt = 0.05, 0.05  # Spatial and temporal discretization.
theta = 1  # Threshold for the activation function.

# Remaining parameters
tau_h = 20      # time constant of the threshold adaptation
h_0 = 0        # initial value of h-level

field_pars = [x_lim, t_lim, dx, dt, theta]

# Input parameters
input_flag = True  # Flag indicating if inputs are present.
input_shape = [3, 1.5]  # parameters of gaussian inputs: amplitude, sigma.
input_position = [-60, -30, 0, 30, 60]  # input_position, input_onset_time and input_duration must have the same length.
input_onset_time = [3, 8, 12, 16, 20]
input_duration = [1, 1, 1, 1, 1]

input_pars = [input_shape, input_position, input_onset_time, input_duration]

# Integrate the model
x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)

history_u = np.zeros([len(t), len(x)])

inputs = get_inputs(x, t, dt, input_pars, input_flag)
u_0 = h_0 * np.ones(np.shape(x))
u_field = u_0

h_u = h_0 * np.ones(np.shape(x))

# kernel and its fft
w_hat = np.fft.fft(kernel_osc(x, *kernel_pars))

for i in range(0, len(t)):
    f = np.heaviside(u_field - theta, 1)
    f_hat = np.fft.fft(f)
    conv = dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * w_hat)))
    h_u = h_u + dt / tau_h * f  # threshold adaptation
    u_field = u_field + dt * (-u_field + conv + inputs[i, :] + h_u)
    history_u[i, :] = u_field

# Plotting

# Plot of u(x,t) at time t=end.
fig = plot_final_state_1d(history_u, field_pars)

# Flat space-time image of the field activity.
# fig = plot_space_time_flat(field_activity, field_pars)

# fig.savefig('figures/sequence_osc_final.png')

# Specify the file name (you can use any file extension, but .npy is commonly used)
file_name_field = "sequence_memory.npy"
file_name_pars = "field_parameters.npy"

# Save the array to the file
np.save(file_name_field, history_u)
np.save(file_name_pars, field_pars)