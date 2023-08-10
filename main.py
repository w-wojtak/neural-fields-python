from src.plotting import *
from src.utils import *


def main():
    # add more kernels
    # Kernel parameters
    kernel_type: 1  # 0: Gaussian, 1: Mex-hat, 2: Oscillatory.

    # Parameters of the mex-hat kernel: a_ex, s_ex, a_in, s_in, w_in.
    kernel_pars = [1, 0.3, 0.4, 0.5, 0.05]

    # Field parameters
    x_lim, t_lim = 6, 20  # Limits for space and time. Space is set as [-x_lim, x_lim], time as [0, t_lim].
    dx, dt = 0.01, 0.01  # Spatial and temporal discretization.
    theta = 0.1  # Threshold for the activation function.
    field_pars = [x_lim, t_lim, dx, dt, theta]

    # Input parameters
    input_flag = True  # Flag indicating if inputs are present.
    input_shape = [0.7, 0.3]  # parameters of gaussian inputs: amplitude, sigma.
    input_position = [-2, 2]  # input_position, input_onset_time and input_duration must have the same length.
    input_onset_time = [2, 5]
    input_duration = [1, 1]

    input_pars = [input_shape, input_position, input_onset_time, input_duration]

    # Initial condition - gaussian (if input_flag = False).
    ic_shape = [0, 0.5, 0.2]  # position, amplitude, sigma

    # Integrate the model
    field_activity, inputs = simulate_amari(field_pars, kernel_pars, input_flag, input_pars, ic_shape)

    # Plotting

    # Plot of u(x,t) at time t=end.
    # plot_final_state_1d(field_activity, field_pars)

    # Flat space-time image of the field activity.
    # plot_space_time_flat(field_activity, field_pars)

    # 3D surface plot of the field activity over space and time.
    # plot_space_time_3d(field_activity, field_pars)

    # 3D surface plot of the field activity over space and time with a contour plot underneath.
    plot_space_time_3d_contour(field_activity, field_pars)

    # Animation of activity u(x,t) and inputs (if present).
    # plot_animate_1d(field_activity, field_pars, inputs, input_flag)

    # Interactive plot with a slider to visualize how activity and inputs change in time.
    # plot_slider_1d(field_activity, field_pars, inputs, input_flag)

    # Plot time courses of bump centers and inputs. Useful only when inputs are present.
    # plot_time_courses(field_activity, field_pars, inputs, input_position)


if __name__ == '__main__':
    main()
    pass
