import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def return_function_val_only(f, arg):
    return f(arg)[0]

def plot_contours_and_paths(x, y):
    print(f'len(x) {len(x)} len(y) {len(y)}')

    # Following code based on https://matplotlib.org/stable/gallery/images_contours_and_fields/contour_demo.html

    delta = 0.025
    # x = np.arange(-3.0, 3.0, delta)
    # y = np.arange(-2.0, 2.0, delta)
    # X, Y = np.meshgrid(x, y)
    # Z1 = np.exp(-X ** 2 - Y ** 2)
    # Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
    # Z = (Z1 - Z2) * 2

    # X1, X2 = np.meshgrid([coordinate[0] for coordinate in x], [coordinate[1] for coordinate in x])
    X1, X2 = [coordinate[0] for coordinate in x], [coordinate[1] for coordinate in x]
    fig, ax = plt.subplots()
    # CS = ax.contour(X, Y, Z)
    CS = ax.contour(X1, X2, y)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('Simplest default with labels')
    plt.show()

def report_iteration(iter_count, cur_location, cur_obj_val, cur_step_len, cur_obj_f_val_change):
    print(f'Iteration number: {iter_count} current location: {cur_location} current obj val: {cur_obj_val} current step length: {cur_step_len} current change in objective function value: {cur_obj_f_val_change}')
