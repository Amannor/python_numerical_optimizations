import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os

MIN_AXIS_REACH = 1.5
PARENT_DIR = os.path.dirname(os.getcwd())

def plot_contours_and_paths(f, x):
    c0 = np.array([coordinate[0] for coordinate in x])
    max_abs_val0 = max(abs(min(c0)), max(c0))
    max_abs_val0 = max(max_abs_val0,MIN_AXIS_REACH)
    linspace0 = np.linspace(-max_abs_val0, max_abs_val0)
    c1 = np.array([coordinate[1] for coordinate in x])
    max_abs_val1 = max(abs(min(c1)), max(c1))
    max_abs_val1 = max(max_abs_val1, MIN_AXIS_REACH)
    linspace1 = np.linspace(-max_abs_val1, max_abs_val1)


    # Following code based on https://matplotlib.org/stable/gallery/images_contours_and_fields/contour_demo.html
    X1, X2 = np.meshgrid(linspace0, linspace1)
    fig, ax = plt.subplots()
    plt.xlim(-max_abs_val0,max_abs_val0)
    plt.ylim(-max_abs_val1,max_abs_val1)
    Z = np.empty(shape=(X1.shape[0],X1.shape[1]))
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i][j] = f(np.array([X1[i][j], X2[i][j]]))[0]
    CS = ax.contour(X1, X2, Z)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title(f'{f.__name__}')
    ax.scatter(c0, c1)


    path = os.path.join(PARENT_DIR, 'output', f'{f.__name__}.png')
    plt.savefig(path)

def report_iteration(iter_count, cur_location, cur_obj_val, cur_step_len, cur_obj_f_val_change):
    print(f'Iteration number: {iter_count} current location: {cur_location} current obj val: {cur_obj_val} current step length: {cur_step_len} current change in objective function value: {cur_obj_f_val_change}')
