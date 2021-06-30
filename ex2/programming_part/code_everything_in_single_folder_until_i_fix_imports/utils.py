import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import patheffects
import os

MIN_AXIS_REACH = 1.5
EXTRA_PLOT_SPACE = 0.2
# PARENT_DIR = os.path.dirname(os.getcwd())

def save_fig(f_name):
    path = os.path.join(os.getcwd(), 'output', f_name)
    plt.savefig(path)

def get_index_of_first_inequality_met(point ,ineq_constraints):
    for i in range(len(ineq_constraints)):
        ineq_res = ineq_constraints[i](point, True)
        # print(f'point[0] {point[0]} i {i} ineq_res {ineq_res}')
        if ineq_constraints[i](point, True)<=0 :
            return i
    return -1

def plot_for_qp(f, x_vals, ineq_constraints, eq_constraints_mat, eq_constraints_rhs):
    #From: https://matplotlib.org/2.0.2/mpl_toolkits/mplot3d/tutorial.html
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    qp_equality_func_for_xy = lambda x,y: 1-x-y

    #Based on example from https://matplotlib.org/2.0.2/mpl_examples/mplot3d/surface3d_demo.py
    X_range = np.arange(-5, 5, 0.25)
    xlen = len(X_range)
    Y_range = np.arange(-5, 5, 0.25)
    ylen = len(Y_range)
    X, Y = np.meshgrid(X_range, Y_range)
    Z = qp_equality_func_for_xy(X,Y)
    colors = np.zeros(X.shape, dtype=str)
    # colortuple = ('b', 'g', 'r', 'y', 'c', 'm', 'k') #For full list see https://matplotlib.org/2.0.2/api/colors_api.html
    color_dict = {'b': 'Blue', 'g': 'Green', 'r': 'Red', 'y': 'Yellow', 'c': 'Cyan', 'm': 'Magenta', 'k': 'Black'} #For full list see https://matplotlib.org/2.0.2/api/colors_api.html
    ineq_to_color = {}
    for y_i in range(ylen):
        for x_i in range(xlen):
            x_val, y_val = X_range[x_i], Y_range[y_i]
            point = np.array([x_val, y_val, qp_equality_func_for_xy(x_val, y_val)])
            ineq_i = get_index_of_first_inequality_met(point, ineq_constraints)
            if ineq_i>=0:
                ineq_str = ineq_constraints[ineq_i](None, return_str_rep=True) 
                cur_color = list(color_dict.keys())[ineq_i % len(color_dict)]
                colors[x_i, y_i] = cur_color
                
                if not ineq_str in ineq_to_color:
                    ineq_to_color[ineq_str] = color_dict[cur_color]

    
    text2d = "Ineqaulities to colors mapping:"
    for ineq_str in sorted(ineq_to_color.keys()):
        text2d+=f'\n{ineq_str}: {ineq_to_color[ineq_str]}'
    ax.text2D(0.05, 0.95, text2d, transform=ax.transAxes)
    ax.plot_surface(X, Y, Z, facecolors=colors)
    ax.scatter([c[0] for c in x_vals], [c[1] for c in x_vals], [c[2] for c in x_vals])
    # ax.set_title(f'Contour lines and objective values {f.__name__}{title_dir_suffix}') #TODO
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    save_fig(f.__name__)

def plot_for_lp(f, x_vals, ineq_constraints):
    
    f_0 = lambda x,y: f(np.array([x,y]))[0]
    f_1 = lambda x,y: ineq_constraints[0](np.array([x,y]), True)
    f_2 = lambda x,y: ineq_constraints[1](np.array([x,y]), True)
    f_3 = lambda x,y: ineq_constraints[2](np.array([x,y]), True)
    f_4 = lambda x,y: ineq_constraints[3](np.array([x,y]), True)
    
    x = np.linspace(-1, 3, 500)
    y = np.linspace(-1, 3, 500)
    X, Y = np.meshgrid(x, y)


    Z = f_0(X, Y)
    plt.contour(X, Y, Z, 50, cmap='RdGy')
    plt.contourf(X, Y, Z, 100, cmap='RdGy', alpha=0.6)

    #From: https://stackoverflow.com/a/51389483
    d = np.linspace(-1,3)
    x,y = np.meshgrid(d,d)

    im = plt.imshow( ((f_1(x,y)<=0) & (f_2(x,y)<=0) & (f_3(x,y)<=0) & (f_4(x,y)<=0)).astype(int), 
                    extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys")

    plt.scatter([c[0] for c in x_vals], [c[1] for c in x_vals])
    for i, x_val in enumerate(x_vals):
        # x_val0_str = "{:.2f}".format(x_val[0])
        # x_val1_str = "{:.2f}".format(x_val[1])
        # plt.annotate(f'({i}) ({x_val0_str},{x_val1_str})', (x_val[0], x_val[1]))
        plt.annotate(f'(x{i})', (x_val[0], x_val[1]))

    save_fig(f.__name__)


def plot_contours_and_paths(f, x, dir_selection_method=None):
    c0 = np.array([coordinate[0] for coordinate in x])
    max_abs_val0 = max(abs(min(c0)), max(c0))
    max_abs_val0 = max(max_abs_val0,MIN_AXIS_REACH)
    max_abs_val0+=EXTRA_PLOT_SPACE
    linspace0 = np.linspace(-max_abs_val0, max_abs_val0)
    c1 = np.array([coordinate[1] for coordinate in x])
    max_abs_val1 = max(abs(min(c1)), max(c1))
    max_abs_val1 = max(max_abs_val1, MIN_AXIS_REACH)
    max_abs_val1+=EXTRA_PLOT_SPACE
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
    title_dir_suffix = f' ({dir_selection_method})' if dir_selection_method else ''
    ax.set_title(f'Contour lines and objective values {f.__name__}{title_dir_suffix}')
    ax.scatter(c0, c1)

    fig_name_dir_suffix = f'_{dir_selection_method}' if dir_selection_method else ''
    save_fig(f'{f.__name__}{fig_name_dir_suffix}.png')

def plot_iter_num_to_obj_val(f, iter_num_to_obj_val, dir_selection_method=None):
    plt.clf()
    x, y = zip(*iter_num_to_obj_val.items())
    plt.scatter(x, y)
    title_dir_suffix = f' ({dir_selection_method})' if dir_selection_method else ''
    plt.title(f'Iterations to objective values {f.__name__}{title_dir_suffix}')
    plt.xlabel("Iteration no.")
    plt.ylabel("Objective function value")
    fig_name_dir_suffix = f'_{dir_selection_method}' if dir_selection_method else ''
    save_fig(f'{f.__name__}{fig_name_dir_suffix}_iterations_to_obj_val.png')


def report_iteration(iter_count, cur_location, cur_obj_val, cur_step_len, cur_obj_f_val_change):
    print(f'Iteration number: {iter_count} current location: {cur_location} current obj val: {cur_obj_val} current step length: {cur_step_len} current change in objective function value: {cur_obj_f_val_change}')
