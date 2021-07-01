import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import patheffects
import os

MIN_AXIS_REACH = 1.5
EXTRA_PLOT_SPACE = 0.2

def save_fig(f_name):
    path = os.path.join(os.getcwd(), 'output', f_name)
    plt.savefig(path)

def get_indexes_of_inequalities_hold(point ,ineq_constraints):
    res = []
    for i in range(len(ineq_constraints)):
        if ineq_constraints[i](point)<=0 :
            res.append(i)
    return res

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
    color_dict = {'b': 'blue', 'g': 'green', 'r': 'red', 'c': 'cyan', 'm': 'magenta','y': 'yellow', 'k': 'black'} #For full list see https://matplotlib.org/2.0.2/api/colors_api.html
    white_id = 'w'
    color_if_all_hold = list(color_dict.keys())[-1]
    ineq_to_color = {}
    for y_i in range(ylen):
        for x_i in range(xlen):
            x_val, y_val = X_range[x_i], Y_range[y_i]
            point = np.array([x_val, y_val, qp_equality_func_for_xy(x_val, y_val)])
            ineq_indexes = get_indexes_of_inequalities_hold(point, ineq_constraints)
            if len(ineq_indexes) == len(ineq_constraints):
                colors[x_i, y_i] = color_if_all_hold
            else:
                colors[x_i, y_i] = white_id

            #Uncomment to color differnt inequilities regions in distinct colors
            # elif len(ineq_indexes)>0:
            #     ineq_i = ineq_indexes[0]
            #     ineq_str = ineq_constraints[ineq_i](None, return_str_rep=True) 
            #     cur_color = list(color_dict.keys())[ineq_i % len(color_dict)-1]
            #     # colors[x_i, y_i] = cur_color
                
            #     if not ineq_str in ineq_to_color:
            #         ineq_to_color[ineq_str] = color_dict[cur_color]

    #Uncomment to color differnt inequilities regions in distinct colors
    # text2d = "Ineqaulities to colors mapping:"
    # for ineq_str in sorted(ineq_to_color.keys()):
    #     text2d+=f'\n{ineq_str}: {ineq_to_color[ineq_str]}'
    # ax.text2D(0.05, 0.95, text2d, transform=ax.transAxes)
    
    ax.plot_surface(X, Y, Z, facecolors=colors)
    ax.scatter([c[0] for c in x_vals], [c[1] for c in x_vals], [c[2] for c in x_vals], c="g")
    ax.plot([c[0] for c in x_vals], [c[1] for c in x_vals], [c[2] for c in x_vals], c="g")
    ax.set_title(f'Equality constraint (x+y+z=1) and feasible region (in {color_dict[color_if_all_hold]}) {f.__name__}')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    save_fig(f.__name__)

def plot_for_lp(f, x_vals, ineq_constraints):
    
    f_0 = lambda x,y: f(np.array([x,y]))[0]
    f_1 = lambda x,y: ineq_constraints[0](np.array([x,y]))
    f_2 = lambda x,y: ineq_constraints[1](np.array([x,y]))
    f_3 = lambda x,y: ineq_constraints[2](np.array([x,y]))
    f_4 = lambda x,y: ineq_constraints[3](np.array([x,y]))
    
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
    plt.plot([c[0] for c in x_vals], [c[1] for c in x_vals])
    plt.annotate(f'(x0)', (x_vals[0][0], x_vals[0][1]))
    # plt.annotate(f'(x{len(x_vals)-1})', (x_vals[-1][0], x_vals[-1][1]))

    # for i, x_val in enumerate(x_vals):
    #     plt.annotate(f'(x{i})', (x_val[0], x_val[1]))

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
    ax.plot(c0, c1)

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


def report_iteration(iter_count, cur_location, cur_obj_val, cur_step_len, cur_obj_f_val_change, dir_method=None):
    dir_method_suffix_str = f' ({dir_method})' if dir_method else ""
    print(f'Iteration number: {iter_count} current location: {cur_location} current obj val: {cur_obj_val} current step length: {cur_step_len} current change in objective function value: {cur_obj_f_val_change}{dir_method_suffix_str}')
