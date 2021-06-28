import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import patheffects
import os

MIN_AXIS_REACH = 1.5
# PARENT_DIR = os.path.dirname(os.getcwd())

def save_fig(f_name):
    path = os.path.join(os.getcwd(), 'output', f_name)
    plt.savefig(path)



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
    # d = np.linspace(-1,3,200)
    d = np.linspace(-1,3)
    x,y = np.meshgrid(d,d)

    im = plt.imshow( ((f_1(x,y)<=0) & (f_2(x,y)<=0) & (f_3(x,y)<=0) & (f_4(x,y)<=0)).astype(int), 
                    extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys")

    plt.scatter([c[0] for c in x_vals], [c[1] for c in x_vals])

    save_fig(f.__name__)

def plot_for_lp_first_take(f, x_vals, ineq_constraints):
    #From: https://matplotlib.org/devdocs/gallery/images_contours_and_fields/contours_in_optimization_demo.html
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set up survey vectors
    xvec = np.linspace(-0.1, 2.1)
    yvec = np.linspace(-0.1, 1.1)

    # Set up survey matrices.  Design disk loading and gear ratio.
    x1, x2 = np.meshgrid(xvec, yvec)

    # Evaluate some stuff to plot
    obj = -x1-x2
    g1 = -x1-x2+1
    g2 = x2-1
    g3 = x1-2
    g4 = -x2

    cntr = ax.contour(x1, x2, obj, [0.01, 0.1, 0.5, 1, 2, 4, 8, 16],
                      colors='black')
    ax.clabel(cntr, fmt="%2.1f", use_clabeltext=True)

    cg1 = ax.contour(x1, x2, g1, [0], colors='sandybrown')
    # plt.setp(cg1.collections, path_effects=[patheffects.withTickedStroke(angle=135)])
    plt.setp(cg1.collections)

    cg2 = ax.contour(x1, x2, g2, [0], colors='orangered')
    # plt.setp(cg2.collections, path_effects=[patheffects.withTickedStroke(angle=60, length=2)])
    plt.setp(cg2.collections)

    cg3 = ax.contour(x1, x2, g3, [0], colors='mediumblue')
    # plt.setp(cg3.collections, path_effects=[patheffects.withTickedStroke(spacing=7)])
    plt.setp(cg3.collections)
    
    cg4 = ax.contour(x1, x2, g4, [0], colors='mediumblue')
    # plt.setp(cg4.collections,   path_effects=[patheffects.withTickedStroke(spacing=7)])
    plt.setp(cg4.collections)
    
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    plt.show()
    # plot_contours_and_paths(f, x_vals)

    # f = lambda x,y : np.sqrt(2*x*y)-x-y
    # g = lambda x,y : np.sqrt(x**2+y**2)-2

    # d = np.linspace(-2,2,2000)
    # x,y = np.meshgrid(d,d)

    # im = plt.imshow( ((g(x,y)<=f(x,y)) & (f(x,y)<=1)).astype(int) , 
    #                 extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys")

    # plt.show()

def ineq_constraint_func_adapter(ineq_constraint, x, y):
    return ineq_constraint(np.array([x,y]))[0]



def plot_contours_and_paths(f, x, dir_selection_method=None, ineq_constraints=None):
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
    title_dir_suffix = f' ({dir_selection_method})' if dir_selection_method else ''
    ax.set_title(f'Contour lines and objective values {f.__name__}{title_dir_suffix}')
    ax.scatter(c0, c1)
    if ineq_constraints and len(ineq_constraints) == 4: #Note: hard-coded for our specific case since I didn't fund a way to that for arbitrary length
        x, y = np.meshgrid(np.linspace(-0.1, 2.1), np.linspace(-0.1, 1.1))
        im = ax.imshow( ((ineq_constraint_func_adapter(ineq_constraints[0], x, y)<=0) & (ineq_constraint_func_adapter(ineq_constraints[1], x, y)<=0) & (ineq_constraint_func_adapter(ineq_constraints[2], x, y)<=0) & (ineq_constraint_func_adapter(ineq_constraints[3], x, y)<=0)).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys")

    fig_name_dir_suffix = f'_{dir_selection_method}' if dir_selection_method else ''
    save_fig(f'{f.__name__}{fig_name_dir_suffix}.png')

def plot_iter_num_to_obj_val(f, iter_num_to_obj_val, dir_selection_method):
    plt.clf()
    # print(f'type(iter_num_to_obj_val) {type(iter_num_to_obj_val)}')
    x, y = zip(*iter_num_to_obj_val.items())
    # plt.plot(x, y)
    plt.scatter(x, y)
    plt.title(f'Iterations to objective values {f.__name__} ({dir_selection_method})')
    plt.xlabel("Iteration no.")
    plt.ylabel("Objective function value")
    # plt.show()
    save_fig(f'{f.__name__}_{dir_selection_method}_iterations_to_obj_val.png')


def report_iteration(iter_count, cur_location, cur_obj_val, cur_step_len, cur_obj_f_val_change):
    print(f'Iteration number: {iter_count} current location: {cur_location} current obj val: {cur_obj_val} current step length: {cur_step_len} current change in objective function value: {cur_obj_f_val_change}')
