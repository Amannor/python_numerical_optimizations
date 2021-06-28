import numpy as np
import utils

#Given consts
DEFAULT_T = 1
DEFAULT_MU = 10
DEFAULT_INIT_STEP_LEN = 1.0
DEFAULT_SLOPE_RATIO = 10**-4
DEFAULT_BACKTRACK_FACTOR = 0.2

#My consts
DEFAULT_EPSILON = 10 ** -2
MAX_ITER_NUM = 2000


#This method is identical to the one in unconstraind_min.py but is here for (in)dependency reasons
def get_step_len_by_first_wolfe(f, df_val_vector, xk, pk, alpha=DEFAULT_INIT_STEP_LEN, c1=DEFAULT_SLOPE_RATIO, back_track_factor=DEFAULT_BACKTRACK_FACTOR):
    #From lecture 3. slides 16-19: the loop stops iff 𝑓(𝑥_𝑘+𝛼*𝑝_𝑘)≤𝑓(𝑥_𝑘)+𝑐_1*𝛼∇𝑓(𝑥_𝑘).𝑇*𝑝_𝑘
    while not f(xk+alpha*pk)[0] <= f(xk)[0]+c1*alpha*df_val_vector.T@pk: 
        alpha*=back_track_factor
    # print(f'{chr(945)}: {alpha}')
    return alpha

def calc_newton_decrment(pnt, dfdfx):
    return np.sqrt(pnt.T @ dfdfx @ pnt)

def get_concatenated_df_dff(func, x, ineq_constraints):
    func_res = func(x,True)
    grad_f, hessian = func_res[1], func_res[2]
    if not ineq_constraints:
        return grad_f, hessian
    for ineq_constraint in ineq_constraints:
        ineq_constraint_res = ineq_constraint(x)
        grad_f+=ineq_constraint_res[1] #Maybe ineq_constraint_res[1].T ??
        hessian+=ineq_constraint_res[2]

    return grad_f, hessian


def newton_method(func, x, n, epsilon, ineq_constraints, eq_constraints_mat, max_iter):
    success = False
    i=0
    x_vals = []
    f_prev = func(x)[0]
    utils.report_iteration(i, x, f_prev, float("NaN"), float("NaN"))

    while (not success and i<max_iter):
        x_vals.append(x)
        grad_f, hessian = get_concatenated_df_dff(func, x, ineq_constraints)
        if eq_constraints_mat and len(eq_constraints_mat)>0:
            rhs = np.append(-grad_f, 0)
            #TODO hessian
        else:
            rhs = -grad_f
            kkt_mat = hessian 
        pnt = np.linalg.solve(kkt_mat, rhs)
        pnt = pnt[:n]
        newton_decrment = np.power(calc_newton_decrment(pnt, hessian), 2) #Lecture 7+8 slide 36
        if (0.5 * newton_decrment < epsilon):
            success = True
        else:
            # ak = get_step_len_by_first_wolfe(func, func(x)[1], x, pnt) #TODO - calc this from first wolfe condition?
            ak = 1 
            x_prev = x
            x=x+ak*pnt
            f_next = func(x)[0]
            cur_param_val_change = np.linalg.norm(x - x_prev)
            cur_obj_val_change = abs(f_next - f_prev)
            utils.report_iteration(i+1, x, func(x)[0], cur_param_val_change, cur_obj_val_change)
            f_prev = func(x_prev)[0]
        i+=1

    return x_vals, success


def barrier_method(func, ineq_constraints, eq_constraints_mat, x0, m, t = DEFAULT_T, mu = DEFAULT_MU, epsilon = DEFAULT_EPSILON, max_iter = MAX_ITER_NUM):
    success = False
    i=0
    x_vals = []
    x=x0
    while (not success and i<max_iter):
        cur_x_vals, success = newton_method(func, x, m, epsilon, ineq_constraints, eq_constraints_mat, max_iter)
        x_vals.extend(cur_x_vals)
        #The following is from lecture 7+8. slide 64
        if (not success and m / t < epsilon):
            success = True
        else:
            t = t * mu
            i=i+1

    return x_vals, success 

def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
    '''
    Minimizes the function func, subject to constraints (using the log-barrier method)
    ineq_constraints: List of inequality constraints
    eq_constraints_mat: Matrix of affine equality constraints 𝐴𝑥=𝑏
    eq_constraints_rhs: The right hand side vector
    x0: Where the outer iterations start at
    '''
    
    #TODO add code that throws "unsopported infeasible starting point" exception in case x0 isn't feasible (i.e eq_constraints_mat.dot(x0) != eq_constraints_rhs)
    
    n = len(x0)
    x_vals, success = barrier_method(func, ineq_constraints, eq_constraints_mat, x0, n)
    last_point = x_vals[-1]
    obj_val = func(last_point)[0]
    constraints_vals = [ineq_func(last_point)[0] for ineq_func in ineq_constraints]

    return x_vals, success, obj_val, constraints_vals