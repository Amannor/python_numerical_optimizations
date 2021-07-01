import numpy as np
import utils

#Given consts
DEFAULT_T = 1
DEFAULT_MU = 10
DEFAULT_INIT_STEP_LEN = 1.0
DEFAULT_SLOPE_RATIO = 10**-4
DEFAULT_BACKTRACK_FACTOR = 0.2

#My consts
DEFAULT_EPSILON = 10 ** -3
MAX_ITER_NUM = 2000
DEFAULT_STEP_TOLERANCE = 10**-16# Setting to 10**-15 will result in one less iteration
DEFAULT_OBJ_TOLERANCE = 10**-12

def newton_func_lp(x, t, should_calc_hessian=False):
    f_val = -t*x[0]-t*x[1]-np.log(x[0]+x[1]-1)-np.log(1-x[1])-np.log(2-x[0])-np.log(x[1])
    x_coordinate = -t-1/(x[0]+x[1]-1)+1/(2-x[0])
    y_coordinate = -t-1/(x[0]+x[1]-1)+1/(1-x[1])-1/x[1]
    grad_vector_val = np.array([x_coordinate, y_coordinate])
    if should_calc_hessian:
        hessian = np.diag([1/((2-x[0])**2), 1/((1-x[1])**2)+1/(x[1]**2)])
        hessian += 1/((x[0]+x[1]-1)**2)
        return f_val, grad_vector_val, hessian
    else:
        return f_val, grad_vector_val, None

def newton_func_qp(x, t, should_calc_hessian=False):
    f_val = t*(x[0]**2+x[1]**2+(x[2]+1)**2)-np.log(x[0])-np.log(x[1])-np.log(x[2])
    x_coordinate = 2*t*x[0]-1/x[0]
    y_coordinate = 2*t*x[1]-1/x[1]
    z_coordinate = 2*t*(x[2]+1)-1/x[2]
    grad_vector_val = np.array([x_coordinate, y_coordinate, z_coordinate])
    if should_calc_hessian:
        hessian = np.diag([1/((x[0])**2), 1/((x[1])**2), 1/((x[2])**2)])
        hessian += 2*t
        return f_val, grad_vector_val, hessian
    else:
        return f_val, grad_vector_val, None


#This method is identical to the one in unconstraind_min.py but is here for (in)dependency reasons
def get_step_len_by_first_wolfe(f, df_val_vector, xk, pk, t, alpha=DEFAULT_INIT_STEP_LEN, c1=DEFAULT_SLOPE_RATIO, back_track_factor=DEFAULT_BACKTRACK_FACTOR):
# def get_step_len_by_first_wolfe(f, df_val_vector, xk, pk, alpha=DEFAULT_INIT_STEP_LEN, c1=DEFAULT_SLOPE_RATIO, back_track_factor=DEFAULT_BACKTRACK_FACTOR):
    #From lecture 3. slides 16-19: the loop stops iff 𝑓(𝑥_𝑘+𝛼*𝑝_𝑘)≤𝑓(𝑥_𝑘)+𝑐_1*𝛼∇𝑓(𝑥_𝑘).𝑇*𝑝_𝑘
    '''
    TODO - since it's possible that we'll send f an invalid input (e.g. non-positive number for np.log() function - need to add a check for that
    Something like:
    import math 
    invalid_input = math.isnan(f(xk+alpha*pk, t)[0]) or math.isnan(f(xk, t)[0])
    while invalid_input or not f(xk+alpha*pk, t)[0] <= f(xk, t)[0]+c1*alpha*df_val_vector.T@pk:
        alpha*=back_track_factor
        invalid_input = math.isnan(f(xk+alpha*pk, t)[0]) or math.isnan(f(xk, t)[0])
    return alpha
    '''
    # while not f(xk+alpha*pk, should_return_only_val=True)[0] <= f(xk, should_return_only_val=True)[0]+c1*alpha*df_val_vector.T@pk:
    while not f(xk+alpha*pk, t)[0] <= f(xk, t)[0]+c1*alpha*df_val_vector.T@pk:
        alpha*=back_track_factor
    return alpha

def calc_newton_decrment(pnt, dfdfx):
    return np.sqrt(pnt.T @ dfdfx @ pnt)

def get_matrix_in_adjusted_shape(mat):
    # Adjusting shape of mat (adding another dimension of size 1 if needed) to make the multiplication work (taken from: https://stackoverflow.com/a/22737220)
    if len(mat.shape) == 1:
        mat =  mat[:, np.newaxis]
    return mat


def check_converge(cur_param_val, cur_obj_val, param_tol=DEFAULT_STEP_TOLERANCE, obj_tol=DEFAULT_OBJ_TOLERANCE):
    return cur_param_val<=param_tol or cur_obj_val<=obj_tol

def newton_method(func, x, n, eq_constraints_mat, epsilon, max_iter, t):
    should_stop = False
    j=0
    x_vals = []
    # f_prev = func(x, should_return_only_val=True)[0]
    f_prev = func(x, t)[0]
    report_prefix = "Newton method inside log-barrier "
    utils.report_iteration(j, x, f_prev, float("NaN"), float("NaN"), special_prefix=report_prefix)

    while (not should_stop and j<max_iter):
        j+=1
        # val_f, grad_f, hessian = func(x, should_calc_hessian=True) #func_for_newton = lambda x, should_calc_hessian=False, should_return_only_val=False
        val_f, grad_f, hessian = func(x, t, True) #func_for_newton = lambda x, should_calc_hessian=False, should_return_only_val=False
        if eq_constraints_mat is not None and len(eq_constraints_mat)>0:
            eq_constraints_mat = get_matrix_in_adjusted_shape(eq_constraints_mat)
            padding_size = eq_constraints_mat.shape[1]
            rhs = np.append(-grad_f, np.zeros([padding_size]))
            kkt_mat = np.append(hessian, eq_constraints_mat, axis=1) #https://stackoverflow.com/a/20688968
            bottom_part_kkt = np.append(eq_constraints_mat.T, np.zeros([padding_size, padding_size]), axis=1)
            kkt_mat = np.concatenate((kkt_mat, bottom_part_kkt))
        else:
            rhs = -grad_f
            kkt_mat = hessian 
        pnt = np.linalg.solve(kkt_mat, rhs)
        pnt = pnt[:n]
        newton_decrment = np.power(calc_newton_decrment(pnt, hessian), 2) #Lecture 7+8 slide 36
        if 0.5 * newton_decrment < epsilon:
            should_stop = True
        else:
            ak = get_step_len_by_first_wolfe(func, func(x, t)[1], x, pnt, t)
            # ak = get_step_len_by_first_wolfe(func, func(x)[1], x, pnt)
            x_prev = x
            x=x+ak*pnt
            x_vals.append(x)
            # f_next = func(x, should_return_only_val=True)[0]
            f_next = func(x, t)[0]
            cur_param_val_change = np.linalg.norm(x - x_prev)
            cur_obj_val_change = abs(f_next - f_prev)
            utils.report_iteration(j, x, f_next, cur_param_val_change, cur_obj_val_change, special_prefix=report_prefix)
            f_prev = func(x_prev, t)[0]
            # f_prev = func(x_prev, should_return_only_val=True)[0]
            should_stop = should_stop or check_converge(cur_param_val_change, cur_obj_val_change)

    return x_vals

# def func_pattern_for_newton(x, obj_func, ineq_constraints, t, should_calc_hessian=False, should_return_only_val=False):
#     f_val, grad_f, hessian = obj_func(x, should_calc_hessian)
#     f_val *= t
#     if not should_return_only_val:
#         grad_f *= t
#         if should_calc_hessian:
#             hessian *= t
#     for ineq_constraint_func in ineq_constraints:
#         ineq_f_val, ineq_grad_f, ineq_hessian = ineq_constraint_func(x, should_return_only_val)
#         f_val += ineq_f_val
#         if not should_return_only_val:
#             grad_f += ineq_grad_f
#             if should_calc_hessian:
#                 hessian += ineq_hessian
#     return f_val, grad_f, hessian



def barrier_method(func, ineq_constraints, eq_constraints_mat, x0, m, t = DEFAULT_T, mu = DEFAULT_MU, epsilon = DEFAULT_EPSILON, max_iter = MAX_ITER_NUM):
    success = False
    i=0
    x_vals = [x0]
    # x=x0

    while (not success and i<max_iter):
        x = x_vals[-1]
        # print(f'x {x}') newton_func_qp
        # func_for_newton = lambda x, should_calc_hessian=False, should_return_only_val=False: func_pattern_for_newton(x, func, ineq_constraints, t, should_calc_hessian, should_return_only_val)
        func_for_newton = newton_func_qp if 'qp' in  func.__name__ else newton_func_lp
        cur_x_vals = newton_method(func_for_newton, x, m, eq_constraints_mat, epsilon, max_iter, t)
        # print(f'Inside log-barrier outer-loop (no. {i}), got back {len(cur_x_vals)} from inner-loop')
        x_vals.extend(cur_x_vals)
        # print(f'i {i} x_vals {x_vals}')
        #The following is from lecture 7+8. slide 64
        if (m / t < epsilon):
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
    
    m = len(ineq_constraints)
    x_vals, success = barrier_method(func, ineq_constraints, eq_constraints_mat, x0, m)
    last_point = x_vals[-1]
    obj_val = func(last_point)[0]
    constraints_vals = [ineq_func(last_point)[0] for ineq_func in ineq_constraints]

    return x_vals, success, obj_val, constraints_vals