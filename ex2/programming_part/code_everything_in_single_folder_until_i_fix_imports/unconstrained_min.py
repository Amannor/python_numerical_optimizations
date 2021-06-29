import numpy as np
# from . import utils
import utils
from collections import OrderedDict

#Given consts
DEFAULT_INIT_STEP_LEN = 1.0
DEFAULT_SLOPE_RATIO = 10**-4
DEFAULT_BACKTRACK_FACTOR = 0.2

#My consts
DEFAULT_EPSILON = 10 ** -3

def check_converge(cur_param_val, param_tol, cur_obj_val, obj_tol):
	return cur_param_val<=param_tol or cur_obj_val<=obj_tol

def get_step_len_by_first_wolfe(f, df_val_vector, xk, alpha, pk, c1, back_track_factor):
	#From lecture 3. slides 16-19: the loop stops iff 𝑓(𝑥_𝑘+𝛼*𝑝_𝑘)≤𝑓(𝑥_𝑘)+𝑐_1*𝛼∇𝑓(𝑥_𝑘).𝑇*𝑝_𝑘
	while not f(xk+alpha*pk)[0] <= f(xk)[0]+c1*alpha*df_val_vector.T@pk: 
		alpha*=back_track_factor
	# print(f'{chr(945)}: {alpha}')
	return alpha


def get_next_B_matrix(B_k, x_k, x_k_plus_1, df_k, df_k_plus_1):
	sk = x_k_plus_1 - x_k   #Lecture 3 slide 30
	yk = df_k_plus_1 - df_k #Lecture 3 slide 30
	
	# Adjusting shape of sk,yk (adding another dimension of size 1) to make the multiplication work (taken from: https://stackoverflow.com/a/22737220)
	if len(sk.shape) == 1:
		sk =  sk[:, np.newaxis]
	if len(yk.shape) == 1:
		yk =  yk[:, np.newaxis]

	lhs_nominator = B_k@sk@sk.T@B_k
	lhs_denominator = sk.T@B_k@sk
	rhs_nominator = yk@yk.T
	rhs_denominator = yk.T@sk
	B_k_plus_1 = B_k - lhs_nominator/lhs_denominator + rhs_nominator/rhs_denominator #Lecture 3 slide 33
	return B_k_plus_1

def should_stop_by_newton_decrement(pnt, hessian, epsilon=DEFAULT_EPSILON):
	newton_decrement = pnt.T @ hessian @ pnt #Lecture 7+8 slide 36
	return 0.5 * newton_decrement < epsilon

def bfgs_dir(f, x0, step_size, obj_tol, param_tol, max_iter, init_step_len, slope_ratio, back_track_factor):
	x_vals = []
	y_vals = []
	x_prev = x0
	f_prev, df_prev, hessian = f(x0, True)
	x_vals.append(x_prev)
	y_vals.append(f_prev)
	i = 0
	success = False
	utils.report_iteration(i, x_prev, f_prev, float("NaN"), float("NaN"))
	iter_num_to_obj_val = OrderedDict()
	iter_num_to_obj_val[i] = f_prev
	B_prev = np.eye(len(x0))
	while not success and i < max_iter:
		success = should_stop_by_newton_decrement(x_prev, hessian)

		pk = np.linalg.solve(B_prev, -df_prev) # Lecture 2 slide 40: 𝑝𝑘≔−∇^2(𝑓(𝑥_𝑘))^(−1)∇𝑓(𝑥_𝑘) + lecture 3 slide 7
		step_len = get_step_len_by_first_wolfe(f, df_prev, x_prev, init_step_len, pk, slope_ratio, back_track_factor)
		x_next = x_prev +step_len*pk
		f_next, df_next, hessian = f(x_next, True)
		i += 1
		iter_num_to_obj_val[i]=f_next
		cur_obj_val = abs(f_next - f_prev)
		cur_param_val = np.linalg.norm(x_next - x_prev)
		utils.report_iteration(i, x_next, f_next, cur_param_val, cur_obj_val)
		success = success or check_converge(cur_param_val, param_tol, cur_obj_val, obj_tol)

		B_prev = get_next_B_matrix(B_prev, x_prev, x_next, df_prev, df_next) #def get_next_B_matrix(B_k, x_k, x_k_plus_1, df_k, df_k_plus_1):
		x_prev = x_next
		f_prev = f_next
		df_prev = df_next
		x_vals.append(x_prev)
		y_vals.append(f_prev)
	print(f'Function {f.__name__} (bfgs) final success status: {"Success" if success else "Fail"}')
	return x_next, success, x_vals, iter_num_to_obj_val

def newton_dir(f, x0, step_size, obj_tol, param_tol, max_iter, init_step_len, slope_ratio, back_track_factor):
	x_vals = []
	y_vals = []
	x_prev = x0
	f_prev, df_prev, dff_prev = f(x0, True)
	x_vals.append(x_prev)
	y_vals.append(f_prev)
	i = 0
	success = False
	utils.report_iteration(i, x_prev, f_prev, float("NaN"), float("NaN"))
	iter_num_to_obj_val = OrderedDict()
	iter_num_to_obj_val[i+1] = f_prev
	while not success and i < max_iter:
		pk = np.linalg.solve(dff_prev, -df_prev) # Lecture 2 slide 40: 𝑝𝑘≔−∇^2(𝑓(𝑥_𝑘))^(−1)∇𝑓(𝑥_𝑘) + lecture 3 slide 7
		step_len = get_step_len_by_first_wolfe(f, df_prev, x_prev, init_step_len, pk, slope_ratio, back_track_factor)
		x_next = x_prev +step_len*pk
		f_next, df_next, dff_next = f(x_next, True)
		i += 1
		iter_num_to_obj_val[i+1]=f_next
		cur_obj_val = abs(f_next - f_prev)
		cur_param_val = np.linalg.norm(x_next - x_prev)
		utils.report_iteration(i, x_next, f_next, cur_param_val, cur_obj_val)
		success = check_converge(cur_param_val, param_tol, cur_obj_val, obj_tol) 
		x_prev = x_next
		f_prev = f_next
		df_prev = df_next
		x_vals.append(x_prev)
		y_vals.append(f_prev)
	print(f'Function {f.__name__} (Newton) final success status: {"Success" if success else "Fail"}')
	return x_next, success, x_vals, iter_num_to_obj_val


def gd_dir(f, x0, step_size, obj_tol, param_tol, max_iter):
	x_vals = []
	y_vals = []
	x_prev = x0
	f_prev, df_prev = f(x0)
	x_vals.append(x_prev)
	y_vals.append(f_prev)
	i = 0
	success = False
	utils.report_iteration(i, x_prev, f_prev, float("NaN"), float("NaN"))
	iter_num_to_obj_val = OrderedDict()
	iter_num_to_obj_val[i] = f_prev
	while not success and i < max_iter:
		pk = -step_size*df_prev
		x_next = x_prev+pk
		f_next, df_next = f(x_next)
		i += 1
		iter_num_to_obj_val[i]=f_next
		cur_obj_val = abs(f_next - f_prev)
		cur_param_val = np.linalg.norm(x_next - x_prev)
		utils.report_iteration(i, x_next, f_next, cur_param_val, cur_obj_val)
		success = check_converge(cur_param_val, param_tol, cur_obj_val, obj_tol)
		x_prev = x_next
		f_prev = f_next
		df_prev = df_next
		x_vals.append(x_prev)
		y_vals.append(f_prev)
	print(f'Function {f.__name__} (GD) final success status: {"Success" if success else "Fail"}')
	return x_next, success, x_vals, iter_num_to_obj_val

# def gradient_descent(f, x0, step_size, obj_tol, param_tol, max_iter):#ex1 line
def line_search(f, x0, step_size, obj_tol, param_tol, max_iter, dir_selection_method, init_step_len=DEFAULT_INIT_STEP_LEN, slope_ratio=DEFAULT_SLOPE_RATIO, back_track_factor=DEFAULT_BACKTRACK_FACTOR):
	'''
	f is the function minimized
	x0 is the starting point
	step_size is the coefficient multiplying the gradient vector in the algorithm update rule
	obj_tol is the numeric tolerance for successful termination in terms of small enough change in objective function values, between two consecutive iterations (𝑓(𝑥𝑖+1) and 𝑓(𝑥𝑖)).
	param_tol is the numeric tolerance for successful termination in terms of small enough distance between two consecutive iterations iteration locations (𝑥𝑖+1 and 𝑥𝑖).
	max_iter is the maximum allowed number of iterations.
	dir_selection_method one of three strings: ‘gd’, ‘nt’, ‘bfgs’, to specify the method for selecting the direction for line search (gradient descent, Newton or BFGS, respectively)
	'''
	if dir_selection_method =='gd':
		return gd_dir(f, x0, step_size, obj_tol, param_tol, max_iter)
	elif dir_selection_method =='nt':
		return newton_dir(f, x0, step_size, obj_tol, param_tol, max_iter, init_step_len, slope_ratio, back_track_factor)
	elif dir_selection_method =='bfgs':
		return bfgs_dir(f, x0, step_size, obj_tol, param_tol, max_iter, init_step_len, slope_ratio, back_track_factor)
	else:
		raise Exception('dir_selection_method param supplied with invalid value')
