import numpy as np
from . import utils
from collections import OrderedDict

DEFAULT_INIT_STEP_LEN = 1.0
DEFAULT_SLOPE_RATIO = 10**-4
DEFAULT_BACKTRACK_FACTOR = 0.2

def check_converge(cur_param_val, param_tol, cur_obj_val, obj_tol):
	return cur_param_val<=param_tol or cur_obj_val<=obj_tol

def does_meet_first_wolfe_condition(f, df_val_vector, xk, alpha, pk, c1):
	#From lecture 3. slides 16-19: this function returns true iff 𝑓(𝑥_𝑘+𝛼*𝑝_𝑘)≤𝑓(𝑥𝑘)+𝑐_1*𝛼∇𝑓(𝑥_𝑘).𝑇*𝑝_𝑘
	return f(xk+alpha*pk)[0] <=f(xk)+c1*alpha*df_val_vector.T@pk



def bfgs_dir():
	pass
	#TODO

def newton_dir(f, x0, step_size, obj_tol, param_tol, max_iter, dir_selection_method, step_len, slope_ratio, back_track_factor):
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
	pk = x0 #TODO - this is tmp of course
	while not success and i < max_iter:
		x_next = x_prev - step_size*df_prev
		f_next, df_next = f(x_next)
		i += 1
		iter_num_to_obj_val[i]=f_next
		cur_obj_val = abs(f_next - f_prev)
		cur_param_val = np.linalg.norm(x_next - x_prev)
		utils.report_iteration(i, x_next, f_next, cur_param_val, cur_obj_val)
		success = check_converge(cur_param_val, param_tol, cur_obj_val, obj_tol) || does_meet_first_wolfe_condition(f, df_prev, x_prev, step_len, pk, slope_ratio)
		x_prev = x_next
		f_prev = f_next
		df_prev = df_next
		x_vals.append(x_prev)
		y_vals.append(f_prev)
	print(f'Function {f.__name__} final success status: {"Success" if success else "Fail"}')
	return x_next, success, x_vals, iter_num_to_obj_val
	#TODO


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
		x_next = x_prev - step_size*df_prev
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
	print(f'Function {f.__name__} final success status: {"Success" if success else "Fail"}')
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
		print("Stam")
		return gd_dir(f, x0, step_size, obj_tol, param_tol, max_iter)
	elif dir_selection_method =='nt':
		print("Stam")
		return newton_dir(f, x0, step_size, obj_tol, param_tol, max_iter, dir_selection_method, init_step_len, slope_ratio, back_track_factor)
	elif dir_selection_method =='bfgs':
		print("Stam")
		return bfgs_dir()
	else:
		raise Exception('dir_selection_method param supplied with invalid value')

	# x_vals = []
	# y_vals = []
	# x_prev = x0
	# f_prev, df_prev = f(x0)
	# x_vals.append(x_prev)
	# y_vals.append(f_prev)
	# i = 0
	# success = False
	# utils.report_iteration(i, x_prev, f_prev, float("NaN"), float("NaN"))
	# while not success and i < max_iter:
	# 	x_next = x_prev - step_size*df_prev
	# 	f_next, df_next = f(x_next)
	# 	i += 1
	# 	cur_obj_val = abs(f_next - f_prev)
	# 	cur_param_val = np.linalg.norm(x_next - x_prev)
	# 	utils.report_iteration(i, x_next, f_next, cur_param_val, cur_obj_val)
	# 	success = check_converge(cur_param_val, param_tol, cur_obj_val, obj_tol)
	# 	x_prev = x_next
	# 	f_prev = f_next
	# 	df_prev = df_next
	# 	x_vals.append(x_prev)
	# 	y_vals.append(f_prev)
	# print(f'Function {f.__name__} final success status: {"Success" if success else "Fail"}')
	# return x_next, success, x_vals
