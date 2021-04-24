import numpy as np

from . import utils

def check_converge(cur_param_val, param_tol, cur_obj_val, obj_tol):
	return cur_param_val<=param_tol or cur_obj_val<=obj_tol


def gradient_descent(f, x0, step_size, obj_tol, param_tol, max_iter):
	'''
	f is the function minimized
	x0 is the starting point
	step_size is the coefficient multiplying the gradient vector in the algorithm update rule
	obj_tol is the numeric tolerance for successful termination in terms of small enough change in objective function values, between two consecutive iterations (𝑓(𝑥𝑖+1) and 𝑓(𝑥𝑖)).
	param_tol is the numeric tolerance for successful termination in terms of small enough distance between two consecutive iterations iteration locations (𝑥𝑖+1 and 𝑥𝑖).
	max_iter is the maximum allowed number of iterations.
	'''
	x_vals = []
	y_vals = []
	x_prev = x0
	f_prev, df_prev = f(x0)
	x_vals.append(x_prev)
	y_vals.append(f_prev)
	i = 0
	success = False
	utils.report_iteration(i, x_prev, f_prev, float("NaN"), float("NaN"))
	while not success and i < max_iter:
		x_next = x_prev - step_size*df_prev
		f_next, df_next = f(x_next)
		i += 1
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
	return x_next, success, x_vals


	# Todo: call the iteration reporting function in utils.py (send to it: the iteration number 𝑖, the current location 𝑥𝑖, the current objective value 𝑓(𝑥𝑖), the current step length taken ‖𝑥𝑖−𝑥𝑖−1‖ and the current change in objective function value |𝑓(𝑥𝑖)−𝑓(𝑥𝑖−1)|.)
