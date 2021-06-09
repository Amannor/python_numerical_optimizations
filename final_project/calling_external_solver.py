import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import math

def objective_function(x, mu, kappa, G):
	return -x.T@mu+kappa*x.T@G@x


eligible_assets_ids = pickle.load( open( os.path.join("concise_data", "eligible_assets_ids.pickle"), "rb" ) )
cov_matrix = pickle.load( open( os.path.join("concise_data", "cov_matrix.pickle"), "rb" ) )
assets_avg_yearly_yields = pickle.load( open( os.path.join("concise_data", "assets_avg_yearly_yields.pickle"), "rb" ) )


PLT_COLORS = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"] #From: https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html

KAPPA_MIN, KAPPA_MAX = 0, 5

MIN_VAL_TO_PLOT = 10**-4

#From https://medium.com/tech-that-works/solving-quadratic-convex-optimization-problems-in-python-e378c7442494

from cvxopt import matrix, solvers

x_coordinates_to_max_y = {}
x_unif = np.full((len(assets_avg_yearly_yields), 1), 1/len(assets_avg_yearly_yields))
kappa_to_obj_vals = {}


for kappa, color_index in zip(np.linspace(KAPPA_MIN,KAPPA_MAX,len(PLT_COLORS)), range(len(PLT_COLORS))):
	kappa_to_obj_vals[kappa] = {}
	kappa = float(kappa)
	P = 2*kappa*matrix(cov_matrix)
	q = matrix(assets_avg_yearly_yields)
	G = matrix(-np.identity(len(assets_avg_yearly_yields)))
	h = matrix([0.0]*len(assets_avg_yearly_yields))
	A = matrix([1.0]*len(assets_avg_yearly_yields), (1,len(assets_avg_yearly_yields)))
	b = matrix([1.0])
	sol=solvers.qp(P, q, G, h, A, b)
	obj_val_unif = objective_function(x_unif, np.array(assets_avg_yearly_yields), kappa, cov_matrix)
	obj_val_optimal = objective_function(sol['x'], np.array(assets_avg_yearly_yields), kappa, cov_matrix)
	kappa_to_obj_vals[kappa]["obj_val_unif"] = obj_val_unif
	kappa_to_obj_vals[kappa]["obj_val_optimal"] = obj_val_optimal
	kappa_to_show = "{:.2f}".format(kappa)
	points = [p for p in zip(range(1,len(assets_avg_yearly_yields)+1), sol['x'][:]) if p[1]>=MIN_VAL_TO_PLOT]
	for p in points:
		cur_x, cur_y = p[0], p[1]
		if not cur_x in x_coordinates_to_max_y:
			x_coordinates_to_max_y[cur_x] = cur_y
		elif x_coordinates_to_max_y[cur_x] < cur_y:
			x_coordinates_to_max_y[cur_x] = cur_y
	# plt.plot(sol['x'],color=PLT_COLORS[color_index], label=f'\u03BA={kappa_to_show}')
	# plt.scatter(range(1,len(assets_avg_yearly_yields)+1), sol['x'][:],color=PLT_COLORS[color_index], label=f'\u03BA={kappa_to_show}')
	plural_suffix = "s" if len(x_coordinates_to_max_y)>1 else ""
	plt.scatter(*zip(*points), color=PLT_COLORS[color_index], label=f'\u03BA={kappa_to_show} ({len(x_coordinates_to_max_y)} asset{plural_suffix})')
for x_coordinate in x_coordinates_to_max_y:
	plt.annotate(eligible_assets_ids[x_coordinate-1], (x_coordinate+0.01, x_coordinates_to_max_y[x_coordinate]+0.01))

plt.title(f'Asset distribution in optimal portfolio for risk-aversion (\u03BA) values\n(Higher values of \u03BA represent higher risk-aversion)')
plt.xlabel("Assets IDs")
h = plt.ylabel("% of\nportfolio")
h.set_rotation(0)


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.legend()
plt.show()



for kappa in np.linspace(KAPPA_MIN,KAPPA_MAX,len(PLT_COLORS)):
	min_val, max_val = math.inf, -math.inf
	i_for_min, i_for_max = -1, -1
	for i in range(len(assets_avg_yearly_yields)):
		x = np.zeros(len(assets_avg_yearly_yields))
		x[i] = 1
		obj_val = objective_function(x, np.array(assets_avg_yearly_yields), kappa, cov_matrix)
		if(obj_val<min_val):
			i_for_min, min_val = i, obj_val
		if(obj_val>max_val):
			i_for_max, max_val = i, obj_val
	kappa_to_obj_vals[kappa]["one_hot_min_val"] = (i_for_min, min_val)
	kappa_to_obj_vals[kappa]["one_hot_max_val"] = (i_for_max, max_val)


for kappa, color_index in zip(kappa_to_obj_vals, range(len(PLT_COLORS))):
	kappa_to_show = "{:.2f}".format(kappa)
	print(f'***\u03BA={kappa_to_show}***')
	print(kappa_to_obj_vals[kappa])
	# plt.plot(kappa, kappa_to_obj_vals[kappa][0], color=PLT_COLORS[color_index])
	# plt.plot(kappa, kappa_to_obj_vals[kappa][1], color=PLT_COLORS[color_index])

#### Testing obj values for 1 hot values of x###




# Original
# P = 2*KAPPA*matrix([ [3.0, 1.0], [1.0, 4.0] ])
# q = matrix([5.0, 1.0])
# G = matrix([[-1.0,0.0],[0.0,-1.0]])
# h = matrix([0.0,0.0])
# A = matrix([1.0, 2.0], (1,2))
# b = matrix(1.0)
# sol=solvers.qp(P, q, G, h, A, b)
# print(sol['x'])


