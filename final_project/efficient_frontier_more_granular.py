import cvxpy as cp
import numpy as np
import pickle
import os
import math
import matplotlib.pyplot as plt

### Consts ###
PLT_COLORS = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"] #From: https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html
KAPPA_MIN, KAPPA_MAX = 0, 5
KAPPA_MIN+=0.001 #Doesn't handle 0 well (the problem becomes linear programming and that's a whole nother ball game)
KAPPA_MAX+=0.001
MIN_VAL_TO_PLOT = 10**-4
TOLERANCE_FOR_SANITY = 10**-14
PLOTS_DIR = "plots"


### Loading data ###
eligible_assets_ids = pickle.load( open( os.path.join("concise_data", "eligible_assets_ids.pkl"), "rb" ) )
cov_matrix = pickle.load( open( os.path.join("concise_data", "cov_matrix.pkl"), "rb" ) )
assets_avg_yearly_yields = pickle.load( open( os.path.join("concise_data", "assets_avg_yearly_yields.pkl"), "rb" ) )
n = len(assets_avg_yearly_yields)
x_unif = np.full((n), 1/n)


def objective_function(x, mu, G):
    #Note: G is alreay multiplied by kappa
    x = np.array(x)
    mu = np.array(mu)
    G = np.array(G)
    return -x.T.dot(mu)+x.T.dot(G).dot(x)


def get_min_val_and_index(q,P):
    min_val = math.inf
    i_for_min = -1
    for i in range(n):
        x = np.zeros(n)
        x[i] = 1
        obj_val = objective_function(x, q, P)
        if(obj_val<min_val):
            min_val, i_for_min = obj_val, i
    return min_val,i_for_min

def get_sanity_tests_results(x_vec, opt_val_solver, opt_val_direct, obj_val_unif, obj_val_one_hot_min):
    sanity_fails= []
    if abs(sum(x_vec)-1)>=TOLERANCE_FOR_SANITY:
        sanity_fails.append(f'Sum does not add up to 1 (difference is {abs(sum(x_vec)-1)})')
    for i in range(len(x_vec)):
        if x_vec[i]<-TOLERANCE_FOR_SANITY:
            sanity_fails.append(f'x_vec[{i}] is negative')
            break
    if abs(opt_val_solver-opt_val_direct)>=TOLERANCE_FOR_SANITY:
        sanity_fails.append(f'Returned optimal results differ (by {abs(opt_val_solver-opt_val_direct)})')
    if opt_val_solver-TOLERANCE_FOR_SANITY>obj_val_unif:
        sanity_fails.append("Optimal result worse than uniform x")
    if opt_val_solver-TOLERANCE_FOR_SANITY>obj_val_one_hot_min:
        sanity_fails.append("Optimal result worse than min one-hot")
    return sanity_fails

#Based on: https://www.cvxpy.org/examples/basic/quadratic_program.html
# Define the CVXPY problem.
q = np.array(assets_avg_yearly_yields) #This is mu

x = cp.Variable(n)
constraints = []
constraints.append(x >= 0)
constraints.append(cp.sum(x) == 1) 

x_coordinates_to_max_y = {}
kappa_to_obj_vals = {}
plt.figure(figsize=(12.0, 7.5))


def get_kappa_to_show(kappa):
    kappa_to_show = "{:.3f}".format(kappa)
    return kappa_to_show

for kappa, color_index in zip(np.linspace(KAPPA_MIN,KAPPA_MAX,len(PLT_COLORS)), range(len(PLT_COLORS))):
    kappa_to_show = get_kappa_to_show(kappa)
    print(f'kappa_to_show {kappa_to_show} color_index {color_index}')
    kappa_to_obj_vals[kappa] = {}
    kappa = float(kappa)
    P = (kappa)*cov_matrix
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) - q.T @ x), constraints)
    prob.solve()
    opt_val_solver = prob.value
    opt_val_direct = objective_function(x.value, q, P)
    obj_val_unif = objective_function(x_unif, q, P)
    obj_val_one_hot_min = get_min_val_and_index(q,P)[0]
    sanity_fails = get_sanity_tests_results(x.value, opt_val_solver, opt_val_direct, obj_val_unif, obj_val_one_hot_min)
    if(len(sanity_fails)>0):
        print("***Sanity failed***")
        print(f'Errors: {sanity_fails}')
    # print(f'opt_val_solver {opt_val_solver} opt_val_direct {opt_val_direct} obj_val_unif {obj_val_unif} obj_val_one_hot_min {obj_val_one_hot_min} sum(x) {sum(x.value)} min(x) {min(x.value)}')
    kappa_to_obj_vals[kappa]["opt_val"] = opt_val_solver
    kappa_to_obj_vals[kappa]["unif_val"] = obj_val_unif
    kappa_to_obj_vals[kappa]["one_hot_min_val"] = obj_val_one_hot_min
    points = [p for p in zip(range(1,n+1), x.value) if p[1]>=MIN_VAL_TO_PLOT]
    for p in points:
        cur_x, cur_y = p[0], p[1]
        if not cur_x in x_coordinates_to_max_y:
            x_coordinates_to_max_y[cur_x] = cur_y
        elif x_coordinates_to_max_y[cur_x] < cur_y:
            x_coordinates_to_max_y[cur_x] = cur_y
    plural_suffix = "s" if len(x_coordinates_to_max_y)>1 else ""
    plt.scatter(*zip(*points), color=PLT_COLORS[color_index], label=f'\u03BA={kappa_to_show} ({len(x_coordinates_to_max_y)} asset{plural_suffix})')

for x_coordinate in x_coordinates_to_max_y:
    plt.annotate(eligible_assets_ids[x_coordinate-1], (x_coordinate+0.01, x_coordinates_to_max_y[x_coordinate]+0.01))

# plt.title(f'Asset distribution in optimal portfolio for risk-aversion (\u03BA) values\n(Higher values of \u03BA represent higher risk-aversion)')
# plt.xlabel("Assets IDs")
# h = plt.ylabel("% of\nportfolio")
# h.set_rotation(0)
# plt.legend(loc=2)
# plt.savefig(os.path.join(PLOTS_DIR, "Asset_distribution_in_optimal_portfolio_for_risk_aversion_values"))
# # plt.show()
# plt.clf()


# kappa_to_obj_val = {}
# kappa_to_unif_val = {}
# kappa_to_min_one_hot_val = {}
# for kappa in np.linspace(KAPPA_MIN,KAPPA_MAX,len(PLT_COLORS)):
#     kappa_to_show = get_kappa_to_show(kappa)
#     kappa_to_obj_val[kappa_to_show] = -kappa_to_obj_vals[kappa]["opt_val"]
#     kappa_to_unif_val[kappa_to_show] = -kappa_to_obj_vals[kappa]["unif_val"]
#     kappa_to_min_one_hot_val[kappa_to_show] = -kappa_to_obj_vals[kappa]["one_hot_min_val"]

# plt.scatter(*zip(*kappa_to_obj_val.items()), color=PLT_COLORS[0], label="optimal values")
# plt.scatter(*zip(*kappa_to_unif_val.items()), color=PLT_COLORS[1], label="uniform x values")
# plt.scatter(*zip(*kappa_to_min_one_hot_val.items()), color=PLT_COLORS[2], label="min one hot x values")
# plt.title(f'\u03BA values to objective value (overall expected yearly return)')
# plt.xlabel("\u03BA values")
# h = plt.ylabel("Overall\nexpected\nreturn")
# h.set_rotation(0)
# plt.legend()
# # plt.show()
# plt.savefig(os.path.join(PLOTS_DIR, f'kappa_to_expected_returns_for_kappa_from_{round(KAPPA_MIN)}_to_{round(KAPPA_MAX)}'))
# plt.clf()

# plt.plot(*zip(*kappa_to_obj_val.items()), color=PLT_COLORS[0], label="optimal values")
# plt.title(f'\u03BA values to objective value (overall expected yearly return)\n Efficient frontier')
# plt.xlabel("\u03BA values")
# h = plt.ylabel("Overall\nexpected\nreturn", labelpad=25)
# h.set_rotation(0)
# plt.legend()
# # plt.show()
# plt.savefig(os.path.join(PLOTS_DIR, f'kappa_to_expected_returns_for_efficient_frontier'))
# plt.clf()


# kappa_to_obj_vals = {}
# kappa_to_unif_vals = {}
# for kappa in np.linspace(KAPPA_MIN,KAPPA_MAX,500*len(PLT_COLORS)):
#     kappa_to_show = get_kappa_to_show(kappa)
#     print(f'kappa_to_show {kappa_to_show}')
#     kappa = float(kappa)
#     P = (kappa)*cov_matrix
#     prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) - q.T @ x), constraints)
#     prob.solve()
#     opt_val_solver = -prob.value
#     kappa_to_obj_vals[1/kappa] = opt_val_solver
#     obj_val_unif = objective_function(x_unif, q, P)
#     kappa_to_unif_vals[1/kappa] = -obj_val_unif




# plt.plot(*zip(*kappa_to_obj_vals.items()), color=PLT_COLORS[0], label="Optimal values")
# plt.plot(*zip(*kappa_to_unif_vals.items()), color=PLT_COLORS[1], label="Values for unifrom distribution")
# plt.title(f'\u03BA values to objective value (overall expected yearly return)\n Efficient frontier')
# plt.xlabel("Risk (in reciprocal of \u03BA values)")
# h = plt.ylabel("Overall\nexpected\nreturn", labelpad=25)
# h.set_rotation(0)
# plt.legend()
# # plt.show()
# plt.savefig(os.path.join(PLOTS_DIR, f'kappa_to_expected_returns_for_efficient_frontier_500_granular'))
# plt.clf()


plt.clf()
KAPPA_MIN = 0.00001
KAPPA_MAX = 1.1

kappa_to_obj_vals = {}
kappa_to_unif_vals = {}
for kappa in np.linspace(KAPPA_MIN,KAPPA_MAX,20*len(PLT_COLORS)):
    kappa_to_show = get_kappa_to_show(kappa)
    print(f'kappa_to_show {kappa_to_show}')
    kappa = float(kappa)
    P = (kappa)*cov_matrix
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) - q.T @ x), constraints)
    prob.solve()
    opt_val_solver = -prob.value
    kappa_to_obj_vals[kappa] = opt_val_solver
    obj_val_unif = objective_function(x_unif, q, P)
    kappa_to_unif_vals[kappa] = -obj_val_unif




plt.plot(*zip(*kappa_to_obj_vals.items()), color=PLT_COLORS[0], label="Optimal values")
plt.plot(*zip(*kappa_to_unif_vals.items()), color=PLT_COLORS[1], label="Values for unifrom distribution")
plt.title(f'\u03BA values to objective value (overall expected yearly return)\n Efficient frontier')
plt.xlabel("Risk (higher \u03BA means higher risk)")
h = plt.ylabel("Overall\nexpected\nreturn", labelpad=25)
h.set_rotation(0)
plt.legend()
# plt.show()
plt.savefig(os.path.join(PLOTS_DIR, f'kappa_to_expected_returns_for_efficient_frontier_kappa_from_0_to_1_1_granularity_20'))
plt.clf()
