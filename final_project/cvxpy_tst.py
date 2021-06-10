import cvxpy as cp
import numpy as np
import pickle
import os
import math

eligible_assets_ids = pickle.load( open( os.path.join("concise_data", "eligible_assets_ids.pkl"), "rb" ) )
cov_matrix = pickle.load( open( os.path.join("concise_data", "cov_matrix.pkl"), "rb" ) )
assets_avg_yearly_yields = pickle.load( open( os.path.join("concise_data", "assets_avg_yearly_yields.pkl"), "rb" ) )
n = len(assets_avg_yearly_yields)


def objective_function(x, mu, G=cov_matrix, kappa=1):
    x = np.array(x)
    mu = np.array(mu)
    G = np.array(G)
    return -x.T.dot(mu)+kappa*x.T.dot(G).dot(x)


#Based on: https://www.cvxpy.org/examples/basic/quadratic_program.html
P = cov_matrix
q = np.array(assets_avg_yearly_yields) #This is mu
G = -np.identity(n)
h = np.zeros(n)
A = np.ones(n).reshape(1,n)
b = np.zeros(1)#.reshape(1)

# Define and solve the CVXPY problem.
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) - q.T @ x),
                 [G @ x <= h,
                  A @ x == b])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print(f'The optimal val back from obj {objective_function(x.value, q, P)}')
x_unif = np.full((n), 1/n)
print(f'The value for of obj for x_unif {objective_function(x_unif, q, P)}')
print(f'sum(x.value) {sum(x.value)}')
print("A solution x is")
print(x.value)
# print(f'x.value.shape {x.value.shape}')
# print(f'x_unif.shape {x_unif.shape}')



min_val, max_val = math.inf, -math.inf
i_for_min, i_for_max = -1, -1
for i in range(n):
    x = np.zeros(n)
    x[i] = 1
    obj_val = objective_function(x, q, P)
    if(obj_val<min_val):
        i_for_min, min_val = i, obj_val
    if(obj_val>max_val):
        i_for_max, max_val = i, obj_val
print(f'i_for_min: {i_for_min}, min_val {min_val}')
print(f'i_for_max: {i_for_max}, max_val {max_val}')