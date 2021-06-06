import pickle
import os
import numpy as np

eligible_assets_ids = pickle.load( open( os.path.join("concise_data", "eligible_assets_ids.pickle"), "rb" ) )
cov_matrix = pickle.load( open( os.path.join("concise_data", "cov_matrix.pickle"), "rb" ) )
assets_avg_yearly_yields = pickle.load( open( os.path.join("concise_data", "assets_avg_yearly_yields.pickle"), "rb" ) )


print(f'type(cov_matrix) {type(cov_matrix)}')
KAPPA = 1

#From https://medium.com/tech-that-works/solving-quadratic-convex-optimization-problems-in-python-e378c7442494

from cvxopt import matrix, solvers
P = 2*KAPPA*matrix(cov_matrix)
q = matrix(assets_avg_yearly_yields)
G = matrix(-np.identity(len(assets_avg_yearly_yields)))
h = matrix([0.0]*len(assets_avg_yearly_yields))
A = matrix([1.0]*len(assets_avg_yearly_yields), (1,len(assets_avg_yearly_yields)))
b = matrix([1.0])
sol=solvers.qp(P, q, G, h, A, b)
# print(sol['x'])
print(f'sum(sol[x]) {sum(sol["x"])}') #Sane test (to see it adds up to 1)


# Original
# P = 2*KAPPA*matrix([ [3.0, 1.0], [1.0, 4.0] ])
# q = matrix([5.0, 1.0])
# G = matrix([[-1.0,0.0],[0.0,-1.0]])
# h = matrix([0.0,0.0])
# A = matrix([1.0, 2.0], (1,2))
# b = matrix(1.0)
# sol=solvers.qp(P, q, G, h, A, b)
# print(sol['x'])


