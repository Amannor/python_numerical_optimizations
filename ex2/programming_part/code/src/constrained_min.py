

DEFAULT_T = 1
DEFAULT_mu = 10

def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
	'''
	Minimizes the function func, subject to constraints (using the log-barrier method)
	ineq_constraints: List of inequality constraints
	eq_constraints_mat: Matrix of affine equality constraints 𝐴𝑥=𝑏
	eq_constraints_rhs: T^he right hand side vector
	x0: Where the outer iterations start at
	'''