import unittest
import numpy as np
import examples
import constrained_min
import utils
# from ..src import unconstrained_min
# from ..src import utils
# from .. import src.unconstrained_min
# from . import src
# import src.utils
# import src.unconstrained_min


# DIR_SELECTION_METHODS = ['gd','nt', 'bfgs']
DIR_SELECTION_METHODS = ['nt', 'bfgs']


### ***Note that we need to calcauate the derivatives for minus lograithm of the minus of each constraint function (lecture 7+8 slide 61)*** ###
def qp_ineq1(x):
    f_val = -x[0]
    # Derivatives are of -log(-f_i(x))
    grad_vector_val = np.array([-1/x[0],0,0])
    hessian = np.zeros([len(x), len(x)])
    hessian[0][0] = 1 / (x[0] ** 2)
    return f_val, grad_vector_val, hessian

def qp_ineq2(x):
    f_val = -x[1]
    # Derivatives are of -log(-f_i(x))
    grad_vector_val = np.array([0,-1/x[1],0])
    hessian = np.zeros([len(x), len(x)])
    hessian[1][1] = 1 / (x[1] ** 2)
    return f_val, grad_vector_val, hessian

def qp_ineq3(x):
    f_val = -x[2]
    # Derivatives are of -log(-f_i(x))
    grad_vector_val = np.array([0,0,-1/x[2]])
    hessian = np.zeros([len(x), len(x)])
    hessian[2][2] = 1 / (x[2] ** 2)
    return f_val, grad_vector_val, hessian

def lp_ineq1(x, should_return_only_val=False):
    f_val = -x[0]-x[1]+1
    if should_return_only_val:
        return f_val
    # Derivatives are of -log(-f_i(x))
    grad_vector_val = np.array([1/(1-x[0]-x[1]),1/(1-x[0]-x[1])])
    hessian=(1/((1-x[0]-x[1])**2))*np.ones([len(x), len(x)])
    return f_val, grad_vector_val, hessian

def lp_ineq2(x, should_return_only_val=False):
    f_val = x[1]-1
    if should_return_only_val:
        return f_val
    # Derivatives are of -log(-f_i(x))
    grad_vector_val = np.array([0,1/(1-x[1])])
    hessian = np.zeros([len(x), len(x)])
    hessian[1][1] = 1/((1-x[1])**2)
    return f_val, grad_vector_val, hessian

def lp_ineq3(x, should_return_only_val=False):
    f_val = x[0]-2
    if should_return_only_val:
        return f_val
    # Derivatives are of -log(-f_i(x))
    grad_vector_val = np.array([1/(2-x[0]),0])
    hessian = np.zeros([len(x), len(x)])
    hessian[0][0] = 1/((2-x[0])**2)
    return f_val, grad_vector_val, hessian

def lp_ineq4(x, should_return_only_val=False):
    f_val = -x[1]
    if should_return_only_val:
        return f_val
    # Derivatives are of -log(-f_i(x))
    grad_vector_val = np.array([0,-1/x[1]])
    hessian = np.zeros([len(x), len(x)])
    hessian[1][1] = 1 / (x[1] ** 2)
    return f_val, grad_vector_val, hessian


class TestConstrainedMinimizaton(unittest.TestCase):

    '''
    #Sample test - START
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())
    #Sample test - END
    '''

    # def test_qp(self):
    #     '''
    #     f0(x) = x1^2+x2^2+(x3+1)^2
    #     h1(x) = x1+x2+x3-1
    #     f1(x) = -x1
    #     f2(x) = -x2
    #     f3(x) = -x3
    #     '''
    #     x0 = np.array([0.1,0.2,0.7])
    #     ineq_constraints = [qp_ineq1, qp_ineq2, qp_ineq3] 
    #     eq_constraints_mat = np.ones(len(x0))
    #     eq_constraints_rhs = np.ones(1)
    #     interior_pt_res = constrained_min.interior_pt(func: examples.f_qp, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0)


        # x0 = np.array([1,1])
        # step_size = 0.1
        # max_iter = 100
        # step_tolerance = 10**-8
        # obj_tolerance = 10**-12
        # for quad_example in [examples.f_b_i, examples.f_b_ii, examples.f_b_iii]:
        #     for dir_selection_method in DIR_SELECTION_METHODS:
        #         print(f'Testing quad function: {quad_example.__name__} dir method: {dir_selection_method}')
        #         grad_desc_res = unconstrained_min.line_search(quad_example, x0, step_size, obj_tolerance, step_tolerance, max_iter, dir_selection_method, )
        #         utils.plot_contours_and_paths(quad_example, grad_desc_res[2], dir_selection_method)
        #         utils.plot_iter_num_to_obj_val(quad_example, grad_desc_res[3], dir_selection_method)
        #         self.assertTrue(grad_desc_res[1])

    def test_lp(self):
        '''
        f0(x) = -x1-x2
        f1(x) = -x1-x2+1
        f2(x) = x2-1
        f3(x) = x1-2
        f4(x) = -x2
        '''
        x0 = np.array([0.5,0.75])
        ineq_constraints = [lp_ineq1, lp_ineq2, lp_ineq3, lp_ineq4] 
        x_vals, success, obj_val, constraints_vals = constrained_min.interior_pt(examples.f_lp, ineq_constraints, None, None, x0)
        print(f'x_vals {x_vals}')
        ineq_constraints_summary_str = [f'{ineq_constraints[i].__name__}: {constraints_vals[i]}' for i in range(len(ineq_constraints))]
        ineq_constraints_summary_str = " ".join(ineq_constraints_summary_str)
        print(f'{examples.f_lp.__name__} {"" if examples.f_lp else "Not "} Succesful.\nFinal candidate: {x_vals[-1]} objective function value: {obj_val} inequality constraints values: {ineq_constraints_summary_str}')
        utils.plot_for_lp(examples.f_lp, x_vals, ineq_constraints)
        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()