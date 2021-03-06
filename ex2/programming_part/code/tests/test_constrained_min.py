import unittest
import numpy as np
import examples
from collections import OrderedDict
#These lines from: https://stackoverflow.com/a/10272919
import sys
import os
sys.path.append(os.path.abspath('src')) #Note: this means this file should run from the '.../ex2/programming_part/code' directory
import constrained_min
import utils

GREATER_THAN_OR_EQAUL_SIGN = u'\u2265'

def qp_ineq1(x, return_str_rep=False):
    if return_str_rep:
        return  u'x{0}0'.format(GREATER_THAN_OR_EQAUL_SIGN)
    f_val = -x[0]
    return f_val

def qp_ineq2(x, return_str_rep=False):
    if return_str_rep:
        return  u'y{0}0'.format(GREATER_THAN_OR_EQAUL_SIGN)
    f_val = -x[1]
    return f_val

def qp_ineq3(x, return_str_rep=False):
    if return_str_rep:
        return  u'z{0}0'.format(GREATER_THAN_OR_EQAUL_SIGN)
    f_val = -x[2]
    return f_val

def lp_ineq1(x):
    f_val = -x[0]-x[1]+1
    return f_val

def lp_ineq2(x):
    f_val = x[1]-1
    return f_val

def lp_ineq3(x):
    f_val = x[0]-2
    return f_val

def lp_ineq4(x):
    f_val = -x[1]
    return f_val

def get_iter_num_to_obj_val_from_x_vals(func, x_vals):
    iter_num_to_obj_val = OrderedDict()
    for x_val in x_vals:
        iter_num_to_obj_val[(len(iter_num_to_obj_val))] = func(x_val)[0]
    return iter_num_to_obj_val

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



    def test_qp(self):
        '''
        f0(x) = x1^2+x2^2+(x3+1)^2
        h1(x) = x1+x2+x3-1
        f1(x) = -x1
        f2(x) = -x2
        f3(x) = -x3
        '''
        func = examples.f_qp
        x0 = np.array([0.1,0.2,0.7])
        ineq_constraints = [qp_ineq1, qp_ineq2, qp_ineq3]
        eq_constraints_mat = np.ones(len(x0))
        eq_constraints_rhs = np.ones(1)
        x_vals, success, obj_val, constraints_vals = constrained_min.interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0)
        ineq_constraints_summary_str = [f'{ineq_constraints[i].__name__}: {constraints_vals[i]}' for i in range(len(ineq_constraints))]
        ineq_constraints_summary_str = " ".join(ineq_constraints_summary_str)
        print(f'Function {func.__name__} result: {"Success" if success else "Failure"}.\nFinal candidate: {x_vals[-1]} objective function value: {obj_val} inequality constraints values: {ineq_constraints_summary_str}')

        ineq_constraints_for_plot = [qp_ineq1, qp_ineq2, qp_ineq3]
        utils.plot_for_qp(func, x_vals, ineq_constraints_for_plot)

        iter_num_to_obj_val = get_iter_num_to_obj_val_from_x_vals(func, x_vals)
        utils.plot_iter_num_to_obj_val(func, iter_num_to_obj_val)

        self.assertTrue(success)

    def test_lp(self):
        '''
        f0(x) = -x1-x2
        f1(x) = -x1-x2+1
        f2(x) = x2-1
        f3(x) = x1-2
        f4(x) = -x2
        '''
        func = examples.f_lp
        x0 = np.array([0.5,0.75])
        ineq_constraints = [lp_ineq1, lp_ineq2, lp_ineq3, lp_ineq4]
        x_vals, success, obj_val, constraints_vals = constrained_min.interior_pt(func, ineq_constraints, None, None, x0)
        ineq_constraints_summary_str = [f'{ineq_constraints[i].__name__}: {constraints_vals[i]}' for i in
                                        range(len(ineq_constraints))]
        ineq_constraints_summary_str = " ".join(ineq_constraints_summary_str)
        print(f'Function {func.__name__} result: {"Success" if success else "Failure"}.\nFinal candidate: {x_vals[-1]} objective function value: {obj_val} inequality constraints values: {ineq_constraints_summary_str}')

        ineq_constraints_for_plot = [lp_ineq1, lp_ineq2, lp_ineq3, lp_ineq4]
        utils.plot_for_lp(func, x_vals, ineq_constraints_for_plot)
        
        iter_num_to_obj_val = get_iter_num_to_obj_val_from_x_vals(func, x_vals)
        utils.plot_iter_num_to_obj_val(func, iter_num_to_obj_val)
        
        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()