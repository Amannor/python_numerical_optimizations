import unittest
import numpy as np
import examples
import constrained_min
import utils
from collections import OrderedDict
# from ..src import unconstrained_min
# from ..src import utils
# from .. import src.unconstrained_min
# from . import src
# import src.utils
# import src.unconstrained_min


def qp_ineq1_val_only(x):
    f_val = -x[0]
    return f_val

def qp_ineq2_val_only(x):
    f_val = -x[1]
    return f_val

def qp_ineq3_val_only(x):
    f_val = -x[2]
    return f_val

def lp_ineq1_val_only(x):
    f_val = -x[0]-x[1]+1
    return f_val

def lp_ineq2_val_only(x):
    f_val = x[1]-1
    return f_val

def lp_ineq3_val_only(x):
    f_val = x[0]-2
    return f_val

def lp_ineq4_val_only(x):
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
        ineq_constraints = [qp_ineq1_val_only, qp_ineq2_val_only, qp_ineq3_val_only]
        eq_constraints_mat = np.ones(len(x0))
        eq_constraints_rhs = np.ones(1)
        x_vals, success, obj_val, constraints_vals = constrained_min.interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0)
        
        ineq_constraints_for_plot = [qp_ineq1_val_only, qp_ineq2_val_only, qp_ineq3_val_only]
        utils.plot_for_qp(func, x_vals, ineq_constraints_for_plot, eq_constraints_mat, eq_constraints_rhs)

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
        ineq_constraints = [lp_ineq1_val_only, lp_ineq2_val_only, lp_ineq3_val_only, lp_ineq4_val_only]
        x_vals, success, obj_val, constraints_vals = constrained_min.interior_pt(func, ineq_constraints, None, None, x0)
        
        ineq_constraints_for_plot = [lp_ineq1_val_only, lp_ineq2_val_only, lp_ineq3_val_only, lp_ineq4_val_only]
        utils.plot_for_lp(func, x_vals, ineq_constraints_for_plot)
        
        iter_num_to_obj_val = get_iter_num_to_obj_val_from_x_vals(func, x_vals)
        utils.plot_iter_num_to_obj_val(func, iter_num_to_obj_val)
        
        self.assertTrue(success)

if __name__ == '__main__':
    unittest.main()