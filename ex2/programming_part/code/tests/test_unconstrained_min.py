import unittest
import numpy as np
import examples
# from ..src import unconstrained_min
# from ..src import utils
# from .. import src.unconstrained_min
from . import src
# import src.utils
# import src.unconstrained_min


# DIR_SELECTION_METHODS = ['nt', 'bfgs']
DIR_SELECTION_METHODS = ['nt']

class TestStringMethods(unittest.TestCase):

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

    def test_quad_min(self):
        x0 = np.array([1,1])
        step_size = 0.1
        max_iter = 100
        step_tolerance = 10**-8
        obj_tolerance = 10**-12
        for quad_example in [examples.f_b_i, examples.f_b_ii, examples.f_b_iii]:
            for dir_selection_method in DIR_SELECTION_METHODS:
                print(f'Testing quad function: {quad_example.__name__} dir method: {dir_selection_method}')
                grad_desc_res = src.unconstrained_min.line_search(quad_example, x0, step_size, obj_tolerance, step_tolerance, max_iter, dir_selection_method, )
                src.utils.plot_contours_and_paths(quad_example, grad_desc_res[2], dir_selection_method)
                src.utils.plot_iter_num_to_obj_val(quad_example, grad_desc_res[3], dir_selection_method)
                self.assertTrue(grad_desc_res[1])

    def test_rosenbrock_min(self):
        x0 = np.array([2,2])
        step_size = 0.001
        max_iter = 10000
        step_tolerance = 10**-8
        obj_tolerance = 10**-7
        for dir_selection_method in DIR_SELECTION_METHODS:
            print(f'Testing function: Rosenbrock dir method: {dir_selection_method}')
            grad_desc_res = src.unconstrained_min.line_search(examples.f_c_rosenbrock, x0, step_size, obj_tolerance,
                                                                   step_tolerance, max_iter, dir_selection_method)
            src.utils.plot_contours_and_paths(examples.f_c_rosenbrock, grad_desc_res[2], dir_selection_method)
            src.utils.plot_iter_num_to_obj_val(examples.f_c_rosenbrock, grad_desc_res[3], dir_selection_method)
            self.assertTrue(grad_desc_res[1])

    ### Skipped in ex2 ###
    # def test_lin_min(self):
    #     x0 = np.array([1,1])
    #     step_size = 0.1
    #     max_iter = 100
    #     step_tolerance = 10**-8
    #     obj_tolerance = 10**-12
    #     print(f'Testing function: linear')
    #     grad_desc_res = src.unconstrained_min.line_search(examples.f_d_lin, x0, step_size, obj_tolerance,
    #                                                            step_tolerance, max_iter)
    #     src.utils.plot_contours_and_paths(examples.f_d_lin, grad_desc_res[2])
    #     self.assertFalse(grad_desc_res[1])



if __name__ == '__main__':
    unittest.main()