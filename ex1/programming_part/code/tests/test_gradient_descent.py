import unittest
import numpy as np
import examples
import src.unconstrained_min
import src.utils



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
            print(f'Testing quad function: {quad_example.__name__}')
            grad_desc_res = src.unconstrained_min.gradient_descent(quad_example, x0, step_size, obj_tolerance, step_tolerance, max_iter)
            src.utils.plot_contours_and_paths(grad_desc_res[2],grad_desc_res[3])
            self.assertTrue(grad_desc_res[1])

    def test_rosenbrock_min(self):
        x0 = np.array([2,2])
        step_size = 0.001
        max_iter = 10000
        step_tolerance = 10**-8
        obj_tolerance = 10**-7
        print(f'Testing function: Rosenbrock')
        grad_desc_res = src.unconstrained_min.gradient_descent(examples.f_c_rosebrock, x0, step_size, obj_tolerance,
                                                               step_tolerance, max_iter)
        self.assertTrue(grad_desc_res[1])

    def test_lin_min(self):
        x0 = np.array([1,1])
        step_size = 0.1
        max_iter = 100
        step_tolerance = 10**-8
        obj_tolerance = 10**-12
        print(f'Testing function: linear')
        grad_desc_res = src.unconstrained_min.gradient_descent(examples.f_d_lin, x0, step_size, obj_tolerance,
                                                               step_tolerance, max_iter)
        self.assertFalse(grad_desc_res[1])



if __name__ == '__main__':
    unittest.main()