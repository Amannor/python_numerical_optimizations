import numpy as np
import math

def eval_quad(Q, x):
    return x.T.dot(Q).dot(x)

def f_b_i(x):
    Q = np.array([[1.,0],[0,1.]])
    f_val = eval_quad(Q, x)
    grad_vector_val = 2 * Q.dot(x)
    return f_val, grad_vector_val

def f_b_ii(x):
    Q = np.array([[5.,0],[0,1.]])
    f_val = eval_quad(Q, x)
    grad_vector_val = 2 * Q.dot(x)
    return f_val, grad_vector_val

def f_b_iii(x):
    q1 = np.array([[math.sqrt(3.)/2 ,-1./2],[1./2,math.sqrt(3.)/2]]).T
    q2 = np.array([[5.,0],[0,1.]])
    q3 = np.array([[math.sqrt(3.)/2.,-0.5],[0.5,math.sqrt(3.)/2.]])
    Q = q1.dot(q2).dot(q3) # Q is np.array([[4, -math.sqrt(3.)],[-math.sqrt(3.)., 2]]) i.e. f = 4x[0]^2 - 2*sqrt(3)*x[0]*x[1]^2 +2x[1]^2
    f_val = eval_quad(Q, x)
    grad_vector_val = np.array([8*x[0] - 2*math.sqrt(3)*x[1], 4*x[1]-2*math.sqrt(3)*x[0]])
    return f_val, grad_vector_val

def f_c_rosebrock(x):
    f_val = 100*(x[1] - x[0]**2)**2 + (1-x[0])**2
    deriv_by_x1 = 400*x[0]**3+2*x[0]-400*x[0]*x[1]
    deriv_by_x2 = 200*x[1]-200*x[0]**2
    grad_vector_val = np.array([deriv_by_x1 , deriv_by_x2])
    return f_val, grad_vector_val

def f_d_lin(x):
    const_val = 2
    a = np.full(x.shape, const_val)
    f_val = a.T.dot(x)
    grad_vector_val = a
    return f_val, grad_vector_val

