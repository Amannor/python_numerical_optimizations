import numpy as np
import math

def eval_quad(Q, x):
    return x.T.dot(Q).dot(x)

def f_b_i(x, should_calc_hessian=False):
    Q = np.array([[1.,0],[0,1.]])
    f_val = eval_quad(Q, x)
    grad_vector_val = 2 * Q.dot(x)
    if should_calc_hessian:
       hessian = 2*np.eye(len(Q))
       return f_val, grad_vector_val, hessian
    else:
        return f_val, grad_vector_val, None

def f_b_ii(x, should_calc_hessian=False):
    Q = np.array([[5.,0],[0,1.]])
    f_val = eval_quad(Q, x)
    grad_vector_val = 2 * Q.dot(x)
    if should_calc_hessian:
        hessian = 2*Q
        return f_val, grad_vector_val, hessian
    else:
        return f_val, grad_vector_val, None

def f_b_iii(x, should_calc_hessian=False):
    q1 = np.array([[math.sqrt(3.)/2 ,-1./2],[1./2,math.sqrt(3.)/2]]).T
    q2 = np.array([[5.,0],[0,1.]])
    q3 = np.array([[math.sqrt(3.)/2.,-0.5],[0.5,math.sqrt(3.)/2.]])
    Q = q1.dot(q2).dot(q3) # Q is np.array([[4, -math.sqrt(3.)],[-math.sqrt(3.)., 2]]) i.e. f = 4x[0]^2 - 2*sqrt(3)*x[0]*x[1]^2 +2x[1]^2
    f_val = eval_quad(Q, x)
    grad_vector_val = np.array([8*x[0] - 2*math.sqrt(3)*x[1], 4*x[1]-2*math.sqrt(3)*x[0]])
    if should_calc_hessian:
        hessian = np.array([[8.,-2*math.sqrt(3)],[-2*math.sqrt(3),4.]]) 
        return f_val, grad_vector_val, hessian
    else:
        return f_val, grad_vector_val, None

def f_c_rosenbrock(x, should_calc_hessian=False):
    f_val = 100*(x[1] - x[0]**2)**2 + (1-x[0])**2
    deriv_by_x1 = 400*x[0]**3+2*x[0]-400*x[0]*x[1]-2
    deriv_by_x2 = 200*x[1]-200*x[0]**2
    grad_vector_val = np.array([deriv_by_x1, deriv_by_x2])
    if should_calc_hessian:
        hessian = np.array([[1200*x[0]**2+2-400*x[1],-400*x[1]],[-400*x[1],200.]])
        return f_val, grad_vector_val, hessian
    else:
        return f_val, grad_vector_val, None

def f_d_lin(x, should_calc_hessian=False):
    const_val = 2
    a = np.full(x.shape, const_val)
    f_val = a.T.dot(x)
    grad_vector_val = a
    if should_calc_hessian:
        hessian = np.zeros(len(x), len(x))
        return f_val, grad_vector_val, hessian
    else:
        return f_val, grad_vector_val, None

def f_qp(x, should_calc_hessian=False):
    f_val = x[0]**+x[1]**2+(x[2]+1)**2
    grad_vector_val = np.array([2*x[0], 2*x[1], 2*x[2]+2])
    if should_calc_hessian:
        hessian = 2*np.eye(len(x))
        return f_val, grad_vector_val, hessian
    else:
        return f_val, grad_vector_val, None

def f_lp(x, should_calc_hessian=False):
    f_val = -x[0]-x[1]
    grad_vector_val = np.array([-1.,-1.])
    if should_calc_hessian:
        hessian = np.zeros([len(x), len(x)])
        return f_val, grad_vector_val, hessian
    else:
        return f_val, grad_vector_val, None

