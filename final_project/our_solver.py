import numpy as np
import pickle
import matplotlib.pyplot as plt


def main():
    eligible_assets_ids = pickle.load(open(r"C:\Users\t-admich\Desktop\IDC\PY\concise_data\eligible_assets_ids.pickle", "rb"));
    cov_matrix = pickle.load(open(r"C:\Users\t-admich\Desktop\IDC\PY\concise_data\cov_matrix.pickle", "rb"));
    assets_avg_yearly_yields =pickle.load(open(r'C:\Users\t-admich\Desktop\IDC\PY\concise_data\assets_avg_yearly_yields.pickle', "rb"));

    kappa = 1
    res,values,iteration = GetOptInvesment(cov_matrix, kappa, np.array(assets_avg_yearly_yields),hessian,df);
    print(values)
    plt.scatter(iteration,values)
    plt.xlabel('Barrier Method Iteration number')
    h= plt.ylabel('f0(x) ')
    h=h.set_rotation(0)
    plt.title("Value of f0 by Barrier iteration number")
    plt.show()
    n = len(assets_avg_yearly_yields)
    for i in range(n):
        x1 = np.zeros(n).reshape(n, 1)
        x1[i] = 1
        x2 = res.reshape(n, 1)
        mu = np.array([assets_avg_yearly_yields]).reshape((n, 1))
        fx1 = f0(x1, mu, cov_matrix, kappa)

        fx2 = f0(x2, mu, cov_matrix, kappa)
        if (fx1 < fx2):
            print("Wrong fx1 < fx2")

    with open('x_first_iter.pkl', 'wb') as handle:
        pickle.dump(res, handle)

    # res_from_pkl = pickle.load(open(r'x_first_iter.pkl', "rb"))
    # print(f'res_from_pkl {res_from_pkl}')
    # print(f'res_from_pkl[0] {res_from_pkl[0]}')


def GetOptInvesment(G, kappa, mu,hessian,df):
    n = len(mu)
    x=1/n*np.ones(n)
    t = 1
    u = 2
    epsilon = 10 ** -2
    xStar = BarrierMethod(x, epsilon, n, t, u, G, mu, kappa,hessian,df)
    return xStar

def BarrierMethod(x, epsilon, m, t, u, G, mu, k,hessian,df):
    isTerminate = False
    i=0
    values = []
    iteration =[]
    while (not isTerminate):
        x = NewtonMethod(x, epsilon, t, G, mu, k,hessian,df)
        iteration.append(i)
        values.append(f0(x, mu, G, k))
        if (m / t < epsilon):
            isTerminate = True
        else:
            t = t * u
            i=i+1

    return x,values,iteration

def NewtonMethod(x, epsilon, t, g, mu, k,hessian,df):
    isTerminate = False
    n = len(x)
    while (not isTerminate):
        hess_f= hessian(x,t,g,k)
        grad_f=df(x,t,mu,g,k)
        rhs = np.append(-grad_f, 0)
        kkt_mat = np.concatenate((hess_f, np.ones(n).reshape((1, n))))
        c = np.append(np.ones(n), 0)
        c = c.reshape(((n + 1, 1)))
        kkt_mat = np.concatenate((kkt_mat, c), axis=1)
        pnt = np.linalg.solve(kkt_mat, rhs)
        pnt = pnt[:n]
        newtonDecrment = np.power(NewtonDecrment(pnt, hess_f), 2)
        if (0.5 * newtonDecrment < epsilon):
            isTerminate = True
        else:
            ak = 1
            x=x+ak*pnt
    return x

def f0(x,mu,g,k):
    return -x.T@ mu + k*x.T @ g @ x
def df(x,t,mu,g,k):
    return (-t*mu+2*t*k*g.dot(x))-np.array(1/x).T
def hessian(x,t,g,k):
   return 2 * t * k * g + np.diag(1 / (x ** 2))


def NewtonDecrment(pnt, dfdfx):
    return np.sqrt(pnt.T @ dfdfx @ pnt)


if __name__ == '__main__':
    main()