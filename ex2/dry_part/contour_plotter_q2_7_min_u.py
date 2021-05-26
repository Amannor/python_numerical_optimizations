import math
import os
OUT_FOLDER = 'plots'

if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)


#Original code taken from: https://jakevdp.github.io/PythonDataScienceHandbook/04.04-density-and-contour-plots.html
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np

def save_fig(fname):
	plt.savefig(os.path.join(OUT_FOLDER, fname))



def f_0(x, y):
    return 0.5*x-y

def f_1(x, y):
    return -x+y-1

def f_2(x, y):
    return -x/3. + y -5./3.

def f_3(x, y):
    return x-4

def f_4(x, y):
    return y-3

def f_5(x, y):
    return -x

def f_6(x, y):
    return -y

def f_2_min_u(x, y):
    return -x/3. + y -5./3.+2./3.

x = np.linspace(-1, 5, 500)
y = np.linspace(-1, 5, 500)
X, Y = np.meshgrid(x, y)

Z = f_0(X, Y)
plt.contour(X, Y, Z, 50, cmap='RdGy')
plt.contourf(X, Y, Z, 100, cmap='RdGy', alpha=0.6)

#From: https://stackoverflow.com/a/51389483
d = np.linspace(-1,5,200)
x,y = np.meshgrid(d,d)

im = plt.imshow( ((f_1(x,y)<=0) & (f_2_min_u(x,y)<=0) & (f_3(x,y)<=0) & (f_4(x,y)<=0) & (f_5(x,y)<=0) & (f_6(x,y)<=0)).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys")


#For q2_7_u_min I already pre calculated the minimum point (https://www.wolframalpha.com/input/?i=min+0.5x-y%2C+-x%2By-1%3C%3D0%2C+-x%2F3%2By-5%2F3%3C%3D0%2C+x-4%3C%3D0%2C+y-3%3C%3D0%2C+x%3E%3D0%2C+y%3E%3D0)
min_x, min_y = 0,1
plt.plot(min_x, min_y, 'bo')
save_fig('q2_7_u_min.png')
