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



def f_1(x, y):
    return abs(x)+abs(y)

def f_2(x, y):
    return (x-1)**2+(y-1)**2



x = np.linspace(-3, 3, 500)
y = np.linspace(-3, 3, 500)
X, Y = np.meshgrid(x, y)

Z = f_1(X, Y)
# plt.contour(X, Y, Z, colors='black')
# plt.contour(X, Y, Z, 150, cmap='RdGy')
plt.contour(X, Y, Z, 50, cmap='RdGy')
plt.contourf(X, Y, Z, 100, cmap='RdGy', alpha=0.6)
save_fig('q1_1.png')

#From: https://stackoverflow.com/a/51389483
d = np.linspace(-3,3,200)
x,y = np.meshgrid(d,d)

im = plt.imshow( ((y<=1) & (f_2(x,y)<=1)).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys")
save_fig('q1_2.png')


#For q1_3 I already pre calculated the minimum point (https://www.wolframalpha.com/input/?i=minimize+abs%28x%29%2Babs%28y%29%2C+%28x-1%29%5E2%2B%28y-1%29%5E2%3C%3D1%2C+y%3C%3D1)
min_x, min_y = 1. - 1/(math.sqrt(2)), 1. - 1/(math.sqrt(2))
plt.plot(min_x, min_y, 'bo')
save_fig('q1_3.png')

#From https://stackoverflow.com/a/63843501
from matplotlib.patches import FancyArrowPatch
fig, ax = plt.subplots()

ax.contour(X, Y, Z, 50, cmap='RdGy')
ax.contourf(X, Y, Z, 100, cmap='RdGy', alpha=0.6)
im = ax.imshow( ((y<=1) & (f_2(x,y)<=1)).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys")
ax.plot(min_x, min_y, 'bo')

# x1 = -20     # position of the gradient
# y1 = 10
# Directiob of gradients from: https://www.geogebra.org/m/sWsGNs86
dz1_dx_for_obj_func = 1  # value of the gradient at that position
dz1_dy_for_obj_func = 1
arrow = FancyArrowPatch((min_x, min_y), (min_x+dz1_dx_for_obj_func, min_y+dz1_dy_for_obj_func),    
                        arrowstyle='simple', color='k', mutation_scale=10)
ax.add_patch(arrow)

dz1_dx_for_circle_constraint = min_x+1  # value of the gradient at that position
dz1_dy_for_circle_constraint = min_y+1
arrow = FancyArrowPatch((min_x, min_y), (min_x-dz1_dx_for_circle_constraint, min_y-dz1_dy_for_circle_constraint),    
                        arrowstyle='simple', color='g', mutation_scale=10)
ax.add_patch(arrow)
save_fig('q1_5.png')