'''
#Original code taken from: https://jakevdp.github.io/PythonDataScienceHandbook/04.04-density-and-contour-plots.html
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np

def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)


x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
plt.contour(X, Y, Z, colors='black')
plt.show()
'''

# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np

def f_1(x, y):
    return abs(x)+abs(y)

def f_2(x, y):
    return (x-1)**2+(y-1)**2


x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(x, y)

Z = f_1(X, Y)
# plt.contour(X, Y, Z, colors='black')
# plt.contour(X, Y, Z, 100, cmap='RdGy')
plt.contourf(X, Y, Z, 100, cmap='RdGy', alpha=0.6)

#From: https://stackoverflow.com/a/51389483
d = np.linspace(-1,3,200)
x,y = np.meshgrid(d,d)

im = plt.imshow( ((y<=1) & (f_2(x,y)<=1)).astype(int) , 
                extent=(x.min(),x.max(),y.min(),y.max()),origin="lower", cmap="Greys")

plt.show()