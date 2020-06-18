import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib.pyplot import *


data_file = np.loadtxt('ex1data1.txt', delimiter=',');
x = data_file[:, 0];
y = data_file[:, 1];

# ----Plotting Data-----
plot(x, y, 'ro');
title('Housing')
xlabel('Population in 10,000s')
ylabel('Profit')
show();

#-------defining inputs--------
m = len(x);
b = np.ones((m, 1));
X = np.asmatrix([x]).T;
X = np.insert(X, [0], b, 1); #input matrix
theta = np.zeros((2, 1)); #initializing theta
y = np.asmatrix([y]).T;
alpha = 0.01;
iters = 1500;

#--------Cost function Calculation-------------
def cost(X,y,theta):
    J = 0;
    m = len(X);
    h = X * theta
    J = (1 / (2 * m)) * sum(np.power(np.asmatrix(h - y), 2));
    return (J)
print(cost(X,y,theta));

#--------Gradient Descent Calculation-------------

def gradientdescent(X, y, theta, alpha, iters):
    n = len(X)
    J_his = np.zeros((iters, 1), dtype=float)
    for i in range(0, iters):
        h = X * theta
        for j in range(0, 2):
            k = X[:, j]
            d = np.array(h - y) * np.array(k)
            theta[j, 0] = theta[j, 0] - (alpha / n) * sum((d))
            print(theta[:])
        J_his[i, :] = cost(X, y, theta)
    plot(J_his) #plotting cost function to see its shape. A decrecing trend with the increasing iteration is the desired trend
    show()
    return (theta, J_his)
gradientdescent(X, y, theta, alpha, iters)


#--------Prediction-------------------------------
predict = X * theta
plot(x, y, 'or')
xlabel('Profit')
ylabel('Population')
plot(x, predict, '+b')
show();

#plotting J 3D


for i in range(0, 5):
    for j in range(0, 5):
         t = [[i], [j]]
         J_new = cost(X, y, t)


