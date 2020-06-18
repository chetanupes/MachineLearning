from numpy import *
from mpl_toolkits.mplot3d import axes3d
from matplotlib.pyplot import *


data_file = loadtxt('ex1data2.txt', delimiter=',');
x1 = data_file[:, 0];
x2 = data_file[:, 1];
y = data_file[:, 2];
x1 = (x1-mean(x1))/std(x1)
x2 = (x2-mean(x2))/std(x2)

tight_layout()


# ----Plotting Data-----
plot(x1, y, 'ro');
plot(x2, y, 'bv');
title('Selling House')
xlabel('Size & Rooms')
ylabel('Price')
show();

#-------defining inputs--------
m = len(x1);
b = ones((m, 1));
X = asmatrix([x1,x2]).T;
X = insert(X, [0], b, 1); #input matrix
theta = zeros((3, 1)); #initializing theta
y = asmatrix([y]).T;
alpha = 0.01;
iters = 400;

#--------Cost function Calculation-------------
def cost(X,y,theta):
    J = 0;
    m = len(X);
    h = X * theta
    J = (1 / (2 * m)) * sum(power((h - y), 2));
    return (J)
print(cost(X,y,theta));

#--------Gradient Descent Calculation-------------

def gradientdescent(X, y, theta, alpha, iters):
    n = len(X)
    J_his = zeros((iters, 1), dtype=float)
    for i in range(0, iters):
        h = X * theta
        for j in range(0, 3):
            k = X[:, j]
            d = array(h - y) * array(k)
            theta[j, 0] = theta[j, 0] - (alpha / n) * sum((d))
        J_his[i, :] = cost(X, y, theta)
    plot(J_his) #plotting cost function to see its shape
    ylabel('Cost Function')
    tight_layout()
    show()
    return (theta, J_his)
gradientdescent(X, y, theta, alpha, iters)

#--------Prediction-------------------------------
predict = X * theta
plot(x1, y, 'or')
plot(x2, y, 'ob')
xlabel('Profit')
ylabel('Population')
plot(x1, predict, '*')
tight_layout()
style.use('seaborn')
show();

X=array([1,1650,3])[0]
p=sum(X*theta)
print(p)