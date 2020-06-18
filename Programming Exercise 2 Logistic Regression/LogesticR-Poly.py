import numpy as np
from matplotlib.pyplot import *
from scipy.optimize import fmin_bfgs
from sklearn.preprocessing import PolynomialFeatures

#importing the file and plotting the data for visulization
data_file = np.loadtxt('ex2data2.txt', delimiter=',')
x1 = data_file[:, 0]
x2 = data_file[:, 1]
pos = data_file[data_file[:, 2]==1]
neg = data_file[data_file[:, 2]==0]
y = data_file[:, [2]]

#Plotting data
plot(pos[:, 0], pos[:, 1], '+r')
plot(neg[:, 0], neg[:, 1], 'ob')
title('Admission Status')
legend(['Accepted','Not Accepted'])
tight_layout()
xlabel('Score 1')
ylabel('Score 2')
show()


#Creating polynomial feature
X = np.array([x1, x2]).T
poly_features = PolynomialFeatures(6)
x_poly = poly_features.fit_transform(X)
X = x_poly
m, n = x_poly.shape
theta = np.zeros((n, 1))
theta1 = theta[1:n, :]
theta2 = np.insert(theta1, 0, 0).T
lamb = 1

#Sigmoid function
def sigmoid(x):
    g = 1 / (1 + np.exp(-x))
    return g

#Cost function and regularization
def costfun(theta):
    h = sigmoid(np.dot(X,theta))
    reg = lamb/(2*m) * sum(np.power(theta, 2))
    J = 1/m*(-y.T.dot(np.log(h))-(1-y).T.dot(np.log(1-h))) + reg
    return J


#gradient
def gradient(theta):
    theta = theta.reshape(-1, 1) #important step to reshape theta, converts theta from 1d to 2D
    grad = np.zeros((n,1))
    h = sigmoid(np.dot(X, theta))
    for j in range(0, n):
        if j==0:
            grad[j, 0] = 1 / m * sum(np.dot((h - y).T, X[:, j]).T)
        else:
            grad[j, 0] = 1 / m * sum(np.dot((h - y).T, X[:, j]).T) + (lamb / m) * theta[j, 0]
        grad[:]
    return grad.flatten()


#Set Options

initial_theta=np.zeros((n,1))
theta=fmin_bfgs(costfun, initial_theta, fprime=gradient, maxiter=500)
theta= theta.reshape(28,1)

#plotting decision boundary

#prediction
def predict(X,theta):
    prediction = sigmoid(np.dot(X, theta)) >= 0.5
    return prediction

#Testing accuracy
p = predict(X, theta)
print(np.mean(p==y)*100)

