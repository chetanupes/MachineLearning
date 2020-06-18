import numpy as np
from matplotlib.pyplot import *
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures

#importing the file and plotting the data for visulization
data_file = np.loadtxt('ex2data1.txt', delimiter=',');
x = data_file[:, [0, 1]];
x1 = data_file[:, 0];
x2 = data_file[:, 1];
y = data_file[:, 2];


#user later to use for prediction
X1 = np.array([x1, x2]).T;

pos = data_file[data_file[:,2] == 1]
neg = data_file[data_file[:,2] == 0]


# ----Plotting Data-----
plot(pos[:, 0], pos[:, 1], '+')
plot(neg[:, 0], neg[:, 1], 'o')
legend(['Admitted'])
title('Data')
xlabel('Exam1 Score')
ylabel('Exam2 Score')
tight_layout()
style.use('seaborn-dark-palette')
show();

#-------defining inputs--------
m = len(x);
b = np.ones((m, 1));
X = np.array([x1,x2]).T;
X = np.insert(X, [0], b, 1); #input matrix
theta = np.ones((3, 1)); #initializing theta
y = np.array([y]).T;

#Sigmoid function
def sigmoid(x):
    g = 1 / (1 + np.exp(-x))
    return g

#Cost function
def costfun(theta):
    h = sigmoid(np.dot(X,theta))
    J = 1/m*(-y.T.dot(np.log(h))-(1-y).T.dot(np.log(1-h)))
    return J
#print(costfun(theta))

#gradient
def gradient(theta):
    theta = theta.reshape(-1, 1)
    h = sigmoid(X.dot(theta))
    grad = 1/m*(np.dot((h-y).T,X))
    return grad.flatten()

#Set Options

initial_theta=np.zeros(3)
res = opt.fmin_bfgs(f=costfun, x0=initial_theta, fprime=gradient)
print(res)

theta=[[-25.16133284] , [0.2062317] ,   [0.2014716]  ]

x_test = np.array([45, 85])
x_test = np.append(np.ones(1),x_test)
prob = sigmoid(np.dot(x_test,theta))
print("For a student with scores 45 and 85, we predict an admission probability of",prob[0])


#prediction
def predict(X,theta):
    prediction = sigmoid(np.dot(X,theta))>0.5
    return prediction

#Testing accuracy
p=predict(X,theta)
print(sum(p==y))

