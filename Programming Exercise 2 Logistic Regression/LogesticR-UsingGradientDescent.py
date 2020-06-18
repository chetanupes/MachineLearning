from numpy import *
import scipy.optimize as opt
from matplotlib.pyplot import *


data_file = loadtxt('ex2data1.txt', delimiter=',');
x = data_file[:, [0, 1]];
x1 = data_file[:, 0];
x2 = data_file[:, 1];
y = data_file[:, 2];


#user later to use for prediction
X1 = asmatrix([x1, x2]).T;
m1 = mean(X1,axis=0)
std1 = std(X1,axis=0)

pos = data_file[data_file[:,2] == 1]
neg = data_file[data_file[:,2] == 0]
x1 = (x1-mean(x1))/std(x1)
x2 = (x2-mean(x2))/std(x2)

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
b = ones((m, 1));
X = asmatrix([x1,x2]).T;
X = insert(X, [0], b, 1); #input matrix
theta = ones((3, 1)); #initializing theta
y = asmatrix([y]).T;
alpha = 1;
iters = 400;

#defining sigmoid function
def sigmoid(x):
    g = 1/(1+exp(-x))
    return (g)

#cost function and gradient
def costfunction(X,y,theta):
    J = 0
    h = sigmoid(X * theta)
    #J = (1/m)*sum(array(-y)*array(log(h))-array(1-y)*array(log(1-h)))
    J = (1 / m) * sum((-np.array(y) * np.array(np.log(h)) - np.array(1 - y) * np.array(np.log(1 - h))))
    return (J)
print(costfunction(X,y,theta))

def gradientdescent(X, y, theta, alpha, iters):
    n = len(X)
    J_his = zeros((iters, 1), dtype=float)
    for i in range(0, iters):
        h = sigmoid(X * theta)
        for j in range(0, 3):
            k = X[:, j]
            d = array(h - y) * array(k)
            theta[j, 0] = theta[j, 0] - (alpha / n) * sum((d))
        J_his[i, :] = costfunction(X, y, theta)
    plot(J_his) #plotting cost function to see its shape
    ylabel('Cost Function')
    tight_layout()
    show()
    return (theta, J_his)
print(gradientdescent(X,y,theta,alpha,iters))



x_test = array([45, 85])
x_test = (x_test - m1)/std1
x_test = append(ones(1),x_test)
prob = sigmoid(dot(x_test,theta))
print("For a student with scores 45 and 85, we predict an admission probability of",prob[0])

#prediction
def predict(X,theta):
    prediction=sigmoid(X*theta)>0.5
    return prediction

#Testing accuracy
p=predict(X,theta)
print(sum(p==y))
