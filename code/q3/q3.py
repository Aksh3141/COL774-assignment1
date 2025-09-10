import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# LOADING DATA 
X = pd.read_csv("/home/aksh/Desktop/Assignment_archive/ML_assignement1/code/data/Q3/logisticX.csv", header = None).values
Y = pd.read_csv("/home/aksh/Desktop/Assignment_archive/ML_assignement1/code/data/Q3/logisticY.csv", header = None).values.flatten()
print("---------DataFrame shape--------")
print()
print("X : " ,X.shape)
print("Y : " ,Y.shape)
print("--------------------------------")

# Normalizing the dataset -- using standardization 
# subtract each point by mean and divide by variance

X = X - X.mean()/X.std()
X = np.hstack([np.ones((X.shape[0], 1)), X])  # adding bias term


# Defining important functions for Newtons methods
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient(X, y, theta):
    preds = sigmoid(X @ theta)           
    return X.T @ (y - preds) 

def hessian(X, theta):
    preds = sigmoid(X @ theta)      
    diagnoal_mat = np.diag(preds * (1 - preds))                
    return -(X.T @ diagnoal_mat @ X)


# Initialize paramters with 0 
theta = np.zeros(X.shape[1])
tol = 1e-12  # tolerance value for stopping
theta_history = [theta.copy()]
iter =0
while(True):
    
    # computing gradient and hessain of logsistic regression loss function wrt paramters theta
    grad = gradient(X,Y,theta)
    hess = hessian(X,theta)

    # Newton's update
    theta -= np.linalg.inv(hess)@grad
    theta_history.append(theta.copy())
    iter +=1
    if(np.sqrt(np.sum(grad**2))<tol):
        print("Reached convergence in ", iter, "iterations")
        print()
        break

print("The optimal parameters are : ", theta)
theta_history = np.array(theta_history) 

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(theta_history[:,0], theta_history[:,1], theta_history[:,2], marker='o')
ax.set_xlabel('Theta0 (bias)')
ax.set_ylabel('Theta1')
ax.set_zlabel('Theta2')
ax.set_title('Parameter Trajectory in 3D')
plt.show()



X1 = X[:,1]
X2 = X[:,2]

# Scatter plot of data
plt.figure(figsize=(8,6))
plt.scatter(X1[Y==0], X2[Y==0], color='red', label='Class 0')
plt.scatter(X1[Y==1], X2[Y==1], color='blue', label='Class 1')

# Decision boundary
x_values = np.linspace(X1.min()-1, X1.max()+1, 100)
y_values = -(theta[0] + theta[1]*x_values) / theta[2]  

plt.plot(x_values, y_values, color='green', label='Decision Boundary')

plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.title('Training Data and Decision Boundary')
plt.show()

    
