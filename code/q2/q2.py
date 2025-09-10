import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

#setting seed to 314 to get the same results for different runs of code
np.random.seed(314)

# SAMPLING FROM NORMAL DISTRIBUTION

X_1 = np.random.normal(3,2,1000000) 
X_2 = np.random.normal(-1,2,1000000)
epsilon = np.random.normal(0,2**(1/2),1000000)

Y = 3 + 1*X_1 + 2*X_2 + epsilon   # the given hypothesis fucntion
df = pd.DataFrame({"X1" : X_1, "X2" : X_2, "Y" : Y})

# In df Dataframe we have our final dataset 
print("Shape of our DataFrame: ", df.shape)
# splitting dataset for testing and training

split_size = int(0.8 * len(df))

training = df[:split_size]
testing = df[split_size:]

print("Shape of Train set: ", training.shape)
print("Shape of Test set: ", testing.shape)

training = training.sample(frac=1) # shuffle the training set

# defining functions for computing loss and gradients
def stochastic_MSE_loss(X1,X2,Y,theta0,theta1,theta2):
    pred = theta0 + theta1*X1 + theta2*X2
    return np.mean((Y-pred)**2)/2

def compute_grads(X1, X2, Y, theta0, theta1, theta2):
    pred = theta0 + theta1*X1 + theta2*X2
    error = Y - pred
    grad0 = -np.mean(error)
    grad1 = -np.mean(X1 * error)
    grad2 = -np.mean(X2 * error)
    return grad0, grad1, grad2


# Stochastic Gradient Descent 


eta = 0.001
batch_size = [1,80,8000,800000]
r=1

m=len(training)
epsilon = 1e-6 
for r in batch_size:
    theta1 = 0
    theta2 = 0
    theta0 = 0
    prev_loss = []
    if r == 1:
        size_prev_loss = 200000
    elif r== 80:
        size_prev_loss = 100000
    elif r==8000:
        size_prev_loss = 100
    else:
        size_prev_loss = 2
        epsilon = 1e-5
    i=0 
    theta_history = []
    X1_train = training["X1"].values
    X2_train = training["X2"].values
    Y_train = training["Y"].values

    X1_test = testing["X1"].values
    X2_test = testing["X2"].values
    Y_test = testing["Y"].values
    while(True):
        # preparing index for batch gradient descent
        index1= (i*r)%m
        index2= ((i+1)*r)%m if ((i+1)*r)%m !=0 else m

        # batch preparation according to index 
        X1_batch = X1_train[index1:index2]
        X2_batch = X2_train[index1:index2]
        Y_batch  = Y_train[index1:index2]

        # compute gradient 
        grad0,grad1,grad2 = compute_grads(X1_batch,X2_batch,Y_batch,theta0,theta1,theta2)

        #gradient descent update rule
        theta0,theta1,theta2 = theta0 - eta*grad0, theta1-eta*grad1, theta2-eta*grad2

        # saving history of paramters for plotting
        theta_history.append((theta0, theta1, theta2))
        i+=1

        #stopping criterion
        prev_loss.append(stochastic_MSE_loss(X1_batch,X2_batch,Y_batch,theta0,theta1,theta2))
        #print(stochastic_MSE_loss(train["X1"],train["X2"], train["Y"],theta0,theta1,theta2))
        if len(prev_loss) > size_prev_loss:
            prev_loss.pop(0)
            #print(np.mean(prev_loss[:size_prev_loss//2])- np.mean(prev_loss[size_prev_loss//2:]))
            if(np.mean(prev_loss[:size_prev_loss//2])- np.mean(prev_loss[size_prev_loss//2:])< epsilon):
                break


    theta_history = np.array(theta_history)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot path
    ax.plot(theta_history[:,0], theta_history[:,1], theta_history[:,2], 
            color="blue", linewidth=1, marker="o", markersize=2)

    # Start & End points
    ax.scatter(theta_history[0,0], theta_history[0,1], theta_history[0,2], 
            color="red", s=50, label="Start")
    ax.scatter(theta_history[-1,0], theta_history[-1,1], theta_history[-1,2], 
            color="green", s=50, label="End")

    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    ax.set_zlabel(r"$\theta_2$")
    ax.set_title(f"Parameter Trajectory (Batch size = {r})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"theta_trajectory_batch_{r}.png", dpi=300) 
    plt.close()

    print()
    print("-------------------------------------------------------------------------")
    print("For Batch Size", r, "we get the following parameter values:")
    print("Theta0 = ", theta0)
    print("Theta1 = ", theta1)
    print("Theta2 = ", theta2)
    print("Number of iterations : ", i)
    print("Training Loss is :",stochastic_MSE_loss(training["X1"],training["X2"], training["Y"],theta0,theta1,theta2))
    print("Testing Loss is:", stochastic_MSE_loss(testing["X1"],testing["X2"], testing["Y"],theta0,theta1,theta2))
    print("-------------------------------------------------------------------------")


##############################################
   # Closed form solution 
##############################################

X = training[["X1","X2"]].to_numpy()
X = np.hstack([np.ones((X.shape[0], 1)), X])
Y = training[["Y"]].to_numpy()

prod = X.T @ X
inv = np.linalg.inv(prod)

sol = inv @ X.T @ Y

theta0, theta1, theta2 = sol[0][0], sol[1][0], sol[2][0]
print()
print(" ----------------------- SOLUTION BY NORMAL EQUATIONS --------------")
print()
print("Theta0 = ", theta0)
print("Theta1 = ", theta1)
print("Theta2 = ", theta2)
print("Training Loss is :",stochastic_MSE_loss(training["X1"],training["X2"], training["Y"],theta0,theta1,theta2))
print("Testing Loss is:", stochastic_MSE_loss(testing["X1"],testing["X2"], testing["Y"],theta0,theta1,theta2))

