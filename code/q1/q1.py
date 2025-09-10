
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# LOADING DATA 
X = pd.read_csv("/home/aksh/Desktop/Assignment_archive/ML_assignement1/code/data/Q1/linearX.csv", header = None).values.flatten()
Y = pd.read_csv("/home/aksh/Desktop/Assignment_archive/ML_assignement1/code/data/Q1/linearY.csv", header = None).values.flatten()

# INITIALIZING PARAMETER VALUES WITH 0 
w = 0
b = 0
print("Learning rate can be atmost =",1/((1/len(X))*(X@X)))

'''
Here we define functions to compute MSE_loss and its gradient 
with respect to the paramters w(slope) and b(y intercept)
'''

def MSE_loss(X, Y, w, b):
    preds = w*X + b
    return np.mean((Y - preds)**2) / 2

def grad_wrt_w(X, Y, w, b):
    preds = w*X + b
    return -np.mean(X * (Y - preds))

def grad_wrt_b(X, Y, w, b):
    preds = w*X + b
    return -np.mean(Y - preds)

'''

Here we define learning rate (eta) and other important hyperparamters :-
upper_limit : gradient descent after 300000 iterations if we don't reach the convergence condition by then
loss_width : we compare the last 10 loss fucntion values, specially the max and min values of loss in this width to check for convergence 
stop_criterion : we do gradient descent until the max - min in loss width is lower than 1e-9.

'''

eta = 0.5
upper_limit = 300000
loss_width = 10         
stop_criterion = 1e-9 

loss_history = []
prev_losses = []
w_path, b_path = [], [] 
for i in range(upper_limit):
    #update rule for gradient descent
    w,b = w - eta * grad_wrt_w(X, Y, w, b),b - eta * grad_wrt_b(X, Y, w, b)

    loss = MSE_loss(X, Y, w, b)
    loss_history.append(loss)
    w_path.append(w)
    b_path.append(b)
    prev_losses.append(loss)
    if len(prev_losses) > loss_width:
        prev_losses.pop(0) 

        #checking the max and min values of loss in the last 10 iterations
        if max(prev_losses) - min(prev_losses) < stop_criterion:
            print(f"Stopping on iteration {i}, loss={loss:.6f}")
            break

print("Final loss:", MSE_loss(X, Y, w, b))
print("w =", w, " b =", b)

# PLOTTING the Training loss 
plt.figure(figsize=(8,5))
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Loss Curve during Training")
plt.legend()
plt.grid(True)
plt.show()

#PLOTTING the DATA and the learned line by GRADIENT DESCENT
plt.scatter(X, Y, color="blue", label="Data")  
x_line = np.linspace(min(X), max(X), 100)       
y_line = w * x_line + b
plt.plot(x_line, y_line, color="red", label="Fitted line")  
plt.xlabel("X (Normalized acidity)")
plt.ylabel("Y (Density of wine)")
plt.legend()
plt.title("Linear Regression Fit")
plt.show()


############################################################
 # PLOTTING 3D PLOTS and CONTOURS (PART 3, PART 4 & PART 5)
############################################################

w_range = np.linspace(w-30, w+30, 100)
b_range = np.linspace(b-20, b+20, 100)
W, B = np.meshgrid(w_range, b_range)

# Compute loss values for grid
Z = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = MSE_loss(X, Y, W[i, j], B[i, j])

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W, B, Z, cmap="viridis", edgecolor='none', alpha=0.9)

ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("MSE Loss")
ax.set_title("Error surface with respect to (w, b)")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# plotting gradient descent points on the 3D loss curve 
w_range = np.linspace(w-30, w+30, 100)
b_range = np.linspace(b-20, b+20, 100)
W, B = np.meshgrid(w_range, b_range)

Z = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = MSE_loss(X, Y, W[i, j], B[i, j])

fig = plt.figure(figsize=(15,11))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(W, B, Z, cmap="viridis", edgecolor='none', alpha=0.8)
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_zlabel("MSE Loss")
ax.set_title("Gradient Descent on Error Surface")

# Animate gradient descent path
for wi, bi in zip(w_path, b_path):
    if not plt.fignum_exists(fig.number):
        break
    ax.scatter(wi, bi, MSE_loss(X, Y, wi, bi), color="red", s=25)
    plt.draw()
    plt.pause(0.2)

### CONTOURS 

fig, ax = plt.subplots(figsize=(10,7))
contours = ax.contour(W, B, Z, levels=50, cmap="viridis")
ax.set_xlabel("w")
ax.set_ylabel("b")
ax.set_title("Gradient Descent Path on Contour of Error Surface")

for wi, bi in zip(w_path, b_path):
    if not plt.fignum_exists(fig.number):  
        break
    ax.scatter(wi, bi, color="red", s=30)
    plt.draw()
    plt.pause(0.2)

plt.show()

# defining gradeint descent fuction to perform it for diffent eta values 
def gradient_descent(X, Y, eta, max_iter=3000, tol=1e-9, loss_width=10):
    w, b = 0, 0
    w_path, b_path, loss_history, prev_losses = [], [], [], []
    
    for i in range(max_iter):
        w = w - eta * grad_wrt_w(X, Y, w, b)
        b = b - eta * grad_wrt_b(X, Y, w, b)
        
        loss = MSE_loss(X, Y, w, b)
        w_path.append(w)
        b_path.append(b)
        loss_history.append(loss)
        
        prev_losses.append(loss)
        if len(prev_losses) > loss_width:
            prev_losses.pop(0)
            if max(prev_losses) - min(prev_losses) < tol:
                break
    return w_path, b_path

etas = [0.001, 0.025, 0.1]
paths = [gradient_descent(X, Y, eta) for eta in etas]

all_w = np.concatenate([p[0] for p in paths])
all_b = np.concatenate([p[1] for p in paths])

w_range = np.linspace(min(all_w)-10, max(all_w)+10, 200)
b_range = np.linspace(min(all_b)-10, max(all_b)+10, 200)
W, B = np.meshgrid(w_range, b_range)

Z = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = MSE_loss(X, Y, W[i, j], B[i, j])

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, eta in zip(axes, etas):
    ax.contour(W, B, Z, levels=50, cmap="viridis")
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_title(f"Learning rate = {eta}")

# GD paths for different eta values 
max_len = max(len(p[0]) for p in paths)
for step in range(max_len):
    if not plt.fignum_exists(fig.number): 
        break
    for ax, (w_path, b_path) in zip(axes, paths):
        if step < len(w_path):
            ax.scatter(w_path[step], b_path[step], color="red", s=20)
    plt.draw()
    plt.pause(0.2)
plt.show()
