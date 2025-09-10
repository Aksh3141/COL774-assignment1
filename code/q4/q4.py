# GDA
import numpy as np
import matplotlib.pyplot as plt


X = np.loadtxt("/home/aksh/Desktop/Assignment_archive/ML_assignement1/code/data/Q4/q4x.dat")  
y = np.loadtxt("/home/aksh/Desktop/Assignment_archive/ML_assignement1/code/data/Q4/q4y.dat",dtype=str)  
label_map = {"Alaska": 0, "Canada": 1}
y = np.array([label_map[val] for val in y])
print()
print("----------------------------------")
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)
print("----------------------------------")

# normalizing X for training
print(X[1].mean())
print(X[1].std())
exit()
X = X - X.mean()/X.std()
m, n = X.shape
mu0 = np.mean(X[y == 0], axis=0)
mu1 = np.mean(X[y == 1], axis=0)

# Shared covariance
Sigma = np.zeros((n, n))
for i in range(m):
    xi = X[i]
    mui = mu0 if y[i] == 0 else mu1
    diff = (xi - mui).reshape(-1, 1)
    Sigma += diff @ diff.T
Sigma /= m

print("mu0 (Alaska):", mu0)
print("mu1 (Canada):", mu1)
print("The covariance matrix is:")
print(Sigma)


# Data Plotting
plt.figure(figsize=(7,6))
plt.scatter(X[y==0,0], X[y==0,1], marker='o', label='Alaska', edgecolor='k')
plt.scatter(X[y==1,0], X[y==1,1], marker='x', label='Canada', s=60)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Training Data")
plt.legend()
plt.grid(True)
plt.show()


# calculating the equation of line from learned parameters
phi = np.mean(y) 
invSigma = np.linalg.inv(Sigma)
w = invSigma @ (mu1 - mu0)
b = 0.5 * (mu0.T @ invSigma @ mu0 - mu1.T @ invSigma @ mu1) + np.log(phi/(1-phi))

# Plot boundary on same figure
x_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
y_vals = -(w[0]*x_vals + b)/w[1]   
plt.figure(figsize=(7,6))
plt.scatter(X[y==0,0], X[y==0,1], marker='o', label='Alaska', edgecolor='k')
plt.scatter(X[y==1,0], X[y==1,1], marker='x', label='Canada', s=60)
plt.plot(x_vals, y_vals, 'g--', linewidth=2, label="Linear GDA boundary")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Linear GDA Boundary")
plt.legend()
plt.grid(True)
plt.show()



m0 = np.sum(y==0)
m1 = np.sum(y==1)

Sigma0 = ((X[y==0] - mu0).T @ (X[y==0] - mu0)) / m0
Sigma1 = ((X[y==1] - mu1).T @ (X[y==1] - mu1)) / m1

print()
print("Printing the convariance matrix")
print(Sigma0)
print()
print(Sigma1)
print("--------------------------------")
invS0 = np.linalg.inv(Sigma0)
invS1 = np.linalg.inv(Sigma1)
a, logdet0 = np.linalg.slogdet(Sigma0)
b, logdet1 = np.linalg.slogdet(Sigma1)

def quad_val(x):
    x = x.reshape(-1,1)
    m0v = mu0.reshape(-1,1)
    m1v = mu1.reshape(-1,1)
    val = ( (x-m0v).T @ invS0 @ (x-m0v) - (x-m1v).T @ invS1 @ (x-m1v) ).item()
    val += (logdet0 - logdet1) + 2*np.log((1-phi)/phi)
    return val

# grid
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
    np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = np.array([quad_val(pt) for pt in grid]).reshape(xx.shape)

plt.figure(figsize=(7,6))
plt.scatter(X[y==0,0], X[y==0,1], marker='o', label='Alaska', edgecolor='k')
plt.scatter(X[y==1,0], X[y==1,1], marker='x', label='Canada', s=60)
plt.plot(x_vals, y_vals, 'g--', label="Linear boundary")
plt.contour(xx, yy, Z, levels=[0], colors='r', linewidths=2)
plt.xlabel("x1 (freshwater)")
plt.ylabel("x2 (marine)")
plt.title("Linear vs Quadratic GDA Boundary")
plt.legend()
plt.grid(True)
plt.show()

