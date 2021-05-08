#plot wave-number vs no of jacobi iterations
import matplotlib.pyplot as plt
import numpy as np



#load data for beta = 1, 10, 100
data = np.loadtxt("Iter.dat")

plt.scatter(data[:,0], data[:,1], marker = 'o')

plt.title("wave-number vs no of jacobi iterations")
plt.xlabel("wave number k")
plt.ylabel("N iterations")
plt.savefig("results/jacobi1")
