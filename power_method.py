import numpy as np
import matplotlib.pyplot as plt
import math

def power_method(matrix):
    n = matrix.shape[0]
    
    # Initialize a random vector as an approximation of the dominant eigenvector
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    
    eigenvalues = []
    eigenvectors = []

    for i in range(30):
        x_new = np.dot(matrix, x)
        eigenvalue = np.dot(x_new, x)
        eigenvalues.append(eigenvalue)
        
        # Normalize the eigenvector
        x_new = x_new / np.linalg.norm(x_new)
        eigenvectors.append(x_new)
        
        x = x_new
    
    return eigenvalues, eigenvectors

# Example matrix
A = np.array([[4, -math.sqrt(2)], [-math.sqrt(2), 3]]) # *** TODO: Change the matrix A here

# Applying power method
eigenvalues, eigenvectors = power_method(A)

# Calculate true eigenvalue and eigenvector
true_eigenvalue = 5
true_eigenvector = np.array([-math.sqrt(2), 11]) # *** TODO: Put the value of the true eigenvector here
true_eigenvector = true_eigenvector / np.linalg.norm(true_eigenvector)

# Predicted convergence rate
predicted_rate = 2/5

# Plotting convergence of eigenvalues
plt.figure(figsize=(10, 6))
error_eigenvalue = np.abs(np.array(eigenvalues) - true_eigenvalue)
plt.semilogy(range(1, len(error_eigenvalue) + 1), error_eigenvalue, marker='o', linestyle='-', label='eigenvalue')
error_eigenvector = [np.minimum(np.linalg.norm(np.abs(approx_eig - true_eigenvector)), np.linalg.norm(np.abs(approx_eig + true_eigenvector))) for approx_eig in eigenvectors]
plt.semilogy(range(1, len(error_eigenvector) + 1), error_eigenvector, marker='o', linestyle='-', label='eigenvector')
plt.semilogy(range(1, len(error_eigenvalue)+1), np.power(predicted_rate, np.arange(1, len(error_eigenvalue)+1)), marker='x', linestyle=':', label='predicted rate')
plt.title('Convergence Error of Dominant Eigenpair')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()
