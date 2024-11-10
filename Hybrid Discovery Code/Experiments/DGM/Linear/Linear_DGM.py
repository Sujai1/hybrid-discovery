#Packages
import numpy as np
from scipy.stats import bernoulli, uniform

# Data Generation
def generate_adjacency_matrix(p, s):
    # Initialize an empty adjacency matrix
    adj_matrix = np.zeros((p, p))
    # Fill the matrix without creating cycles
    for i in range(p):
        for j in range(i):
            adj_matrix[i][j] = bernoulli.rvs(s)
    # Return the lower triangular part of the matrix which represents the DAG
    return adj_matrix

def replace_nonzero_entries(adj_matrix):
    nonzero_indices = np.where(adj_matrix == 1)
    for i, j in zip(nonzero_indices[0], nonzero_indices[1]):
        adj_matrix[i][j] = np.random.choice([np.random.uniform(low=-0.5, high=-1.5), np.random.uniform(low=0.5, high=1.5)])
    return adj_matrix

def generate_external_influences(p, n):
    influences = np.zeros((p, n))
    for i in range(p):
        variance = np.random.choice([1, 3])
        range_width = np.sqrt(12 * variance)
        influences[i, :] = np.random.uniform(low=-range_width / 2, high=range_width / 2, size=n)
    return influences

def generate_observed_variables(adj_matrix, external_influences):
    inverse_term = np.linalg.inv(np.eye(adj_matrix.shape[0]) - adj_matrix)
    observed_variables = np.dot(inverse_term, external_influences).T
    mean = observed_variables.mean(axis=0)
    std = observed_variables.std(axis=0)
    standardized_observed_variables = (observed_variables - mean) / std
    return standardized_observed_variables

def permute_data(data):
    permutation = np.random.permutation(data.shape[1])  # Use number of columns (features p) for permutation
    permuted_data = data[:, permutation]  # Apply permutation to columns (features p)
    return permuted_data, permutation

def permute_adjacency_matrix(adj_matrix, permutation):
    permuted_matrix = adj_matrix[permutation, :][:, permutation]
    return permuted_matrix

def experimental_procedure(p, s, n):
    adj_matrix = generate_adjacency_matrix(p, s)
    adj_matrix = replace_nonzero_entries(adj_matrix)
    external_influences = generate_external_influences(p, n)
    observed_variables = generate_observed_variables(adj_matrix, external_influences)
    permuted_data, permutation = permute_data(observed_variables)  # Correct permutation for observed data
    adj_matrix = permute_adjacency_matrix(adj_matrix, permutation)  # Use the same permutation
    return adj_matrix, permuted_data