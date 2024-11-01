# Packages
import numpy as np
from CausalDisco.analytics import r2_sortability


# Data Generation Method
def generate_quadratic_data(n, d, avg_edges, max_r2_sortability=0.7, max_attempts=100):
    from CausalDisco.analytics import r2_sortability
    
    def generate_data():
        p = avg_edges
        adjacency_matrix = generate_adjacency_matrix(d, p)
        topological_order = topological_sort(adjacency_matrix)
        X = np.zeros((n, d))

        for node in topological_order:
            parents = np.where(adjacency_matrix[:, node] == 1)[0]
            if len(parents) == 0:
                # Root node initialization
                # Uniform Noise
                #variance = np.where(np.random.rand(n) < 0.5, 1, np.sqrt(3))
                #X[:, node] = np.random.uniform(0, variance, n)
                
                # Normal Noise
                #variance = np.where(np.random.rand(n) < 0.5, (1/12)**0.5, (1/4)**0.5)
                #X[:, node] = np.random.normal(0, variance, n)

                # Laplace Noise
                #variance = np.where(np.random.rand(n) < 0.5, np.sqrt(1/24), np.sqrt(9/24))
                #X[:, node] = np.random.laplace(0, variance, n)
            else:
                #Quadratic causal mechanism
                parent_data = X[:, parents]

                # Random Weights
                lower_range = np.random.uniform(-2.5, -1.5, parent_data.shape[1])
                upper_range = np.random.uniform(1.5, 2.5, parent_data.shape[1])
                random_multipliers = np.where(np.random.rand(parent_data.shape[1]) < 0.5, lower_range, upper_range)
                parent_data = parent_data * random_multipliers  
                
                # Quadratic Mechanisms
                quadratic_sum = np.sum(parent_data, axis=1)**2

                # Add Noise
                variance = np.where(np.random.rand(n) < 0.5, 1, np.sqrt(3))
                X[:, node] = quadratic_sum + np.random.uniform(0, variance, n)
                #variance = np.where(np.random.rand(n) < 0.5, (1/12)**0.5, (1/4)**0.5)
                #X[:, node] = quadratic_sum + np.random.normal(0, variance, n)
                #variance = np.where(np.random.rand(n) < 0.5, np.sqrt(1/24), np.sqrt(9/24))
                #X[:, node] = quadratic_sum + np.random.laplace(0, variance, n)
                
                # Normalize generated variable to prevent values from collapsing to 0 due to quadratic 			# mechanisms
                X[:, node] = normalize_vector(X[:, node])

        # Normalize all variables at the end
        for node in range(d):
            X[:, node] = normalize_vector(X[:, node])
        
        return X, adjacency_matrix, topological_order
    
    # Try to produce data with low-R^2 sortability.
    attempt = 0
    while attempt < max_attempts:
        X, adjacency_matrix, topological_order = generate_data()
        try:
            r2_value = r2_sortability(X, adjacency_matrix)
        except Exception as e:
            continue

        if r2_value <= max_r2_sortability:
            break
        attempt += 1

    parents_list = [set(np.where(adjacency_matrix[:, node] == 1)[0]) for node in range(d)]
    
    if attempt == max_attempts:
        print(f"Reached maximum attempts ({max_attempts}) without achieving desired sortability.")

    return X, adjacency_matrix, topological_order, parents_list

def normalize_vector(v):
    return (v - np.mean(v)) / np.std(v)

##########

# Submethods
def generate_adjacency_matrix(d, p):
    adjacency_matrix = np.zeros((d, d), dtype=int)
    for i in range(d):
        for j in range(i + 1, d):
            if np.random.rand() < p:
                adjacency_matrix[i, j] = 1
    return adjacency_matrix

def topological_sort(adjacency_matrix):
    d = adjacency_matrix.shape[0]
    in_degree = np.sum(adjacency_matrix, axis=0)
    zero_in_degree = [node for node in range(d) if in_degree[node] == 0]
    topological_order = []

    while zero_in_degree:
        node = zero_in_degree.pop()
        topological_order.append(node)
        for i in range(d):
            if adjacency_matrix[node, i] == 1:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    zero_in_degree.append(i)

    if len(topological_order) != d:
        raise ValueError("The graph has cycles or is disconnected.")

    return topological_order

   