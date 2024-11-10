
# D_top metric for topological sorting methods

def count_topological_errors(M, k):
    """
    Counts the number of topological sorting errors in a DAG given its adjacency matrix and a topological order.

    :param M: A 2D list (list of lists) representing the adjacency matrix of the DAG.
              M[i][j] != 0 means there is a directed edge from j to i.
    :param k: A list representing the nodes in topological order.
    :return: The percentage of topological errors.
    """
    # Index each node based on its position in the topological order for quick lookup.
    index_map = {node: idx for idx, node in enumerate(k)}

    #Sum of potential errors
    sum = 0
    
    errors = 0
    # Check each pair (i, j) based on their indices in the topological order.
    for idx_i, i in enumerate(k):
        for idx_j, j in enumerate(k):
            if M[i][j] != 0:
                sum+=1
            # If i appears after j in the topological order but i causes j,
            # it's an error because i -> j should mean i should come before j.
                if idx_i > idx_j :
                    errors += 1

    if sum == 0:
        return 1

    # This function returns the % of correct ancestral relations determined (number of necessary ancestral relations)
    return (sum-errors)/sum