#Packages
import numpy as np
from sklearn.linear_model import LinearRegression
from conditional_independence import hsic_test

#Method

def LHTS(data, alpha = 0.05, condition = 100, r = 0.5):
    '''Returns a topological ordering, given data.'''
    # Find ancestor table
    ancestors = ancestor_table(data, alpha)
    # Find linear topological ordering
    lin_sort =  linear_sort(ancestors)
    lay_sort = hierarchical_sort(ancestors)

    return lin_sort, lay_sort, ancestors

# Bunch of submethods

def marg_dep(data, alpha = 0.05):
    '''
    Determine marginal dependencies between variables.

    Parameters:
    data (np.array): Dataset with d variables as columns and n samples as rows.
    alpha (float): Threshold for determining dependence, default is 0.01.

    Returns:
    list of lists: Each sublist contains indices of variables that are marginally dependent with the variable at the index of the sublist.
    '''
    # Number of variables
    d = data.shape[1]
    # Initialize list of lists for dependencies
    ind_collection = [[] for _ in range(d)]
    # prune original data to only use k samples (no need to use all data in pairwise)
    k = min(data.shape[0],10000)
    random_indices = np.random.choice(data.shape[0], size = k, replace = False)
    sampled_data = data[random_indices,:]
    # Test for dependencies between all pairs (i, j) where neither are sorted
    for i in range(d):
        for j in range(i+1, d):
            #Append if x_i and x_j are dependent
            if (hsic_test(sampled_data, i,j,[])['p_value'] < alpha):
                ind_collection[i].append(j)
                ind_collection[j].append(i)
    return ind_collection

def marg_pop(marg_dep, ancestors):
    # Number of variables
    d = data.shape[1]
    #Populate ancestor lists using marg_dep information
    for i in range(d):
        for j in range(i+1,d):
            if j not in marg_dep[i]:
                ancestors[i][j] = 1
                ancestors[j][i] = 1

    return ancestors


def get_mutual_ancestors(i, j, ancestors):
    """ Returns indices of mutual ancestors of x_i and x_j """
    d = len(ancestors)
    return [k for k in range(d) if ancestors[i][k] == 3 and ancestors[j][k] == 3 and k != i and k != j]



def is_y_ancestor_of_x_ancestors(A, x, y):
    """
    Check if node y is an ancestor of any of x's ancestors in the given table A.

    :param A: 2D list representing the ancestor relationships (matrix)
    :param x: Node whose ancestors to check against y
    :param y: Node to check if it is an ancestor of any of x's ancestors
    :return: True if y is an ancestor of any of x's ancestors, False otherwise
    """
    # check if y is direct parent of x
    if A[y][x] == 2:
        return True
    n = len(A)  # Assuming A is a square matrix
    for i in range(n):
        # Check if i is an ancestor of x
        if A[i][x] == 2:
            # Check if y is an ancestor of i
            if A[y][i] == 2:
                return True
    return False

def find_max_location(lists,ancestors):
    """
    Finds the location of the maximum value in a list of lists.

    :param lists: A list of lists containing numerical values.
    :return: A tuple (i, j) where i is the index of the sublist containing the max value,
             and j is the index of the max value within that sublist.
    """
    if not lists:
        return None  # Return None if the list of lists is empty

    max_value = float('-inf')
    max_location = (-1, -1)  # Initialize to an invalid index
    for i, sublist in enumerate(lists):
        for j, value in enumerate(sublist):
            if is_y_ancestor_of_x_ancestors(ancestors, i, j) == False:
                
                if value > max_value:
                    max_value = value
                    max_location = (i, j)
                
                
    return max_location

def ancestor_update(ancestors):
    # for each node, make sure that all of the ancestors of its ancestors are ancestors of that node
    counter = False
    for i in range(len(ancestors)):
        for j in range(len(ancestors)):
            if is_y_ancestor_of_x_ancestors(ancestors, i, j):
                for k in range(len(ancestors)):
                    if is_y_ancestor_of_x_ancestors(ancestors, j, k) and ancestors[k][i] != 2:
                        ancestors[k][i] = 2
                        ancestors[i][k] = 3
                        counter = True
    return ancestors, counter

def pair_reg(data, ancestors, ancestor_residuals, alpha = 0.01):
    '''Subroutine that updates the ancestor table after each round of conditional regressions.'''
    # -1 for i's relation to i
    # 0 means i,j unknown relations, 
    # 1 means i,j have no ancestral relation, 
    # 2 means i < j,  
    # 3 means j < i,

    # Number of variables
    d = data.shape[1]
    # Initialize list of lists for p-values
    plist = [[0 for j in range(d)] for i in range(d)]
   

    #Populate p-list using pairwise regression + adding mutual ancestors as covariates
    for i in range(d):
        for j in [x for x in range(d) if x!=i and ancestors[i][x] == 0]:
    
            #Grab mutual ancestors
            mutual_ancestor_indices = get_mutual_ancestors(i,j,ancestors)
            #Grab x_i, x_j
            x_i = data[:,i].reshape(-1,1)
            x_j = data[:,j].reshape(-1,1)
            
            #Regress mutual ancestors out of x_i,x_j
            i_storage = False
            j_storage = False
            for o in range(len(ancestor_residuals)):
                           if ancestor_residuals[o][0] == set(mutual_ancestor_indices) and ancestor_residuals[o][1] == [i]:
                               x_i = ancestor_residuals[o][2]
                               i_storage = True
                           if ancestor_residuals[o][0] == set(mutual_ancestor_indices) and ancestor_residuals[o][1] == [j]:
                               x_j = ancestor_residuals[o][2]
                               j_storage = True
            for k in mutual_ancestor_indices:
                #Obtain ancestor
                x_k = data[:,k].reshape(-1,1)
                if i_storage == False:
                    #Fit Linear Regression x_i on x_k
                    reg = LinearRegression().fit(x_k,x_i)
                    x_i_pred = reg.predict(x_k)
                    #Set residual = x_i
                    x_i = x_i - x_i_pred
                if j_storage == False:
                    #Fit Linear Regression x_j on x_k
                    reg = LinearRegression().fit(x_k,x_j)
                    x_j_pred = reg.predict(x_k)
                    #Set residual = x_i
                    x_j = x_j - x_j_pred
            # Store residuals
            if i_storage == False:
                ancestor_residuals.append([mutual_ancestor_indices, [i], x_i])
            if j_storage == False:
                ancestor_residuals.append([mutual_ancestor_indices, [j], x_j])
                    
            #Fit Linear Regression x_j on x_i
            reg = LinearRegression().fit(x_i,x_j)
            x_j_pred = reg.predict(x_i)
            #Residuals
            residuals_ij = x_j - x_j_pred
            #Check for independence of residuals with x_i
            resid_data = np.hstack((x_i,residuals_ij.reshape(-1,1)))
            #Using HSIC - Append p-value (0 IFF x_i is not independent of residual)
            plist[i][j] = hsic_test(resid_data, 0,1,[])['p_value']

    # Update Tracker
    tracker = 0

    #Populate ancestor list using p-list
    for i in range(d):
        for j in [x for x in range(d) if x!=i and ancestors[i][x] == 0]:
            #x_i and x_j share no ancestral relation (high bar set for consistency)
            if plist[i][j] >= .5 and plist[j][i] >= .5:
                ancestors[i][j] = 1
                ancestors[j][i] = 1
                tracker = 1
               
            
            #x_i is ancestor of x_j
            if plist[i][j] >= .1 and plist[j][i] <= .05:
                ancestors[i][j] = 2
                ancestors[j][i] = 3
                tracker = 1
                
            #x_j is ancestor of x_i
            elif plist[i][j] <= .05 and plist[j][i] >= .1:
                ancestors[i][j] = 3
                ancestors[j][i] = 2
                tracker = 1
             
    
    
    # Robustness 

    # Find Maximally Independent Node - try the node that is maximally independent on average
    if tracker == 0:
        i,j = find_max_location(plist, ancestors)
        ancestors[i][j] = 2
        ancestors[j][i] = 3
    
    # Update as much as possible
    counter = True
    while counter: 
        ancestors, counter = ancestor_update(ancestors)
    
    return ancestors


def ancestor_table(data, alpha):
    '''
    Returns a completed ancestor table, given a dataset.
    '''
    d = data.shape[1]
    ancestors = [[-1 if i == j else 0 for j in range(d)] for i in range(d)]

    # Obtain marginal independence table
    marg = marg_dep(data)

    # Obtain initial ancestor set
    ancestors = marg_pop(marg,ancestors)
    ancestor_residuals = []
    i = 0
    while any(0 in sublist for sublist in ancestors):
        ancestors = pair_reg(data,ancestors, ancestor_residuals, alpha = alpha)
        i+= 1
        #print(i)
        if i == 2000:
            print("Loop")
            return []
    
    return ancestors


def linear_sort(ancestors):
    '''
    Returns a topological ordering, given a completed ancestor table.
    '''
    # Initialize
    d = len(ancestors)
    # If the ancestor table failed to complete
    if d == 0:
        return "Error"
    order = []
    remaining = set(range(d))  # Set of indices of nodes still in the graph

    def has_no_descendants(node):
        """ Check if the node has no descendants in the remaining graph """
        return all(ancestors[node][j] != 2 for j in remaining)
    
    def has_fewest_descendants():
        min_descend = float('inf')  # Start with a very large number
        node_with_fewest_descendants = None
    
        # Iterate over each node in 'remaining' to count its ancestors
        for node in remaining:
        # Count how many times `2` appears indicating an descendants
            descendant_count = sum(1 for j in remaining if ancestors[node][j] == 2)
            # Update if the current node has fewer ancestors than the previous minimum
            if descendant_count < min_descend:
                min_descend = descendant_count
                node_with_fewest_descendants = node

        return node_with_fewest_descendants

    while remaining:
        # Find a node that has no descendants/ node with fewest descendants
        #sink = next(node for node in remaining if has_no_descendants(node))
        sink = has_fewest_descendants()
        # Add it to the order
        order.append(sink)
        # Remove it from the set of remaining nodes
        remaining.remove(sink)
    order.reverse()

    return order

def hierarchical_sort(ancestors):
    '''
    Returns a topological ordering, given a completed ancestor table.
    '''
    # Initialize
    d = len(ancestors)
    # If the ancestor table failed to complete
    if d == 0:
        return "Error"
    order = []

    remaining = set(range(d))  # Set of indices of nodes still in the graph

    def has_no_ancestors(node):
        """ Check if the node has no ancestors in remaining """
        return all(ancestors[node][j] != 3 for j in remaining)
    
    def has_fewest_ancestors():
        min_ancestors = float('inf')  # Start with a very large number
        node_with_fewest_ancestors = None
    
        # Iterate over each node in 'remaining' to count its ancestors
        for node in remaining:
        # Count how many times `3` appears indicating an ancestor
            ancestor_count = sum(1 for j in remaining if ancestors[node][j] == 3)
            # Update if the current node has fewer ancestors than the previous minimum
            if ancestor_count < min_ancestors:
                min_ancestors = ancestor_count
                node_with_fewest_ancestors = node

        return node_with_fewest_ancestors

    while remaining:
        # Initialize layer
        layer = []
        # Find all nodes with no ancestors in remaining
        for node in remaining:
            if has_no_ancestors(node):
                layer.append(node)
        
        if layer == []:
            layer.append(has_fewest_ancestors())
    
        # Add it to the order
        order.append(layer)
        # Remove layer from the set of remaining nodes
        set_layer = set(layer)
        remaining = list(filter(lambda x: x not in set_layer, remaining))
    
    return order




