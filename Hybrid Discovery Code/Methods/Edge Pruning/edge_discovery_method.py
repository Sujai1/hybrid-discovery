#Packages
import numpy as np
from causallearn.utils.cit import CIT


#Edge Discovery Method
def edge_discovery(topo_sort, data, cit_test = "kci", alpha = 0.01):
    '''
    Inputs:
    topo_sort - a topological sort, where index(x)<index(y) implies y cannot cause x
    data - table that has columns for each variable and rows of samples (np matrix)
    '''
    #Initialization
    d = data.shape[1]
    #find out how to build xrange
    parents = [set() for _ in range(d)]
    children = [set() for _ in range(d)]
    ind_collection = marg_ind(data, cit_test, alpha)
    #CIT Algorithm
    for i in range(1,d):
        b = topo_sort[i]
        for j in range(i-1,-1,-1):
            a = topo_sort[j]
            #only check for edge if a and b are marginally dependent
            if b in ind_collection[a]:
                parents, children = edge_detection(a,b,
                                                    parents,
                                                    children,
                                                    data,
                                                    ind_collection,
                                                    cit_test,
                                                    alpha)
    return parents, children

# Subroutine
def edge_detection(a, b, parents, children, data, ind_collection, cit_test = "kci", alpha = 0.01):
    '''
    Inputs:
    a - the possible parent
    b - the possible child
    parents - a list of lists, where the list at index i contains all parents of node i collected so far
    child - a list of lists, where the list at index i contains all children of node i collected so far
    
    Returns:
    the parent set and children set at this step of the edge detection
    '''
    #Potential Confounder Set
    confounders = parents[a]
    #Potential Mediator Set
    mediators = parents[b] 
    # object views vs copy constructing new objects
    cond_set = confounders.union(mediators)
    #Include only marginally dependent (on b) potential mediators and confounders
    cond_set = cond_set.intersection(ind_collection[b])
    fisherz_obj = CIT(data, cit_test)
    pvalue = fisherz_obj(a,b,list(cond_set))
    if pvalue < alpha:
        parents[b].add(a)
        children[a].add(b)
    return parents, children

# Subroutine
def marg_ind(data, cit_test = "kci", alpha = 0.01):
    '''Inputs: 
    data - table that has d columns for each variable and n rows of samples (np matrix)
    Returns:
    dxd symmetric matrix where matrix[i,j] = matrix[j,i] = 1 if x_i ind x_j, 0 otherwise.
    '''
    #Find Data Dimensionality
    d = data.shape[1]
    #Create matrix: there is a 0 in i,j (and j,i) if x_i \ind x_j, 1 otherwise.
    ind_collection = [set() for _ in range(d)]
    #Initialize Kernel Independence Test
    fisherz_obj = CIT(data, cit_test)
    for i in range(d):
        for j in range(i+1, d):
            if fisherz_obj(i,j,[]) < alpha:
                ind_collection[i].add(j)
                ind_collection[j].add(i)
    return ind_collection