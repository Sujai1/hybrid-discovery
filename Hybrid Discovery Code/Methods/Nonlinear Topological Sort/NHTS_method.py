import numpy as np
from sklearn.kernel_ridge import KernelRidge
from npeet import entropy_estimators as ee
from causallearn.utils.cit import CIT
from conditional_independence import hsic_test


def check_independence(xi, xj, thresh):
    """
    Check if xi and xj are independent using Kernel Conditional Independence (KCI) test.
    """
    data = np.column_stack((xi, xj))
    kci_obj = CIT(data, "kci")
    pValue = kci_obj(0, 1, [])
    return pValue > thresh

def check_conditional_independence(xi, xj, given, thresh):
    """
    Check if xi and xj are conditionally independent given 'given' using Kernel Conditional Independence (KCI) test.
    """
    data = np.column_stack((xi, xj, given))
    kci_obj = CIT(data, "kci")
    pValue = kci_obj(0, 1, list(range(2, data.shape[1])))
    return pValue > thresh

def calculate_residual(y, X):
    """
    Calculate the residual of y regressed on X using Kernel Ridge Regression.
    """
    krr = KernelRidge(kernel='polynomial', alpha=1, degree=3, coef0=1)
    krr.fit(X, y)
    y_pred = krr.predict(X)
    residuals = y - y_pred
    return residuals

def get_Pij(i, j, ind, features, d):
    """
    Get the set of features that are independent of xi but not independent of xj.
    """
    Pij = []
    for k in range(d):
        if k != i and k != j:
            if k not in ind[i] and k in ind[j]:
                Pij.append(features[k])
    return np.array(Pij).T

def check_PP2(i, PRS, d):
    '''Checks whether PP2 criterion holds for i: i must be identified in PP2 relation with at least one j to be a root, and if a j is in PP2 relation with i,
    i cannot be a root.'''
    pot_root = True
    for j in range(d):
        if j!=i:
            if (j,i) in PRS and PRS[(j,i)] == 'PP2':
                pot_root = False
    return pot_root


def hierarchical_topological_sort(features, ind):
    d = len(features)
    PRS = {}
    pi_H = {}

    # Stage 1: Not-PP1 Relations
    for i in range(d):
        for j in range(d):
            if i != j:
                if i in ind[j] or j in ind[i]:
                    PRS[(i, j)] = 'Not in PP1'

    for i in range(d):
        if ind[i] == []:
            PRS[i] = 'Isolated'
            pi_H[i] = 1

    # Stage 2: PP2 Relations
    for i in range(d):
        for j in range(d):
            if (i, j) not in PRS or PRS[(i, j)] != 'Not in PP1':
                continue
            Pij = get_Pij(i, j, ind, features, d)
            xj_residual = calculate_residual(features[j], features[i].reshape(-1, 1))
            if Pij.size > 0:
                xj_residual_P = calculate_residual(features[j], np.hstack((features[i].reshape(-1, 1), Pij)))
            else:
                xj_residual_P = xj_residual
            if check_independence(features[i], xj_residual, thresh=0.05) or check_independence(features[i], xj_residual_P, thresh=0.05):
                PRS[(i, j)] = 'PP2'

    # Stage 3: Root Identification
    for i in range(d):
        if i in PRS and PRS[i] == 'Isolated':
            continue

        dependents = [features[k] for k in range(d) if k != i and (i, k) in PRS and PRS[(i, k)] != 'PP2']
        flag = True
        for xk in dependents:
            if all(check_conditional_independence(features[j], xk, features[i], thresh=0.05) for j in range(d) if (i, j) in PRS and PRS[(i, j)] == 'PP2'):
                flag = False
                # If the above condition holds, i cannot be a root, so we stop immediately
                break
        if flag == True:
            pi_H[i] = 1
        
    roots = [i for i in range(d) if i in pi_H and pi_H[i] == 1]

    return roots

def marg_dep(data, alpha=0.01):
    d = data.shape[1]
    ind_collection = [[] for _ in range(d)]
    for i in range(d):
        for j in range(i + 1, d):
            if hsic_test(data, i, j, [])['p_value'] < alpha:
                ind_collection[i].append(j)
                ind_collection[j].append(i)
    return ind_collection

def nonlinear_sort(sorted_list, unsorted_list, ind, data):
    while unsorted_list:
        measures = np.full(data.shape[1], np.inf)
        for x in unsorted_list:
            anc_x = ind[x]
            features = list(set(anc_x) & set(sorted_list))
            if not features:
                # If no features are found, set measure to a high value (indicating low priority)
                measures[x] = np.inf
                continue
            X = np.array([data[:, y] for y in features]).T
            y = np.array(data[:, x])
            krr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.01)
            krr.fit(X, y)
            residuals = y - krr.predict(X)
            mi_values = []
            for y in features:
                mi = ee.mi(data[:, y], residuals)
                mi_values.append(max(0, mi))
 
            # Use a cutoff to decide if x in next layer, for comparison to linear sort (yields a linear sort not nonlinear sort)
            if all(mi_values[j] < 0.05 for j in range(0,len(features))):
                measures[x] = 0
            #Else, use avg to ensure at least one vertex gets selected
            else:
                measures[x] = np.mean(mi_values)

        
        # Check if all measures are np.inf
        if np.all(measures == np.inf):
            # If all measures are np.inf, randomly select an element from unsorted_list
            min_index = np.random.choice(unsorted_list)
        else:
            # Select just one vertex for comparison with linear topological sorts
            min_index = np.argmin(measures)
        
        sorted_list.append(min_index)
        unsorted_list.remove(min_index)
    return sorted_list


def NHTS(data):
    """
    Nonlinear Hierarchical Topological Sort (NHTS) function.
    
    Parameters:
    data (np.array): Dataset with d variables as columns and n samples as rows.
    
    Returns:
    list: Topological ordering of the variables, where if y comes after x, y cannot cause x.
    """
    ind = marg_dep(data)
    roots = hierarchical_topological_sort(data.T, ind)
    unsorted = [i for i in range(data.shape[1]) if i not in roots]
    output = nonlinear_sort(roots, unsorted, ind, data)
    return output
