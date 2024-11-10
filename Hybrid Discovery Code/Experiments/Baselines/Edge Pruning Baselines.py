import numpy as np
from sklearn.linear_model import LassoCV
from patsy import dmatrix
from statsmodels.gam.api import GLMGam, BSplines
from dcor import independence
from sklearn.ensemble import RandomForestRegressor

# Edge Pruning Baselines

def lasso_parents(data, causal_order):
    n, d = data.shape
    parents = [set() for _ in range(d)]

    # Initial parent determination using all preceding variables in the causal order
    for i, target in enumerate(causal_order):
        if i == 0:
            continue
        parents[target] = set(causal_order[:i])

    for i, target in enumerate(causal_order):
        X = data[:, causal_order[:i]]
        y = data[:, target]
        if X.shape[1] == 0:
            continue
        lasso = LassoCV().fit(X, y)
        zero_indices = np.where(lasso.coef_ == 0)[0]
        for j in zero_indices:
            parents[target].remove(causal_order[j])

    return parents

def gam_parents(data, causal_order):
    n, d = data.shape
    parents = [set() for _ in range(d)]
    
    for i, target in enumerate(causal_order):
        X = data[:, causal_order[:i]]
        y = data[:, target]
        if X.shape[1] == 0:
            continue

        # Create B-splines basis for GAM
        X_dict = {f'x{i}': X[:, i] for i in range(X.shape[1])}
        formula = " + ".join([f"bs(x{i}, df=3, include_intercept=False)" for i in range(X.shape[1])])
        x_splines = dmatrix(formula, X_dict, return_type='dataframe')

        # Ensure the dimensions match
        gam_bs = BSplines(x_splines, df=[3]* x_splines.shape[1], degree= [2]* x_splines.shape[1], include_intercept = False)
        gam = GLMGam(y, smoother=gam_bs).fit()
        
        # Test each covariate
        for j in range(X.shape[1]):
            p_value = gam.pvalues[j]  # Use integer index
            if p_value < 0.05:  # Threshold for significance
                parents[target].add(causal_order[j])
    return parents



def hsic_test(X, Y):
    # Perform distance covariance test
    test_statistic, p_value = independence.distance_covariance_test(X, Y, num_resamples=100)
    return p_value

def resit_parents(data, causal_order):
    n, d = data.shape
    parents = [set() for _ in range(d)]

    # Initial parent determination using all preceding variables in the causal order
    for i, target in enumerate(causal_order):
        if i == 0:
            continue
        parents[target] = set(causal_order[:i])

    # Phase 2: Remove superfluous edges
    for k in range(1, d):
        target = causal_order[k]
        for parent in list(parents[target]):
            X = data[:, [i for i in parents[target] if i != parent]]
            y = data[:, target]
            if X.shape[1] == 0:
                continue

            rf = RandomForestRegressor(max_depth=2, n_estimators=100, random_state=0)
            rf.fit(X, y)
            residuals = y - rf.predict(X)

            # Check if residuals are independent of previous variables in the causal order
            independent = True
            for prev in causal_order[:k]:
                pvalue = hsic_test(residuals.reshape(-1, 1), data[:, prev].reshape(-1, 1))
                if pvalue < 0.05:  # Threshold for dependence 
                    independent = False
                    break

            if independent:
                parents[target].remove(parent)

    return parents