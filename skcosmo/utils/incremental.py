import numpy as np

def update_covariance(cov, X):
    return cov + X.T @ X

# Update by Sherman-Morrison formula:
# https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
# TODO: find better reference
# https://stats.stackexchange.com/questions/81740/recursive-online-regularised-least-squares-algorithm
def update_prefactor(inv_cov, X):
    a = np.eye(X.shape[0]) + np.linalg.multi_dot([X, inv_cov, X.T])
    inv = np.linalg.lstsq(a, np.eye(X.shape[0]), rcond=None)[0]
    return np.linalg.multi_dot([inv_cov, X.T, inv])

def update_inverse_covariance(inv_cov, X):
    return inv_cov - np.linalg.multi_dot([update_prefactor(inv_cov, X), X, inv_cov])

def update_ridge_weights(inv_cov, coef, X, y):
    return coef + np.linalg.multi_dot([y.T - coef @ X.T, update_prefactor(inv_cov, X).T])
