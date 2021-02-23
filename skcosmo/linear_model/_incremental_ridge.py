import numpy as np
from sklearn.base import RegressorMixin, MultiOutputMixin
from ..base import _BaseIncremental
from ..utils.incremental import update_ridge_weights, update_inverse_covariance

class IncrementalRidge(MultiOutputMixin, RegressorMixin, _BaseIncremental):

    def __init__(self, alpha=1.0, *, fit_intercept=True, normalize=False,
            copy_X=True, batch_size=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.batch_size = batch_size

    def partial_fit(self, X, y):
        # TODO: make sure X is 2D with samples as rows
        # TODO: make sure y is also the appropriate shape
        # TODO: need to make sure X is centered to get covariance to work?
        # TODO: account for intercept
        self.intercept_ = 0.0
        # TODO: account for normalization
        # TODO: account for sample weights?

        # TODO: accept sparse?
        X, y = self._validate_data(
            X, y,
            multi_output=True, y_numeric=True
        )

        ravel = False
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            ravel = True

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if not hasattr(self, 'inv_cov_'):
            self.inv_cov_ = 1.0 / self.alpha * np.eye(X.shape[1])
        if not hasattr(self, 'coef_'):
            self.coef_ = np.zeros((y.shape[1], X.shape[1]))

        self.coef_ = update_ridge_weights(self.inv_cov_, self.coef_, X, y)
        if ravel:
            self.coef_ = self.coef_.squeeze()

        self.inv_cov_ = update_inverse_covariance(self.inv_cov_, X)
