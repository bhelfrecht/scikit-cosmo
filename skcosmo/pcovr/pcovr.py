import numpy as np
from sklearn.linear_model import RidgeCV as LR
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils import check_array
from sklearn.decomposition._base import _BasePCA
from sklearn.linear_model._base import LinearModel
from scipy.sparse.linalg import eigs

from .pcovr_distances import pcovr_covariance, pcovr_kernel_distance


class PCovR(_BasePCA, LinearModel):
    """
    Performs Principal Covariates Regression, as described in `[S. de Jong and
    H. A. L. Kiers, 1992] <https://doi.org/10.1016/0169-7439(92)80100-I>`_.

    :param mixing: mixing parameter,
                   as described in PCovR as :math:`{\\alpha}`, defaults to 1
    :type mixing: float

    :param n_components: Number of components to keep.
    :type n_components: int

    :param regularization: regularization parameter for linear models
    :type regularization: float, default 1E-6

    :param tol: tolerance below which to consider eigenvalues = 0
    :type tol: float, default 1E-12

    :param full_eig: whether to compute the full eigendecomposition
    :type full_eig: boolean, default False

    :param space: whether to compute the PCovR in `structure` or `feature` space
                  defaults to `structure` when :math:`{n_{samples} < n_{features}}` and
                  `feature` when :math:`{n_{features} < n_{samples}}``
    :type space: {'feature', 'structure', 'auto'}

    :param lr_args: dictionary of arguments to pass to the Ridge Regression
                    in estimating :math:`{\\mathbf{\\hat{Y}}}`

    References
        1.  S. de Jong, H. A. L. Kiers, 'Principal Covariates
            Regression: Part I. Theory', Chemometrics and Intelligent
            Laboratory Systems 14(1): 155-164, 1992
        2.  M. Vervolet, H. A. L. Kiers, W. Noortgate, E. Ceulemans,
            'PCovR: An R Package for Principal Covariates Regression',
            Journal of Statistical Software 65(1):1-14, 2015
    """

    def __init__(
        self,
        mixing=0.0,
        n_components=None,
        regularization=1e-6,
        tol=1e-12,
        full_eig=False,
        space=None,
        lr_args={"cv": 2, "fit_intercept": False},
    ):

        self.mixing = mixing
        self.regularization = regularization
        self.tol = tol
        self.space = space
        self.lr_args = lr_args
        self.full_eig = full_eig
        self.n_components = n_components
        self.whiten = False

    def fit(self, X, Y, Yhat=None, W=None):
        """

        Fit the model with X and Y. Depending on the dimensions of X,
        calls either `_fit_feature_space` or `_fit_structure_space`

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        Y : array-like, shape (n_samples, n_properties)
            Training data, where n_samples is the number of samples and
            n_properties is the number of properties
        Yhat : array-like, shape (n_samples, n_properties), optional
            Regressed training data, where n_samples is the number of samples and
            n_properties is the number of properties. If not supplied, computed
            by ridge regression.
        W : array-like, shape (n_features, n_properties), optional
            Weights of regressed training data. If not supplied, computed
            by ridge regression.

        """

        X, Y = check_X_y(X, Y, y_numeric=True, multi_output=True)

        if self.n_components is None:
            self.n_components = min(X.shape)

        # Sparse eigensolvers will not work when seeking N-1 eigenvalues
        if min(X.shape) <= self.n_components:
            self.full_eig = True

        # self._check_space(X)

        if Yhat is None or W is None:
            Yhat, W = self._compute_Yhat(X, Y, Yhat=Yhat, W=W)

        self._fit(X, Yhat, W)

        self.mean_ = np.mean(X, axis=0)
        self.pxy_ = self.pxt_ @ self.pty_
        if len(Y.shape) == 1:
            self.pxy_ = self.pxy_.reshape(
                X.shape[1],
            )
            self.pty_ = self.pty_.reshape(
                self.n_components,
            )

        self.components_ = self.pxt_.T  # for sklearn compatibility

    def _fit(self, X, Yhat, W):
        if self.space is not None and self.space not in [
            "feature",
            "structure",
            "auto",
        ]:
            raise ValueError("Only feature and structure space are supported.")

        if self.space is None:
            if X.shape[0] > X.shape[1]:
                self.space = "feature"
            else:
                self.space = "structure"

        if self.space == "feature":
            return self._fit_feature_space(X, Yhat, W)
        else:
            return self._fit_structure_space(X, Yhat, W)

    def _compute_Yhat(self, X, Y, Yhat=None, W=None):
        """
        Method for computing the approximation of Y to fit the PCovR
        """

        if Yhat is None:
            if W is None:
                lr = LR(**self.lr_args)  # some sort of args
                lr.fit(X, Y)
                Yhat = lr.predict(X)
                W = lr.coef_.T
            else:
                Yhat = X @ W

        elif W is None:
            W = np.linalg.pinv(np.dot(X.T, X), rcond=self.regularization)
            W = np.linalg.multi_dot([W, X.T, Y])

        Yhat = Yhat.reshape(X.shape[0], -1)
        W = W.reshape(X.shape[1], -1)
        return Yhat, W

    def _fit_feature_space(self, X, Yhat, W=None):
        """
        In feature-space PCovR, the projectors are determined by:

        .. math::

            \\mathbf{\\tilde{C}} = \\alpha \\mathbf{X}^T \\mathbf{X} +
            (1 - \\alpha) \\left(\\left(\\mathbf{X}^T
            \\mathbf{X}\\right)^{-\\frac{1}{2}} \\mathbf{X}^T
            \\mathbf{\\hat{Y}}\\mathbf{\\hat{Y}}^T \\mathbf{X} \\left(\\mathbf{X}^T
            \\mathbf{X}\\right)^{-\\frac{1}{2}}\\right)

        where

        .. math::

            \\mathbf{P}_{XT} = (\\mathbf{X}^T \\mathbf{X})^{-\\frac{1}{2}}
                                \\mathbf{U}_\\mathbf{\\tilde{C}}^T
                                \\mathbf{\\Lambda}_\\mathbf{\\tilde{C}}^{\\frac{1}{2}}

        .. math::

            \\mathbf{P}_{TX} = \\mathbf{\\Lambda}_\\mathbf{\\tilde{C}}^{-\\frac{1}{2}}
                                \\mathbf{U}_\\mathbf{\\tilde{C}}^T
                                (\\mathbf{X}^T \\mathbf{X})^{\\frac{1}{2}}

        .. math::

            \\mathbf{P}_{TY} = \\mathbf{\\Lambda}_\\mathbf{\\tilde{C}}^{-\\frac{1}{2}}
                               \\mathbf{U}_\\mathbf{\\tilde{C}}^T (\\mathbf{X}^T
                               \\mathbf{X})^{-\\frac{1}{2}} \\mathbf{X}^T
                               \\mathbf{Y}

        """

        Ct, iCsqrt = pcovr_covariance(
            mixing=self.mixing,
            X_proxy=X,
            Y_proxy=Yhat,
            rcond=self.tol,
            return_isqrt=True,
        )
        Csqrt = np.linalg.inv(iCsqrt)

        v, U = self._eig_solver(Ct)
        S = v ** 0.5
        S_inv = np.linalg.pinv(np.diagflat(S))

        self.singular_values_ = S.copy()
        self.explained_variance_ = (S ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / self.explained_variance_.sum()
        )

        self.pxt_ = np.linalg.multi_dot([iCsqrt, U, np.diagflat(S)])
        self.ptx_ = np.linalg.multi_dot([S_inv, U.T, Csqrt])
        self.pty_ = np.linalg.multi_dot([S_inv, U.T, iCsqrt, X.T, Yhat])

    def _fit_structure_space(self, X, Yhat, W):
        """
        In sample-space PCovR, the projectors are determined by:

        .. math::

            \\mathbf{\\tilde{K}} = \\alpha \\mathbf{X} \\mathbf{X}^T +
            (1 - \\alpha) \\mathbf{\\hat{Y}}\\mathbf{\\hat{Y}}^T

        where

        .. math::

            \\mathbf{P}_{XT} = \\left(\\alpha \\mathbf{X}^T + (1 - \\alpha)
                               \\mathbf{W} \\mathbf{\\hat{Y}}^T\\right)
                               \\mathbf{U}_\\mathbf{\\tilde{K}}
                               \\mathbf{\\Lambda}_\\mathbf{\\tilde{K}}^{-\\frac{1}{2}}

        .. math::

            \\mathbf{P}_{TX} = \\mathbf{\\Lambda}_\\mathbf{\\tilde{K}}^{-\\frac{1}{2}}
                                \\mathbf{U}_\\mathbf{\\tilde{K}}^T \\mathbf{X}

        .. math::

            \\mathbf{P}_{TY} = \\mathbf{\\Lambda}_\\mathbf{\\tilde{K}}^{-\\frac{1}{2}}
                               \\mathbf{U}_\\mathbf{\\tilde{K}}^T \\mathbf{Y}

        """

        Kt = pcovr_kernel_distance(mixing=self.mixing, X_proxy=X, Y_proxy=Yhat)

        v, U = self._eig_solver(Kt)
        S = v ** 0.5

        P = (self.mixing * X.T) + (1.0 - self.mixing) * np.dot(W, Yhat.T)

        self.singular_values_ = S.copy()
        self.explained_variance_ = (S ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / self.explained_variance_.sum()
        )

        T = U @ np.diagflat(1 / S)

        self.pxt_ = P @ T
        self.pty_ = T.T @ Yhat
        self.ptx_ = T.T @ X

    def _eig_solver(self, matrix):

        if not self.full_eig:
            v, U = eigs(matrix, k=self.n_components, tol=self.tol)
        else:
            v, U = np.linalg.eig(matrix)

        U = np.real(U[:, np.argsort(v)])
        v = np.real(v[np.argsort(v)])

        U = U[:, v > self.tol]
        v = v[v > self.tol]

        if len(v) == 1:
            U = U.reshape(-1, 1)

        if self.n_components > len(v):
            print(
                f"There are fewer than {self.n_components} "
                f"significant eigenpair(s). Adding "
                f"{self.n_components - len(v)} null eigenpair(s)."
            )
            for i in range(len(v), self.n_components):
                v = np.array([*v, 0])
                U = np.array([*U.T, np.zeros(U.shape[0])]).T

        return v[: self.n_components], U[:, : self.n_components]

    def inverse_transform(self, T):
        """Transform data back to its original space.

        .. math::

            \\mathbf{\\hat{X}} = \\mathbf{T} \\mathbf{P}_{TX}
                              = \\mathbf{X} \\mathbf{P}_{XT} \\mathbf{P}_{TX}


        Parameters
        ----------
        T : array-like, shape (n_samples, n_components)
            Projected data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)
        """

        return T @ self.ptx_ + self.mean_

    def predict(self, X=None, T=None):
        """Predicts the property values using regression on X or T"""

        check_is_fitted(self)

        if X is None and T is None:
            raise ValueError("Either X or T must be supplied.")

        if X is not None:
            X = check_array(X)
            return X @ self.pxy_
        else:
            T = check_array(T)
            return T @ self.pty_

    def transform(self, X=None):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components as determined by the
        modified PCovR distances.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        """

        return super().transform(X)
