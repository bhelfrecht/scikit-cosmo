import unittest
from skcosmo.pcovr import PCovR
from sklearn.datasets import load_boston
import numpy as np
from sklearn import exceptions
from sklearn.utils.validation import check_X_y


def rel_error(A, B):
    return np.linalg.norm(A - B) ** 2.0 / np.linalg.norm(A) ** 2.0


class PCovRTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = lambda mixing, **kwargs: PCovR(
            mixing, full_eig=False, regularization=1e-8, **kwargs
        )
        error_tol = 1e-3
        self.rounding = -int(round(np.log10(error_tol)))

        self.X, self.Y = load_boston(return_X_y=True)

    def setUp(self):
        pass

    def test_lr_with_x_errors(self):

        prev_error = -1.0

        for i, mixing in enumerate(np.linspace(0, 1, 11)):

            pcovr = self.model(mixing=mixing, n_components=2, tol=1e-12)
            pcovr.fit(self.X, self.Y)

            Yp = pcovr.predict(self.X)
            error = rel_error(self.Y, Yp)

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error):
                self.assertGreaterEqual(error, prev_error)

            prev_error = error

    def test_lr_with_t_errors(self):

        prev_error = -1.0

        for i, mixing in enumerate(np.linspace(0, 1, 11)):
            pcovr = self.model(mixing=mixing, n_components=2, tol=1e-12)
            pcovr.fit(self.X, self.Y)

            Yp = pcovr.predict(T=pcovr.transform(self.X))
            error = rel_error(self.Y, Yp)

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error):
                self.assertGreaterEqual(error, prev_error)

            prev_error = error

    def test_reconstruction_errors(self):

        prev_error = 1.0

        for i, mixing in enumerate(np.linspace(0, 1, 11)):
            pcovr = self.model(mixing=mixing, n_components=2, tol=1e-12)
            pcovr.fit(self.X, self.Y)

            error = rel_error(self.X, pcovr.inverse_transform(pcovr.transform(self.X)))

            with self.subTest(error=error):
                self.assertFalse(np.isnan(error))
            with self.subTest(error=error):
                self.assertLessEqual(error, prev_error)

            prev_error = error

    def test_nonfitted_failure(self):
        model = self.model(mixing=0.5, n_components=2, tol=1e-12)
        with self.assertRaises(exceptions.NotFittedError):
            _ = model.transform(self.X)

    def test_no_arg_predict(self):
        model = self.model(mixing=0.5, n_components=2, tol=1e-12)
        model.fit(self.X, self.Y)
        with self.assertRaises(ValueError):
            _ = model.predict()

    def test_T_shape(self):
        pcovr = self.model(mixing=0.5, n_components=2, tol=1e-12)
        pcovr.fit(self.X, self.Y)
        T = pcovr.transform(self.X)
        self.assertTrue(check_X_y(self.X, T, multi_output=True))


class PCovRSpaceTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = lambda mixing, **kwargs: PCovR(
            mixing, full_eig=False, regularization=1e-8, **kwargs
        )
        error_tol = 1e-3
        self.rounding = -int(round(np.log10(error_tol)))

        self.X, self.Y = load_boston(return_X_y=True)

    def test_select_feature_space(self):
        pcovr = self.model(mixing=0.5, n_components=2, tol=1e-12)
        pcovr.fit(self.X, self.Y)

        self.assertTrue(pcovr.space == "feature")

    def test_select_structure_space(self):
        pcovr = self.model(mixing=0.5, n_components=2, tol=1e-12)

        n_structures = self.X.shape[1] - 1
        pcovr.fit(self.X[:n_structures], self.Y[:n_structures])

        self.assertTrue(pcovr.space == "structure")

    def test_bad_space(self):
        with self.assertRaises(ValueError):
            pcovr = self.model(mixing=0.5, n_components=2, tol=1e-12, space="bad")
            pcovr.fit(self.X, self.Y)

    def test_override_space_selection(self):
        pcovr = self.model(mixing=0.5, n_components=2, tol=1e-12, space="structure")
        pcovr.fit(self.X, self.Y)

        self.assertTrue(pcovr.space == "structure")


if __name__ == "__main__":
    unittest.main(verbosity=2)
