from sklearn.utils import gen_batches
from sklearn.linear_model._base import LinearModel
from scipy import sparse

# What is the best way to inherit?
class _BaseIncremental(LinearModel):
    def fit(X, y=None):
        # Adapted from sklearn.decomposition.IncrementalPCA.fit()
        n_samples, n_features = X.shape
        # TODO: check X and y shapes

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for batch in gen_batches(
            n_samples, self.batch_size_, min_batch_size=0
        ):
            X_batch = X[batch]

            if y is not None:
                y_batch = y[batch]
            else:
                y_batch = None

            if sparse.issparse(X_batch):
                X_batch = X_batch.toarray()

            # TODO: how to make sure the correct partial_fit is called?
            # Use abstractmethod decorator?
            self.partial_fit(X_batch, y_batch)
