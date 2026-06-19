import numpy as np

class Log1pScaler:
    def __init__(self):
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        """Compute the necessary statistics for scaling based on input data."""
        self.min_ = np.min(X)
        self.scale_ = np.max(np.log1p(X - self.min_))

    def transform(self, X):
        """Apply log1p scaling to the input data."""
        return np.log1p(X - self.min_) / self.scale_

    def inverse_transform(self, X_scaled):
        """Inverse the log1p scaling."""
        return np.expm1(X_scaled * self.scale_) + self.min_