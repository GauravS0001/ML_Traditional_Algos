import numpy as np

class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        self.components = eigenvectors[:, np.argsort(eigenvalues)[::-1][:self.n_components]]

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered.dot(self.components)


X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [8, 8], [10, 10]])
pca = PCA(n_components=1)
pca.fit(X_train)
transformed = pca.transform(X_train)
print(transformed)  