import numpy as np

class HierarchicalClustering:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters

    def fit(self, X):
        self.n_samples = X.shape[0]
        self.labels = np.arange(self.n_samples)
        self.distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        np.fill_diagonal(self.distances, np.inf)
        self.linkage_matrix = self._perform_clustering()

    def _perform_clustering(self):
        clusters = np.arange(self.n_samples)
        distances = self.distances.copy()
        while len(np.unique(clusters)) > self.n_clusters:
            i, j = np.unravel_index(np.argmin(distances), distances.shape)
            clusters[clusters == clusters[j]] = clusters[i]
            distances[i, :] = np.minimum(distances[i, :], distances[j, :])
            distances[:, i] = distances[i, :]
            distances[j, :] = np.inf
            distances[:, j] = np.inf
        return clusters

    def predict(self, X):
        return self.labels


X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [8, 8], [10, 10]])
hc = HierarchicalClustering(n_clusters=2)
hc.fit(X_train)
predictions = hc.predict(X_train)
print(predictions) 