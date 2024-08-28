import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        for _ in range(self.max_iters):
            self.labels = self._assign_labels(X)
            new_centroids = self._compute_centroids(X)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X):
        return np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def predict(self, X):
        return self._assign_labels(X)



X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [8, 8], [10, 10]])
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)
predictions = kmeans.predict(X_train)
print(predictions)