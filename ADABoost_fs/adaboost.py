import numpy as np

class DecisionTree:
  
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1:
            return y[0]
        if depth == self.max_depth:
            return np.mean(y)
        
        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            return np.mean(y)
        
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return (best_feature, best_threshold, left_tree, right_tree)

    def _find_best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_impurity = float('inf')
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold
                
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                
                impurity = self._calculate_impurity(y[left_indices], y[right_indices])
                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold

    def _calculate_impurity(self, left_y, right_y):
        left_size = len(left_y)
        right_size = len(right_y)
        total_size = left_size + right_size
        left_impurity = 1 - sum((np.sum(left_y == c) / left_size) ** 2 for c in set(left_y))
        right_impurity = 1 - sum((np.sum(right_y == c) / right_size) ** 2 for c in set(right_y))
        return (left_size / total_size) * left_impurity + (right_size / total_size) * right_impurity

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        
        feature, threshold, left_tree, right_tree = tree
        if x[feature] < threshold:
            return self._predict_one(x, left_tree)
        else:
            return self._predict_one(x, right_tree)


class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples
        
        for _ in range(self.n_estimators):
            model = DecisionTree(max_depth=1)
            model.fit(X, y)
            predictions = model.predict(X)
            
            error = np.sum(weights * (predictions != y)) / np.sum(weights)
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            
            weights *= np.exp(alpha * (predictions != y))
            weights /= np.sum(weights)
            
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for model, alpha in zip(self.models, self.alphas):
            predictions += alpha * model.predict(X)
        return np.sign(predictions)



if __name__ == "__main__":
    X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [8, 8], [10, 10]])
    y_train = np.array([1, 1, -1, -1, 1, -1])

    ada = AdaBoost(n_estimators=10)
    ada.fit(X_train, y_train)
    predictions = ada.predict(X_train)
    print(predictions) 
