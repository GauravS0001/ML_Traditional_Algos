import random

class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return [self._predict(x, self.tree) for x in X]

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1:
            return y[0]
        if depth == self.max_depth:
            return self._most_common_label(y)

        best_split = self._best_split(X, y)
        if best_split is None:
            return self._most_common_label(y)

        left_indices = [i for i in range(len(X)) if X[i][best_split['feature']] <= best_split['value']]
        right_indices = [i for i in range(len(X)) if X[i][best_split['feature']] > best_split['value']]

        left_tree = self._build_tree([X[i] for i in left_indices], [y[i] for i in left_indices], depth + 1)
        right_tree = self._build_tree([X[i] for i in right_indices], [y[i] for i in right_indices], depth + 1)

        return {'feature': best_split['feature'], 'value': best_split['value'], 'left': left_tree, 'right': right_tree}

    def _best_split(self, X, y):
        best_split = None
        best_score = float('inf')

        for feature in range(len(X[0])):
            values = set(x[feature] for x in X)
            for value in values:
                left_labels = [y[i] for i in range(len(X)) if X[i][feature] <= value]
                right_labels = [y[i] for i in range(len(X)) if X[i][feature] > value]

                score = self._gini_impurity(left_labels, right_labels)
                if score < best_score:
                    best_score = score
                    best_split = {'feature': feature, 'value': value}

        return best_split

    def _gini_impurity(self, left_labels, right_labels):
        total_size = len(left_labels) + len(right_labels)
        left_size = len(left_labels)
        right_size = len(right_labels)
        
        left_impurity = 1 - sum((left_labels.count(label) / left_size) ** 2 for label in set(left_labels))
        right_impurity = 1 - sum((right_labels.count(label) / right_size) ** 2 for label in set(right_labels))

        return (left_size / total_size) * left_impurity + (right_size / total_size) * right_impurity

    def _most_common_label(self, labels):
        return max(set(labels), key=labels.count)

    def _predict(self, x, tree):
        if not isinstance(tree, dict):
            return tree

        if x[tree['feature']] <= tree['value']:
            return self._predict(x, tree['left'])
        else:
            return self._predict(x, tree['right'])


class Bagging:
    def __init__(self, base_model, n_models=10):
        self.base_model = base_model
        self.n_models = n_models

    def fit(self, X, y):
        self.models = []
        for _ in range(self.n_models):
            X_bootstrap, y_bootstrap = self._bootstrap_sampling(X, y)
            model = self.base_model()
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return [self._majority_vote(p) for p in zip(*predictions)]

    def _bootstrap_sampling(self, X, y):
        indices = [random.randint(0, len(X) - 1) for _ in range(len(X))]
        X_bootstrap = [X[i] for i in indices]
        y_bootstrap = [y[i] for i in indices]
        return X_bootstrap, y_bootstrap

    def _majority_vote(self, predictions):
        return max(set(predictions), key=predictions.count)


X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train = [0, 0, 1, 1]
X_test = [[1, 2.5], [3, 3.5]]

bagging_model = Bagging(base_model=DecisionTree, n_models=5)
bagging_model.fit(X_train, y_train)
predictions = bagging_model.predict(X_test)
print(predictions)
