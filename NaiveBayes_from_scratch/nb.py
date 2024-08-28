import random
from collections import defaultdict
import math

class NaiveBayes:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = defaultdict(lambda: defaultdict(lambda: 0))
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.total_samples = 0

    def fit(self, X, y):
        self.total_samples = len(y)
        for features, label in zip(X, y):
            self.class_counts[label] += 1
            for feature in features:
                self.feature_counts[label][feature] += 1
        
        self.class_probs = {label: count / self.total_samples for label, count in self.class_counts.items()}

        for label in self.class_counts:
            for feature in self.feature_counts[label]:
                self.feature_probs[label][feature] = self.feature_counts[label][feature] / self.class_counts[label]

    def predict(self, X):
        return [self._predict(features) for features in X]

    def _predict(self, features):
        best_label = None
        best_prob = -1

        for label in self.class_probs:
            prob = self.class_probs[label]
            for feature in features:
                prob *= self.feature_probs[label].get(feature, 1e-6)  # Smoothing

            if prob > best_prob:
                best_prob = prob
                best_label = label
        
        return best_label


X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train = [0, 0, 1, 1]
X_test = [[1, 2.5], [3, 3.5]]

model = NaiveBayes()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions) 
