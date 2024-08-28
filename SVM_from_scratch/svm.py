import random
import math

class SVM:
    def __init__(self, learning_rate=0.001, epochs=1000, C=1.0):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.C = C

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.alpha = [0] * len(X)
        self.b = 0.0

        for epoch in range(self.epochs):
            for i in range(len(self.X_train)):
                if self._predict(self.X_train[i]) * self.y_train[i] < 1:
                    self.alpha[i] += self.learning_rate * (1 - self._predict(self.X_train[i]) * self.y_train[i])
                    self.b += self.learning_rate * self.y_train[i]
                else:
                    self.alpha[i] -= self.learning_rate * self.alpha[i]

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        return sum(self.alpha[i] * self.y_train[i] * self._kernel(x, self.X_train[i]) for i in range(len(self.X_train))) + self.b

    def _kernel(self, x1, x2):
        return sum(a * b for a, b in zip(x1, x2))


X_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
y_train = [1, -1, -1, 1]
X_test = [[1, 2.5], [3, 3.5]]

model = SVM()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions) 
