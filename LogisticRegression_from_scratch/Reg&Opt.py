import random
import math

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, regularization=0.0, early_stopping_patience=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.early_stopping_patience = early_stopping_patience
        self.coefficients = []
        self.intercept = 0.0
        self.best_loss = float('inf')
        self.patience_counter = 0

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + math.exp(-z))

    def fit(self, X, Y):
        n_samples, n_features = len(X), len(X[0])
        self.coefficients = [0.0] * n_features
        
        X_normalized = self.normalize_features(X)
        
        for epoch in range(self.epochs):
            Y_pred = [self.sigmoid(sum(X_normalized[i][j] * self.coefficients[j] for j in range(n_features)) + self.intercept) for i in range(n_samples)]
            
            
            D_m = [(-1/n_samples) * sum(X_normalized[i][j] * (Y[i] - Y_pred[i]) for i in range(n_samples)) + 2 * self.regularization * self.coefficients[j] for j in range(n_features)]
            D_c = (-1/n_samples) * sum(Y[i] - Y_pred[i] for i in range(n_samples))
            
            # Update coefficients and intercept
            for j in range(n_features):
                self.coefficients[j] -= self.learning_rate * D_m[j]
            self.intercept -= self.learning_rate * D_c

          
            loss = -sum(Y[i] * math.log(Y_pred[i]) + (1 - Y[i]) * math.log(1 - Y_pred[i]) for i in range(n_samples)) / n_samples + self.regularization * sum(coef ** 2 for coef in self.coefficients)

            # early stopp
            if loss < self.best_loss:
                self.best_loss = loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f'Early stopping at epoch {epoch} with loss {loss:.4f}')
                    break

            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Coefficients = {self.coefficients}, Intercept = {self.intercept:.4f}, Loss = {loss:.4f}')

    def normalize_features(self, X):
        """ Normalize features to have zero mean and unit variance """
        means = [sum(feature) / len(feature) for feature in zip(*X)]
        std_devs = [math.sqrt(sum((x - mean) ** 2 for x in feature) / len(feature)) for feature, mean in zip(zip(*X), means)]
        X_normalized = [[(X[i][j] - means[j]) / std_devs[j] for j in range(len(X[0]))] for i in range(len(X))]
        return X_normalized

    def predict(self, X):
        X_normalized = self.normalize_features(X)
        return [self.sigmoid(sum(X_normalized[i][j] * self.coefficients[j] for j in range(len(self.coefficients))) + self.intercept) for i in range(len(X))]

# Example usage
def generate_synthetic_data(n_samples, n_features):
    X = [[random.uniform(0, 10) for _ in range(n_features)] for _ in range(n_samples)]
    Y = [1 if sum(X[i][j] * (j + 1) for j in range(n_features)) + random.uniform(-1, 1) > 0 else 0 for i in range(n_samples)]
    return X, Y

n_samples = 100
n_features = 5
learning_rate = 0.01
epochs = 1000
regularization = 0.01 
early_stopping_patience = 10


X, Y = generate_synthetic_data(n_samples, n_features)

model = LogisticRegression(learning_rate=learning_rate, epochs=epochs, regularization=regularization, early_stopping_patience=early_stopping_patience)
model.fit(X, Y)

X_new = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
predictions = model.predict(X_new)
print(f'Predicted probabilities: {predictions}')
