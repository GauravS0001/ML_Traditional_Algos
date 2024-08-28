import random

"""
Techniques used to reduce loss:
1. **Normalization**
2. **Regularization**
3. **Early Stopping**
"""

class LinearRegression:
    def __init__(self, learning_rate=0.001, epochs=1000, regularization=0.0, early_stopping_patience=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.early_stopping_patience = early_stopping_patience
        self.coefficients = []
        self.intercept = 0.0
        self.best_loss = float('inf')
        self.patience_counter = 0

    def fit(self, X, Y):
        n_samples, n_features = len(X), len(X[0])
        self.coefficients = [0.0] * n_features
        
        # Normalized features to have zero mean and unit variance
        X_normalized = self.normalize_features(X)
        
        for epoch in range(self.epochs):
            # Predict Y using current coefficients and intercept
            Y_pred = [sum(X_normalized[i][j] * self.coefficients[j] for j in range(n_features)) + self.intercept for i in range(n_samples)]
            
            # Calculate gradients for coefficients (with L2) and intercept
            D_m = [(-2/n_samples) * sum(X_normalized[i][j] * (Y[i] - Y_pred[i]) for i in range(n_samples)) + 2 * self.regularization * self.coefficients[j] for j in range(n_features)]
            D_c = (-2/n_samples) * sum(Y[i] - Y_pred[i] for i in range(n_samples))
            
            # Update coefficients and intercept using the calculated gradients
            for j in range(n_features):
                self.coefficients[j] -= self.learning_rate * D_m[j]
            self.intercept -= self.learning_rate * D_c

            # Compute loss with L2 
            loss = sum((Y[i] - Y_pred[i]) ** 2 for i in range(n_samples)) / n_samples + self.regularization * sum(coef ** 2 for coef in self.coefficients)

            # early stopping
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
        n_samples = len(X)
        n_features = len(X[0])
        
        means = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
        std_devs = [((sum((X[i][j] - means[j]) ** 2 for i in range(n_samples)) / n_samples) ** 0.5) for j in range(n_features)]
        
        X_normalized = []
        for i in range(n_samples):
            normalized_row = [(X[i][j] - means[j]) / std_devs[j] if std_devs[j] != 0 else 0 for j in range(n_features)]
            X_normalized.append(normalized_row)
        
        return X_normalized

    def predict(self, X):
        X_normalized = self.normalize_features(X)
        return [sum(X_normalized[i][j] * self.coefficients[j] for j in range(len(self.coefficients))) + self.intercept for i in range(len(X))]


def generate_synthetic_data(n_samples, n_features):
    X = [[random.uniform(0, 10) for _ in range(n_features)] for _ in range(n_samples)]
    Y = [sum(X[i][j] * (j + 1) for j in range(n_features)) + random.uniform(-1, 1) for i in range(n_samples)]
    return X, Y


n_samples = 100
n_features = 5
learning_rate = 0.001
epochs = 1000
regularization = 0.01  
early_stopping_patience = 10


X, Y = generate_synthetic_data(n_samples, n_features)


model = LinearRegression(learning_rate=learning_rate, epochs=epochs, regularization=regularization, early_stopping_patience=early_stopping_patience)
model.fit(X, Y)


X_new = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
predictions = model.predict(X_new)
print(f'Predicted Y values: {predictions}')
