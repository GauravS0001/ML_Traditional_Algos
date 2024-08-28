class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coefficients = None  # To store the weights m for each feature
        self.intercept = 0.0  # To store the intercept c
    
    def fit(self, X, Y):
        n = float(len(X))  # Number of data points
        num_features = len(X[0])  # Number of features


        self.coefficients = [0.0] * num_features

        for epoch in range(self.epochs):
            # Predict Y using current coefficients and intercept
            Y_pred = [self._predict_single(x) for x in X]

            # Calculating gradients
            D_coefficients = [-2/n * sum([(y - y_pred) * x_i for x_i, y, y_pred in zip(x, Y, Y_pred)]) for x in zip(*X)]
            D_intercept = -2/n * sum([y - y_pred for y, y_pred in zip(Y, Y_pred)])

            # Updating coefficients and intercept
            self.coefficients = [coef - self.learning_rate * D_coef for coef, D_coef in zip(self.coefficients, D_coefficients)]
            self.intercept -= self.learning_rate * D_intercept

            # Print progress every 100 epochs
            if epoch % 100 == 0:
                loss = sum([(y - y_pred) ** 2 for y, y_pred in zip(Y, Y_pred)]) / n  # MSE
                print(f'Epoch {epoch}: Coefficients = {self.coefficients}, Intercept = {self.intercept:.4f}, Loss = {loss:.4f}')
    
    def _predict_single(self, x):
        return sum(coef * x_i for coef, x_i in zip(self.coefficients, x)) + self.intercept
    
    def predict(self, X):
        return [self._predict_single(x) for x in X]


if __name__ == "__main__":
   
    X = [
        [6.5, 300, 15, 50, 5],  
        [7.0, 400, 10, 45, 4],
        [6.0, 350, 12, 55, 6],
        [5.5, 280, 20, 60, 5],
        [7.2, 390, 8, 40, 4],
        [6.8, 320, 18, 50, 3]
    ]
    Y = [24, 28, 22, 20, 26, 25] 

    model = LinearRegression(learning_rate=0.01, epochs=1000)

    model.fit(X, Y)

    new_X = [
        [6.7, 310, 14, 52, 4], 
        [7.1, 330, 16, 48, 3]
    ]
    predicted_Y = model.predict(new_X)
    print(f'Predicted Y values: {predicted_Y}')
