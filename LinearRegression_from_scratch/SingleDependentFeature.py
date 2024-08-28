class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 0.0  # Slope
        self.c = 0.0  # Intercept
    
    def fit(self, X, Y):
        n = float(len(X)) 
        
        for epoch in range(self.epochs):

            Y_pred = [self.m * x + self.c for x in X]
            
            # Calculating gradients
            D_m = (-2/n) * sum([x * (y - y_pred) for x, y, y_pred in zip(X, Y, Y_pred)])  # Derivative wrt m
            D_c = (-2/n) * sum([(y - y_pred) for y, y_pred in zip(Y, Y_pred)])  # Derivative wrt c
            
            # Update m and c
            self.m -= self.learning_rate * D_m
            self.c -= self.learning_rate * D_c
            
            # Print progress every 100 epochs
            if epoch % 100 == 0:
                loss = sum([(y - y_pred) ** 2 for y, y_pred in zip(Y, Y_pred)]) / n  # MSE
                print(f'Epoch {epoch}: m = {self.m:.4f}, c = {self.c:.4f}, Loss = {loss:.4f}')
    
    def predict(self, x):
        return self.m * x + self.c

'''
if __name__ == "__main__":
    # X: independent, Y: dependent
    X = [1, 2, 3, 4, 5]
    Y = [3, 4, 2, 4, 5]

    model = LinearRegression(learning_rate=0.01, epochs=1000)

    model.fit(X, Y)

    # Predict
    new_X = 6
    predicted_Y = model.predict(new_X)
    print(f'Predicted Y for X={new_X}: {predicted_Y:.4f}')
'''
