from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data (X = input, y = output)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict
prediction = model.predict([[6]])

print("Prediction for 6:", prediction[0])
