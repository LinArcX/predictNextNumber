import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Given data
X = np.array([[1], [2], [3], [4], [5], [6]])  # Input values
y = np.array([2, 4, 6, 8, 10, 12])         # Output values

# Create polynomial features
poly = PolynomialFeatures(degree=2)  # You can adjust the degree as needed
X_poly = poly.fit_transform(X)

# Train the model
model = LinearRegression()
model.fit(X_poly, y)

# Function to predict the output for a new value
def predict_value(new_value):
    new_value_poly = poly.transform([[new_value]])
    return model.predict(new_value_poly)[0]

# Example usage
new_value = 12
result = predict_value(new_value)
print(f"The predicted result for {new_value} is: {result}")
