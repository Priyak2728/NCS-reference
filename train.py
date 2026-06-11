from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

# Training Data
X = np.array([
    [1000],
    [1500],
    [2000],
    [2500]
])

y = np.array([
    50,
    75,
    100,
    125
])

# Train Model
model = LinearRegression()

model.fit(X, y)

# Prediction
prediction = model.predict([[1800]])

print(f"Predicted Price: ₹{prediction[0]:.2f} Lakhs")

# Save Model
joblib.dump(model, "house_price_model.pkl")

print("Model Saved Successfully")