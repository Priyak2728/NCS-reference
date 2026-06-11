import joblib

model = joblib.load("house_price_model.pkl")

predicted_price = 90
actual_price = 120

error = abs(actual_price - predicted_price)

print("Prediction:", predicted_price)
print("Actual:", actual_price)
print("Error:", error)

if error > 20:
    print("Retraining Required")
else:
    print("Model Healthy")