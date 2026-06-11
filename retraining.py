from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

X_new = np.array([
    [1000],
    [1500],
    [2000],
    [2500],
    [3000]
])

y_new = np.array([
    50,
    75,
    100,
    125,
    180
])

model = LinearRegression()

model.fit(X_new, y_new)

joblib.dump(model, "house_price_model_v2.pkl")

print("Version 2 Created")