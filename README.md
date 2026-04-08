# Weather-forecasting-system
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset (replace with real dataset)
days = np.array([1,2,3,4,5,6,7]).reshape(-1,1)
temperature = np.array([30,32,34,33,35,36,37])

# Model
model = LinearRegression()
model.fit(days, temperature)

# Prediction
future_days = np.array([8,9,10]).reshape(-1,1)
predicted_temp = model.predict(future_days)

# Plot
plt.scatter(days, temperature, color='blue', label='Actual')
plt.plot(days, model.predict(days), color='red', label='Model')
plt.scatter(future_days, predicted_temp, color='green', label='Predicted')

plt.xlabel("Days")
plt.ylabel("Temperature")
plt.title("Weather Forecasting")
plt.legend()
plt.show()
plt.savefig("prediction_graph.png")