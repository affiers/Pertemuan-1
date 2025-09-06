import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Contoh dataset sederhana
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10])

#membuat model regresi linier
model = LinearRegression()
model.fit(x, y)

#Prediksi nilai baru
y_pred = model.predict(x)

#Visualisasi hasil
plt.scatter(x, y, color="blue", label="Data Asli")
plt.plot(x, y_pred, color="red", linestyle="--", label="Regresi Linier")
plt.legend()
plt.show()