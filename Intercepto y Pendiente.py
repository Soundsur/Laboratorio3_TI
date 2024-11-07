import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Datos de distancia y potencia
distancia_m = np.array([
    1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
    2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
    3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
    4.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
    23.0, 24.0
])

potencia_dbm = np.array([
    -56.2, -57.4, -65.3, -62.0, -65.0, -61.2, -58.3, -61.3, -62.7, -60.1,
    -62.0, -62.2, -65.6, -65.1, -68.3, -70.0, -67.2, -69.4, -66.2, -63.7,
    -64.6, -69.5, -73.5, -68.2, -67.3, -71.2, -70.3, -71.4, -66.8, -69.4,
    -67.8, -70.5, -70.3, -71.5, -68.0, -70.8, -70.1, -66.3, -67.4, -67.2,
    -74.7, -71.0, -69.1, -72.3, -72.7, -69.9, -69.3, -73.7, -73.5, -69.7,
    -71.5, -71.4
])

# Aplicando logaritmo a la distancia para el modelo log-normal
log_distancia = np.log10(distancia_m)

# Ajuste lineal para encontrar la pendiente
pendiente, intercepto, r_value, p_value, std_err = linregress(log_distancia, potencia_dbm)

# Graficando los datos y la línea de regresión
plt.figure(figsize=(10, 6))
plt.scatter(log_distancia, potencia_dbm, label="Datos experimentales")
plt.plot(log_distancia, pendiente * log_distancia + intercepto, color="red", label=f"Ajuste lineal: pendiente = {pendiente:.2f}")
plt.xlabel("Log10(Distancia) [m]")
plt.ylabel("Potencia [dBm]")
plt.title("Gráfico de Potencia vs Log(Distancia)")
plt.legend()
plt.grid(True)
plt.show()

# Mostrando los resultados de la pendiente, intercepto y error estándar
print("Pendiente:", pendiente)
print("Intercepto:", intercepto)
print("Error estándar de la pendiente:", std_err)
