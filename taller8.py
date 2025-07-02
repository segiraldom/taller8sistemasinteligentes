import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# 1. Leer archivo CSV y cargar datos
ruta_archivo = 'caesarian.csv'
matriz = []

with open(ruta_archivo, newline='') as archivo:
    lector = csv.reader(archivo)
    next(lector)  # Saltar encabezado
    for fila in lector:
        fila_entera = [int(valor) for valor in fila]
        matriz.append(fila_entera)

# 2. Convertir a numpy y separar X, y
datos = np.array(matriz)
X = datos[:, :-1]  # Todas las columnas menos la última
y = datos[:, -1]   # Última columna como etiqueta

# 3. Normalizar los datos
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)
