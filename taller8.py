import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
from tabulate import tabulate

# ------------------------------
# Función sigmoide y su derivada
# ------------------------------
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# ------------------------------
# 1. Leer archivo CSV
# ------------------------------
ruta_archivo = 'caesarian.csv'
matriz = []

with open(ruta_archivo, newline='') as archivo:
    lector = csv.reader(archivo)
    next(lector)  # Saltar encabezado
    for fila in lector:
        fila_entera = [int(valor) for valor in fila]
        matriz.append(fila_entera)

# ------------------------------
# 2. Separar datos y etiquetas
# ------------------------------
datos = np.array(matriz)
X = datos[:, :-1]
y = datos[:, -1]

# ------------------------------
# 3. Normalizar los datos
# ------------------------------
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# ------------------------------
# 4. Barajar y dividir los datos
# ------------------------------
datos_combinados = np.hstack((X_norm, y.reshape(-1, 1)))
np.random.seed(42)
np.random.shuffle(datos_combinados)

X_shuffled = datos_combinados[:, :-1]
y_shuffled = datos_combinados[:, -1]

n_total = len(X_shuffled)
n_entrenamiento = int(n_total * 0.85)

X_train = X_shuffled[:n_entrenamiento]
y_train = y_shuffled[:n_entrenamiento]
X_test = X_shuffled[n_entrenamiento:]
y_test = y_shuffled[n_entrenamiento:]

# ------------------------------
# 5. Inicializar pesos aleatorios
# ------------------------------
np.random.seed(42)
syn0 = np.random.rand(X_train.shape[1], 9)  # capa oculta de 9 neuronas
syn1 = np.random.rand(9, 1)

# ------------------------------
# 6. Entrenamiento con backpropagation
# ------------------------------
eta = 0.55
iterac = []
vecerror = []

for iter in range(20000):
    l1 = nonlin(np.dot(X_train, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # Cálculo del error
    l2_error = y_train.reshape(-1, 1) - l2
    errorabs = np.mean(np.abs(l2_error))

    # Backpropagation
    l2_delta = l2_error * nonlin(l2, deriv=True) * eta
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True) * eta

    # Actualizar pesos
    syn1 += l1.T.dot(l2_delta)
    syn0 += X_train.T.dot(l1_delta)

    # Guardar error
    if iter % 10 == 0:
        iterac.append(iter)
        vecerror.append(errorabs)


print ('Output After Training:')
#print('Salida red:','\n',l2)
print ('Error:' + str(np.mean(np.abs(l2_error))))
print('Pesos Nuevos','\n')
print (syn1)
print (syn0)

# Calcular predicciones finales para los datos de entrenamiento
l1_train = nonlin(np.dot(X_train, syn0))
l2_train = nonlin(np.dot(l1_train, syn1))  # salida final de la red

# Graficar comparación
plt.figure(figsize=(10, 5))
plt.plot(y_train, label="Y real (y_train)", marker='o')
plt.plot(l2_train.flatten(), label="Salida red neuronal", marker='x')
plt.title("Comparación entre salida real y salida de la red neuronal (entrenamiento)")
plt.xlabel("Índice de muestra")
plt.ylabel("Valor")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# 9. Gráfica de evolución del error
# ------------------------------
plt.plot(iterac, vecerror)
plt.title("Evolución del error (MAE) durante el entrenamiento")
plt.xlabel("Iteraciones")
plt.ylabel("MAE")
plt.grid(True)
plt.show()