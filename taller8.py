import csv #import de la lectura csv
import numpy as np # Import de calculos numéricos
from sklearn.preprocessing import MinMaxScaler # Import de la normalización de datos
import matplotlib.pyplot as plt # Import de la visualización de datos
import time # Import para medir el tiempo de ejecución

# ------------------------------
# Función sigmoide y su derivada
# ------------------------------
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x)) # Función sigmoide

# ------------------------------
# 1. Leer archivo CSV
# ------------------------------
ruta_archivo = 'caesarian.csv'
matriz = [] # Lista para almacenar los datos

with open(ruta_archivo, newline='') as archivo: # Abrir archivo CSV
    lector = csv.reader(archivo) # Leer el archivo
    next(lector)  # Saltar encabezado
    for fila in lector: # Iterar sobre cada fila
        fila_entera = [int(valor) for valor in fila]
        matriz.append(fila_entera) # Convertir cada fila a enteros y agregar a la matriz

# ------------------------------
# 2. Separar datos y etiquetas
# ------------------------------
datos = np.array(matriz) # Convertir la lista a un array de NumPy
X = datos[:, :-1] # Todas las columnas excepto la última
y = datos[:, -1] # Última columna como etiquetas

# ------------------------------
# 3. Normalizar los datos
# ------------------------------
scaler = MinMaxScaler() # Crear un escalador
X_norm = scaler.fit_transform(X) # Normalizar los datos

# ------------------------------
# 4. Barajar y dividir los datos
# ------------------------------
datos_combinados = np.hstack((X_norm, y.reshape(-1, 1))) # Combinar X y y
np.random.seed(42) # Fijar semilla para reproducibilidad
np.random.shuffle(datos_combinados) # Barajar los datos combinados

X_shuffled = datos_combinados[:, :-1] # Todas las columnas excepto la última
y_shuffled = datos_combinados[:, -1] # Última columna como etiquetas

n_total = len(X_shuffled) # Número total de muestras
n_entrenamiento = int(n_total * 0.85) # Número de muestras para entrenamiento

X_train = X_shuffled[:n_entrenamiento] # Primer 85% de las muestras
y_train = y_shuffled[:n_entrenamiento] # Primer 85% de las etiquetas
X_test = X_shuffled[n_entrenamiento:] # Resto de las muestras
y_test = y_shuffled[n_entrenamiento:] # Resto de las etiquetas

# ------------------------------
# 5. Inicializar pesos aleatorios
# ------------------------------
np.random.seed(42) # Fijar semilla para reproducibilidad
syn0 = np.random.rand(X_train.shape[1], 9)  # capa oculta de 9 neuronas
syn1 = np.random.rand(9, 1) # capa de salida de 1 neurona

# Medir el tiempo de inicio del entrenamiento
inicio = time.time()

# ------------------------------
# 6. Entrenamiento con backpropagation
# ------------------------------
eta = 0.45 # Tasa de aprendizaje
iterac = [] # Lista para almacenar iteraciones
vecerror = [] # Lista para almacenar errores

max_iters = 10000000 # Máximo número de iteraciones
patience = 100 # Paciencia para early stopping
tolerance = 1e-7 # Tolerancia para early stopping
early_stop_counter = 0 # Contador para early stopping
last_error = None # Último error para comparar

for iter in range(max_iters): # Ciclo de entrenamiento
    # Propagación hacia adelante
    l1 = nonlin(np.dot(X_train, syn0)) # Capa oculta
    l2 = nonlin(np.dot(l1, syn1)) # Capa de salida

    # Cálculo del error
    l2_error = y_train.reshape(-1, 1) - l2 # Error de la capa de salida
    errorabs = np.mean(np.abs(l2_error)) # Error absoluto medio (MAE)

    # Backpropagation
    l2_delta = l2_error * nonlin(l2, deriv=True) * eta # Delta de la capa de salida
    l1_error = l2_delta.dot(syn1.T) # Error de la capa oculta
    l1_delta = l1_error * nonlin(l1, deriv=True) * eta # Delta de la capa oculta

    # Actualizar pesos
    syn1 += l1.T.dot(l2_delta)
    syn0 += X_train.T.dot(l1_delta)

    # Guardar error
    if iter % 10 == 0:
        iterac.append(iter)
        vecerror.append(errorabs)
        print(f"Iteración {iter} - MAE: {errorabs:.6f}")

        # Early stopping
        if last_error is not None and abs(last_error - errorabs) < tolerance: # Si el error no cambia
            early_stop_counter += 1 # Contador de early stopping
            if early_stop_counter > patience: # Si se supera la paciencia
                print("⚠️ Early stopping: error estabilizado.")
                break
        else:
            early_stop_counter = 0 # Reiniciar contador si el error cambia
        last_error = errorabs # Actualizar el último error

fin = time.time() # Tiempo final de entrenamiento
print(f"\nTiempo de entrenamiento: {fin - inicio:.2f} segundos") # Mostrar tiempo de entrenamiento

# ------------------------------
# 7. Predicciones finales (entrenamiento y prueba)
# ------------------------------
# Predicción sobre entrenamiento
l2_train = nonlin(np.dot(nonlin(np.dot(X_train, syn0)), syn1))
# Umbralizar salida
l2_train_bin = (l2_train > 0.5).astype(int) # Convertir a binario (0 o 1)

# Accuracy
acc_train = np.mean(l2_train_bin.flatten() == y_train) # Calcular precisión

print(f"\nExactitud del entrenamiento: {round(acc_train * 100, 2)}%") # Mostrar precisión

# Predicción sobre prueba
l2_test = nonlin(np.dot(nonlin(np.dot(X_test, syn0)), syn1)) # Predicción sobre la prueba
l2_bin_test = (l2_test > 0.5).astype(int) # Umbralizar salida
acc_test = np.mean(l2_bin_test.flatten() == y_test)
print(f"Exactitud de la prueba: {round(acc_test * 100, 2)}%")

# Predicción para nuevos datos de 2 instancias nuevas
datos1 = [22, 0, 1, 0, 0,] 
datos2 = [26, 3, 2, 2, 1,]
datos1Norm = scaler.transform([datos1]) # Normalizar datos de entrada
datos2Norm = scaler.transform([datos2]) # Normalizar datos de entrada
# Predicción para nuevos datos
prediccion1 = nonlin(np.dot(nonlin(np.dot(datos1Norm, syn0)), syn1))
prediccion2 = nonlin(np.dot(nonlin(np.dot(datos2Norm, syn0)), syn1))
print(f"\nPredicción para datos1: {prediccion1[0][0]:.4f}")
print(f"Predicción para datos2: {prediccion2[0][0]:.4f}")

# ------------------------------
# 8. Gráfica comparación salida real vs predicha (entrenamiento)
# ------------------------------
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