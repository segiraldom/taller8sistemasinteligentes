import csv

# Ruta al archivo CSV
ruta_archivo = 'caesarian.csv'

# Lista para almacenar los datos
matriz = []

# Leer el archivo CSV
with open(ruta_archivo, newline='') as archivo:
    lector = csv.reader(archivo)
    next(lector)  # Saltar la primera fila (encabezados)
    for fila in lector:
        # Convertir cada valor a entero
        fila_entera = [int(valor) for valor in fila]
        matriz.append(fila_entera)