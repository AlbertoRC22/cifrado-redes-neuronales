import numpy as np

# Se generan n mensajes binarios aleatoriamente de los bits especificados
def generar_mensajes(n, bits):
    return np.random.randint(0, 2, size=(n, bits)).astype(np.float32)