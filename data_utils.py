import numpy as np

def generar_mensajes(n, bits):
    return np.random.randint(0, 2, size=(n, bits)).astype(np.float32)
