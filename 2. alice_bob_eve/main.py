from train import entrenar
from eval import evaluar

if __name__ == '__main__':

    entrenamiento_conjunto = True

    bits = 32
    epochs = 3000
    batch_size = 64
    gamma = 1.0
    entrenar(bits, epochs, batch_size, gamma, entrenamiento_conjunto)

    muestras = 10
    evaluar(bits, muestras)