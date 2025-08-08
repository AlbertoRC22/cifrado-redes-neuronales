from train import entrenar
from eval import evaluar

if __name__ == '__main__':

    entrenamiento_conjunto = True

    bits = 32
    epochs = 3000
    batch_size = 128
    alfa = 1.0
    beta = 2.0
    gamma = 7.0
    entrenar(bits, epochs, batch_size, beta, gamma)

    muestras = 10
    evaluar(bits, muestras)