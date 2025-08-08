from train import entrenar
from eval import evaluar

if __name__ == '__main__':
    bits = 32
    epochs = 2000
    batch_size = 64
    entrenar(bits, epochs, batch_size)

    muestras = 10
    evaluar(bits, muestras)