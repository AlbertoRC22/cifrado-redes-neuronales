from train import entrenar
from eval import evaluar

if __name__ == '__main__':
    key = True
    
    bits = 32

    if key:
        epochs = 2000
    else:
        epochs = 2000

    batch_size = 64
    entrenar(key, bits, epochs, batch_size)

    muestras = 10
    evaluar(key, bits, muestras)