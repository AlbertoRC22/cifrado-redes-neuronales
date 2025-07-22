from train import entrenar
from eval import evaluar

if __name__ == '__main__':
    
    bits = 16
    epochs = 1000
    batch_size = 64
    alfa = 1.0
    beta = 2.0
    gamma = 1.0
    entrenar(bits, epochs, batch_size, alfa, beta, gamma)

    muestras = 10
    evaluar(bits, muestras)