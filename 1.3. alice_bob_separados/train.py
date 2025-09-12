from models import crear_modelo_alice, crear_modelo_bob
from data_utils import generar_mensajes
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import time as t

def generar_mensajes_y_clave(n_mensajes, bits): 
    # Para simplificar por tener entrenamiento separado, se genera una sola clave
    clave_fija = np.random.randint(0, 2, size=(1, bits)).astype(np.float32)
    
    # Generamos los mensajes
    mensajes = generar_mensajes(n_mensajes, bits)
    
    return clave_fija, mensajes

def entrenar(n_mensajes, bits, epochs, batch_size, adam_optimizer):
    # Generamos una única clave y los mensajes
    clave_fija, mensajes = generar_mensajes_y_clave(n_mensajes, bits)

    # Entrenamiento de Alice
    alice = crear_modelo_alice(bits, key=True)
    alice.compile(optimizer=Adam(adam_optimizer), loss=BinaryCrossentropy())

    time_0 = t.time()
    for epoch in range(epochs):
        idx = np.random.choice(n_mensajes, batch_size)
        mensajes_batch = mensajes[idx]
        claves_batch = np.repeat(clave_fija, batch_size, axis=0)

        cifrados_batch = alice.predict([mensajes_batch, claves_batch])  # inicialización
        alice.train_on_batch([mensajes_batch, claves_batch], cifrados_batch)

        print(f"Época {epoch+1} Alice")

    # Guardamos el modelo de Alice
    alice.save("modelo_alice_separado.keras")

    # Ahora generamos los cifrados que Bob tendrá que aprender a descifrar
    claves_completas = np.repeat(clave_fija, n_mensajes, axis=0)
    cifrados = alice.predict([mensajes, claves_completas])

    # Instanciamos y entrenamos a Bob
    bob = crear_modelo_bob(bits, key=True)
    bob.compile(optimizer=Adam(adam_optimizer), loss=BinaryCrossentropy())

    for epoch in range(epochs):
        idx = np.random.choice(n_mensajes, batch_size)
        cifrados_batch = cifrados[idx]
        mensajes_batch = mensajes[idx]

        # Repite la clave para usar la misma en todo el batch
        claves_batch = np.repeat(clave_fija, batch_size, axis=0)

        bob.train_on_batch([cifrados_batch, claves_batch], mensajes_batch)
        reconstruidos = bob.predict([cifrados_batch, claves_batch])

        acc = np.mean((reconstruidos > 0.5).astype(int) == mensajes_batch)
        print(f"Época {epoch+1} Bob - Precisión del descifrado: {acc:.3f}")

    # Guardamos el modelo de Bob
    bob.save("modelo_bob_separado.keras")

    time = t.time() - time_0
    return time
