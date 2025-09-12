from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from models import crear_modelo_alice, crear_modelo_bob
import numpy as np
import time as t

from data_utils import generar_mensajes

def crear_modelo_conjunto(bits):
    alice = crear_modelo_alice(bits)
    bob = crear_modelo_bob(bits)

    # Modelo combinado con clave
    mensaje_input, clave_input = alice.input
    cifrado_output = alice.output

    # Recibe la input de Alice y Bob recibe su salida y la clave, se entrenarán conjuntamente
    modelo = Model([mensaje_input, clave_input], bob([cifrado_output, clave_input]))

    return alice, bob, modelo

def entrenar_modelo(modelo, n_mensajes, epochs, mensajes, batch_size):
    for epoch in range(epochs):

        # Se selecciona un batch aleatorio de mensajes, que es lo que va en mensajes_batch para evitar overfitting
        idx = np.random.choice(n_mensajes, batch_size)
        mensajes_batch = mensajes[idx]

        # Se generan claves en cada batch
        claves_batch = np.random.randint(0, 2, size=(batch_size, bits)).astype(np.float32)
            
        # Entrenamiento con clave
        modelo.train_on_batch([mensajes_batch, claves_batch], mensajes_batch)
        reconstruidos = modelo.predict([mensajes_batch, claves_batch])
           
        # Media de todos los reonstruidos correctamente
        precision = np.mean((reconstruidos > 0.5).astype(int) == mensajes_batch)
        print(f"Epochs totales: {epochs} | Epoch {epoch+1} - Precisión del descifrado: {precision:.3f}")

def entrenar(n_mensajes, bits, epochs, batch_size, adam_optimizer):
    # Generamos los mensajes
    mensajes = generar_mensajes(n_mensajes, bits)

    alice, bob, modelo = crear_modelo_conjunto(bits)

    # Se prepara el modelo para entrenarlo con Adam y BinaryCrossEntropy
    modelo.compile(optimizer=Adam(adam_optimizer), loss=BinaryCrossentropy())

    time_0 = t.time()
    
    # Comienza el entrenamiento
    entrenar_modelo(modelo, n_mensajes, epochs, mensajes, batch_size)

    time = t.time() - time_0
  
    alice.save('modelo_alice_key.keras')
    bob.save('modelo_bob_key.keras')

    return time
