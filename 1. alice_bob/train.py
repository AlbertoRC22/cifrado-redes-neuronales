# train.py
from models import crear_modelo_alice, crear_modelo_bob
from data_utils import generar_mensajes
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np

def entrenar(key, bits, epochs, batch_size):
    n_mensajes = 10000
    print("GENERANDO MENSAJES")
    mensajes = generar_mensajes(n_mensajes, bits)

    print("INSTANCIANDO MODELOS")
    alice = crear_modelo_alice(bits, key)
    bob = crear_modelo_bob(bits, key)

    if key:
        # Modelo combinado con clave
        mensaje_input, clave_input = alice.input
        cifrado_output = alice.output
        # Recibe la input de Alice y Bob recibe su salida y la clave, se entrenan conjuntamente
        modelo = Model([mensaje_input, clave_input], bob([cifrado_output, clave_input]))
    else:
        # Modelo combinado sin clave
        modelo = Model(alice.input, bob(alice.output))

    # Se prepara el modelo para entrenarlo con Adam y BinaryCrossEntropy
    # Adam = algoritmo que calcula y actualiza los pesos, con 0.001 como un ratio estándar
    # BinaryCrossEntropy = función que mide el error de la estimación
    modelo.compile(optimizer=Adam(0.001), loss=BinaryCrossentropy())


    # COMIENZA EL ENTRENAMIENTO
    for epoch in range(epochs):

        # Se selecciona un lote aleatorio de mensajes, que es lo que va en mensajes_batch para evitar overfitting
        idx = np.random.choice(n_mensajes, batch_size)
        mensajes_batch = mensajes[idx]

        if key:
            # Se generan claves en cada lote
            claves_batch = np.random.randint(0, 2, size=(batch_size, bits)).astype(np.float32)
            # Entrenamiento con clave
            modelo.train_on_batch([mensajes_batch, claves_batch], mensajes_batch)
            reconstruidos = modelo.predict([mensajes_batch, claves_batch])
        else:
            # Entrenamiento sin clave
            modelo.train_on_batch(mensajes_batch, mensajes_batch)
            reconstruidos = modelo.predict(mensajes_batch)
        
        # Media de todos los reonstruidos correctamente
        precision = np.mean((reconstruidos > 0.5).astype(int) == mensajes_batch)
        print(f"Época {epoch+1} - Precisión del descifrado: {precision:.3f}")

    print("GUARDANDO MODELOS")
    if key:
        alice.save('modelo_alice_key.keras')
        bob.save('modelo_bob_key.keras')
    else:
        alice.save('modelo_alice.keras')
        bob.save('modelo_bob.keras')
