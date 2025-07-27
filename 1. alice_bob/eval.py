import numpy as np
from tensorflow.keras.models import load_model
from models import crear_modelo_alice, crear_modelo_bob
from data_utils import generar_mensajes

def evaluar(key, bits, muestras):
    print("CARGANDO MODELOS")
    alice = crear_modelo_alice(bits, key)
    bob = crear_modelo_bob(bits, key)

    # Se cargan los pesos del entrenamiento anterior
    if key:
        alice.load_weights('modelo_alice_key.keras')
        bob.load_weights('modelo_bob_key.keras')
    else:
        alice.load_weights('modelo_alice.keras')
        bob.load_weights('modelo_bob.keras')

    print("GENERANDO MENSAJES")
    mensajes = generar_mensajes(n=muestras, bits=bits)
    
    # Se cogen los mensajes y se generan las claves para tener suficientes
    print("CIFRANDO Y DESCIFRANDO")
    if key:
        claves = np.random.randint(0, 2, size=(mensajes.shape[0], bits))
        cifrados = alice.predict([mensajes, claves])
        reconstruidos = bob.predict([cifrados, claves])
    else:
        cifrados = alice.predict(mensajes)
        reconstruidos = bob.predict(cifrados)

    print("\nEVALUACIÓN:\n")
    
    precisiones = []
    distancias = []

    for i in range(len(reconstruidos)):
        
        original = mensajes[i].astype(int)
        reconstruido = (reconstruidos[i] > 0.5).astype(int)
        
        # Coges el original y el reconstruido y haces la media de cuántos bits se parecen
        precision = np.mean(original == reconstruido)
        distancia_hamming = np.sum(original != reconstruido)

        if i < muestras:
            print(f"Original     --> {original}")
            print(f"Reconstruido --> {reconstruido} | Precisión: {precision:.2f}")
            print(f"Distancia de Hamming: {distancia_hamming}")
            print("-" * 50)

        precisiones.append(precision)
        distancias.append(distancia_hamming)
    
    media_precision = np.mean(precisiones)
    media_distancias = np.mean(distancias)
    print(f"La media de la precisión del descifrado es {media_precision:.4f}")
    print(f"Distancia media de Hamming = {media_distancias:.4f} | Número de bits: {bits}")
