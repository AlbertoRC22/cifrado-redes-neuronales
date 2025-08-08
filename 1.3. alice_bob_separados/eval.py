import numpy as np
from tensorflow.keras.models import load_model
from models import crear_modelo_alice, crear_modelo_bob
from data_utils import generar_mensajes

def evaluar(bits, muestras):
    print("CARGANDO MODELOS")
    alice = crear_modelo_alice(bits)
    bob = crear_modelo_bob(bits)

    alice.load_weights('modelo_alice_separado.keras')
    bob.load_weights('modelo_bob_separado.keras')

    print("GENERANDO MENSAJES")
    mensajes = generar_mensajes(n=muestras, bits=bits)

    clave_fija = np.random.randint(0, 2, size=(1, bits)).astype(np.float32)
    claves = np.repeat(clave_fija, mensajes.shape[0], axis=0)

    cifrados = alice.predict([mensajes, claves])
    reconstruidos = bob.predict([cifrados, claves])

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
    print(f"\nPrecisión promedio: {media_precision:.4f}")
    print(f"Distancia de Hamming media: {media_distancias:.4f} (de {bits} bits)")
