from tensorflow.keras.models import load_model
from models import crear_modelo_alice, crear_modelo_bob
import numpy as np
from data_utils import generar_mensajes

def evaluar(bits, muestras):
    print("CARGANDO MODELOS")
    alice = crear_modelo_alice(bits)
    bob = crear_modelo_bob(bits)

    # Se cargan los pesos del entrenamiento anterior
    alice.load_weights('modelo_alice_key.keras')
    bob.load_weights('modelo_bob_key.keras')

    print("GENERANDO MENSAJES")
    mensajes = generar_mensajes(n=muestras, bits=bits)
    
    # Se cogen los mensajes y se generan las claves para tener suficientes
    print("CIFRANDO Y DESCIFRANDO")
    claves = np.random.randint(0, 2, size=(mensajes.shape[0], bits))
    cifrados = alice.predict([mensajes, claves])
    reconstruidos = bob.predict([cifrados, claves])

    print("\nEVALUACIÓN:\n")
    
    precisiones = []
    distancias = []
    reconstrucciones_perfectas = 0

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
        if distancia_hamming == 0:
            reconstrucciones_perfectas += 1
    
    media_precision = np.mean(precisiones)
    media_distancias = np.mean(distancias)
    print(f"La media de la precisión del descifrado es {media_precision:.4f}")
    print(f"Distancia media de Hamming = {media_distancias:.4f} | Número de bits: {bits}")
    print (f"Número de reconstrucciones perfectas = {reconstrucciones_perfectas} ")
    

    print('\n EVALUACIÓN DE BOB CON CLAVES ERRÓNEAS')

    precision_errores = []
    dist_errores = []

    claves_erroneas = np.random.randint(0, 2, size=(mensajes.shape[0], bits)).astype(np.float32)
    reconstruidos_bob_errados = bob.predict([cifrados, claves_erroneas])

    for i in range(muestras):
        original = mensajes[i].astype(int)
        reconstruidos_bob_err = (reconstruidos_bob_errados[i] > 0.5).astype(int)
        
        acc_err = np.mean(original == reconstruidos_bob_err)
        hamming_err = np.sum(original != reconstruidos_bob_err)

        print(f"[{i+1}] Bob (clave errada) --> {reconstruidos_bob_err} | Precisión: {acc_err:.2f} | Hamming: {hamming_err}")
        print("-" * 60)

        precision_errores.append(acc_err)
        dist_errores.append(hamming_err)

    print("\nRESULTADOS CON CLAVES ERRÓNEAS\n")
    print(f"Bob → Precisión media (clave errada): {np.mean(precision_errores):.4f} | Hamming media: {np.mean(dist_errores):.2f}")
