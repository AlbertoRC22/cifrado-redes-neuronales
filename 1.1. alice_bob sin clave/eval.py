from tensorflow.keras.models import load_model
from models import crear_modelo_alice, crear_modelo_bob
import numpy as np

from data_utils import generar_mensajes

def cargar_modelos(bits):
    alice = crear_modelo_alice(bits)
    bob = crear_modelo_bob(bits)

    alice.load_weights('modelo_alice.keras')
    bob.load_weights('modelo_bob.keras')

    return alice, bob

def analizar_resultados(muestras, res_file_name, mensajes, reconstruidos):
    
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

            str_mensaje_original = f"Original     --> {original}\n"
            str_precision_reconstruido = f"Reconstruido --> {reconstruido} | Precisión: {precision:.2f}\n"
            str_distancia_hamming = f"Distancia de Hamming: {distancia_hamming}\n"
            str_mensaje_delimiter = ("-" * 50) + "\n"
            str_muestra = str_mensaje_original + str_precision_reconstruido + str_distancia_hamming + str_mensaje_delimiter
            with open(res_file_name, "a") as f:
                f.write(str_muestra)


        precisiones.append(precision)
        distancias.append(distancia_hamming)
        if distancia_hamming == 0:
            reconstrucciones_perfectas += 1
    
    return precisiones, distancias, reconstrucciones_perfectas

def evaluar(n_mensajes, bits, muestras, res_file_name, epochs):
    
    # Cargamos los modelos y los pesos del entrenamiento anterior
    alice, bob = cargar_modelos(bits)

    # Generamos los mensajes
    mensajes = generar_mensajes(n_mensajes, bits)
    
    # Se cifran y descifran los mensajes
    cifrados = alice.predict(mensajes)
    reconstruidos = bob.predict(cifrados)

    with open(res_file_name, "a") as f:
        f.write(f"\nEVALUACIÓN CON {epochs}:\n\n")

    precisiones, distancias, reconstrucciones_perfectas = analizar_resultados(muestras, res_file_name, mensajes, reconstruidos)
    
    media_precision = np.mean(precisiones)
    media_distancias = np.mean(distancias)
    
    str_media_precision = f"La media de la precisión del descifrado es {media_precision:.4f}\n"
    str_media_distancias = f"Distancia media de Hamming = {media_distancias:.4f} | Número de bits: {bits}\n"
    str_reconstrucciones_perfectas = f"Número de reconstrucciones perfectas = {reconstrucciones_perfectas}\n"
    str_medidas = str_media_precision + str_media_distancias + str_reconstrucciones_perfectas +"\n\n"
    with open(res_file_name, "a") as f:
        f.write(str_medidas)

    return [media_precision, media_distancias, reconstrucciones_perfectas]