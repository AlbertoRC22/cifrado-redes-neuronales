from tensorflow.keras.models import load_model
from models import crear_modelo_alice, crear_modelo_bob
import numpy as np
from data_utils import generar_mensajes

def evaluar(n_mensajes, bits, muestras, res_file_name, epochs):
    print("CARGANDO MODELOS")
    alice = crear_modelo_alice(bits)
    bob = crear_modelo_bob(bits)

    # Se cargan los pesos del entrenamiento anterior
    alice.load_weights('modelo_alice_key.keras')
    bob.load_weights('modelo_bob_key.keras')

    print("GENERANDO MENSAJES")
    mensajes = generar_mensajes(n_mensajes, bits)
    
    # Se cogen los mensajes y se generan las claves para tener suficientes
    print("CIFRANDO Y DESCIFRANDO")
    claves = np.random.randint(0, 2, size=(mensajes.shape[0], bits))
    cifrados = alice.predict([mensajes, claves])
    reconstruidos = bob.predict([cifrados, claves])

    with open(res_file_name, "a") as f:
        f.write(f"\nEVALUACIÓN CON {epochs}:\n\n")
    
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
            str_mensaje_reconstruido = f"Reconstruido --> {reconstruido} | Precisión: {precision:.2f}\n"
            str_distancia_hamming = f"Distancia de Hamming: {distancia_hamming}\n"
            str_mensaje_delimiter = ("-" * 50) + "\n"
            str_muestra = str_mensaje_original + str_mensaje_reconstruido + str_distancia_hamming + str_mensaje_delimiter
            with open(res_file_name, "a") as f:
                f.write(str_muestra)


        precisiones.append(precision)
        distancias.append(distancia_hamming)
        if distancia_hamming == 0:
            reconstrucciones_perfectas += 1
    
    media_precision = np.mean(precisiones)
    media_distancias = np.mean(distancias)
    str_media_precision = f"La media de la precisión del descifrado es {media_precision:.4f}\n"
    str_media_distancias = f"Distancia media de Hamming = {media_distancias:.4f} | Número de bits: {bits}\n"
    str_reconstrucciones_perfectas = f"Número de reconstrucciones perfectas = {reconstrucciones_perfectas}\n"
    str_medidas = str_media_precision + str_media_distancias + str_reconstrucciones_perfectas + "\n\n"
    with open(res_file_name, "a") as f:
        f.write(str_medidas)

    
    with open(res_file_name, "a") as f:
        f.write('\n EVALUACIÓN DE BOB CON CLAVES ERRÓNEAS\n\n')

    precision_errores = []
    dist_errores = []
    reconstrucciones_perfectas_errores = 0

    claves_erroneas = np.random.randint(0, 2, size=(mensajes.shape[0], bits)).astype(np.float32)
    reconstruidos_bob_errados = bob.predict([cifrados, claves_erroneas])

    for i in range(muestras):
        original = mensajes[i].astype(int)
        reconstruidos_bob_err = (reconstruidos_bob_errados[i] > 0.5).astype(int)
        
        acc_err = np.mean(original == reconstruidos_bob_err)
        hamming_err = np.sum(original != reconstruidos_bob_err)

        str_errores = f"[{i+1}] Bob (clave errada) --> {reconstruidos_bob_err} | Precisión: {acc_err:.2f} | Hamming: {hamming_err}\n"
        str_errores_delimiter = ("-" * 60) + "\n"
        str_errores_final = str_errores + str_errores_delimiter

        with open(res_file_name, "a") as f:
            f.write(str_errores_final)

        precision_errores.append(acc_err)
        dist_errores.append(hamming_err)
        if(hamming_err == 0):
            reconstrucciones_perfectas_errores += 1

    media_precision_errores = np.mean(precision_errores)
    media_distancia_errores = np.mean(dist_errores)

    with open(res_file_name, "a") as f:
        f.write("\nRESULTADOS CON CLAVES ERRÓNEAS\n\n")
        f.write(f"Bob -> Precisión media (clave errada): {media_precision_errores:.4f} | Hamming media: {media_distancia_errores:.2f}\n")
        f.write(f"Bob -> Reconstrucciones perfectas (con clave errada): {reconstrucciones_perfectas_errores}")
    
    return [media_precision, media_distancias, reconstrucciones_perfectas, media_precision_errores, media_distancia_errores, reconstrucciones_perfectas_errores]
