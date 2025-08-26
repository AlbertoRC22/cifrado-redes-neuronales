import numpy as np
from tensorflow.keras.models import load_model
from models import crear_modelo_alice, crear_modelo_bob, crear_modelo_eve
from data_utils import generar_mensajes

def evaluar(n_mensajes, bits, muestras, epochs, res_file_name):
    print("CARGANDO MODELOS")
    alice = crear_modelo_alice(bits)
    bob = crear_modelo_bob(bits)
    eve = crear_modelo_eve(bits)

    # Se cargan los pesos del entrenamiento anterior
    alice.load_weights('modelo_alice.keras')
    bob.load_weights('modelo_bob.keras')
    eve.load_weights('modelo_eve_entrenada.keras')

    mensajes = generar_mensajes(n_mensajes, bits).astype(np.float32)
    claves = np.random.randint(0, 2, size=(mensajes.shape[0], bits)).astype(np.float32)

    # Tanto Bob como Eve intentan reconstruir
    cifrados = alice.predict([mensajes, claves])
    reconstruidos_bob = bob.predict([cifrados, claves])
    reconstruidos_eve = eve.predict(cifrados)

    with open(res_file_name, "a") as f:
        f.write(f"\nEVALUACIÓN CON {epochs}:\n\n")

    precisiones_bob = []
    precisiones_eve = []
    dist_bob = []
    dist_eve = []

    reconstrucciones_perfectas_bob = 0
    reconstrucciones_perfectas_eve = 0

    for i in range(len(cifrados)):
        original = mensajes[i].astype(int)

        reconstruidos_b = (reconstruidos_bob[i] > 0.5).astype(int)
        reconstruidos_e = (reconstruidos_eve[i] > 0.5).astype(int)

        precicion_bob = np.mean(original == reconstruidos_b)
        precicion_eve = np.mean(original == reconstruidos_e)

        distancia_hamming_bob = np.sum(original != reconstruidos_b)
        distancia_hamming_eve = np.sum(original != reconstruidos_e)

        if i < muestras:
            str_original = f"[{i+1}] Original  --> {original}\n"
            str_reconstruido_bob = f"     Bob       --> {reconstruidos_b}   | Precisión: {precicion_bob:.2f} | Hamming: {distancia_hamming_bob}\n"
            str_reconstruido_eve = f"     Eve       --> {reconstruidos_e}   | Precisión: {precicion_eve:.2f} | Hamming: {distancia_hamming_eve}\n"
            str_mensaje_delimiter = ("-" * 60) + "\n"
            str_muestra = str_original + str_reconstruido_bob + str_reconstruido_eve + str_mensaje_delimiter
            with open(res_file_name, "a") as f:
                f.write(str_muestra)

        precisiones_bob.append(precicion_bob)
        precisiones_eve.append(precicion_eve)
        dist_bob.append(distancia_hamming_bob)
        dist_eve.append(distancia_hamming_eve)

        if distancia_hamming_bob == 0:
            reconstrucciones_perfectas_bob += 1
        
        if distancia_hamming_eve == 0:
            reconstrucciones_perfectas_eve += 1

    media_precisiones_bob = np.mean(precisiones_bob)
    media_precisiones_eve = np.mean(np.mean(precisiones_eve))
    media_distancias_bob = np.mean(np.mean(dist_bob))
    media_distancias_eve = np.mean(np.mean(dist_eve))
    str_titulo = "\nRESULTADOS FINALES\n\n"
    str_precision_bob = f"Bob -> Precisión media: {media_precisiones_bob:.4f} | Hamming media: {media_distancias_bob:.2f}\n"
    str_precision_eve = f"Eve -> Precisión media: {media_precisiones_eve:.4f} | Hamming media: {media_distancias_eve:.2f}\n"
    str_reconstrucciones_perfectas = f"Bob -> Perfectas: {reconstrucciones_perfectas_bob} | Eve -> Perfectas: {reconstrucciones_perfectas_eve}\n"

    str_resultados_finales = str_titulo + str_precision_bob + str_precision_eve + str_reconstrucciones_perfectas
    with open(res_file_name, "a") as f:
        f.write(str_resultados_finales)

    # En este trozo se generan claves erradas y se intenta que Bob descifre con ellas
    # Es lo mismo, pero con un set de claves nuevo
    with open(res_file_name, "a") as f:
        f.write("\nEVALUACIÓN CON CLAVES ERRÓNEAS PARA BOB\n\n")

    precisiones_bob_errado = []
    distancias_bob_errado = []
    reconstrucciones_perfectas_errado = 0

    claves_erradas = np.random.randint(0, 2, size=(mensajes.shape[0], bits)).astype(np.float32)
    reconstruidos_bob_errado = bob.predict([cifrados, claves_erradas])

    for i in range(len(cifrados)):
        original = mensajes[i].astype(int)
        reconstruidos_bob_err = (reconstruidos_bob_errado[i] > 0.5).astype(int)
        
        acc_err = np.mean(original == reconstruidos_bob_err)
        hamming_err = np.sum(original != reconstruidos_bob_err)

        if i < muestras:
            str_bob_errado = f"[{i+1}] Bob (clave errada) --> {reconstruidos_bob_err} | Precisión: {acc_err:.2f} | Hamming: {hamming_err}\n"
            str_mensaje_delimiter_errado = ("-" * 60) + "\n"
            str_errado = str_bob_errado + str_mensaje_delimiter_errado
            with open(res_file_name, "a") as f:
                f.write(str_errado)

        precisiones_bob_errado.append(acc_err)
        distancias_bob_errado.append(hamming_err)

        if(distancias_bob_errado == 0):
            reconstrucciones_perfectas_errado += 1

    media_precisiones_bob_errado = np.mean(precisiones_bob_errado)
    media_distancias_bob_errado = np.mean(distancias_bob_errado)
    with open(res_file_name, "a") as f:
        f.write("\nRESULTADOS CON CLAVES ERRÓNEAS\n\n")
        f.write(f"Bob -> Precisión media (clave errada): {media_precisiones_bob_errado:.4f} | Hamming media: {media_distancias_bob_errado:.2f}\n")

    return [media_precisiones_bob, media_precisiones_eve, media_distancias_bob, media_distancias_eve, reconstrucciones_perfectas_bob, reconstrucciones_perfectas_eve, media_precisiones_bob_errado, media_distancias_bob_errado, reconstrucciones_perfectas_errado]
