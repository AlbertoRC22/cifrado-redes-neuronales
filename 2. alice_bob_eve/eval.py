import numpy as np
from tensorflow.keras.models import load_model
from models import crear_modelo_alice, crear_modelo_bob, crear_modelo_eve
from data_utils import generar_mensajes

def evaluar(bits, muestras):
    print("CARGANDO MODELOS")
    alice = crear_modelo_alice(bits)
    bob = crear_modelo_bob(bits)
    eve = crear_modelo_eve(bits)

    # Se cargan los pesos del entrenamiento anterior
    alice.load_weights('modelo_alice.keras')
    bob.load_weights('modelo_bob.keras')
    eve.load_weights('modelo_eve_entrenada.keras')

    mensajes = generar_mensajes(n=muestras, bits=bits).astype(np.float32)
    claves = np.random.randint(0, 2, size=(mensajes.shape[0], bits)).astype(np.float32)

    # Tanto Bob como Eve intentan reconstruir
    cifrados = alice.predict([mensajes, claves])
    reconstruidos_bob = bob.predict([cifrados, claves])
    reconstruidos_eve = eve.predict(cifrados)

    print("\nEVALUACIÓN FINAL:\n")

    precisiones_bob = []
    precisiones_eve = []
    dist_bob = []
    dist_eve = []

    for i in range(muestras):
        original = mensajes[i].astype(int)

        reconstruidos_bob = (reconstruidos_bob[i] > 0.5).astype(int)
        reconstruidos_e = (reconstruidos_eve[i] > 0.5).astype(int)

        precicion_bob = np.mean(original == reconstruidos_bob)
        precicion_eve = np.mean(original == reconstruidos_e)

        hamming_b = np.sum(original != reconstruidos_bob)
        hamming_e = np.sum(original != reconstruidos_e)

        print(f"[{i+1}] Original  --> {original}")
        print(f"     Bob       --> {reconstruidos_bob}   | Precisión: {precicion_bob:.2f} | Hamming: {hamming_b}")
        print(f"     Eve       --> {reconstruidos_e}   | Precisión: {precicion_eve:.2f} | Hamming: {hamming_e}")
        print("-" * 60)

        precisiones_bob.append(precicion_bob)
        precisiones_eve.append(precicion_eve)
        dist_bob.append(hamming_b)
        dist_eve.append(hamming_e)

    print("\nRESULTADOS FINALES\n")
    print(f"Bob → Precisión media: {np.mean(precisiones_bob):.4f} | Hamming media: {np.mean(dist_bob):.2f}")
    print(f"Eve → Precisión media: {np.mean(precisiones_eve):.4f} | Hamming media: {np.mean(dist_eve):.2f}")


    # En este trozo se generan claves erradas y se intenta que Bob descifre con ellas
    # Es lo mismo, pero con un set de claves nu
    print("\nEVALUACIÓN CON CLAVES ERRÓNEAS PARA BOB\n")

    accs_bob_errado = []
    dist_bob_errado = []

    claves_erradas = np.random.randint(0, 2, size=(mensajes.shape[0], bits)).astype(np.float32)
    reconstruidos_bob_errado = bob.predict([cifrados, claves_erradas])

    for i in range(muestras):
        original = mensajes[i].astype(int)
        reconstruidos_bob_err = (reconstruidos_bob_errado[i] > 0.5).astype(int)
        
        acc_err = np.mean(original == reconstruidos_bob_err)
        hamming_err = np.sum(original != reconstruidos_bob_err)

        print(f"[{i+1}] Bob (clave errada) --> {reconstruidos_bob_err} | Precisión: {acc_err:.2f} | Hamming: {hamming_err}")
        print("-" * 60)

        accs_bob_errado.append(acc_err)
        dist_bob_errado.append(hamming_err)

    print("\n📉 RESULTADOS CON CLAVES ERRÓNEAS\n")
    print(f"Bob → Precisión media (clave errada): {np.mean(accs_bob_errado):.4f} | Hamming media: {np.mean(dist_bob_errado):.2f}")
