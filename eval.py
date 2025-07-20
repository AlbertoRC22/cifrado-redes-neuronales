import numpy as np
from tensorflow.keras.models import load_model
from models import crear_modelo_alice, crear_modelo_bob, crear_modelo_charlie, crear_modelo_eve
from data_utils import generar_mensajes

def evaluar(bits, muestras):
    
    print("CARGANDO MODELOS")
    alice = crear_modelo_alice(bits, 2)
    bob = crear_modelo_bob(bits)
    charlie = crear_modelo_charlie(bits)
    eve = crear_modelo_eve(bits)

    alice.load_weights('modelo_alice.keras')
    bob.load_weights('modelo_bob.keras')
    charlie.load_weights('modelo_charlie.keras')

    mensajes = generar_mensajes(n=muestras, bits=bits)
    dst_bob = np.tile([1, 0], (muestras, 1))
    dst_charlie = np.tile([0, 1], (muestras, 1))

    cifrado_bob = alice.predict([mensajes, dst_bob])
    cifrado_charlie = alice.predict([mensajes, dst_charlie])

    reconstruidos_bob = bob.predict(cifrado_bob)
    reconstruidos_charlie = charlie.predict(cifrado_charlie)

    bob_sobre_charlie = bob.predict(cifrado_charlie)
    charlie_sobre_bob = charlie.predict(cifrado_bob)

    reconstruidos_eve_bob = eve.predict(cifrado_bob)
    reconstruidos_eve_charlie = eve.predict(cifrado_charlie)

    print("\nEVALUACIÓN EXTENDIDA:\n")
    for i in range(muestras):
        original = mensajes[i].astype(int)

        # Mira la probabilidad en el mensaje y atribuye un 0 (<0.5) o un 1 (>0.5)
        bob_out = (reconstruidos_bob[i] > 0.5).astype(int)
        charlie_out = (reconstruidos_charlie[i] > 0.5).astype(int)
        eve_bob_out = (reconstruidos_eve_bob[i] > 0.5).astype(int)
        eve_charlie_out = (reconstruidos_eve_charlie[i] > 0.5).astype(int)

        bob_wrong = (bob_sobre_charlie[i] > 0.5).astype(int)
        charlie_wrong = (charlie_sobre_bob[i] > 0.5).astype(int)

        # Comprobamos la media de aciertos
        acc_bob = np.mean(original == bob_out)
        acc_charlie = np.mean(original == charlie_out)
        acc_eve_bob = np.mean(original == eve_bob_out)
        acc_eve_charlie = np.mean(original == eve_charlie_out)
        acc_wrong_bob = np.mean(original == bob_wrong)
        acc_wrong_charlie = np.mean(original == charlie_wrong)

        print(f"Original ----------> {original}")
        print(f"Bob (correcto) ----> {bob_out} | Precisión: {acc_bob:.2f}")
        print(f"Charlie -----------> {charlie_out} | Precisión: {acc_charlie:.2f}")
        print(f"Bob sobre Charlie -> {bob_wrong} | Precisión indebida: {acc_wrong_bob:.2f}")
        print(f"Charlie sobre Bob -> {charlie_wrong} | Precisión indebida: {acc_wrong_charlie:.2f}")
        print(f"Eve sobre Bob -----> {eve_bob_out} | Precisión: {acc_eve_bob:.2f}")
        print(f"Eve sobre Charlie -> {eve_charlie_out} | Precisión: {acc_eve_charlie:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    evaluar()
