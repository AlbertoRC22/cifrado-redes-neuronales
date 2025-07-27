from models import crear_modelo_alice, crear_modelo_bob
from data_utils import generar_mensajes
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np

def entrenar(bits, epochs, batch_size):
    n_mensajes = 10000

    # S gener auna clave para cifrar todos los mensajes, ya que se entrenan por separado
    print("🔐 Usando clave fija")
    clave_fija = np.random.randint(0, 2, size=(1, bits)).astype(np.float32)

    print("🚀 Generando mensajes")
    mensajes = generar_mensajes(n_mensajes, bits)

    print("🧠 Entrenando ALICE")
    alice = crear_modelo_alice(bits, key=True)
    alice.compile(optimizer=Adam(0.001), loss=BinaryCrossentropy())

    for epoch in range(epochs):
        idx = np.random.choice(n_mensajes, batch_size)
        mensajes_batch = mensajes[idx]
        claves_batch = np.repeat(clave_fija, batch_size, axis=0)

        cifrados_batch = alice.predict([mensajes_batch, claves_batch])  # inicialización
        alice.train_on_batch([mensajes_batch, claves_batch], cifrados_batch)

        print(f"Época {epoch+1} Alice ✅")

    print("💾 Guardando modelo de Alice")
    alice.save("modelo_alice_separado.keras")

    print("🔄 Generando cifrados con Alice entrenada")
    claves_completas = np.repeat(clave_fija, n_mensajes, axis=0)
    cifrados = alice.predict([mensajes, claves_completas])

    print("🧠 Entrenando BOB")
    bob = crear_modelo_bob(bits, key=True)
    bob.compile(optimizer=Adam(0.001), loss=BinaryCrossentropy())

    for epoch in range(epochs):
        idx = np.random.choice(n_mensajes, batch_size)
        cifrados_batch = cifrados[idx]
        mensajes_batch = mensajes[idx]

        # Repite la clave para usar la misma en todo el lote
        claves_batch = np.repeat(clave_fija, batch_size, axis=0)

        bob.train_on_batch([cifrados_batch, claves_batch], mensajes_batch)
        reconstruidos = bob.predict([cifrados_batch, claves_batch])

        acc = np.mean((reconstruidos > 0.5).astype(int) == mensajes_batch)
        print(f"Época {epoch+1} Bob - Precisión del descifrado: {acc:.3f}")

    print("💾 Guardando modelo de Bob")
    bob.save("modelo_bob_separado.keras")
