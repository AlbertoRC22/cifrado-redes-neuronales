from models import crear_modelo_alice, crear_modelo_bob, crear_modelo_charlie, crear_modelo_eve
from data_utils import generar_mensajes
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Input
from monitor import inicializar_historial, registrar_pérdidas, guardar_si_mejora, graficar_historial
import numpy as np

def entrenar(bits, epochs, batch_size, alfa, beta, gamma):
    
    n_mensajes = 10000
    print("GENERANDO MENSAJES")
    mensajes = generar_mensajes(n_mensajes, bits)
    n_destinatarios = 2

    print("INSTANCIANDO MODELOS")
    alice = crear_modelo_alice(bits, n_destinatarios)
    bob = crear_modelo_bob(bits)
    charlie = crear_modelo_charlie(bits)
    eve = crear_modelo_eve(bits)

    bob.compile(optimizer=Adam(0.001), loss='binary_crossentropy')
    charlie.compile(optimizer=Adam(0.001), loss='binary_crossentropy')
    eve.compile(optimizer=Adam(0.001), loss='binary_crossentropy')

    # Se entrenan combinados para facilitar el entrenamiento
    modelo_ab = Model([alice.input[0], alice.input[1]], bob(alice.output))
    modelo_ac = Model([alice.input[0], alice.input[1]], charlie(alice.output))
    modelo_ab.compile(optimizer=Adam(0.001), loss='binary_crossentropy')
    modelo_ac.compile(optimizer=Adam(0.001), loss='binary_crossentropy')

    historial = inicializar_historial()
    bce = BinaryCrossentropy()

    for epoch in range(epochs):

        idx = np.random.permutation(len(mensajes))
        mensajes_batch = mensajes[idx[:batch_size]]

        # Se replican las etiquetas en cada batch
        dst_bob = np.tile([1, 0], (batch_size, 1))
        dst_charlie = np.tile([0, 1], (batch_size, 1))

        # Bob y Charlie entrenan para reconstruir sus mensajes
        loss_bob = modelo_ab.train_on_batch([mensajes_batch, dst_bob], mensajes_batch)
        loss_charlie = modelo_ac.train_on_batch([mensajes_batch, dst_charlie], mensajes_batch)

        # Alice cifra los mensajes
        cifrado_para_bob = alice.predict([mensajes_batch, dst_bob])
        cifrado_para_charlie = alice.predict([mensajes_batch, dst_charlie])

        # Eve intenta interceptar los mensajes de Bob y charlie
        # Los apila con vtsack y también tiene los mensaje normales, duplicados para que cuadre
        loss_eve = eve.train_on_batch(np.vstack([cifrado_para_bob, cifrado_para_charlie]),
                                      np.vstack([mensajes_batch, mensajes_batch]))

        # Receptores equivocados intentan descifrar
        reconstruido_bob_sobre_charlie = bob.predict(cifrado_para_charlie)
        reconstruido_charlie_sobre_bob = charlie.predict(cifrado_para_bob)

        # Se calula la pérdida cruzada, de cuando no es el destinatario
        loss_bob_sobre_charlie = bce(mensajes_batch, reconstruido_bob_sobre_charlie).numpy()
        loss_charlie_sobre_bob = bce(mensajes_batch, reconstruido_charlie_sobre_bob).numpy()
        loss_non_dst = (loss_bob_sobre_charlie + loss_charlie_sobre_bob) / 2

        # Cálculo de pérdida total con pesos ajustables
        # Ajustable con parámetros y se recompensa:
        # que Bob y Charlie lo hagan bien

        # se penaliza que:
        # Eve logre descifrar mensajes
        # el destinatario incorrecto descifre bien

        loss_total = gamma * ((loss_bob + loss_charlie)/2) + beta * loss_non_dst - alfa * loss_eve


        registrar_pérdidas(historial,
                        loss_bob,
                        loss_charlie,
                        loss_eve,
                        loss_non_dst,
                        loss_total)

        guardar_si_mejora(loss_total, historial, alice, bob, charlie)

        # Formateo de la salida en cada epoch
        print(f"[Epoch {epoch+1:03}] Bob: {loss_bob:.4f} | Charlie: {loss_charlie:.4f} | No-dest: {loss_non_dst:.4f} | Eve: {loss_eve:.4f} | Total: {loss_total:.4f}")

    graficar_historial(historial)

