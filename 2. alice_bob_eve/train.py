import numpy as np
import time
from models import crear_modelo_alice, crear_modelo_bob, crear_modelo_eve
from data_utils import generar_mensajes
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Método auxiliar que genera las claves necesarias
def generar_claves(n, bits):
    return np.random.randint(0, 2, size=(n, bits)).astype(np.float32)

# Método auxiliar que evalua la predicción de un mensaje
def evaluar_salida(modelo, x, threshold=0.5):
    return (modelo.predict(x) > threshold).astype(int)

# Método auxiliar que calcula precisión de una salida
def calcular_precision(salida, original):
    return np.mean(salida == original.astype(int))

def entrenar(bits, epochs, batch_size, gamma, entrenamiento_conjunto):
    print("Generando mensajes")
    mensajes = generar_mensajes(10000, bits)
    alice = crear_modelo_alice(bits)
    bob = crear_modelo_bob(bits)
    eve = crear_modelo_eve(bits)

    # Se entrena con Eve vs No se entrena con Eve
    if entrenamiento_conjunto:
        cifrado, mensajes_batch = entrenar_conjunto(alice, bob, eve, mensajes, bits, epochs, batch_size, gamma)
    else:
        cifrado, mensajes_batch = entrenar_individual(alice, bob, mensajes, bits, epochs, batch_size)

    # En ambos casos, Eve ataca
    atacar_con_eve(eve, cifrado, mensajes_batch, epochs, batch_size)

def entrenar_conjunto(alice, bob, eve, mensajes, bits, epochs, batch_size, gamma):
    bce = BinaryCrossentropy()
    mejor_loss = float('inf') # Punto de partida inicial apra la función de pérdida

    # Se entrenan Alice y Bob en conjunto para cifrar y Eve por otro lado para atacar
    modelo_ab = Model([alice.input[0], alice.input[1]], bob([alice.output, alice.input[1]]))
    modelo_ab.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')
    eve.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')

    for epoch in range(epochs):
        t0 = time.time() # Variable temporal que sirve para calcular luego el tiempo de la epoch

        idx = np.random.permutation(len(mensajes))
        batch_mensajes = mensajes[idx[:batch_size]]
        batch_claves = generar_claves(batch_size, bits)

        # Se obtiene las pérdidas de Bob y de Eve, así como los cifrados
        loss_bob = modelo_ab.train_on_batch([batch_mensajes, batch_claves], batch_mensajes)
        cifrado = alice.predict([batch_mensajes, batch_claves])
        loss_eve = eve.train_on_batch(cifrado, batch_mensajes)

        # Con eto se entrena a Bob a no descifrar mal
        claves_erradas = generar_claves(batch_size, bits)
        cifrado_errado = alice.predict([batch_mensajes, claves_erradas])
        reconstruido = bob.predict([cifrado_errado, claves_erradas])
        # Función nueva para este caso específico de descifrar lo que no toca
        loss_non_dst = bce(batch_mensajes, reconstruido).numpy()

        # Se usan las funciones auxiliares para evaluar salidas y cauclar precisiones
        salida_bob = evaluar_salida(bob, [cifrado, batch_claves])
        salida_eve = evaluar_salida(eve, cifrado)
        precision_bob = calcular_precision(salida_bob, batch_mensajes)
        precision_eve = calcular_precision(salida_eve, batch_mensajes)

        # Función de pérdida total
        loss_total = loss_bob + loss_eve + gamma * loss_non_dst
        
        # Se imprime el cómo va para ir teniendo datos
        print(f"[Epoch {epoch+1:03}] Bob: {loss_bob:.4f} | NDst: {loss_non_dst:.4f} | Eve: {loss_eve:.4f} | Precision_Bob: {precision_bob:.3f} | Precision_Eve: {precision_eve:.3f} | ⏱️ {time.time()-t0:.2f}s")

        # Hay que minimizar esa función, por lo que se va actualizando
        if loss_total < mejor_loss:
            mejor_loss = loss_total
            alice.save("modelo_alice.keras")
            bob.save("modelo_bob.keras")
            eve.save("modelo_eve_entrenada.keras")
            print("Modelos guardados")

    print("Entrenamiento conjunto terminado")
    return cifrado, batch_mensajes

def entrenar_individual(alice, bob, mensajes, bits, epochs, batch_size):
    print("Entrenamiento ALICE/BOB (SIN EVE)")
    mejor_loss = float('inf') # Misma lógica d eponer un número altísimo

    # Aquí Eve ni pincha ni corta, por lo que solo se compilan Alice y Bob
    modelo_ab = Model([alice.input[0], alice.input[1]], bob([alice.output, alice.input[1]]))
    modelo_ab.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')

    for epoch in range(epochs):
        t0 = time.time()

        idx = np.random.permutation(len(mensajes))
        batch_mensajes = mensajes[idx[:batch_size]]
        batch_claves = generar_claves(batch_size, bits)

        # Como en otros casos, se va calculando la pérdida
        loss = modelo_ab.train_on_batch([batch_mensajes, batch_claves], batch_mensajes)
        cifrado = alice.predict([batch_mensajes, batch_claves])
        salida_bob = evaluar_salida(bob, [cifrado, batch_claves])
        precision_bob = calcular_precision(salida_bob, batch_mensajes)

        # Se va documentando el progreso
        print(f"[Epoch {epoch+1:03}] LossBob: {loss:.4f} | Precision_Bob: {precision_bob:.3f} | ⏱️ {time.time()-t0:.2f}s")

        # Vamos guardando la mejor pérdida, que es la mínima
        if loss < mejor_loss:
            mejor_loss = loss
            alice.save("modelo_alice.keras")
            bob.save("modelo_bob.keras")
            print("Modelos guardados")

    print("Entrenamiento Alice/Bob finalizado")
    return cifrado, batch_mensajes

def atacar_con_eve(eve, cifrado, mensajes, epochs, batch_size):
    print("Entrenando EVE para atacar SIN CLAVES...")
    # Se compila a Eve porque es la que se va a entrenar ahora
    eve.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')


    for epoch in range(epochs):
        t0 = time.time()
        idx = np.random.choice(batch_size, batch_size)
        x_batch = cifrado[idx]
        y_batch = mensajes[idx]

        loss = eve.train_on_batch(x_batch, y_batch)
        salida = evaluar_salida(eve, x_batch)
        precision = calcular_precision(salida, y_batch)

        print(f"[Epoch {epoch+1:02}] LossEve: {loss:.4f} | Precision_Eve: {precision:.3f} | ⏱️ {time.time()-t0:.2f}s")

    eve.save("modelo_eve_atacante.keras")
    print("Ataque de Eve completado")
