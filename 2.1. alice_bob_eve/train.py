from models import crear_modelo_alice, crear_modelo_bob, crear_modelo_eve
from data_utils import generar_mensajes
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import time as t

def generar_claves_y_mensajes(n_mensajes, bits):
    # Generando mensajes
    mensajes = generar_mensajes(n_mensajes, bits)

    # Generando claves
    claves = np.random.randint(0, 2, (n_mensajes, bits)).astype(np.float32)

    return mensajes, claves

def crear_modelos(bits):
    alice = crear_modelo_alice(bits)
    bob = crear_modelo_bob(bits)
    eve = crear_modelo_eve(bits)

    return alice, bob, eve


def entrenar(n_mensajes, bits, epochs, batch_size, adam_optimizer_rate, alfa, beta, gamma):
    
    mensajes, claves = generar_claves_y_mensajes(n_mensajes, bits)

    # Creando los modelos
    alice, bob, eve = crear_modelos(bits)

    # Compilamos a Eve (solo una loss)
    eve.compile(optimizer=Adam(adam_optimizer_rate), loss='binary_crossentropy')
   
    mensajes_input = alice.input[0]
    claves_input = alice.input[1]
    claves_erroneas = Input(shape=(bits,), name='clave_err') 

    # Construimos Alice, Bob e Eve con las entradas correspondientes
    # Nótese que hay 2 Bobs, para segurarnos de que Bob descifra con claves correctas
    cifrado = alice([mensajes_input, claves_input])
    bob_bien = bob([cifrado, claves_input])
    bob_err = bob([cifrado, claves_erroneas])

    eve.trainable = False
    eve_adv = eve(cifrado)

    model_ab = Model(
        inputs  = [mensajes_input, claves_input, claves_erroneas],
        outputs = [bob_bien, bob_err, eve_adv]
    )
    
    # Se especifican tres funciones de pérdida diferentes
    model_ab.compile(
        optimizer    = Adam(adam_optimizer_rate),
        loss         = ['binary_crossentropy','binary_crossentropy','binary_crossentropy'],
        loss_weights = [gamma, beta, alfa]
    )

    mejor_loss = float('inf')

    time_0 = t.time()
    for epoch in range(epochs):
        
        idx = np.random.permutation(n_mensajes)[:batch_size]
        mensajes_batch = mensajes[idx]
        claves_batch = claves[idx]

        cifrado_batch = alice.predict([mensajes_batch, claves_batch])
        loss_eve = eve.train_on_batch(cifrado_batch, mensajes_batch)

        claves_dummy = np.random.randint(0,2,(batch_size,bits)).astype(np.float32)
        y_estimada = np.zeros_like(mensajes_batch)

        y_eve_invertida = 1. - mensajes_batch

        resultados = model_ab.train_on_batch(
            [mensajes_batch, claves_batch, claves_dummy],
            [mensajes_batch, y_estimada, y_eve_invertida]
        )
        # Recuperamos las losses
        loss_total, loss_bob, loss_bob_errores, loss_eve_invertida = resultados
        # Ese loss_total es gamma * loss_bob + beta * loss_bob_errores + alfa *loss_eve_invertida
        
        if loss_total < mejor_loss:
            mejor_loss = loss_total
            alice.save('modelo_alice.keras')
            bob.save('modelo_bob.keras')
            eve.save('modelo_eve.keras')
            print(f"Modelos guardados en epoch {epoch+1} (loss_total={loss_total:.4f})")

        # Métricas de precisión que s eimprimen por epoch
        prediccion_bob = (bob.predict([cifrado_batch, claves_batch]) > 0.5).astype(int)
        prediccion_errores = (bob.predict([cifrado_batch, claves_dummy]) > 0.5).astype(int)

        precision_bob = np.mean(prediccion_bob  == mensajes_batch)
        precision_errores = np.mean(prediccion_errores == mensajes_batch)
        precision_eve = np.mean((eve.predict(cifrado_batch) > 0.5).astype(int) == mensajes_batch)

        print(
            f"EPOCHS TOTALES = {epochs} |"
            f"[Epoch {epoch+1:03}] | "
            f"Bob_ok={precision_bob:.3f} | "
            f"Bob_err={precision_errores:.3f} | "
            f"Eve_acc={precision_eve:.3f} | "
            f"loss_bob={loss_bob:.4f} | "
            f"loss_bob_errores={loss_bob_errores:.4f} | "
            f"loss_eve={loss_eve:.4f} | "
            f"loss_eve_inv={loss_eve_invertida:.4f}"
        )
    time = t.time() - time_0
    return time