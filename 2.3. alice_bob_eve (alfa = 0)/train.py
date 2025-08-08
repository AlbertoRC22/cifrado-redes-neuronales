from models import crear_modelo_alice, crear_modelo_bob, crear_modelo_eve
from data_utils import generar_mensajes
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import time as t

def entrenar(bits, epochs, batch_size, alfa, beta, gamma):
    
    n_mensajes = 10000
    adam_optimizer_rate = 0.001
    print("GENERANDO MENSAJES")
    mensajes = generar_mensajes(n_mensajes, bits)

    print("GENERANDO CLAVES")
    claves = np.random.randint(0, 2, (n_mensajes, bits)).astype(np.float32)

    print("INSTANCIANDO MODELOS")
    alice = crear_modelo_alice(bits)
    bob = crear_modelo_bob(bits)
    eve = crear_modelo_eve(bits)

    # Compilamos a Eve (solo un loss)
    eve.compile(optimizer=Adam(adam_optimizer_rate), loss='binary_crossentropy')
   
    mensajes_input = alice.input[0]
    claves_input = alice.input[1]
    claves_erroneas = Input(shape=(bits,), name='clave_err') # Se define que las claves son un tensor de ese tamaño

    # Construimos Alice, Bob e Eve con las entradas correspondientes
    # Nótese que hay 2 Bobs
    cifrado = alice([mensajes_input, claves_input])
    bob_bien = bob([cifrado, claves_input])
    bob_err = bob([cifrado, claves_erroneas])

    model_ab = Model(
        inputs  = [mensajes_input, claves_input, claves_erroneas],
        outputs = [bob_bien, bob_err]
    )
    
    # Se especifican dos funciones de pérdida diferentes
    model_ab.compile(
        optimizer    = Adam(adam_optimizer_rate),
        loss         = ['binary_crossentropy','binary_crossentropy'],
        loss_weights = [gamma, beta]
    )

    mejor_loss = float('inf')

    cifrados_para_eve = []
    mensajes_batch_para_eve = []

    for epoch in range(epochs):
        time_0 = t.time()
        idx = np.random.permutation(n_mensajes)[:batch_size]
        mensajes_batch = mensajes[idx]
        claves_batch = claves[idx]
        claves_dummy = np.random.randint(0,2,(batch_size,bits)).astype(np.float32)
        y_estimada = np.zeros_like(mensajes_batch)  # Así obligamos a Bob a fallar

        resultados = model_ab.train_on_batch(
            [mensajes_batch, claves_batch, claves_dummy],
            [mensajes_batch, y_estimada]
        )
        # Recuperamos las losses, la de loss_bob y loss_bob_errores
        _, loss_bob, loss_bob_errores = resultados

        cifrado_batch = alice.predict([mensajes_batch, claves_batch])
        loss_eve = eve.train_on_batch(cifrado_batch, mensajes_batch)

        # Metemos los datos en las listas para Eve
        mensajes_batch_para_eve.append(mensajes_batch)
        cifrados_para_eve.append(cifrado_batch)

        loss_total = gamma * loss_bob + beta * loss_bob_errores - alfa * loss_eve
        
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

        print(f"[Epoch {epoch+1:03}] | "
              f"Bob_ok={precision_bob:.3f} | "
              f"Bob_err={precision_errores:.3f} | "
              f"Eve_acc={precision_eve:.3f} | "
              f"loss_bob={loss_bob:.4f} | "
              f"loss_bob_errores={loss_bob_errores:.4f} | "
              f"TIME={(t.time() - time_0):.4f}]")

    
    mejor_loss_eve = float('inf')
    for epoch in range(epochs):
        time_0 = t.time()
        mensajes_batch = mensajes_batch_para_eve[epoch]
        cifrado_batch = cifrados_para_eve[epoch]

        loss_eve = loss_eve = eve.train_on_batch(cifrado_batch, mensajes_batch)

        if (loss_eve < mejor_loss_eve):
            mejor_loss_eve = loss_eve
            eve.save('modelo_eve.keras')

        print(f"[Epoch {epoch+1:03}] | "
              f"loss_eve={loss_eve:.4f} | "
              f"TIME={(t.time() - time_0):.4f}]")


