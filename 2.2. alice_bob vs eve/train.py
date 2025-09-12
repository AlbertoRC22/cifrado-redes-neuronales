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

def generar_batch(n_mensajes, batch_size, mensajes, claves):
    idx = np.random.permutation(n_mensajes)[:batch_size]
    mensajes_batch = mensajes[idx]
    claves_batch = claves[idx]

    return mensajes_batch, claves_batch

def actualizar_mejor_loss(alice, bob, loss_total, epoch):
    mejor_loss = loss_total
    alice.save('modelo_alice.keras')
    bob.save('modelo_bob.keras')
    print(f"Modelos guardados en epoch {epoch+1} (loss_total={loss_total:.4f})")
    
    return mejor_loss

def toma_de_medidas(mensajes_batch, cifrado_batch, claves_batch, claves_dummy, bob):

    prediccion_bob = (bob.predict([cifrado_batch, claves_batch]) > 0.5).astype(int)
    prediccion_errores = (bob.predict([cifrado_batch, claves_dummy]) > 0.5).astype(int)

    precision_bob = np.mean(prediccion_bob  == mensajes_batch)
    precision_errores = np.mean(prediccion_errores == mensajes_batch)

    return precision_bob, precision_errores

def entrenar_eve(eve, epochs, mensajes_batch_para_eve, cifrados_para_eve):
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

def entrenar(n_mensajes, bits, epochs, batch_size, adam_optimizer_rate, beta, gamma):
 
    mensajes, claves = generar_claves_y_mensajes(n_mensajes, bits)

    alice, bob, eve = crear_modelos(bits)

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

    tiempo_inicial = t.time()
    for epoch in range(epochs):
        
        # Generamos los mensajes y las claves de este batch
        mensajes_batch, claves_batch = generar_batch(n_mensajes, batch_size, mensajes, claves)

        # Generamos claves dummy y lo que hay que estimar a partir de ellas.
        # Con esto enseñaremos a Bob a cifrar mal si no tiene la clave correcta
        claves_dummy = np.random.randint(0,2,(batch_size,bits)).astype(np.float32)
        y_estimada = np.zeros_like(mensajes_batch)

        resultados = model_ab.train_on_batch(
            [mensajes_batch, claves_batch, claves_dummy],
            [mensajes_batch, y_estimada]
        )
        # Recuperamos las losses, la de loss_bob y loss_bob_errores
        _, loss_bob, loss_bob_errores = resultados

        cifrado_batch = alice.predict([mensajes_batch, claves_batch])

        # Metemos los datos en las listas para Eve
        mensajes_batch_para_eve.append(mensajes_batch)
        cifrados_para_eve.append(cifrado_batch)

        loss_total = gamma * loss_bob + beta * loss_bob_errores

        # Actualizamos la pérdida, y los modelos, si esta ha mejorado
        if loss_total < mejor_loss:
            mejor_loss = actualizar_mejor_loss(alice, bob, eve, loss_total, epoch)

        # Métricas de precisión que se imprimen por epoch
        precision_bob, precision_errores = toma_de_medidas(mensajes_batch, cifrado_batch, claves_batch, claves_dummy, bob)

        print(
            f"[Epoch {epoch+1:03}] | "
            f"Bob_ok={precision_bob:.3f} | "
            f"Bob_err={precision_errores:.3f} | "
            f"loss_bob={loss_bob:.4f} | "
            f"loss_bob_errores={loss_bob_errores:.4f} | "
        )
    
    entrenar_eve(eve, epochs, mensajes_batch_para_eve, cifrados_para_eve)

    tiempo_total = t.time() - tiempo_inicial
    return tiempo_total