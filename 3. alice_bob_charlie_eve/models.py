from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.regularizers import l2

def crear_modelo(bits, nombre):
    input_cifrado = Input(shape=(bits,), name=f'mensaje_cifrado_{nombre}')
    x = Dense(64, activation='relu')(input_cifrado)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    salida = Dense(bits, activation='sigmoid', name=f'mensaje_reconstruido_{nombre}')(x)
    return Model (input_cifrado, salida, name=nombre.capitalize())


def crear_modelo_alice(bits, n_destinatarios):
    
    # Cotenido del mensaje. Es una secuencia de bits
    # El destinatario se representa con [1, 0] o [0, 1]
    input_msg = Input(shape=(bits,), name='mensaje_original')
    input_dst = Input(shape=(n_destinatarios,), name='destinatario')

    # Le da a Alice el contexto, de forma que sabe qué cifrado aplicar
    # en función del contendio y del destinatario
    x = Concatenate()([input_msg, input_dst])

    # Capas ocultas que permiten aprender una función de cifrado
    # x = Dense(256, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dense(64, activation='tanh')(x)
    # x = Dense(32, activation='sigmoid')(x)

    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # Lineal para preservar el cifrado. L2 ayuda a evitar el overfitting
    cifrado = Dense(bits, activation='linear',
                    kernel_regularizer=l2(0.01),
                    name='mensaje_cifrado')(x)

    return Model([input_msg, input_dst], cifrado, name='Alice')

def crear_modelo_bob(bits):
    nombre = 'bob'
    modelo = crear_modelo(bits, nombre)
    return modelo

def crear_modelo_charlie(bits):
    nombre = 'charlie'
    modelo = crear_modelo(bits, nombre)
    return modelo

def crear_modelo_eve(bits):
    nombre = 'eve'
    modelo = crear_modelo(bits, nombre)
    return modelo
