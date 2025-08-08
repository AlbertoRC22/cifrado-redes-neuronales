from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.regularizers import l2

def crear_modelo_alice(bits, key=True):
    
    # Se define la entrada, tanto con mensaje como con clave
    input_msg = Input(shape=(bits,), name='mensaje_original')
    input_key = Input(shape=(bits,), name='clave_simetrica')
    
    # Se conbinan el mensaje y al clave para formar la entrada
    x = Concatenate()([input_msg, input_key])
    
    # Va a tener dos capas, con 128 y 64 neuronas, respectivamente
    # Reciben como entrada los mensajes y claves 
    # Dense significa que las neuronas reciben como entrada toda la entrada anterior
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # bits = número de neuronas, cada una tiene una salida
    # Activación lineal
    # kernel_regularizer = función que evita el sobreajuste
    cifrado = Dense(bits, activation='linear', kernel_regularizer=l2(0.01))(x)

    return Model([input_msg, input_key], cifrado, name='Alice')

def crear_modelo_bob(bits, key=True):
    input_cifrado = Input(shape=(bits,), name='mensaje_cifrado')
    input_key = Input(shape=(bits,), name='clave_simetrica')

    x = Concatenate()([input_cifrado, input_key])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # Uso de la función sigmoide para reconstruir los mensajes porque es bianria
    reconstruido = Dense(bits, activation='sigmoid')(x)

    return Model([input_cifrado, input_key], reconstruido, name='Bob')
