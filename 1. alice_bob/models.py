from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.regularizers import l2

# Crea una red neuronal llamada Alice
def crear_modelo_alice(bits, key):
    input_msg = Input(shape=(bits,), name='mensaje_original') # Se define la entrada, que es el mensaje

    if key:
        input_key = Input(shape=(bits,), name='clave_simetrica') # Si hay clave, también la recibe
        x = Concatenate()([input_msg, input_key]) # Juntamos clave y mensaje para usarlo de input
        final_input = [input_msg, input_key]
    else:
        x = input_msg
        final_input = input_msg

    # Va a tener dos capas, con 128 y 64 neuronas, respectivamente
    # Reciebn como entrada los mensajes/mensajes y claves 
    # Dense significa que las neuronas reciben como entrada toda la entrada anterior
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # bits = número de neuronas, cada una tiene una salida
    # Activación lineal
    # kernel_regularizer = función que evita el sobreajuste
    cifrado = Dense(bits, activation='linear', kernel_regularizer=l2(0.01))(x)
    return Model(final_input, cifrado, name='Alice')

def crear_modelo_bob(bits, key):
    input_cifrado = Input(shape=(bits,), name='mensaje_cifrado')
    if key:
        input_key = Input(shape=(bits,), name='clave_simetrica')
        x = Concatenate()([input_cifrado, input_key])
        final_input = [input_cifrado, input_key]
    else:
        x = input_cifrado
        final_input = input_cifrado
    
    
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # Uso de la función sigmoide para reconstruir los mensajes porque es bianria
    reconstruido = Dense(bits, activation='sigmoid')(x)
    return Model(final_input, reconstruido, name='Bob')
