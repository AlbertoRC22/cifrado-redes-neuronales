from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from tensorflow.keras.regularizers import l2

def crear_modelo_alice(bits):
    input_msg = Input(shape=(bits,), name='mensaje_original')

    input_key = Input(shape=(bits,), name='clave_simetrica')
    x = Concatenate()([input_msg, input_key])
    final_input = [input_msg, input_key]

    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    cifrado = Dense(bits, activation='linear', kernel_regularizer=l2(0.01))(x)
    return Model(final_input, cifrado, name='Alice')

def crear_modelo_bob(bits):
    input_cifrado = Input(shape=(bits,), name='mensaje_cifrado')
    input_key = Input(shape=(bits,), name='clave_simetrica')
    x = Concatenate()([input_cifrado, input_key])
    final_input = [input_cifrado, input_key]
    
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    reconstruido = Dense(bits, activation='sigmoid')(x)
    return Model(final_input, reconstruido, name='Bob')

# Eve es igual, pero no recibe ninguna clave
def crear_modelo_eve(bits):
    input_cifrado = Input(shape=(bits,), name=f'mensaje_cifrado_eve')
    x = Dense(64, activation='relu')(input_cifrado)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    salida = Dense(bits, activation='sigmoid', name=f'mensaje_reconstruido_eve')(x)
    return Model (input_cifrado, salida, name='Eve')

