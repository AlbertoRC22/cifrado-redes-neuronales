import tensorflow as tf

# Crear un modelo simple con una capa densa
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compilar el modelo
model.compile(optimizer='sgd', loss='mean_squared_error')

# Datos de entrenamiento
import numpy as np
X = np.array([1, 2, 3, 4], dtype=float)
Y = np.array([2, 4, 6, 8], dtype=float)

# Entrenar el modelo
model.fit(X, Y, epochs=500)

# Predicción
print(model.predict(np.array([5], dtype=float)))

