import tensorflow as tf
from data import get_data
import numpy as np

# Cargar los datos
train_data, _, _ = get_data()

# Separar las características y las etiquetas del conjunto de entrenamiento
X_train, y_train = zip(*train_data)
X_train = np.array(X_train)
y_train = np.array(y_train)
n, m, _ = X_train.shape
X_train = X_train.reshape(n, m)

# Obtener los datos de prueba y sus etiquetas
test_data, _, _ = get_data()
X_test, y_test = zip(*test_data)
X_test = np.array(X_test)
y_test = np.array(y_test)
n, m, _ = X_test.shape
X_test = X_test.reshape(n, m)

# Definir la arquitectura de la red neuronal con activación sigmoide logística
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X_train.shape[1:]),
    tf.keras.layers.Dense(32, activation='sigmoid'),
    tf.keras.layers.Dense(16, activation='sigmoid'),
    tf.keras.layers.Dense(10, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=100, batch_size=1)

# Evaluar el modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", test_accuracy)
