from sklearn.neural_network import MLPClassifier
import numpy as np
from data import get_data

# Obtener los datos de entrenamiento, prueba y validación
train_data, test_data, validation_data = get_data()

# Separar las características (X_train) y las etiquetas (y_train) del conjunto de entrenamiento
X_train, y_train = zip(*train_data)
X_train = np.array(X_train)
y_train = np.array(y_train)
n, m, _ = X_train.shape
X_train = X_train.reshape(n, m)

# Definir la arquitectura de la red neuronal
hidden_layer_sizes = (32, 16)
regularization_term = 0.001
learning_rate = 0.01
max_iter = 100
batch_size = 1

# Crear un objeto MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                    activation="logistic", solver="sgd", 
                    alpha=regularization_term, max_iter=max_iter, 
                    batch_size=batch_size, learning_rate_init=learning_rate, random_state=161)

# Entrenar el modelo
mlp.fit(X_train, y_train)

# Calcular la precisión en el conjunto de entrenamiento
train_accuracy = mlp.score(X_train, y_train)
print("Training accuracy:", train_accuracy)
