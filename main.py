# Importar las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from data import get_data 
import neural_network
import pandas as pd
import seaborn as sns

# Obtener los datos
train_data, test_data, validation_data = get_data()

accuracy_list = []

# Definir la función para evaluar los datos de validación
def val_data_evaluation(net):
    print("\nTotal accuracy of the network with validation data: ")
    correct_predictions = net.evaluate(validation_data)
    n_test = len(test_data)
    accuracy = (correct_predictions / n_test) * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Crear una red neuronal y entrenarla
net = neural_network.Network([64, 32, 16, 10])  # Se crea una red neuronal con 4 capas
accuracy_list.append(net.stochastic_gradient_descent(train_data, 150, 1, 0.01, "cross_entropy", "sigmoid", test_data=test_data))
val_data_evaluation(net)

# Descomenta para guardar datos sobre la red neuronal
# Guardar pesos
for i, layer_weights in enumerate(net.weights):
    filename = f'weights/weights_layer_{i}.npy'
    np.save(filename, layer_weights)

# Guardar sesgos
for i, layer_biases in enumerate(net.biases):
    filename = f'biases/biases_layer_{i}.npy'
    np.save(filename, layer_biases)

# Crear un DataFrame con las precisiones obtenidas durante el entrenamiento
accuracy_df = pd.DataFrame(accuracy_list).T
accuracy_df.columns = [f'Model {i+1}' for i in range(len(accuracy_list))]

# Crear el gráfico
plt.figure(figsize=(10, 6))
for i in range(len(accuracy_df.columns)):
    sns.lineplot(data=accuracy_df, x=accuracy_df.index, y=accuracy_df.columns[i])

# Añadir etiquetas y título
plt.xlabel('Iteraciones')
plt.ylabel('Precisión')
plt.title('Cross Entropy - Sigmoid')

# Guardar y mostrar el gráfico
plt.savefig('accuracy_plot.png')
plt.show()
