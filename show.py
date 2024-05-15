from collections import Counter
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from data import get_data

def see_digits():
    # Cargar los dígitos y verlos
    digits = load_digits()
    digits.images.shape
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i + 150])
    plt.show()

def see_test_train_val_set():
    # Obtener los datos de entrenamiento, prueba y validación
    train_data, test_data, validation_data = get_data()

    # Obtener las etiquetas de los conjuntos de datos
    train_labels = [item[1] for item in train_data]
    test_labels = [item[1] for item in test_data]
    val_labels = [item[1] for item in validation_data]

    # Ordenar las etiquetas
    train_labels.sort()
    test_labels.sort()
    val_labels.sort()

    # Contar la frecuencia de cada etiqueta en cada conjunto
    train_c = Counter(train_labels)
    test_c = Counter(test_labels)
    val_c = Counter(val_labels)
    counters = [train_c, test_c, val_c]

    # Obtener los números y las frecuencias de cada conjunto
    numbers = [list(counter.keys()) for counter in counters]
    frequencies = [list(counter.values()) for counter in counters]

    bar_width = 0.25
    indices = np.arange(len(numbers[0]))
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#99FFFF', '#FFD700', '#FF69B4', '#BA55D3']
    v = ["Entrenamiento", "Prueba", "Validación"]

    # Graficar las frecuencias de cada número por conjunto
    for i, (nums, freqs) in enumerate(zip(numbers, frequencies)):
        plt.bar(indices + i * bar_width, freqs, bar_width, label=v[i], color=colors[i])

    plt.xlabel('Número')
    plt.ylabel('Frecuencia')
    plt.title('Frecuencia de cada número por conjunto')
    plt.xticks(indices + bar_width, numbers[0])
    plt.legend()
    plt.tight_layout()
    plt.show()

see_test_train_val_set()
