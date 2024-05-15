# Importar las funciones necesarias de scikit-learn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Definir una función para obtener los datos divididos en conjuntos de entrenamiento, validación y prueba
def get_data():
    # Cargar el conjunto de datos de dígitos
    digits = load_digits()
    
    # Dividir los datos en conjuntos de entrenamiento y datos temporales
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        digits.data, digits.target, 
        train_size=0.6,  # 60% de los datos para entrenamiento
        test_size=0.4,   # 40% restante para datos temporales
        random_state=1   # Semilla aleatoria para reproducibilidad
    )
    
    # Dividir los datos temporales en conjuntos de validación y prueba
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels,
        train_size=0.5,  # 50% de los datos temporales para validación
        test_size=0.5,   # 50% de los datos temporales para prueba
        random_state=1   # Mismo estado aleatorio para coherencia
    )
    
    # Convertir los datos en una lista de tuplas, donde cada tupla contiene una imagen de un dígito y su etiqueta
    train = [(train_data[i].reshape(-1, 1), train_labels[i]) for i in range(len(train_data))]
    val = [(val_data[i].reshape(-1, 1), val_labels[i]) for i in range(len(val_data))]
    test = [(test_data[i].reshape(-1, 1), test_labels[i]) for i in range(len(test_data))]
    
    # Devolver los conjuntos de datos divididos
    return train, test, val
