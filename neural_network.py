# Importar las librerías necesarias
import random
import numpy as np

# Configurar las semillas aleatorias para reproducibilidad
random.seed(163)
np.random.seed(163)

# Definir la clase Network
class Network:
    def __init__(self, *args):
        if len(args) == 1:  # Si se proporciona solo un argumento, se asume que es la estructura
            self.initialize_from_structure(args[0])
        elif len(args) == 2:  # Si se proporcionan dos argumentos, se asume que son pesos y sesgos
            self.initialize_from_weights_biases(args[0], args[1])
        else:
            raise ValueError("Invalid number of arguments provided")

    def initialize_from_structure(self, structure):
        # Inicializar la red neuronal con una estructura dada
        self.num_layers = len(structure)
        self.structure = structure
        # Inicializar los pesos y sesgos de manera aleatoria con una distribución normal
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(structure[:-1], structure[1:])]
        self.biases = [np.random.randn(y, 1) for y in structure[1:]]

    def initialize_from_weights_biases(self, weights, biases):
        # Inicializar la red neuronal con pesos y sesgos dados
        self.structure = [len(layer) for layer in range(len(biases))]
        self.num_layers = len(self.structure)
        self.weights = weights
        self.biases = biases
    
    def feedforward(self, a, store_values=False, f="sigmoid"):
        # Propagación hacia adelante de la entrada a través de la red neuronal
        a = np.array(a)
        n, m, _ = a.shape
        a = a.reshape(n, m)

        outputs = []
        weighted_inputs = [a]
        activations = [a]
        
        layer = a
        
        for w, b in zip(self.weights, self.biases):
            # Calcular las entradas ponderadas para toda la capa
            z = np.dot(w, layer.T).T + b.T
            
            # Aplicar la función de activación a toda la capa
            layer = self.activation_function(z, f)
            
            if store_values:
                weighted_inputs.append(z)
                activations.append(layer)
        
        outputs = layer
        
        if store_values:
            return outputs, weighted_inputs, activations
        
        return outputs
    
    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, alpha, cost_function, activation_function, test_data=None):
        # Algoritmo de descenso de gradiente estocástico para el entrenamiento de la red neuronal
        n = len(training_data)
        learning_accuracy_list = []
        for epoch in range(epochs):
            # Obtener subconjuntos de los datos de entrenamiento
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            training_images = [[image for image, _ in mini_batch] for mini_batch in mini_batches]
            training_labels = [[label for _, label in mini_batch] for mini_batch in mini_batches]
            for training_images, training_labels in zip(training_images, training_labels):
                # Por cada subconjunto, aplicar el algoritmo de decenso por gradiente
                self.update_mini_batch(training_images, training_labels, alpha, cost_function, activation_function)
                # Evaluar el modelo
            if test_data:
                n_test = len(test_data)
                correct_predictions = self.evaluate(test_data)
                accuracy = (correct_predictions / n_test) * 100
                learning_accuracy_list.append(accuracy)
                print(f"Epoch {epoch}: {correct_predictions} / {n_test} || Accuracy: {accuracy:.2f}%")
            else:
                print(f"Epoch {epoch} complete")
        return learning_accuracy_list
    
    def update_mini_batch(self, training_images, training_labels, alpha, cost_function, activation_function):
        # Actualizar los pesos y sesgos de la red neuronal utilizando mini lotes de datos
        n = len(training_images)
        # Retropropagación
        delta_nabla_b, delta_nabla_w = self.backprop(training_images, training_labels, activation_function, cost_function)
        a_div_n = alpha / n
        # Descenso por gradiente
        self.weights = [1*w - (a_div_n) * nw for w, nw in zip(self.weights, delta_nabla_w)]
        self.biases = [b - (a_div_n) * nb for b, nb in zip(self.biases, delta_nabla_b)]

    def backprop(self, x, y, activation_function, cost_function):
        # Algoritmo de retropropagación para calcular el gradiente del costo
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Calcular el error de la útlima capa
        _, weighted_inputs, activations = self.feedforward(x, store_values=True) 
        last_z = weighted_inputs[-1]
        last_a = activations[-1]
        almost_last_a = activations[-2]
        target = np.array([np.eye(self.structure[-1], dtype=int)[i] for i in y])
        delta = self.cost_derivative(last_a, last_z, target, cost_function, activation_function) * self.derivative_function(last_z, activation_function)
        nabla_b[-1] = np.sum(delta, axis=0).reshape(self.structure[self.num_layers-1], 1)
        nabla_w[-1] = np.dot(delta.T, almost_last_a)

        # Propagar el error a cada capa
        for l in range(2, self.num_layers):
            l_z = weighted_inputs[-l]
            l_a_before = activations[-l-1]        
            s =  self.derivative_function(l_z, activation_function)
            delta = np.dot(delta, self.weights[-l+1]) * s
            nabla_b[-l] = np.sum(delta, axis=0, keepdims=True).reshape(self.structure[-l], 1)
            nabla_w[-l] = np.dot(delta.T, l_a_before)
        
        return (nabla_b, nabla_w)
    
    def evaluate(self , test_data):
        # Evaluar la red neuronal utilizando un conjunto de datos de prueba
        test_data = list(test_data)
        test_data_inputs = [x for x, _ in test_data]
        test_data_targets = [y for _, y in test_data]

        results = [np.argmax(subarray) for subarray in self.feedforward(test_data_inputs)]
        return sum(int(x == y) for (x, y) in zip(results, test_data_targets))
    
    def activation_function(self, x, f):
        if f == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif f == "tanh":
            return np.tanh(x)
        elif f == "relu":
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unknown activation function: {f}")

    def derivative_function(self, x, f):
        if f == "sigmoid":
            sig = self.activation_function(x, f)
            return sig * (1 - sig)
        elif f == "tanh":
            return 1 - np.tanh(x)**2
        elif f == "relu":
            return np.where(x > 0, 1, 0)
        else:
            raise ValueError(f"Unknown activation function: {f}")

    def cost_derivative(self, y_pred, z, y_true, cost, activation_function):
        if cost == "cross_entropy":
            return y_pred - y_true
        elif cost == "square":
            return (y_pred-y_true) * self.derivative_function(z,activation_function)
        elif cost == "absolute":
            return np.sign(y_true - y_pred)
        else:
            raise ValueError(f"Unknown cost function: {cost}")
