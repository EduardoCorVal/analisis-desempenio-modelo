"""
Archivo principal

Aquí se ejecuta el programa principal.

Entregable: Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución.
"""

__author__ = "Eduardo Joel Cortez Valente"

import graph
import evaluation
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = load_digits()
X = data.data
y = data.target

# Graficación de los datos
graph.PCA_graph(X, y)

X_train_initial, X_test, Y_train_initial, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train_initial, Y_train_initial, test_size=0.2, random_state=0)

# Graficación de las muestras del dataset
graph.graph_dataset_digits(data)

# Imprimir algunas muestras de los conjuntos de entrenamiento, validación y prueba
print("Muestras de Conjunto de Entrenamiento:")
print(X_train[:5])
print(Y_train[:5])
print("\nMuestras de Conjunto de Validación:")
print(X_validation[:5])
print(Y_validation[:5])
print("\nMuestras de Conjunto de Prueba:")
print(X_test[:5])
print(Y_test[:5])

# Modelo de clasificación
classifier = DecisionTreeClassifier(max_depth=5, min_samples_split=2)
classifier.fit(X_train, Y_train)

Y_train_pred = classifier.predict(X_train)
Y_validation_pred = classifier.predict(X_validation)
Y_test_pred = classifier.predict(X_test)

# Evaluación de los modelos de clasificación
evaluation.y_evaluation(Y_train_pred, Y_validation_pred, Y_test_pred, Y_train, Y_validation, Y_test)

