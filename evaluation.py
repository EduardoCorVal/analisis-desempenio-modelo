"""
Archivo de evaluación

Aquí se realizan las evaluaciones de los modelos de clasificación.
Adicionalmnente, se grafican para su visualización.
"""

__author__ = "Eduardo Joel Cortez Valente"

import graph
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

def y_evaluation(Y_train_pred, Y_validation_pred, Y_test_pred, Y_train, Y_validation, Y_test): 
    ''''''
    accuracy_train = accuracy_score(Y_train, Y_train_pred)
    accuracy_validation = accuracy_score(Y_validation, Y_validation_pred)
    accuracy_test = accuracy_score(Y_test, Y_test_pred)

    # Graficado de la precisión en los conjuntos de datos
    accuracies = [accuracy_train, accuracy_validation, accuracy_test]
    graph.accuarcy_graph(accuracies)

    confusion_matrix_train = confusion_matrix(Y_train, Y_train_pred)
    confusion_matrix_validation = confusion_matrix(Y_validation, Y_validation_pred)
    confusion_matrix_test = confusion_matrix(Y_test, Y_test_pred)

    # Graficado de las matrices de confusión
    graph.confusion_matrix_graph(confusion_matrix_train, confusion_matrix_validation, confusion_matrix_test)

    recall_train = recall_score(Y_train, Y_train_pred, average='weighted')
    recall_validation = recall_score(Y_validation, Y_validation_pred, average='weighted')
    recall_test = recall_score(Y_test, Y_test_pred, average='weighted')

    # Graficado de la sensibilidad en los conjuntos de datos
    recalls = [recall_train, recall_validation, recall_test]
    graph.recall_graph(recalls)
