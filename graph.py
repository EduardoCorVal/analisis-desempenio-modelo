"""
Archivo de graficación

Aquí se ejecutan las funciones de graficación.
"""

__author__ = "Eduardo Joel Cortez Valente"

import matplotlib.pyplot as plt
import seaborn as sns

# Graficación de los resultados
# labels = ['Validación', 'Prueba']
# accuracy_scores = [score_validation, score]
# recall_scores = [recall_validation, recall_test]

def graph_dataset_digits(data):
    images = data.images
    labels = data.target

    # Visualizar algunas muestras del dataset
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Dígito {labels[i]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
def accuarcy_graph(accuracies: list):
    labels = ['Entrenamiento', 'Validación', 'Prueba']
    colors = ['blue', 'green', 'orange']

    # Crear el gráfico de barras apiladas
    plt.bar(labels, accuracies, color=colors)
    plt.xlabel('Conjunto de Datos')
    plt.ylabel('Precisión')
    plt.title('Precisión en los Conjuntos de Datos')
    plt.ylim(0, 1)
    plt.show()

def confusion_matrix_graph(cm_train, cm_validation, cm_test):
    # Sumar las matrices para obtener un recuento total de predicciones correctas e incorrectas
    total_confusion_matrix = cm_train + cm_validation + cm_test

    # Crear el gráfico de barras apiladas para las matrices de confusión
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Gráfico para el conjunto de entrenamiento
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0])
    axes[0].set_title('Matriz de Confusión (Entrenamiento)')
    axes[0].set_xlabel('Predicciones')
    axes[0].set_ylabel('Valores Reales')

    # Gráfico para el conjunto de validación
    sns.heatmap(cm_validation, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1])
    axes[1].set_title('Matriz de Confusión (Validación)')
    axes[1].set_xlabel('Predicciones')
    axes[1].set_ylabel('Valores Reales')

    # Gráfico para el conjunto de prueba
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=axes[2])
    axes[2].set_title('Matriz de Confusión (Prueba)')
    axes[2].set_xlabel('Predicciones')
    axes[2].set_ylabel('Valores Reales')

    # Gráfico para el recuento total
    sns.heatmap(total_confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[3])
    axes[3].set_title('Matriz de Confusión (Total)')
    axes[3].set_xlabel('Predicciones')
    axes[3].set_ylabel('Valores Reales')

    plt.tight_layout()
    plt.show()

def recall_graph(recalls: list):
    labels = ['Entrenamiento', 'Validación', 'Prueba']
    colors = ['blue', 'green', 'orange']

    # Crear el gráfico de barras apiladas para los valores de recall
    plt.bar(labels, recalls, color=colors)
    plt.xlabel('Conjunto de Datos')
    plt.ylabel('Recall Ponderado')
    plt.title('Recall en los Conjuntos de Datos')
    plt.ylim(0, 1)  # Establecer el rango del eje y de 0 a 1 para el recall
    plt.show()