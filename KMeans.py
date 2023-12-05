
from sklearn.cluster import KMeans
import numpy as np

class KMeansClustering:

    #Se establecen 3 clústers
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

        #Armado del modelo, se asigna la "semilla" para la asignación de centroides en 42, para poder replicar los resultados
        self.kmeans_model = KMeans(n_clusters=self.n_clusters, random_state=42)

    def fit(self, X):
        self.kmeans_model.fit(X)

    def predict(self, X):
        return self.kmeans_model.predict(X)

if __name__ == "__main__":
    
    #Conjuntoi para entrenamiento
    X_train = np.array([
        [160, 60], [175, 70], [155, 50], [180, 80], [165, 65],
        [150, 45], [185, 90], [170, 75], [140, 40], [195, 100],
        [180, 70], [165, 50], [190, 85], [175, 65], [200, 110],
        [155, 60], [180, 90], [170, 65], [185, 95], [175, 80],
        [125, 25], [130, 30], [110, 22], [140, 18], [135, 27]
    ])

    #
    kmeans_model = KMeansClustering(n_clusters=3)

    #Entrenamiento del modelo
    kmeans_model.fit(X_train)

    #Clasificación de las instancias en los clústers
    predictions = kmeans_model.predict(X_train)

    #Muestreo de resultados
    print("Predicciones de clúster para el conjunto de entrenamiento:")
    for i, pred in enumerate(predictions):
        print(f"Instancia {i + 1}: Clúster {pred}")
