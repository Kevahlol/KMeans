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
    
    #Conjuntoi para entrenamiento, ingresos y gastos en MXN, y nivel educativo. 0 = Primaria, 1 = Secundaria, 2 = Preparatoria, 3 = Universidad
    X_train = np.array([
        [15000, 12000, 1], [50000, 40000, 3], [33000, 32000, 2], [15000, 12000, 0], [70000, 45000, 3],
        [28000, 20000, 2], [85000, 60000, 3], [19000, 15000, 1], [7000, 6000, 0], [90000, 70000, 3],
        [40000, 33000, 2], [75000, 50000, 3], [35000, 20000, 2], [100000, 80000, 3], [26000, 26000, 1], 
        [39000, 35000, 2], [20000, 18000, 1], [85000, 55000, 3], [75000, 55000, 3], [9000, 8000, 0],
        [11000, 8000, 0], [15000, 15000, 0], [17000, 17000, 1], [33000, 24000, 2], [36000, 32000, 2],
        [20000, 16000, 1], [8000, 8000, 0], [90000, 60000, 3], [31000, 28000, 2], [24000, 23000, 1]
    ])

    #
    kmeans_model = KMeansClustering(n_clusters=3)

    #Entrenamiento del modelo
    kmeans_model.fit(X_train)

    #Clasificación de las instancias en los clústers
    predictions = kmeans_model.predict(X_train)

    #Muestreo de resultados por instancias
    print("Predicciones de clúster para el conjunto de entrenamiento:")
    for i, pred in enumerate(predictions):
        print(f"Instancia {i + 1}: Clúster {pred}")

    #Muestreo de resultados por clúster
    for cluster in range(kmeans_model.n_clusters):
        instances_in_cluster = X_train[predictions == cluster]
        print(f"\nInstancias en el Cluster {cluster}:\n{instances_in_cluster}")
