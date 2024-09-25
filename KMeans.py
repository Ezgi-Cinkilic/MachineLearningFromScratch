import math
import random
from typing import List


class KMeansClusterClassifier:
    centerL = []
    labelL = []
    data_list=[];
    
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.centroids = []

    def fit(self, X: List[List[float]], y: List[int] = None):
        tol = 1e-10 #centroidler degisimi bu degerden az olursa iterasyon durur.
        max_iterations = 1000  
        
        random.seed(42)
        self.centroids = random.sample(X, self.n_clusters) # random centroid belirlenmesi

        for iteration in range(max_iterations):
            clusters = [[] for _ in range(self.n_clusters)]

            for sample in X: #samplelar en yakin centroide eklenir
                distances = [self.euclidean_distance(sample, centroid) for centroid in self.centroids]
                closest_centroid_index = distances.index(min(distances))
                clusters[closest_centroid_index].append(sample)

           
            new_centroids = []
            for cluster in clusters:#yeni centroidleri hesaplar
                if cluster:
                    new_centroids.append(self.calculate_mean(cluster))
                else:
                    new_centroids.append(random.choice(X))

          
            if self.has_converged(self.centroids, new_centroids, tol): #centroid tol dan az degismisse fit islemi tamamlanır
                break

            self.centroids = new_centroids

        else:
            print(f"iterations: {max_iterations}")

    def predict(self, X: List[List[float]]) -> List[int]:
        predictions = []
        for sample in X: # samplelar en yakin centroide sınıfına göre etiketlenir
            distances = [self.euclidean_distance(sample, centroid) for centroid in self.centroids]
            closest_centroid_index = distances.index(min(distances))
            predictions.append(closest_centroid_index)
        return predictions

    @staticmethod
    def euclidean_distance(point1: List[float], point2: List[float]) -> float: # distance hesaplama
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

    @staticmethod
    def calculate_mean(cluster: List[List[float]]) -> List[float]: # centroidleri hesaplar
        n = len(cluster)
        mean = [sum(features) / n for features in zip(*cluster)]
        return mean

    @staticmethod
    def has_converged(centroids: List[List[float]], new_centroids: List[List[float]], tol: float) -> bool: #eski-yeni merkezler arasindaki mesafeyi hesaplar, tol dan buyuk mu check eder.
        total_movement = sum(
            math.sqrt(sum((x - y) ** 2 for x, y in zip(old, new)))
            for old, new in zip(centroids, new_centroids)
        )
        return total_movement < tol


# if __name__ == '__main__':  
#     col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
#     data = pd.read_csv("iris.csv", skiprows=1, header=None, names=col_names)
#     SpeciesToint = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
#     data['type'] = data['type'].replace(SpeciesToint)
#     X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values.tolist()
#     y = data['type'].values.tolist()

#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle= True)

#     classifier = KMeansClusterClassifier(3)
#     classifier.fit(X_train, y_train)

#     y_pred = classifier.predict(X_test)
#     print(y_test)
#     print(y_pred)
#     from sklearn.metrics import accuracy_score
#     accuracy = accuracy_score(y_test, y_pred)
#     print("Accuracy:", accuracy)
 




















