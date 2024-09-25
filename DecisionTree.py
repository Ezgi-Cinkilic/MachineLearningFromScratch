from typing import List 

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value  # 0 1 2

class DecisionTreeClassifier():
    
    def __init__(self, max_depth: int):
        self.root = None
        self.max_depth = max_depth

    def fit(self, X: List[List[float]], y: List[int]):
        self.num_class= len(set(y))
        dataset = self.create_dataset(X, y)
        self.root = self.build_tree(dataset)

    def build_tree(self, dataset: List[List[float]], curr_depth=0):
        num_samples = len(dataset)
        num_features = len(dataset[0]) - 1

        if curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split and best_split["info_gain"] > 0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["info_gain"])

        leaf_value = self.calculate_leaf_value([row[-1] for row in dataset])
        return Node(value=leaf_value)

    def get_best_split(self, dataset: List[List[float]], num_samples: int, num_features: int):
        best_split = {"info_gain": -float("inf")}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_values = self.get_feature_values(dataset, feature_index)
            possible_thresholds = self.possible_thresholds(feature_values)

            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y = [row[-1] for row in dataset]
                    left_y = [row[-1] for row in dataset_left]
                    right_y = [row[-1] for row in dataset_right]
                    curr_info_gain = self.gini_calculate(y, left_y, right_y)
                    if curr_info_gain > max_info_gain:
                        best_split = {
                            "feature_index": feature_index,
                            "threshold": threshold,
                            "dataset_left": dataset_left,
                            "dataset_right": dataset_right,
                            "info_gain": curr_info_gain
                        }
                        max_info_gain = curr_info_gain

        return best_split if max_info_gain != -float("inf") else None

    def split(self, dataset: List[List[float]], feature_index: int, threshold: float):
        dataset_left = [row for row in dataset if row[feature_index] <= threshold]
        dataset_right = [row for row in dataset if row[feature_index] > threshold]
        return dataset_left, dataset_right
    
    def possible_thresholds(self, X):
        return sorted(set(X))
    
    def get_feature_values(self, dataset: List[List[float]], feature_index: int):
        return [row[feature_index] for row in dataset]
        
    def gini_calculate(self, y: List[int], l_y: List[int], r_y: List[int]):
        weight_l = len(l_y) / len(y)
        weight_r = len(r_y) / len(y)
        gain = self.gini_index(y) - (weight_l * self.gini_index(l_y) + weight_r * self.gini_index(r_y))
        return gain

    def gini_index(self, y: List[int]):
        class_labels = set(y)
        gini = 0
        for cls in class_labels:
            class_count = sum(1 for label in y if label == cls)
            probability = class_count / len(y)
            gini += probability ** 2
        return 1 - gini

    def calculate_leaf_value(self, y: List[int]):
        return max(y, key=y.count)

    def create_dataset(self, X: List[List[float]], y: List[int]):
        return [x + [y[i]] for i, x in enumerate(X)]

    def predict(self, X: List[List[float]]):
        return [self.make_prediction(x, self.root) for x in X]

    def make_prediction(self, X: List[float], tree: Node):
        if tree.value is not None:
            return tree.value
        feature_val = X[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(X, tree.left)
        else:
            return self.make_prediction(X, tree.right)
    
    #ROC Curve olusturmak icin ekledim. Sadece 1 ve 0 ile olasilik hesapliyor.
    def predict_proba(self, X: List[List[float]]):
        return [self.make_predict_proba(x, self.root) for x in X]

    def make_predict_proba(self,  X: List[float], tree):
        if tree.value is not None:
            return [1.0 if i == tree.value else 0.0 for i in range(self.num_class)]
            
        if X[tree.feature_index] <= tree.threshold:
            return self.make_predict_proba(X, tree.left)
        else:
            return self.make_predict_proba(X, tree.right)

# if __name__ == '__main__':
#     col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
#     data = pd.read_csv("iris.csv", skiprows=1, header=None, names=col_names)
#     SpeciesToint = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
#     data['type'] = data['type'].replace(SpeciesToint)
#     X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values.tolist()
#     y = data['type'].values.tolist()

#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

#     classifier = DecisionTreeClassifier(max_depth=4)
#     classifier.fit(X_train, y_train)

#     y_pred = classifier.predict(X_test)
#     y_prob = classifier.predict_proba(X_test)
#     print(y_prob)
#     from sklearn.metrics import accuracy_score
#     accuracy = accuracy_score(y_test, y_pred)
#     print("Accuracy:", accuracy)
