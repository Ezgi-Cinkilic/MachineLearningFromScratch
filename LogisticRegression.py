from typing import List
import random
import math

class LogisticRegression:
    def __init__(self, learning_rate, epochs, batch_size):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def fit(self, X: List[List[float]], y: List[List[int]]):
        n_samples, n_features = len(X), len(X[0])
        n_classes = len(y[0])
        self.weights = [[0.0] * n_features for _ in range(n_classes)]
        self.bias = [0.0] * n_classes
        
        #Egitim dongusu
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for start in range(0, n_samples, self.batch_size): # bu dongu ve end degeri ile batc_sizelik batchler alinir
                end = start + self.batch_size
                batch_X = X[start:end]
                batch_y = y[start:end]
                
                logits = self.forward(batch_X) #tahmin et
                predictions = [softmax(logit) for logit in logits] #tahminleri olasiliksal dağilima cevir
                
                gradients_w, gradient_b = compute_gradients(batch_X, batch_y, predictions) #weight ve bias icin gradyanlari hesapla
                self.weights, self.bias = update_parameters(self.weights, self.bias, gradients_w, gradient_b, self.learning_rate) #weight ve bias guncelle
                
                # Bu kisim egitimi gozlemlemek icin
                batch_loss = sum(cross_entropy_loss(pred, true) for pred, true in zip(predictions, batch_y))
                epoch_loss += batch_loss
            
            # Print the epoch number and loss
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss/n_samples:.4f}")
    
    def predict(self, X: List[List[float]]) -> List[List[float]]:
        logits = self.forward(X)
        return [softmax(logit) for logit in logits]

    def forward(self, X: List[List[float]]) -> List[List[float]]: #1
        return [[self._dot_product(x, self.weights[c]) + self.bias[c] for c in range(len(self.weights))] for x in X]

    def _dot_product(self, x: List[float], w: List[float]) -> float:
        return sum(x_i * w_i for x_i, w_i in zip(x, w))

def softmax(logits: List[float]) -> List[float]: #2
    max_logit = max(logits)
    exp_logits = [math.exp(logit - max_logit) for logit in logits] #Formulde yer almamasina ragmen sayisal kararlilik icin yapilmasi onerildigi icin ekledim.
    sum_exp_logits = sum(exp_logits)
    return [exp_logit / sum_exp_logits for exp_logit in exp_logits]

def cross_entropy_loss(predictions: List[float], targets: List[int]) -> float:  #3
    return -sum(target * math.log(prediction + 1e-15) for target, prediction in zip(targets, predictions)) # Logaritma hesaplama hatasini onlemek icin kucuk bir deger eklenir

def compute_gradients(X: List[List[float]], y: List[List[int]], predictions: List[List[float]]) -> (List[List[float]], List[float]): #4
    n_samples = len(y)
    n_features = len(X[0])
    n_classes = len(y[0])
    
    gradients_w = [[0.0] * n_features for _ in range(n_classes)]
    gradient_b = [0.0] * n_classes
    
    for i in range(n_samples):
        for c in range(n_classes):
            error = predictions[i][c] - y[i][c]
            for j in range(n_features):
                gradients_w[c][j] += error * X[i][j]
            gradient_b[c] += error
    # her sinif icin weight ve bias gradyanlari gunceller
    gradients_w = [[grad / n_samples for grad in class_grad] for class_grad in gradients_w]
    gradient_b = [grad / n_samples for grad in gradient_b]
    # Ortalama gradyan
    return gradients_w, gradient_b

def update_parameters(weights: List[List[float]], bias: List[float], gradients_w: List[List[float]], gradient_b: List[float], learning_rate: float) -> (List[List[float]], List[float]): #5
    # Ogrenme oranini kullanarak weight ve biasi gunceller
    weights = [[w - learning_rate * grad_w for w, grad_w in zip(class_w, class_grad_w)] for class_w, class_grad_w in zip(weights, gradients_w)]
    bias = [b - learning_rate * grad_b for b, grad_b in zip(bias, gradient_b)]
    return weights, bias