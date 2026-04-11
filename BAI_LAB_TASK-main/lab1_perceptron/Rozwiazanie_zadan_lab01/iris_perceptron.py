import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Ładowanie danych Iris
data = load_iris()
X = data.data[:100, 2:4]  # dwie cechy: długość i szerokość płatka
y = data.target[:100]     # 0 i 1 (Setosa i Versicolor)

# Podział na trening i test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Klasa Perceptron
class Perceptron:
    def __init__(self, n, bias=True):
        self.w = np.random.randn(n) * 0.01
        self.b = np.random.randn() if bias else 0

    def predict(self, x):
        s = np.dot(x, self.w) + self.b
        return 1 if s >= 0 else 0

    def train(self, xx, d, eta=0.1, tol=0.0, max_epochs=1000):
        for epoch in range(max_epochs):
            errors = 0
            for x, target in zip(xx, d):
                y = self.predict(x)
                e = target - y
                if e != 0:
                    errors += 1
                    self.w += eta * e * x
                    self.b += eta * e
            error_rate = errors / len(xx)
            print(f"Epoka {epoch+1}, błąd: {error_rate}")
            if error_rate <= tol:
                print("Zatrzymano wcześniej (osiągnięto tolerancję błędu)")
                break

    def evaluate_test(self, xx, d):
        preds = np.array([self.predict(x) for x in xx])
        error_rate = np.mean(preds != d)
        return error_rate, preds

# Trening Perceptronu
p = Perceptron(n=2)
p.train(X_train, y_train, eta=0.1, tol=0.0, max_epochs=100)

# Ocena skuteczności
err, preds = p.evaluate_test(X_test, y_test)
print("\nPredykcje na zbiorze testowym:", preds)
print("Skuteczność:", (1-err)*100, "%")

# Wizualizacja granicy decyzyjnej
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = np.array([p.predict(np.array([i, j])) for i, j in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label='Trening')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='s', label='Test')
plt.xlabel('Długość płatka')
plt.ylabel('Szerokość płatka')
plt.title('Perceptron na Iris (Setosa vs Versicolor)')
plt.legend()
plt.show()