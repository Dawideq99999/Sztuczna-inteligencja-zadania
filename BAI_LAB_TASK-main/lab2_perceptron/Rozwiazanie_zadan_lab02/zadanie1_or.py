# plik: zadanie1_or.py

import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# ========================================
# Dane dla bramki OR
# ========================================

# Wejścia logiczne
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Wyjścia dla bramki OR:
# 0 OR 0 = 0
# 0 OR 1 = 1
# 1 OR 0 = 1
# 1 OR 1 = 1
y_or = np.array([0, 1, 1, 1])

# ========================================
# Konwersja klas:
# perceptron używa klas {-1, 1}
# więc zamieniamy 0 -> -1
# ========================================

y_or = np.where(y_or == 0, -1, 1)

# ========================================
# Klasa Perceptron
# (jeśli masz już klasę z wcześniejszej sekcji,
# możesz usunąć tę część)
# ========================================

class Perceptron:
    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0

            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update

                if update != 0.0:
                    errors += 1

            self.errors_.append(errors)

        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# ========================================
# Trenowanie perceptronu
# ========================================

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y_or)

# ========================================
# Wykres błędów w epokach
# ========================================

plt.plot(
    range(1, len(ppn.errors_) + 1),
    ppn.errors_,
    marker='o'
)

plt.xlabel("Epoka")
plt.ylabel("Liczba błędnych klasyfikacji")
plt.title("Perceptron — bramka OR")
plt.show()

# ========================================
# Wizualizacja regionów decyzyjnych
# ========================================

plot_decision_regions(X, y_or, clf=ppn)

plt.title("Regiony decyzyjne — bramka OR")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
