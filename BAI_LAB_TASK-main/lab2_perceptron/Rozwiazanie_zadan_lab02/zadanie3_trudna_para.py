
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ========================================
# Klasa Perceptron
# ========================================

class Perceptron:
    def __init__(self, eta=0.01, n_iter=20):
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
# Wczytanie danych MNIST
# ========================================

train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

print("Train dataset:", train_df.shape)
print("Test dataset:", test_df.shape)


# ========================================
# Zadanie 3 — trudna para cyfr
# przykład: 1 vs 7
# ========================================

DIGIT_A = 1
DIGIT_B = 7

print(f"\nKlasyfikacja cyfr: {DIGIT_A} vs {DIGIT_B}")


# ========================================
# 1. Filtracja danych
# ========================================

train_binary = train_df[
    (train_df["label"] == DIGIT_A) |
    (train_df["label"] == DIGIT_B)
].copy()

test_binary = test_df[
    (test_df["label"] == DIGIT_A) |
    (test_df["label"] == DIGIT_B)
].copy()

print("Train shape:", train_binary.shape)
print("Test shape:", test_binary.shape)


# ========================================
# 2. Binaryzacja etykiet
# DIGIT_A -> -1
# DIGIT_B -> +1
# ========================================

y_train = np.where(
    train_binary["label"] == DIGIT_A,
    -1,
    1
)

y_test = np.where(
    test_binary["label"] == DIGIT_A,
    -1,
    1
)


# ========================================
# 3. Normalizacja /255
# ========================================

X_train = train_binary.drop("label", axis=1).values / 255.0
X_test = test_binary.drop("label", axis=1).values / 255.0

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# ========================================
# 4. Trening perceptronu
# ========================================

ppn = Perceptron(eta=0.01, n_iter=20)
ppn.fit(X_train, y_train)

print("\nTrening zakończony.")


# ========================================
# 5. Wykres błędów
# ========================================

plt.plot(
    range(1, len(ppn.errors_) + 1),
    ppn.errors_,
    marker="o"
)

plt.xlabel("Epoka")
plt.ylabel("Liczba błędów")
plt.title("Krzywa błędu — trudna para cyfr (1 vs 7)")
plt.show()


# ========================================
# 6. Ewaluacja
# ========================================

y_pred = ppn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1)
rec = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print("\nWyniki:")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")


# ========================================
# Odpowiedzi do analizy
# ========================================

print("\nANALIZA:")

print("1. Jeśli krzywa błędu nie spada do zera,")
print("   oznacza to, że cyfry nie są idealnie")
print("   liniowo separowalne w przestrzeni 784D.")

print("\n2. Accuracy zwykle jest gorsze niż dla 0 vs 1,")
print("   ponieważ 1 i 7 są bardziej podobne wizualnie.")

print("\n3. W confusion matrix można sprawdzić,")
print("   czy dominuje FP czy FN — zależy od danych.")

