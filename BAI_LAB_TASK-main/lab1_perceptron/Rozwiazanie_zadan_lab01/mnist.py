import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sys

# ===============================
# 1. Wczytanie danych MNIST z konwersją na int
# ===============================
try:
    train_df = pd.read_csv("mnist_train.csv", dtype=int)
    test_df = pd.read_csv("mnist_test.csv", dtype=int)
except FileNotFoundError:
    print("Pliki mnist_train.csv lub mnist_test.csv nie znaleziono w katalogu.")
    sys.exit(1)

# Pierwsza kolumna = etykieta, reszta = piksele
y_train_full = train_df.iloc[:, 0].values
X_train_full = train_df.iloc[:, 1:].values

y_test_full = test_df.iloc[:, 0].values
X_test_full = test_df.iloc[:, 1:].values

# ===============================
# 2. Wybór cyfr do klasyfikacji
# ===============================
digit_a = 3
digit_b = 8

available_train = set(y_train_full)
available_test = set(y_test_full)

if digit_a not in available_train or digit_b not in available_train:
    print(f"Wybrane cyfry ({digit_a}, {digit_b}) nie są w zbiorze treningowym.")
    print("Dostępne cyfry w treningowym:", sorted(available_train))
    sys.exit(1)

if digit_a not in available_test or digit_b not in available_test:
    print(f"Wybrane cyfry ({digit_a}, {digit_b}) nie są w zbiorze testowym.")
    print("Dostępne cyfry w testowym:", sorted(available_test))
    sys.exit(1)

# ===============================
# 3. Filtracja tylko wybranych cyfr
# ===============================
train_filter = np.isin(y_train_full, [digit_a, digit_b])
test_filter = np.isin(y_test_full, [digit_a, digit_b])

X_train = X_train_full[train_filter]
y_train = y_train_full[train_filter]

X_test = X_test_full[test_filter]
y_test = y_test_full[test_filter]

print("Liczba próbek w train:", X_train.shape[0])
print("Liczba próbek w test:", X_test.shape[0])

# ===============================
# 4. Normalizacja pikseli do [0,1]
# ===============================
X_train = X_train / 255.0
X_test = X_test / 255.0

# ===============================
# 5. Zamiana etykiet na 0 i 1
# ===============================
y_train = np.where(y_train == digit_a, 0, 1)
y_test = np.where(y_test == digit_a, 0, 1)

# ===============================
# 6. Tworzenie i trenowanie perceptronu
# ===============================
perceptron = Perceptron(max_iter=1000, eta0=0.01, random_state=42)
perceptron.fit(X_train, y_train)

# ===============================
# 7. Predykcje i ocena dokładności
# ===============================
y_pred = perceptron.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność perceptronu dla cyfr {digit_a} i {digit_b}: {accuracy*100:.2f}%")

# ===============================
# 8. Wizualizacja kilku przykładów
# ===============================
num_to_show = min(15, X_test.shape[0])
fig, axes = plt.subplots(3, 5, figsize=(10,6))
axes = axes.flatten()

for i in range(num_to_show):
    axes[i].imshow(X_test[i].reshape(28,28), cmap='gray')
    axes[i].set_title(f"Prawda: {y_test[i]}, Pred: {y_pred[i]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()