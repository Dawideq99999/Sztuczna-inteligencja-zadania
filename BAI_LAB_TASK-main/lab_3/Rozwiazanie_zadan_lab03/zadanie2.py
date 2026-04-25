import numpy as np
import matplotlib.pyplot as plt

# =========================
# Funkcje aktywacji
# =========================

def sigmoid(x, beta):
    return 1.0 / (1.0 + np.exp(-beta * x))

def tanh(x, beta):
    return np.tanh(beta * x)

def sigmoid_diff(y, beta):
    return beta * y * (1 - y)

def tanh_diff(y, beta):
    return beta * (1 - y * y)


# =========================
# Forward pass (Zadanie 1)
# =========================

def mlp(x, w1, w2, beta):
    # Warstwa ukryta
    z_hidden = w1 @ x
    v_hidden = tanh(z_hidden, beta)

    # Dodanie biasu
    v = np.concatenate(([1], v_hidden))

    # Warstwa wyjściowa
    z_output = w2 @ v
    y = sigmoid(z_output, beta)

    return y, v, v_hidden


# =========================
# Dane XOR
# =========================

# Bias + sygnały bipolarne zamiast 0/1
xx = np.array([
    [1, -1, -1],
    [1, -1,  1],
    [1,  1, -1],
    [1,  1,  1]
])

d = np.array([0, 1, 1, 0])


# =========================
# Wariant 1:
# aktualizacja po każdej próbce
# =========================

def train_sample(xx, d, eta, beta):
    np.random.seed(0)

    # 2 neurony ukryte, 2 wejścia + bias
    w1 = np.random.randn(2, 3) * 0.5

    # 1 neuron wyjściowy, 2 ukryte + bias
    w2 = np.random.randn(1, 3) * 0.5

    errors = []

    for epoch in range(100000):
        mse = 0
        class_error = 0

        for i in range(len(xx)):
            x = xx[i]
            target = d[i]

            # Forward pass
            y, v, v_hidden = mlp(x, w1, w2, beta)
            y = y[0]

            # Błąd
            e = target - y
            mse += e ** 2

            # Klasyfikacja
            predicted = 1 if y > 0.9 else 0 if y < 0.1 else -1
            if predicted != target:
                class_error += 1

            # =================
            # Backpropagation
            # =================

            # Delta wyjścia
            delta_out = e * sigmoid_diff(y, beta)

            # Gradient dla W2
            grad_w2 = delta_out * v

            # Delta warstwy ukrytej
            delta_hidden = (
                delta_out
                * w2[0, 1:]   # bez biasu
                * tanh_diff(v_hidden, beta)
            )

            # Gradient dla W1
            grad_w1 = np.outer(delta_hidden, x)

            # Aktualizacja wag
            w2 += eta * grad_w2.reshape(1, -1)
            w1 += eta * grad_w1

        mse /= len(xx)
        errors.append(mse)

        # Stop jeśli brak błędu klasyfikacji
        if class_error == 0:
            print(f"train_sample zakończono po {epoch+1} epokach")
            break

    return errors


# =========================
# Wariant 2:
# aktualizacja po epoce
# =========================

def train_epoch(xx, d, eta, beta):
    np.random.seed(0)

    w1 = np.random.randn(2, 3) * 0.5
    w2 = np.random.randn(1, 3) * 0.5

    errors = []

    for epoch in range(100000):
        mse = 0
        class_error = 0

        # Akumulacja gradientów
        sum_grad_w1 = np.zeros_like(w1)
        sum_grad_w2 = np.zeros_like(w2)

        for i in range(len(xx)):
            x = xx[i]
            target = d[i]

            # Forward pass
            y, v, v_hidden = mlp(x, w1, w2, beta)
            y = y[0]

            # Błąd
            e = target - y
            mse += e ** 2

            # Klasyfikacja
            predicted = 1 if y > 0.9 else 0 if y < 0.1 else -1
            if predicted != target:
                class_error += 1

            # Delta wyjścia
            delta_out = e * sigmoid_diff(y, beta)

            # Gradient W2
            grad_w2 = delta_out * v

            # Delta hidden
            delta_hidden = (
                delta_out
                * w2[0, 1:]
                * tanh_diff(v_hidden, beta)
            )

            # Gradient W1
            grad_w1 = np.outer(delta_hidden, x)

            # Akumulacja
            sum_grad_w2 += grad_w2.reshape(1, -1)
            sum_grad_w1 += grad_w1

        # Aktualizacja dopiero po całej epoce
        w2 += eta * sum_grad_w2
        w1 += eta * sum_grad_w1

        mse /= len(xx)
        errors.append(mse)

        if class_error == 0:
            print(f"train_epoch zakończono po {epoch+1} epokach")
            break

    return errors


# =========================
# Trening + wykres
# =========================

errors_sample = train_sample(xx, d, eta=0.5, beta=1.0)
errors_epoch = train_epoch(xx, d, eta=0.5, beta=1.0)

plt.figure(figsize=(10, 5))

plt.plot(errors_sample, label='Po każdej próbce', alpha=0.8)
plt.plot(errors_epoch, label='Po epoce', alpha=0.8)

plt.xlabel("Epoka")
plt.ylabel("Błąd MSE")
plt.title("Porównanie dwóch wariantów aktualizacji wag")
plt.legend()
plt.yscale("log")
plt.grid(alpha=0.3)

plt.show()