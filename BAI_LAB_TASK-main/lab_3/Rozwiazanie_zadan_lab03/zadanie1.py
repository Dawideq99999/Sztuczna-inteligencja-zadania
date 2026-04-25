import numpy as np

def sigmoid(x, beta):
    return 1.0 / (1.0 + np.exp(-beta * x))

def tanh(x, beta):
    return np.tanh(beta * x)

# x - sygnal wejsciowy [1, x1, x2, ..., xN]
# w1 - wagi warstwy ukrytej, macierz (K x N+1)
# w2 - wagi warstwy wyjsciowej, wektor (1 x K+1)
# beta - parametr funkcji aktywacji
def mlp(x, w1, w2, beta):
    # 1. Oblicz wyjścia warstwy ukrytej
    #    z_hidden = w1 @ x
    #    v_hidden = tanh(z_hidden)
    z_hidden = w1 @ x
    v_hidden = tanh(z_hidden, beta)

    # 2. Dodaj bias do warstwy ukrytej
    #    v = [1, v1, v2, ...]
    v = np.concatenate(([1], v_hidden))

    # 3. Oblicz wyjście sieci
    #    z_output = w2 @ v
    #    y = sigmoid(z_output)
    z_output = w2 @ v
    y = sigmoid(z_output, beta)

    # 4. Zwróć:
    #    y - odpowiedź sieci
    #    v - wyjścia warstwy ukrytej z biasem
    #    v_hidden - same wyjścia warstwy ukrytej (bez biasu)
    return y, v, v_hidden


# Test: losowe wagi, wejście [1, 0, 1]
np.random.seed(0)

w1_test = np.random.randn(2, 3) * 0.5
w2_test = np.random.randn(1, 3) * 0.5

result = mlp(
    np.array([1, 0, 1]),
    w1_test,
    w2_test,
    beta=1.0
)

print(f"Wyjście sieci: {result[0]}")
print(f"Warstwa ukryta z biasem: {result[1]}")
print(f"Warstwa ukryta bez biasu: {result[2]}")