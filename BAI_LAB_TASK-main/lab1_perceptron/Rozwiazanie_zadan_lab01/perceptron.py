import numpy as np

class Perceptron:
    def __init__(self, n, bias=True):
        # Inicjalizacja wag
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
if __name__ == "__main__":
    import numpy as np
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])
    p = Perceptron(n=2)
    p.train(X, y, eta=0.1, tol=0.0)
    err, preds = p.evaluate_test(X, y)
    print("Predykcje:", preds)
    print("Skuteczność:", (1-err)*100, "%")