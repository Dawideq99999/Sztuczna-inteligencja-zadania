import pandas as pd
from sklearn.model_selection import train_test_split
from perceptron import Perceptron

def run_model(X, y, n_features, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    model = Perceptron(n=n_features)
    model.train(X_train, y_train, eta=0.1, tol=0.0)

    err, preds = model.evaluate_test(X_test, y_test)
    print("\n=== WYNIKI ===")
    print("Predykcje:", preds)
    print(f"Skuteczność: {(1 - err) * 100:.2f}%")
    print("================\n")


# ============ DANE 2D ============
print("==== DANE 2D ====")
df2d = pd.read_csv("data/2D.csv", sep=';')
X2 = df2d[['X1', 'X2']].values
y2 = df2d['L'].astype(int).values
run_model(X2, y2, n_features=2, test_size=0.3)

# ============ DANE 3D ============
print("==== DANE 3D ====")
df3d = pd.read_csv("data/3D.csv", sep=';')
X3 = df3d[['X1', 'X2', 'X3']].values
y3 = df3d['L'].astype(int).values
run_model(X3, y3, n_features=3, test_size=0.3)