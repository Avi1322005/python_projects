import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([5, 7, 9, 11, 13], dtype=float)

learning_rate = 0.01
epochs = 100


def batch_gd(X, y):
    w, b = 0, 0
    n = len(X)
    loss_history = []

    for epoch in range(epochs):
        y_pred = w * X + b

        dw = (-2/n) * np.sum(X * (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        loss = np.mean((y - y_pred) ** 2)
        loss_history.append(loss)

    return w, b, loss_history


def mini_batch_gd(X, y, batch_size=2):
    w, b = 0, 0
    n = len(X)
    loss_history = []

    for epoch in range(epochs):
        for i in range(0, n, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            y_pred = w * X_batch + b

            m = len(X_batch)

            dw = (-2/m) * np.sum(X_batch * (y_batch - y_pred))
            db = (-2/m) * np.sum(y_batch - y_pred)

            w = w - learning_rate * dw
            b = b - learning_rate * db

        full_pred = w * X + b
        loss = np.mean((y - full_pred) ** 2)
        loss_history.append(loss)

    return w, b, loss_history


def sgd(X, y):
    w, b = 0, 0
    n = len(X)
    loss_history = []

    for epoch in range(epochs):
        for i in range(n):
            xi = X[i]
            yi = y[i]

            y_pred = w * xi + b

            dw = -2 * xi * (yi - y_pred)
            db = -2 * (yi - y_pred)

            w = w - learning_rate * dw
            b = b - learning_rate * db

        full_pred = w * X + b
        loss = np.mean((y - full_pred) ** 2)
        loss_history.append(loss)

    return w, b, loss_history


w1, b1, loss1 = batch_gd(X, y)
w2, b2, loss2 = mini_batch_gd(X, y)
w3, b3, loss3 = sgd(X, y)

print("Batch GD     :", w1, b1)
print("Mini Batch GD:", w2, b2)
print("SGD          :", w3, b3)

plt.plot(loss1, label="Batch GD")
plt.plot(loss2, label="Mini Batch GD")
plt.plot(loss3, label="SGD")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Gradient Descent Comparison")
plt.legend()
plt.show()
