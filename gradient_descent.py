def predict(x, m, b):
    y_pred = []
    for xi in x:
        y_pred.append(m * xi + b)
    return y_pred


def compute_loss(y_true, y_pred):
    n = len(y_true)
    total_error = 0
    
    for i in range(n):
        error = y_true[i] - y_pred[i]
        total_error += error ** 2
    
    return total_error / n


def gradient_step(x, y, m, b, learning_rate):
    n = len(x)
    
    m_gradient = 0
    b_gradient = 0
    
    for i in range(n):
        y_pred = m * x[i] + b
        error = y[i] - y_pred
        
        m_gradient += -(2 / n) * x[i] * error
        b_gradient += -(2 / n) * error
    
    m = m - learning_rate * m_gradient
    b = b - learning_rate * b_gradient
    
    return m, b


def gradient_descent(x, y, learning_rate, epochs):
    m = 0
    b = 0
    
    for epoch in range(epochs):
        y_pred = predict(x, m, b)
        loss = compute_loss(y, y_pred)
        
        m, b = gradient_step(x, y, m, b, learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, m = {m:.4f}, b = {b:.4f}")
    
    return m, b


x = [1, 2, 3, 4, 5]
y = [5, 7, 9, 11, 13]

m, b = gradient_descent(x, y, learning_rate=0.01, epochs=1000)

print("\nFinal values:")
print("m =", m)
print("b =", b)