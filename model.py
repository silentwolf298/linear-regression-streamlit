import numpy as np

def train_linear_regression(X, y):
    # Add bias column to X
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add x0 = 1
    theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta_best  # theta = [b, w1, w2, ...]

def save_model_equation(theta, feature_names, filepath="model_equation.txt"):
    equation = f"y = {theta[0]:.4f}"
    for i, coef in enumerate(theta[1:]):
        equation += f" + ({coef:.4f} * {feature_names[i]})"
    with open(filepath, "w") as f:
        f.write(equation)
    return equation

def predict(X_new, theta):
    X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]
    return X_new_b @ theta
