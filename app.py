import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === Linear Regression Closed-form ===
def train_linear_regression(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta_best

def predict(X_new, theta):
    X_new_b = np.c_[np.ones((X_new.shape[0], 1)), X_new]
    return X_new_b @ theta

# === Session State Initialization ===
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'theta' not in st.session_state:
    st.session_state.theta = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'predicted_value' not in st.session_state:
    st.session_state.predicted_value = None
if 'prediction_input' not in st.session_state:
    st.session_state.prediction_input = None

st.set_page_config(page_title="Linear Regression Trainer", layout="centered")
st.title("📈 Train Linear Regression & Predict")

# === Step 1: Define Features ===
st.header("1️⃣ Define Features and Input Training Data")
num_features = st.number_input("How many features?", min_value=1, max_value=5, value=1)
feature_names = []

for i in range(num_features):
    feature_names.append(st.text_input(f"Feature {i+1} name", value=f"x{i+1}"))

num_samples = st.number_input("How many training samples?", min_value=2, max_value=100, value=5)

X_vals = []
y_vals = []

st.subheader("✍️ Enter your training data:")
for i in range(int(num_samples)):
    row = []
    st.markdown(f"**Sample {i+1}**")
    cols = st.columns(num_features + 1)
    for j in range(num_features):
        val = cols[j].number_input(f"{feature_names[j]} (Sample {i+1})", key=f"{j}_{i}")
        row.append(val)
    target = cols[-1].number_input("Target (y)", key=f"y_{i}")
    X_vals.append(row)
    y_vals.append(target)

# === Step 2: Train Model ===
if st.button("🔧 Train Model"):
    X = np.array(X_vals)
    y = np.array(y_vals)
    theta = train_linear_regression(X, y)

    st.session_state.model_trained = True
    st.session_state.theta = theta
    st.session_state.feature_names = feature_names
    st.session_state.X_train = X
    st.session_state.y_train = y
    st.session_state.predicted_value = None
    st.session_state.prediction_input = None

    # Display equation
    equation = f"y = {theta[0]:.4f}"
    for i, f in enumerate(feature_names):
        equation += f" + ({theta[i+1]:.4f} * {f})"
    st.success("✅ Model Trained!")
    st.code(equation)

# === Step 3: Always Show Plot if Model Exists ===
if st.session_state.model_trained:
    st.subheader("📊 Regression Plot")

    X = st.session_state.X_train
    y = st.session_state.y_train
    theta = st.session_state.theta
    feature_names = st.session_state.feature_names

    if len(feature_names) == 1:
        x = X[:, 0]
        y_pred_line = theta[0] + theta[1] * x

        fig, ax = plt.subplots()
        ax.scatter(x, y, color='blue', label='Training Data')
        ax.plot(x, y_pred_line, color='red', label='Regression Line')

        # Plot predicted point if exists
        if st.session_state.predicted_value is not None:
            pred_x = st.session_state.prediction_input[0]
            pred_y = st.session_state.predicted_value
            ax.scatter(pred_x, pred_y, color='green', s=100, marker='X', label='Prediction')

        ax.set_xlabel(feature_names[0])
        ax.set_ylabel("Target")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    else:
        df_plot = pd.DataFrame(X, columns=feature_names)
        df_plot["Target"] = y
        st.write("🔍 Training Data Overview")
        st.dataframe(df_plot)

# === Step 4: Prediction ===
if st.session_state.model_trained:
    st.header("🔮 Predict Target Value")
    input_vals = []
    for f in st.session_state.feature_names:
        val = st.number_input(f"Enter value for {f}:", key=f"predict_{f}")
        input_vals.append(val)

    if st.button("📌 Predict"):
        prediction = predict(np.array([input_vals]), st.session_state.theta)
        st.success(f"📍 Predicted Target: **{prediction[0]:.4f}**")
        st.session_state.predicted_value = prediction[0]
        st.session_state.prediction_input = input_vals
# === Step 5: Reset Button ===
st.divider()
if st.button("🔄 Reset Model"):
    st.session_state.model_trained = False
    st.session_state.theta = None
    st.session_state.feature_names = []
    st.session_state.X_train = None
    st.session_state.y_train = None
    st.session_state.predicted_value = None
    st.session_state.prediction_input = None
    st.experimental_rerun()
