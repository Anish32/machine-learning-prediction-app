import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import plotly.graph_objects as go

# Page Setup
st.set_page_config(page_title="📊 Machine Prediction App", layout="wide")
st.title("📊 Machine Prediction App")

# Sidebar Config
st.sidebar.header("⚙️ Configuration")
task = st.sidebar.selectbox("Select Task Type", ["Classification", "Regression"])
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])

# Sample fallback data
def load_sample_data():
    from sklearn.datasets import load_iris, make_regression
    if task == "Classification":
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
    else:
        X, y = make_regression(n_samples=500, n_features=4, noise=0.5, random_state=42)
        X = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(4)])
        y = pd.Series(y, name="target")
    return X, y

# Load data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Uploaded Dataset Preview")
    st.dataframe(df.head())

    # User selects the target
    target_column = st.sidebar.selectbox("Select Target Column", df.columns)
    X_raw = df.drop(columns=[target_column])
    y_raw = df[target_column]

    # Encode features
    X = pd.get_dummies(X_raw)

    # Encode target if classification and it's object
    if task == "Classification" and y_raw.dtype == "object":
        y = y_raw.astype("category").cat.codes
    else:
        y = y_raw
else:
    st.sidebar.info("ℹ️ Upload a CSV or use sample dataset.")
    X, y = load_sample_data()
    st.subheader("📄 Sample Dataset Preview")
    st.dataframe(pd.concat([X, y], axis=1).head())

# Models
classification_models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier()
}
regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "SVR": SVR()
}

model_options = classification_models if task == "Classification" else regression_models
model_name = st.sidebar.selectbox("Choose Algorithm", list(model_options.keys()))
model = model_options[model_name]
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

# Train button
if st.sidebar.button("🚀 Train and Predict"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Output metrics
    st.subheader("📈 Model Evaluation")
    if task == "Classification":
        acc = accuracy_score(y_test, predictions)
        st.metric("Accuracy", f"{acc:.3f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, predictions))
    else:
        mse = mean_squared_error(y_test, predictions)
        st.metric("Mean Squared Error", f"{mse:.3f}")

    # Plot actual vs predicted
    st.subheader("📉 Actual vs Predicted")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=predictions, mode='markers', name="Predicted"))
    fig.add_trace(go.Scatter(y=y_test.values, mode='lines+markers', name="Actual", line=dict(color='red')))
    fig.update_layout(xaxis_title="Sample Index", yaxis_title="Target", height=400)
    st.plotly_chart(fig, use_container_width=True)
