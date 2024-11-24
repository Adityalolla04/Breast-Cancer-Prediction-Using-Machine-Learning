import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# Title of the app
st.title("Breast Cancer Prediction App")

# Load the cleaned dataset
@st.cache_data
def load_data():
    return pd.read_csv("breast_cancer_cleaned.csv")

# Load the dataset
data = load_data()

# Fixed target column
target_column = "diagnosis"

# Feature selection dropdown
st.write("### Select Features for the Model")
features = st.multiselect(
    "Choose Features",
    options=data.columns.drop(target_column),
    default=list(data.columns.drop(target_column))
)

if features:
    # Prepare data
    X = data[features]
    y = data[target_column]

    # Train-test split
    test_size = st.slider("Test Size (%)", min_value=10, max_value=50, value=20, step=5) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model selection dropdown
    st.write("### Choose Model")
    model_choice = st.selectbox("Model Type", ["Logistic Regression", "Random Forest", "Support Vector Machine"])
    
    # Initialize the model
    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "Support Vector Machine":
        model = SVC(probability=True)

    # Train the model
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Add diagnosis interpretation to predictions
    diagnosis_outcomes = ["Benign (0)", "Malignant (1)"]
    y_pred_diagnosis = [diagnosis_outcomes[int(pred)] for pred in y_pred]

    # Display metrics
    st.write("### Model Performance Metrics")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.2f}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
    if model_choice != "Support Vector Machine":
        st.write(f"ROC AUC: {roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]):.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Show predicted outcomes as diagnosis
    st.write("### Prediction Results")
    prediction_results = pd.DataFrame({
        "Actual Diagnosis": [diagnosis_outcomes[int(actual)] for actual in y_test],
        "Predicted Diagnosis": y_pred_diagnosis
    })
    st.write(prediction_results)

    # Confusion Matrix Visualization
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)
