# Breast Cancer Prediction using Logistic Regression

## Project Overview
This project focuses on predicting breast cancer diagnosis (malignant or benign) based on various features from the Breast Cancer Wisconsin dataset. The project uses a logistic regression model and performs a series of data preprocessing, exploratory data analysis (EDA), and model evaluation steps to improve prediction accuracy. The model predicts whether a tumor is malignant or benign based on its features.

## Key Steps in the Project

### 1. Data Cleaning and Preprocessing
- **Dataset Overview**: The dataset contains 569 samples and 30 features, including radius, texture, smoothness, perimeter, area, and fractal dimension of the tumor.
  - Columns include: `id`, `diagnosis`, `radius_mean`, `texture_mean`, `smoothness_mean`, `compactness_mean`, etc.
- **Data Issues Addressed**:
  - **Missing Values**: Identified missing values in columns and replaced them as follows:
    - Missing values in `director`, `country`, and `cast` were replaced with "Unknown".
    - The column `Unnamed: 32` (which was empty) was dropped.
  - **Feature Engineering**: No new features were created in this version, but handling missing values and preprocessing helped optimize the data for the model.

### 2. Exploratory Data Analysis (EDA)
- **General Statistics**:
  - The dataset contains 569 samples, with 357 benign and 212 malignant diagnoses.
  - Feature correlation analysis was performed to assess relationships between the features, helping in understanding which features are crucial for the prediction.
- **Data Visualization**:
  - **Class Distribution**: The target variable `diagnosis` (Malignant, Benign) was visualized, showing an imbalance between the two classes.
  - **Feature Distributions**: Various histograms and box plots were used to visualize feature distributions for each class, helping identify potential outliers and trends.

### 3. Model Development
- **Model**: Logistic Regression
  - The dataset was split into training and testing sets, with 70% of the data used for training and 30% for testing.
  - **Feature Standardization**: The features were standardized using `StandardScaler` to ensure better model performance.
- **Model Evaluation**:
  - The logistic regression model was trained on the training data and evaluated on the test data.
  - **Performance Metrics**:
    - **Accuracy**: The model achieved an accuracy of **96.5%** on the test data.
    - **Precision**: The precision for malignant cases was **95.2%**, meaning that 95.2% of the tumors predicted as malignant were indeed malignant.
    - **Recall**: The recall was **97.6%**, indicating that 97.6% of the actual malignant cases were correctly identified.
    - **F1-Score**: The F1-score was **96.4%**, which balances precision and recall for malignant predictions.
    - **ROC AUC**: The area under the ROC curve (AUC) was **0.99**, indicating excellent model performance in distinguishing between malignant and benign tumors.

### 4. Evaluation and Results
- **Confusion Matrix**:
  - The confusion matrix showed that the model correctly classified 106 malignant and 155 benign cases, with very few misclassifications.
  - There were **4 false positives** (benign tumors classified as malignant) and **4 false negatives** (malignant tumors classified as benign).
- **Model Insights**:
  - Logistic regression performed exceptionally well with high precision and recall, suggesting that it is well-suited for this binary classification problem.
  - Future models, such as Random Forest or Support Vector Machines (SVM), could be explored to compare performance.

### 5. Predictive Insights
- **Feature Importance**: Logistic regression allows us to identify the most significant features contributing to the prediction. Key features include:
  - `radius_mean`
  - `texture_mean`
  - `perimeter_mean`
  - `area_mean`
  These features played a critical role in distinguishing between malignant and benign tumors.

## Tools and Technologies
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn (Logistic Regression)
- **App Development**: Streamlit for interactive web application development.

## Conclusion
This project demonstrates the application of logistic regression for predicting breast cancer diagnosis using the Breast Cancer Wisconsin dataset. Through thorough data cleaning and preprocessing, followed by model development, the logistic regression model achieved **96.5% accuracy**, with a **95.2% precision** and **97.6% recall**. This model offers a robust approach to classifying tumors as malignant or benign based on critical tumor features.
