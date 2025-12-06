# Smart ECG: Machine Learning for Cardiac Rhythm Classification

Smart ECG is a lightweight, reproducible machine-learning pipeline that classifies ECG heartbeat segments into five clinically meaningful classes using classical ML algorithms.
This project was developed as part of the Machine Learning Laboratory — SRM University–AP (AY 2024–2025).

📌 Project Overview

Electrocardiograms (ECGs) are crucial for detecting cardiac abnormalities. This project uses pre-segmented heartbeat windows from the MIT-BIH Arrhythmia Dataset to build ML models that can classify heartbeats into:

0 — Normal

1 — Supraventricular ectopic

2 — Ventricular ectopic

3 — Fusion

4 — Unknown beat

Each heartbeat sample contains 187 values, representing one ECG beat segment.

The project evaluates baseline ML models and their improved, optimized versions, ultimately identifying which model generalizes best on unseen data.

📂 Repository Structure
smart-ecg/
├── data/
│   ├── mitbih_train.csv
│   ├── mitbih_test.csv
├── notebooks/
│   ├── smart_ecg_notebook.ipynb
├── models/
│   ├── vanilla_rf.joblib
│   ├── vanilla_svm_linear.joblib
│   ├── improved_rf.joblib
│   ├── improved_svm_linear.joblib
│   ├── improved_nb_pca.joblib
├── outputs/
│   ├── confusion_matrices/
│   ├── metrics_summary.csv
├── requirements.txt
└── README.md

⚙️ Installation

Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

requirements.txt (recommended)
numpy
pandas
scikit-learn
imbalanced-learn
matplotlib
seaborn
joblib
scipy

🚀 How to Run

Place the dataset CSVs (mitbih_train.csv, mitbih_test.csv) inside data/.

Open the Jupyter notebook:

jupyter lab notebooks/smart_ecg_notebook.ipynb


Run all cells to:

Load and preprocess data

Create train/validation/test splits

Train baseline & improved ML models

Generate evaluation metrics

Save trained models in /models

Produce confusion matrices & summary plots

All output files are automatically stored in /models and /outputs.

🧠 Models Used
Baseline Models

Gaussian Naive Bayes

Random Forest

Linear SVM (LinearSVC)

Improved Models

Random Forest (GridSearchCV + class balancing)

Linear SVM (RandomizedSearchCV + class balancing)

Naive Bayes + PCA (Dimensionality reduction → GaussianNB)

📊 Results Summary
Validation Set

Improved Linear SVM → Balanced Accuracy ≈ 0.91

Test Set (Generalization Performance)

Improved Random Forest →

Balanced Accuracy ≈ 0.83

Macro F1-score ≈ 0.86

Confusion matrices and performance comparison charts are available in /outputs.

🧪 Evaluation Metrics

To handle class imbalance, the project uses:

Balanced Accuracy (Primary)

Macro F1-Score

Confusion Matrix (Per Class)

Train/Val/Test split accuracy comparisons

📈 Key Features of This Project

✔️ End-to-end reproducible ML pipeline

✔️ Handles severe class imbalance (class weights + PCA)

✔️ Saves trained models for deployment

✔️ Clean modular code inside Jupyter Notebook

✔️ Suitable for embedded devices / wearables due to classical ML approach

🔮 Future Improvements

Apply SMOTE oversampling for minority ECG classes

Try lightweight 1D CNN architectures

Add SHAP-based interpretability

Deploy as a real-time ECG classification microservice

👥 Authors

G. Jahnava

B. Bhanu priya

Arya Shinde 

Project completed for the Machine Learning Laboratory (SRM University–AP).
