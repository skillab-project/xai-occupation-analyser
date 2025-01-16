# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 14:43:12 2024

@author: Dimitris Tsoukalas (tsoukj)

This script performs the following tasks:

- Data Preparation:
    * Reads a CSV file containing job-related data and drops unnecessary columns.
    * Filters data to include only occupational categories with at least 
    min_category_sample
- Train-Test Split:
    * Splits the data into training and testing sets.
- Model training:
    * Initializes a OneVsRestClassifier with a RandomForestClassifier as the 
    base model.
    * Fits the model to the training data.
    * Saves the model to designated folders for future reference. 
- Performance Metrics Calculation:
    * Makes predictions on the test data.
    * Calculates metrics for each class (occupation category).
- SHAP Analysis:
    * Uses SHAP to calculate and store feature importance for each class.
    * Uses SHAP to generate and store plots for each class.
"""

import pandas as pd
import numpy as np
import shap
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.multiclass import OneVsRestClassifier

target = 'occupation4d'
min_category_samples = 50 # Min number of samples per category to be considered for OvR model building

# Define output folders
file_path = 'all_occupation4d_skills.csv'
output_folder_models = "models"
output_folder_models_performance = "models_performance"
output_folder_features = "feature_importance"
output_folder_shap = "shap_values"
output_folder_plots = "plots"

# Ensure output folders exist
for folder in [output_folder_models, output_folder_models_performance, output_folder_features, output_folder_shap, output_folder_plots]:
    os.makedirs(folder, exist_ok=True)

# Code to read processed CSV
result_df = pd.read_csv(file_path)

# Create X and y
columns_to_drop = ["job_id", "upload_date", "occupation4d"]
X = result_df.drop(columns=columns_to_drop)
y = result_df["occupation4d"]

# Data filtering
category_counts = y.value_counts()
valid_categories = category_counts[category_counts >= min_category_samples].index
X_filtered = X[y.isin(valid_categories)]
y_filtered = y[y.isin(valid_categories)]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.1, random_state=42, stratify=y_filtered)

# Model training
ovr_model = OneVsRestClassifier(RandomForestClassifier(random_state=42)) # , class_weight='balanced'
ovr_model.fit(X_train, y_train)

# Save the trained model to a file
# model_file = os.path.join(output_folder_models, "ovr_model_features_all_occupation4d.pkl")
# with open(model_file, "wb") as model_file:
#     pickle.dump(ovr_model, model_file)

# Calculate performance metrics
y_pred_proba = ovr_model.predict_proba(X_test)

roc_auc_scores = {}
for i, class_label in enumerate(ovr_model.classes_):
    roc_auc_scores[class_label] = roc_auc_score(y_test == class_label, y_pred_proba[:, i])

precision_scores = {}
recall_scores = {}
f1_scores = {}

for i, class_label in enumerate(ovr_model.classes_):
    precision_scores[class_label] = precision_score(y_test == class_label, y_pred_proba[:, i] > 0.5)
    recall_scores[class_label] = recall_score(y_test == class_label, y_pred_proba[:, i] > 0.5)
    f1_scores[class_label] = f1_score(y_test == class_label, y_pred_proba[:, i] > 0.5)

metrics_df = pd.DataFrame({
    'AUC': roc_auc_scores,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1 Score': f1_scores
}).sort_values(by='F1 Score', ascending=False)

# Save the performance metrics to a file
models_performance_file = os.path.join(output_folder_models_performance, "Models Performance Metrics.csv")
metrics_df.to_csv(models_performance_file)

# Calculate SHAP values and plot feature importance results
for i, class_label in enumerate(ovr_model.classes_):
    print(f"Processing SHAP values for class: {class_label}")
    
    occupation_id = class_label.split('/')[-1]
    
    # Initialize SHAP TreeExplainer for the i-th estimator
    explainer = shap.TreeExplainer(model=ovr_model.estimators_[i], feature_perturbation='tree_path_dependent')
    # Compute SHAP values for the i-th estimator
    shap_values = explainer.shap_values(X_filtered, approximate=True, check_additivity=False) # numpy array with shape (n_samples, n_variables, n_classes)
    
    # Save SHAP values for the current class as a .npz file
    # shap_values_file = os.path.join(output_folder_shap, f"shap_values_{occupation_id.split('/')[-1]}.npz")
    # np.savez_compressed(shap_values_file, shap_values=shap_values)
    
    # Extract SHAP values for the current class
    shap_values_class = shap_values[:, :, 1]
    
    # Calculate mean absolute SHAP values for feature importance
    feature_importance_values = np.abs(shap_values_class).mean(axis=0)
    # Create a DataFrame to store feature importance
    feature_importance = pd.DataFrame({'Feature': X_filtered.columns, 'Importance': feature_importance_values})
    # Sort features by importance
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # Save values for feature importance to CSV
    feature_importance_file = os.path.join(output_folder_features, f"feature_importance_{occupation_id}.csv")
    feature_importance.to_csv(feature_importance_file, index=False)
    
    # Generate SHAP bar plot
    bar_plot_name = f"shap_bar_plot_{occupation_id}.png"
    bar_plot_path = os.path.join(output_folder_plots, bar_plot_name)
    plt.figure()
    shap.summary_plot(shap_values_class, features=X_filtered, plot_type='bar', feature_names=X_filtered.columns, show=False)
    plt.savefig(bar_plot_path, bbox_inches='tight')
    plt.close()
    
    # Generate SHAP dot plot
    dot_plot_name = f"shap_dot_plot_{occupation_id}.png"
    dot_plot_path = os.path.join(output_folder_plots, dot_plot_name)
    plt.figure()
    shap.summary_plot(shap_values_class, features=X_filtered, plot_type='dot', feature_names=X_filtered.columns, show=False)
    plt.savefig(dot_plot_path, bbox_inches='tight')
    plt.close()
