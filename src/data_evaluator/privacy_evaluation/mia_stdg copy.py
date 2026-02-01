import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, precision_score

COLUMN_ID = 'ID'

def is_identified(record, synthetic_data, th):
    """Check if a record can be identified in the synthetic data based on a threshold.
    
    Parameters:
        record (pd.Series): The record to check.
        synthetic_data (pd.DataFrame): The synthetic data.
        th (float): The threshold for identification based on Hamming distance.

    Returns:
        bool: True if the record can be identified, otherwise False.
    """
    distances = distance.cdist([record.values], synthetic_data, metric='hamming')
    return (distances < th).any()

def get_true_labels(train_data_indexes, attacker_data_indexes):
    """Get the true labels indicating if records were part of the training data.
    
    Parameters:
        train_data_indexes (list): Indexes of the training data.
        attacker_data_indexes (list): Indexes of the attacker's data.

    Returns:
        list: True labels (1 if in training data, 0 otherwise).
    """
    return [1 if idx in train_data_indexes else 0 for idx in attacker_data_indexes]

def predict_labels(attacker_data, synthetic_data, th):
    """Predict if records were part of the training data based on their presence in synthetic data.
    
    Parameters:
        attacker_data (pd.DataFrame): The attacker's data.
        synthetic_data (pd.DataFrame): The synthetic data.
        th (float): The threshold for identification.

    Returns:
        list: Predicted labels (1 if predicted to be in training data, 0 otherwise).
    """
    return [is_identified(row.drop(COLUMN_ID), synthetic_data, th) for _, row in attacker_data.iterrows()]

def evaluate_membership_attack(attacker_data, train_data_indexes, synthetic_data, th):
    """Evaluate the membership inference attack.
    
    Parameters:
        attacker_data (pd.DataFrame): The attacker's data.
        train_data_indexes (list): Indexes of the training data.
        synthetic_data (pd.DataFrame): The synthetic data.
        th (float): The threshold for identification.

    Returns:
        tuple: Precision and accuracy of the attack.
    """
    true_labels = get_true_labels(train_data_indexes, attacker_data[COLUMN_ID].tolist())
    predicted_labels = predict_labels(attacker_data, synthetic_data, th)
    
    precision = precision_score(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    return precision, accuracy
