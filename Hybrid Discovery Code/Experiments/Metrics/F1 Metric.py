#Packages
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, recall_score, f1_score


# F1 Metric used for Edge Pruning Methods

def calculate_accuracy_measures(true_parents_list, predicted_parents_list):#
    """
    Calculate accuracy, precision, and false negative rate for the predicted parent sets.
    
    Parameters:
    - true_parents_list: List of sets, where each set contains the true parents of a variable.
    - predicted_parents_list: List of sets, where each set contains the predicted parents of a variable.
    
    Returns:
    A dictionary containing accuracy, precision, and FNR.
    """
    true_flat = []
    predicted_flat = []
    
    for true_parents, predicted_parents in zip(true_parents_list, predicted_parents_list):
        # Create binary vectors for each set of parents
        true_vector = [1 if node in true_parents else 0 for node in range(len(true_parents_list))]
        predicted_vector = [1 if node in predicted_parents else 0 for node in range(len(predicted_parents_list))]
        
        true_flat.extend(true_vector)
        predicted_flat.extend(predicted_vector)
    
    # Calculate metrics
    precision = precision_score(true_flat, predicted_flat, zero_division=0)
    accuracy = accuracy_score(true_flat, predicted_flat)
    recall = recall_score(true_flat, predicted_flat, zero_division=0)
    f1 = f1_score(true_flat, predicted_flat, zero_division=0)
    
    return f1, precision, recall