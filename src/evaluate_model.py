from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import json

# Evaluation du modèle

def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_prob > threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("Confusion Matrix:\n", cm)
    
    metrics_dict = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
    
    with open("../metrics/metrics.json", "w") as f:
        json.dump(metrics_dict,f,indent=4)
                  
    return accuracy, precision, recall, cm

    


if __name__== "_main_":
    evaluate_model()