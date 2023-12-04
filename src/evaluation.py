import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

def evaluation(model: LogisticRegression, features_test: pd.DataFrame, target_test: pd.DataFrame) -> dict[str, str]:
    """
    Assesses trained model's performance using a test dataset and computes metrics like accuracy, precision, recall, and ROC-AUC.

    Returns:
        dict: Dictionary with key performance metrics of the model.
    """

    preds = model.predict_proba(features_test)

    precision, recall, _ = precision_recall_curve(target_test, preds[:, 1])

    model_metrics = {
        "Precision": precision,
        "Recall": recall
    }

    return model_metrics
