import pandas as pd
from sklearn.linear_model import LogisticRegression

def modeling(features: pd.DataFrame, target: pd.DataFrame) -> LogisticRegression:
    """
    Selects a machine learning algorithm, trains the model with processed data,
    and learns patterns for churn prediction.

    Returns:
        Model: Trained machine learning model for churn prediction.
    """

    model = LogisticRegression(penalty="l1", C=1, solver="saga")
    model.fit(features, target)

    return model
