import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def modeling(
    features: pd.DataFrame, target: pd.Series, logistic_regression_params, logger
) -> Pipeline:
    """
    Prepares a machine learning pipeline that scales features and trains a logistic regression model
    with processed data to predict churn.

    Args:
        features (pd.DataFrame): The features to train the model on.
        target (pd.Series): The target variable to predict.

    Returns:
        Pipeline: A pipeline with a standard scaler and a trained logistic regression model.
    """

    logger.info("Starting Modeling")

    # Directly use the logistic_regression_params global variable
    logger.info("Building model pipeline")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    penalty=logistic_regression_params["penalty"],
                    C=logistic_regression_params["C"],
                    solver=logistic_regression_params["solver"],
                    max_iter=logistic_regression_params["max_iter"],
                    random_state=42,
                ),
            ),
        ]
    )
    """
    pipeline = Pipeline(
        [
            (
                "lightgbm",
                lgb.LGBMClassifier(
                    boosting_type=lightgbm_params["boosting_type"],
                    num_leaves=lightgbm_params["num_leaves"],
                    max_depth=lightgbm_params["max_depth"],
                    learning_rate=lightgbm_params["learning_rate"],
                    n_estimators=lightgbm_params["n_estimators"],
                    random_state=42,
                ),
            ),
        ]
    )"""

    logger.info("Training model")
    model = pipeline.fit(features, target)

    logger.info("Completed model training!")

    return model
