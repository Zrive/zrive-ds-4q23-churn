import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import pickle
import os


def modeling(
    features: pd.DataFrame,
    target: pd.Series,
    lightgbm_params,
    logger,
    model_save_path: str = None,
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
    """
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
    )"""
    pipeline = Pipeline(
        [
            (
                "lightgbm",
                lgb.LGBMClassifier(
                    learning_rate=lightgbm_params["learning_rate"],
                    n_estimators=lightgbm_params["n_estimators"],
                    num_leaves=lightgbm_params["num_leaves"],
                    colsample_bytree=lightgbm_params["colsample_bytree"],
                    subsample=lightgbm_params["subsample"],
                    reg_alpha=lightgbm_params["reg_alpha"],
                    reg_lambda=lightgbm_params["reg_lambda"],
                    random_state=lightgbm_params["random_state"],
                ),
            ),
        ]
    )

    logger.info("Training model")
    model = pipeline.fit(features, target)

    logger.info("Completed model training!")

    if model_save_path:
        path_name_model = os.path.join(model_save_path, "model.lgb")
        pickle.dump(model, open(path_name_model, "wb"))
        logger.info(f"Model saved in {path_name_model}")

    return model
