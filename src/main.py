import configparser
from column_config import diff_cols, keep_cols, users_cols, target_col, transform_cols
from data_gathering import data_gathering
from data_cleaning import data_cleaning
from feature_computation import feature_computation
from modeling import modeling
from evaluation import evaluation
from utils.logger import get_logger


# Instantiate logger
logger = get_logger(__name__)


def get_initial_params():
    """
    Loads all relevant parameters into a global variable. These parameters are then accessible to other functions in the script.
    Each function can consume the parameters it requires for its operation.
    This approach ensures centralized management and consistency of parameters across different functions.

    Returns:
        None: This function does not return a value but populates a global variable
        with necessary parameters.
    """
    global train_from, train_to, logistic_regression_params, lightgbm_params

    config = configparser.ConfigParser()
    config.read("src/params.ini")

    train_from = config.get("PARAMS", "train_from")
    train_to = config.get("PARAMS", "train_to")

    # Load Logistic Regression parameters
    logistic_regression_params = {
        "penalty": config.get("LOGISTIC_REGRESSION", "penalty", fallback="l2"),
        "C": config.getfloat("LOGISTIC_REGRESSION", "C", fallback=1.0),
        "solver": config.get("LOGISTIC_REGRESSION", "solver", fallback="saga"),
        "max_iter": config.getint("LOGISTIC_REGRESSION", "max_iter", fallback=10000),
    }

    # Load LightGBM parameters
    lightgbm_params = {
        "boosting_type": config.get("LIGHTGBM", "boosting_type", fallback="gbdt"),
        "num_leaves": config.getint("LIGHTGBM", "num_leaves", fallback=12),
        "max_depth": config.getint("LIGHTGBM", "max_depth", fallback=-1),
        "learning_rate": config.getfloat("LIGHTGBM", "learning_rate", fallback=0.005),
        "n_estimators": config.getint("LIGHTGBM", "n_estimators", fallback=100),
        "random_state": config.getint("LIGHTGBM", "random_state", fallback=500),
        "colsample_bytree": config.getfloat(
            "LIGHTGBM", "colsample_bytree", fallback=0.64
        ),
        "subsample": config.getfloat("LIGHTGBM", "subsample", fallback=0.7),
        "reg_alpha": config.getint("LIGHTGBM", "reg_alpha", fallback=0),
        "reg_lambda": config.getint("LIGHTGBM", "reg_lambda", fallback=1),
        "path_smooth": config.getfloat("LIGHTGBM", "path_smooth", fallback=0.2),
    }


def main_orchestrator():
    """
    Orchestrates the entire ML pipe.
    It starts by initializing parameters and then sequentially executes the following steps:
    data gathering, data cleaning, feature computation, model training, and model evaluation.
    Each step is handled by a dedicated function, and the flow of data between these functions
    is managed within this orchestrator.

    Steps:
    1. Initialize parameters using `get_initial_params()`.
    2. Gather raw data with `data_gathering()`.
    3. Clean the gathered data using `data_cleaning()`.
    4. Compute relevant features from the clean data in `feature_computation()`.
    5. Train the machine learning model in `modeling()`.
    6. Evaluate the model's performance with `evaluation()`.

    Returns:
        dict: Evaluation results of the trained model, containing key performance metrics.
    """

    query = f"""
        WITH all_periods AS
        (
            SELECT *
            FROM   `mm-bi-catedras-upm.ESTIMACION_CHURN.multibrand_monthly_customer_base_mp2022`
            UNION ALL
            SELECT *
            FROM   `mm-bi-catedras-upm.ESTIMACION_CHURN.multibrand_monthly_customer_base_mp2023_1` ), selectable_customer AS
        (
                SELECT   customer_id
                FROM     all_periods
                GROUP BY customer_id ), customer_selected AS
        (
            SELECT customer_id AS selected_customer
            FROM   selectable_customer
            WHERE  MOD(ABS(FARM_FINGERPRINT(CAST(customer_id AS STRING))), 10) < 2 )
        SELECT     {", ".join(diff_cols + keep_cols + users_cols + target_col + transform_cols)}
        FROM       all_periods
        INNER JOIN customer_selected
        ON         customer_id = selected_customer
        WHERE      IS_CUST_SEGM_RESI > 0
        AND        IS_CUST_BILL_POST_CURR = TRUE
        AND        CUST_BUNDLE_CURR = 'FMC'
        AND        NUM_IMPAGOS = 0
        AND        pago_final_0 IS NOT NULL
    """

    # TO-DO: PARAMETRIZE THIS
    save_curves_path = "src/models"
    save_features_path = "src/features"
    save_target_path = "src/target"

    get_initial_params()
    raw_data = data_gathering(query, logger)
    clean_data = data_cleaning(raw_data, logger)
    features, target, features_test, target_test = feature_computation(
        clean_data,
        train_from,
        train_to,
        logger,
        save_features_path=save_features_path,
        save_target_path=save_target_path,
    )
    model = modeling(features, target, lightgbm_params, logger)
    model_metrics, precision_decile, uplift_by_decile, feature_importance = evaluation(
        model, features_test, target_test, logger, save_curves_path
    )


if __name__ == "__main__":
    print(main_orchestrator())
