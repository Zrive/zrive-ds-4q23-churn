import configparser
from sklearn.pipeline import Pipeline
from utils.logger import get_logger
from data_gathering import data_gathering
from data_cleaning import data_cleaning
from feature_computation import compute_features
import pandas as pd
import pickle
import os
from column_config import users_cols, diff_cols, keep_cols, transform_cols, target_col


logger = get_logger(__name__)


def load_model(models_path: str = "src/models") -> Pipeline:
    """
    Load sklearn Pipeline from memory.

    Args:
        models_path (str): The path the model is saved on.

    Returns:
        Pipeline: sklearn pipeline of the model
    """
    path_name_model = os.path.join(models_path, "model.lgb")

    return pickle.load(open(path_name_model, "rb"))


def get_month_to_predict_from_params():
    """
    Loads predict month parameter into a global variable. This parameter is then accessible to other functions in the script.
    Each function can consume the parameters it requires for its operation.
    This approach ensures centralized management and consistency of parameters across different functions.

    Returns:
        None: This function does not return a value but populates a global variable
        with necessary parameters.
    """
    global predict_month

    config = configparser.ConfigParser()
    config.read("src/params.ini")

    predict_month = config.get("PARAMS", "predict_month")


def feature_computation_to_predict(
    df: pd.DataFrame, predict_month_dt: pd.Timestamp
) -> pd.DataFrame:
    """
    Compute the data in order to predict with the model.
    Args:
        df: The cleaned dataset with customer, month, and payment information.
        predict_month_dt: The pandas timestamp of the month we want to predict.

    Returns:
        DataFrame: Pandas DataFrame with computed features for model predicting.
    """
    logger.info(f"Computing features to predict churn in {predict_month_dt}")
    predict_df = df[
        users_cols + transform_cols + keep_cols + diff_cols + target_col
    ].copy()
    predict_df["date"] = pd.to_datetime(
        predict_df["YEAR"].astype(str) + "-" + predict_df["MONTH"].astype(str) + "-01"
    )
    previous_churned_users_ = predict_df[
        (predict_df["date"] <= predict_month_dt) & (predict_df[target_col[0]] > 0)
    ]["customer_id"].unique()

    predict_df = predict_df[~predict_df["customer_id"].isin(previous_churned_users_)]

    logger.info(
        f"Removing {len(previous_churned_users_)} previous churned users from predict set"
    )
    logger.info(
        f"Unique customers in predict set: {predict_df['customer_id'].nunique()}"
    )

    predict_df_features = compute_features(predict_df, target_col, predict_month_dt)
    predict_df_features = predict_df_features.drop(columns=["MONTH", "YEAR"])

    logger.info(f"Final number of features computed: {predict_df_features.shape[1]}")
    logger.info(f"Length predict data: {len(predict_df_features)}")
    logger.info("Computation done!")

    return predict_df_features


def predict(features: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    """
    Predicts the probability of the features dataset.
    Args:
        features: The ready for model dataframe.
        model: sklearn pipeline of the model.

    Returns:
        pd.DataFrame: Dataframe with the predictions for customer_id.
    """
    logger.info(f"Predicting for {len(features)} customers")
    customer_id = features["customer_id"]
    features = features.drop(columns="customer_id")
    preds = model.predict_proba(features)[:, 1]
    preds_df = pd.DataFrame({"preds": preds})

    preds_per_customer = pd.concat([customer_id, preds_df], axis=1)
    logger.info(preds_per_customer.head(10))

    return preds_per_customer


def gather_data_in_chunks(EOP_from, EOP_to, limit: int, offset: int) -> pd.DataFrame:
    """
    Gather data in chunks as all the data doesn't fit in memory.
    Chunks are managged by the LIMIT and OFFSET functions
    Args:
        EOP_from: date from we get features,  EOP (yearmonth).
        EOP_to: date to we get features, EOP (yearmonth).
        limit: max numbers of rows in the query.
        offset: skipped rows in the query result.

    Returns:
        DataFrame: Pandas DataFrame with chunked data.
    """
    query = f"""
    WITH all_periods AS (
    SELECT *
    FROM `mm-bi-catedras-upm.ESTIMACION_CHURN.multibrand_monthly_customer_base_mp2022`
    UNION ALL 
    SELECT *
    FROM `mm-bi-catedras-upm.ESTIMACION_CHURN.multibrand_monthly_customer_base_mp2023_1`
    ), 

    selectable_customer AS (
        SELECT customer_id
        FROM all_periods
        GROUP BY customer_id
    ), 

    customer_selected AS (
        SELECT customer_id AS selected_customer
        FROM selectable_customer
        WHERE MOD(ABS(FARM_FINGERPRINT(CAST(customer_id AS STRING))), 1) = 0
        LIMIT {limit}
        OFFSET {offset}
    )

    SELECT {", ".join(diff_cols + keep_cols + users_cols + target_col + transform_cols)}
    FROM all_periods
    INNER JOIN customer_selected
    ON customer_id = selected_customer
    WHERE IS_CUST_SEGM_RESI > 0
    AND IS_CUST_BILL_POST_CURR = TRUE
    AND CUST_BUNDLE_CURR = 'FMC'
    AND NUM_IMPAGOS = 0
    AND pago_final_0 IS NOT NULL
    AND EOP >= "{EOP_from}"
    AND EOP <= "{EOP_to}"
    """

    return data_gathering(query, logger)


def predict_orchestrator() -> pd.DataFrame:
    """
    Orchestrates the entire Predict ML pipe.
    It starts by initializing parameter and then sequentially executes the following steps:
    load the model, data gathering, data cleaning, feature computation, prediction.
    Each step is handled by a dedicated function, and the flow of data between these functions
    is managed within this orchestrator.

    Steps:
    1. Initialize parameter using `get_month_to_predict_from_params()`.
    2. Gather raw data in chunks with `gather_data_in_chunks()`.
    3. Clean the gathered data using `data_cleaning()`.
    4. Compute relevant features from the clean data in `feature_computation_to_predict()`.
    5. Predict the probability for each customer in `predict()`.
    6. Repeat 2, 3, 4 and 5 for each chunk

    Returns:
        pd.DataFrame: Dataframe with prediction for each customer_id.
    """
    get_month_to_predict_from_params()
    predict_month_dt = pd.to_datetime(predict_month)
    predict_month_to_dt = predict_month_dt - pd.DateOffset(months=2)
    predict_month_from_dt = predict_month_to_dt - pd.DateOffset(months=6)
    EOP_to = str(predict_month_to_dt.year) + str(
        "{:02d}".format(predict_month_to_dt.month)
    )
    EOP_from = str(predict_month_from_dt.year) + str(
        "{:02d}".format(predict_month_from_dt.month)
    )

    logger.info(
        f"Predicting for {predict_month_dt}, computing features from {predict_month_from_dt} to {predict_month_to_dt}"
    )

    query = """
    WITH all_periods AS (
    SELECT *
    FROM `mm-bi-catedras-upm.ESTIMACION_CHURN.multibrand_monthly_customer_base_mp2022`
    UNION ALL 
    SELECT *
    FROM `mm-bi-catedras-upm.ESTIMACION_CHURN.multibrand_monthly_customer_base_mp2023_1`
    ), 

    selectable_customer AS (
        SELECT customer_id
        FROM all_periods
        GROUP BY customer_id
    )

    SELECT count(*) as n_lines
    FROM selectable_customer
    """

    n_chunks = 3
    number_lines = data_gathering(query, logger)["n_lines"].values[0]

    limit = (number_lines // n_chunks) + 1

    preds_per_customer = pd.DataFrame()

    model = load_model()
    for chunk in range(0, n_chunks - 1):
        logger.info(f"Executing chunk number {chunk}")
        raw_df = gather_data_in_chunks(EOP_from, EOP_to, limit, limit * chunk)
        clean_data = data_cleaning(raw_df, logger)
        features_predict = feature_computation_to_predict(
            clean_data, predict_month_to_dt
        )
        if preds_per_customer.empty:
            preds_per_customer = predict(features_predict, model)
        else:
            print("HERE")
            preds_per_customer = pd.concat(
                [preds_per_customer, predict(features_predict, model)],
                ignore_index=True,
            )

    return preds_per_customer


if __name__ == "__main__":
    print(predict_orchestrator())
