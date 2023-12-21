```python
import sys
import os
import pandas as pd
import numpy as np
import configparser
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import lightgbm as lgb
import lightgbm as lgb
from plotnine import ggplot, aes, geom_boxplot, labs, scale_color_manual, geom_point
import configparser
import random


project_root = "/home/dan1dr/zrive-ds-4q24-churn"

# Define the project root path
current_wd = os.getcwd()

# Change the working directory if necessary
if current_wd != project_root:
    print(f"Changing working directory from {current_wd} to {project_root}")
    os.chdir(project_root)
else:
    print("Already in the correct path")

# Add 'src' directory to sys.path
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    print(f"Adding {src_path} to sys.path")
    sys.path.insert(0, src_path)

# Import the modules
from db_connectors.bigquery_service import BigqueryService
from data_gathering import data_gathering
from utils.logger import get_logger
from data_cleaning import data_cleaning
from column_config import users_cols, diff_cols, keep_cols, transform_cols, target_col


config = configparser.ConfigParser()
config.read("src/params.ini")

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
    global train_from, train_to, end_date, logistic_regression_params, lightgbm_params

    config = configparser.ConfigParser()
    config.read("src/params.ini")

    train_from = config.get("PARAMS", "train_from")
    train_to = config.get("PARAMS", "train_to")
    end_date = config.get("PARAMS", "end_date")
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
    }


get_initial_params()
save_curves_path = "src/models"
save_features_path = "src/features"
save_target_path = "src/target"
```

    Changing working directory from /home/dan1dr/zrive-ds-4q24-churn/src/eda to /home/dan1dr/zrive-ds-4q24-churn
    Adding /home/dan1dr/zrive-ds-4q24-churn/src to sys.path



```python
get_initial_params()
```


```python
def feature_computation(
    clean_data: pd.DataFrame,
    train_from: str,
    train_to: str,
    logger,
    keep_gap_month_churns: bool = False,
    save_features_path: str = "",
    save_target_path: str = "",
) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    """
    Split data into train and test features set, aggregate the data into historical behavior for those cols needed.
    It also joins it with already calculated features, and extract the needed target from 2 months ahead.
    Args:
        clean_data: The cleaned dataset with customer, month, and payment information.
        train_from: The starting date of the training period.
        train_to: The ending date of the training period.

    Returns:
        DataFrame: Pandas DataFrame with computed features for model training.
        Series: Pandas Series representing the target variable for train set.
        DataFrame: Pandas DataFrame with computed features for model testing.
        Series: Pandas Series representing the target variable for test set.
    """
    logger.info("Starting feature computation")

    # TO-DO: Catch exceptions
    # TO-DO: Potential unit tests validating same length for features/targets
    # TO-DO: Instead of defining the cols every time import them somewhere else (they're need in data_cleaning also)

    # Convert the train_from and train_to to datetime
    train_from_dt = pd.to_datetime(train_from)
    train_to_dt = pd.to_datetime(train_to)

    # Filter train and test data before feature computation
    test_from_dt = train_from_dt + pd.DateOffset(months=1)
    test_to_dt = train_to_dt + pd.DateOffset(months=1)
    target_train_month = train_to_dt + pd.DateOffset(months=2)
    target_test_month = test_to_dt + pd.DateOffset(months=2)

    logger.info(
        f"Train computation from {train_from} to {train_to}. Target for {target_train_month}"
    )
    logger.info(
        f"Test computation from {test_from_dt} to {test_to_dt}. Target for {target_test_month}"
    )

    compute_ready_data = clean_data[
        users_cols + transform_cols + keep_cols + diff_cols + target_col
    ].copy()
    train_df, test_df = split_train_test(
        compute_ready_data, train_from_dt, train_to_dt, test_from_dt, test_to_dt
    )

    # For test set, we will remove all users that previously churned
    previous_churned_users_test = test_df[
        (test_df["date"] <= test_to_dt) & (test_df[target_col[0]] > 0)
    ]["customer_id"].unique()
    test_df = test_df[~test_df["customer_id"].isin(previous_churned_users_test)]

    logger.info(
        f"Removing {len(previous_churned_users_test)} previous churned users from test set"
    )

    logger.info(f"Unique customers in train: {train_df['customer_id'].nunique()}")
    logger.info(f"Unique customers in test: {test_df['customer_id'].nunique()}")

    logger.info("Starting features and target computation")
    logger.info(f"Initial number of features passed: {train_df.shape[1]}")
    logger.info("Starting computation")


    train_df_features = compute_features(train_df, target_col)
    test_df_features = compute_features(test_df, target_col)
    train_df_target = compute_target(train_df_features,
        compute_ready_data, target_col, keep_gap_month_churns
    )
    test_df_target = compute_target(test_df_features,
        compute_ready_data, target_col, keep_gap_month_churns
    )
    logger.info(f"Final number of features computed: {train_df_features.shape[1]}")
    logger.info(f"Length train data: {len(train_df_features)}")
    logger.info(f"Length test data: {len(test_df_features)}")
    logger.info("Computation done!")

    # As there are customer that leave between the month we use for training and the target month
    # We have to join the features and the targets and drop those that don't have target. By doing this,
    # we exclude customer that churned in gap month or those with no corresponding record in the target dataset.
    features_and_target_train = train_df_features.merge(
        train_df_target, on="customer_id", how="left"
    )
    features_and_target_test = test_df_features.merge(
        test_df_target, on="customer_id", how="left"
    )    
    
    features_and_target_train = features_and_target_train[
        features_and_target_train[target_col[0]].notna()
    ]

    features_and_target_test = features_and_target_test[
        features_and_target_test[target_col[0]].notna()
    ]
    
    date_cols = ['date', 'target_month', 'gap_month', 'cutoff_date']

    # Split train and test features + target (squeeze into 1D array)
    features = features_and_target_train.drop(columns=target_col + users_cols + date_cols)
    features_test = features_and_target_test.drop(columns=target_col + users_cols + date_cols)
    target = features_and_target_train[target_col].squeeze()
    target_test = features_and_target_test[target_col].squeeze()

    logger.info(f"Features: {features.columns.tolist()}")
    logger.info(f"Target: {target.name}")
    logger.info("Completed feature computation!")


    try:
        features.to_parquet(f"{save_features_path}/features.parquet", index=False)
        features_test.to_parquet(
            f"{save_features_path}/features_test.parquet", index=False
        )
        target.to_frame().to_parquet(f"{save_target_path}/target.parquet", index=False)
        target_test.to_frame().to_parquet(
            f"{save_target_path}/target_test.parquet", index=False
        )
        logger.info(f"Features saved on {save_features_path}")
        logger.info(f"Targets saved on {save_target_path}")
    except:
        logger.info(f"Features not saved on {save_features_path}")
        logger.info(f"Targets not saved on {save_target_path}")

    return features, target, features_test, target_test



def split_train_test(
    df: pd.DataFrame,
    train_from_dt: pd.Series,
    train_to_dt: pd.Series,
    test_from_dt: pd.Series,
    test_to_dt: pd.Series,
    window_months: int = 6,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Split data into train and test and randomly assigns a specific month for each of the users.
    Args:
        df: The clean dataset with the columns we want to use as features.
        train_from: The starting date of the training period.
        train_to: The ending date of the training period.
        test_from: The starting date of the testing period.
        test_to: The ending date of the testing period.

    Returns:
        DataFrame: Pandas DataFrame with training months only.
        DataFrame: Pandas DataFrame with testing months only.
    """
    # Create date col to mix month and year
    window = pd.DateOffset(months=window_months)
    df["date"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01"
    )

    # Filter compute_data for the specific date intervals.
    df = df[(df["date"] >= train_from_dt - window) & (df["date"] <= test_to_dt)]

    train_df = df[(df["date"] >= train_from_dt) & (df["date"] <= train_to_dt)]
    test_df = df[(df["date"] >= test_from_dt) & (df["date"] <= test_to_dt)]

    selected_train_df = (
        train_df.groupby('customer_id')['date']
        .apply(lambda x: random.choice(x.unique()))
        .reset_index()
        .rename(columns={'date': 'cutoff_date'})
    )
    
    train_df = df.merge(selected_train_df, how='inner', on=['customer_id'])
    train_df = train_df[(train_df['date'] <= train_df['cutoff_date']
                         ) & (train_df['date'] > (train_df['cutoff_date'] - window))]
    
    # Set the cutoff date to test_to_dt for all test data    
    test_df['cutoff_date'] = test_to_dt

    return train_df, test_df

def compute_features(
    df: pd.DataFrame, target_col: list[str]) -> pd.DataFrame:
    """
    Compute the features and adds them to the df.
    Args:
        df: The clean dataset with the columns we want to use as features.
        target_col: Name of the target column.
        add_churn_label: Whether to add the 'WHEN_USER_CHURNED' column.

    Returns:
        DataFrame: Pandas DataFrame with new computed variables.
    """

    # TO-DO: The rolling function is propagated backwards for each single month.
    # we just need it for the last one (but for that we need past data also). Didn't find
    # any option to do it with pandas

    #df = df.drop(columns=target_col)

    df = df.sort_values(by=["customer_id", "date"])
    df = df.set_index("date")

    # Dynamically compute features for each col in transform_cols
    for col in transform_cols:
        df[f"{col}_prev_month"] = df.groupby("customer_id")[col].shift(1)
        df[f"{col}_prev_month"] = df[f"{col}_prev_month"].fillna(0)
        df[f"{col}_avg_3_months"] = compute_x_months_avg(df, col, 3)
        df[f"{col}_avg_6_months"] = compute_x_months_avg(df, col, 6)
        df[f"{col}_std_3_months"] = compute_x_months_std(df, col, 3)
        df[f"{col}_std_6_months"] = compute_x_months_std(df, col, 6)
    
    df = df.reset_index().drop(columns=target_col)
    df = df[df['date'] == df['cutoff_date']]


    return df


def compute_x_months_avg(
    df: pd.DataFrame, col_name: list[str], months: int
) -> pd.DataFrame:
    """
    Compute the mean of the last months for a column.
    Args:
        df: The clean dataset with the columns we want to use as features.
        col_name: Name of the column to compute
        months: Number of months we want to compute the feature back

    Returns:
        DataFrame: Pandas DataFrame the new computed column.
    """
    return (
        df.groupby("customer_id")[col_name]
        .rolling(window=months, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


def compute_x_months_std(
    df: pd.DataFrame, col_name: list[str], months: int
) -> pd.DataFrame:
    """
    Compute the mean of the last months for a column.
    Args:
        df: The clean dataset with the columns we want to use as features.
        col_name: Name of the column to compute
        months: Number of months we want to compute the feature back

    Returns:
        DataFrame: Pandas DataFrame the new computed column.
    """
    return (
        df.groupby("customer_id")[col_name]
        .rolling(window=months, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )


def compute_target(
    df: pd.DataFrame,
    compute_ready_data: pd.DataFrame,
    target_col: list[str],
    keep_gap_month_churns: bool = False,
) -> pd.DataFrame:
    """
    Compute the target column for a df.
    Args:
        df: The clean dataset with all the data.
        target_col: Name of the target column.
        target_month: The date where the target has to be computed.
        keep_gap_month_churns: A boolean parameter that determines the treatment of churns occurring in the gap month.
        If True, these churns are treated as actual churns (assigned a value of 1);
        otherwise, they are excluded from the analysis. Defaults to False.
    Returns:
        DataFrame: Pandas DataFrame with the customer_id and the target computed.
    """

    df['target_month'] = df['date'] + pd.DateOffset(months=2)
    df['gap_month'] = df['date'] + pd.DateOffset(months=1)

    # Create two separate DataFrames for target_month and gap_month
    df_target = df[['customer_id', 'target_month']].copy()
    df_target['reference_date'] = df_target['target_month']

    df_gap = df[['customer_id', 'gap_month']].copy()
    df_gap['reference_date'] = df_gap['gap_month']

    # Combine the target and gap month DataFrames
    combined_df = pd.concat([df_target, df_gap], ignore_index=True)

    # Merge with compute_ready_data
    target_df = compute_ready_data.merge(combined_df, 
                                         left_on=['customer_id', 'date'], 
                                         right_on=['customer_id', 'reference_date'])

    for col in target_col:
        target_df[col].fillna(0, inplace=True)
        target_df[col] = np.where(
            ((target_df[col] > 0) & (target_df["date"] == target_df["target_month"])),
            1,
            target_df[col],
        )
        target_df[col] = np.where(
            ((target_df[col] > 0) & (target_df["date"] == target_df['gap_month'])),
            2,
            target_df[col],
        )

    if keep_gap_month_churns:
        # Convert all values that are 2 (gap month churns) into 1 (real churns)
        target_df[col] = np.where(target_df[col] == 2, 1, target_df[col])

    # Add categorical data for churned month
    target_df = add_month_churned_column(target_df, target_col)
    target_df['WHICH_MONTH_CHURNED'] = target_df['WHICH_MONTH_CHURNED'].astype("category")

    # Exclude the records that are still marked as 2 (gap month churns) if keep_gap_month_churns is False
    target_df = target_df[
        (target_df["NUM_DAYS_LINE_TYPE_FIXE_POST_DEA"] != 2)
        & (target_df["date"] != target_df['gap_month'])
    ][["customer_id", "WHICH_MONTH_CHURNED"] + target_col]

    #target_df = add_month_churned_column(df)

    target_df[target_col] = target_df[target_col].astype("int")

    return target_df

def add_month_churned_column(df: pd.DataFrame, target_col: list[str]) -> pd.DataFrame:
    """ 
    Add a column indicating the month in which churn occurred.
    """
    month_to_letter = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 
                       7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L',
                       }

    # For each target column, determine the churn month
    for col in target_col:
        df['WHICH_MONTH_CHURNED'] = np.where(
            df[col] == 1, df['target_month'].dt.month.map(month_to_letter),
            np.nan  # Assign NaN if not churned in target month
        )

    return df
```


```python
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
    
```


```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb


def modeling(
    features: pd.DataFrame, target: pd.Series, lightgbm_params, logger
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

    # Convert Categoricals
    features['WHICH_MONTH_CHURNED'] = features['WHICH_MONTH_CHURNED'].astype('category')

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

    return model

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
import os


def evaluation(
    model,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    logger,
    save_curves_path,
) -> dict[str, str]:
    """
    Assesses trained model's performance using a test dataset and computes metrics like accuracy, precision, recall, and ROC-AUC.

    Returns:
        dict: Dictionary with key performance metrics of the model.
    """
    logger.info(f"Started evaluation for {model}")
    preds = model.predict_proba(features_test)[:, 1]

    # Plotting
    logger.info("Generating plots")

    generate_evaluation_curves(
        model,
        preds,
        target_test,
        save_curves_path,
    )

    precision, recall, _ = precision_recall_curve(target_test, preds)
    model_metrics = {
        "Precision Curve": precision,
        "Recall Curve": recall,
        # "ROC AUC": roc_auc
    }

    # Calculate Precision in the First Decile
    precision_decile = calculate_precision_first_decile(target_test, preds)
    logger.info(f"Precision in the first decile: {precision_decile:.2f}")

    # Calculate Uplift for Each Decile
    uplift_by_decile = calculate_uplift(target_test, preds)
    logger.info("Uplift by decile:")
    logger.info(uplift_by_decile)

    logger.info("Completed evaluation!")

    model_name = type(model["lightgbm"]).__name__

    if model_name == "LGBMClassifier":
        feature_importance = get_feature_importance_lightgbm(model, features_test)
    else:
        feature_importance = get_feature_importance_logistic_regression(
            model, features_test
        )

    logger.info("Feature importance")
    logger.info(feature_importance.head(10))

    return model_metrics, precision_decile, uplift_by_decile, feature_importance


def calculate_precision_first_decile(target, y_pred_proba):
    """
    Calculate the precision in the first decile of predictions.

    Args:
    - y_true (array-like): True labels.
    - y_pred_proba (array-like): Predicted probabilities.

    Returns:
    - precision_decile (float): Precision in the first decile.
    """
    data = pd.DataFrame({"y_true": target, "y_pred_proba": y_pred_proba})
    data_sorted = data.sort_values(by="y_pred_proba", ascending=False)
    decile_cutoff = int(len(data_sorted) * 0.1)
    first_decile = data_sorted.head(decile_cutoff)
    true_positives = first_decile["y_true"].sum()
    precision_decile = true_positives / decile_cutoff

    return precision_decile


def calculate_uplift(target, y_pred_proba):
    """
    Calculate the uplift for each decile.

    Args:
    - y_true (array-like): True labels.
    - y_pred_proba (array-like): Predicted probabilities.

    Returns:
    - pd.Series: Uplift for each decile.
    """
    data = pd.DataFrame({"y_true": target, "y_pred_proba": y_pred_proba})
    data_sorted = data.sort_values(by="y_pred_proba", ascending=False)
    try:
        data_sorted["decile"] = pd.qcut(data_sorted["y_pred_proba"], q=10, labels=list(reversed(range(10))))
    except ValueError:
        # Handle fewer bins than desired
        bins = pd.qcut(data_sorted["y_pred_proba"], q=137, duplicates='drop').categories
        unique_bins = len(bins)
        labels = list(reversed(range(unique_bins-1)))  # Create one less label
        data_sorted["decile"] = pd.qcut(data_sorted["y_pred_proba"], q=unique_bins, labels=labels, duplicates='drop')
    decile_churn_rate = data_sorted.groupby("decile", observed=True)["y_true"].mean()

    overall_churn_rate = data["y_true"].mean()
    uplift = decile_churn_rate / overall_churn_rate

    # return by ascending deciles
    return uplift.sort_index(ascending=False)


def generate_evaluation_curves(
    model: str, y_pred, y_test, save_curves_path: str = None
):
    """
    Generate ROC and Precision-Recall curves for a binary classification model
    and save them in a single figure.

    Parameters:
    - model_name (str): Name of the model for labeling the curves.
    - y_pred (array-like): Predicted probabilities or scores.
    - y_test (array-like): True labels.
    - save_curves_path (str, optional): Directory to save the generated figure.
    If None, the figure will not be saved.

    Returns:
    - None
    """

    # Create a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y_%m_%d")
    model_type = type(model[-1]).__name__  # Assuming 'model' is your pipeline

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f}) - {model_type}")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.2f}) - {model_type}")
    plt.xlim([-0.005, 1.0])  # Adjusted to start slightly before 0 for a clearer view
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")

    plt.tight_layout()

    if save_curves_path:
        # Define the filename with a timestamp
        figure_filename = f"Evaluation_Curves_{timestamp}.png"
        figure_path = os.path.join(save_curves_path, figure_filename)

        plt.savefig(figure_path)

    plt.show()


def get_feature_importance_logistic_regression(model, features):
    """
    Get feature importance for Logistic Regression model.

    Args:
    - model: Trained Logistic Regression model.
    - feature_names (list or array-like): List of feature names.

    Returns:
    - pd.DataFrame: DataFrame containing feature names and their corresponding coefficients.
    """
    feature_names = features.columns
    # Extract coefficients
    lr_model = model.named_steps["logistic_regression"]
    coefficients = lr_model.coef_[0]  # for Logistic Regression

    # Create a DataFrame for easy visualization
    feature_importance = pd.DataFrame(
        {"Feature": feature_names, "Coefficient": coefficients}
    )

    # Sort by absolute value of coefficients in descending order
    feature_importance = feature_importance.reindex(
        feature_importance.Coefficient.abs().sort_values(ascending=False).index
    )

    return feature_importance


def get_feature_importance_lightgbm(model, features):
    """
    Get feature importance for Lightgbm model.

    Args:
    - model: Trained Lightgbm model.
    - feature_names (list or array-like): List of feature names.

    Returns:
    - pd.DataFrame: DataFrame containing feature names and their corresponding coefficients.
    """
    feature_names = features.columns
    # Extract coefficients
    lr_model = model.named_steps["lightgbm"]
    coefficients = lr_model.feature_importances_

    # Create a DataFrame for easy visualization
    feature_importance = pd.DataFrame(
        {"Feature": feature_names, "Coefficient": coefficients}
    )

    # Sort by absolute value of coefficients in descending order
    feature_importance = feature_importance.reindex(
        feature_importance.Coefficient.abs().sort_values(ascending=False).index
    )

    return feature_importance
```


```python
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
```

    INFO - Started querying data
    INFO - Data succesfully retrieved! Length: 1019032
    INFO - Starting cleaning data
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Dropping column Import_Rest_quota_disp
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Dropping column NUM_FIX_PORT
    INFO - Dropping column NUM_FIX_PORT_LAST_1_MONTH
    INFO - Dropping column NUM_FIX_PORT_LAST_3_MONTHS
    INFO - Dropping column NUM_FIX_PORT_LAST_6_MONTHS
    INFO - Should fillna
    INFO - Dropping column NUM_MOB_PORT
    INFO - Dropping column NUM_MOB_PORT_LAST_1_MONTH
    INFO - Dropping column NUM_MOB_PORT_LAST_3_MONTHS
    INFO - Dropping column NUM_MOB_PORT_LAST_6_MONTHS
    INFO - Dropping column NUM_MOB_PORT_REQS_LAST_1_MONTH
    INFO - Dropping column NUM_MOB_PORT_REQS_LAST_3_MONTHS
    INFO - Dropping column NUM_MOB_PORT_REQS_LAST_6_MONTHS
    INFO - Dropping column NUM_MOB_PORT_TRANS_CURR
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Dropping column NUM_PORT_OPER_DONO_MASMOVIL_GRP_LAST_THREE_MONTHS
    INFO - Dropping column NUM_PORT_REQS_OPER_DONO_MASMOVIL_GRP_LAST_THREE_MONTHS
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Dropping column cust_max_days_between_fix_port
    INFO - Dropping column cust_max_days_between_mob_port
    INFO - Dropping column cust_max_months_between_fix_port
    INFO - Dropping column cust_max_months_between_mob_port
    INFO - Dropping column cust_min_days_between_fix_port
    INFO - Dropping column cust_min_days_between_mob_port
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Completed cleaning data!
    INFO -   customer_id MONTH  YEAR  dif_pago_final_prev_month  \
    0     7195022    01  2022                      -0.07   
    1     6198249    04  2022                       0.00   
    2     6404316    07  2022                      17.08   
    3     6993728    01  2022                      -0.30   
    4     4142131    04  2022                       0.60   
    
       dif_pago_final_prev_2_month  dif_pago_final_prev_3_month  \
    0                        -1.00                        -9.71   
    1                        -0.30                         0.00   
    2                       -11.38                         3.00   
    3                         1.30                        53.13   
    4                        -0.97                       -18.76   
    
       dif_consumo_prev_month  dif_consumo_prev_2_month  dif_consumo_prev_3_month  \
    0                  -33.77                   -136.56                    -45.00   
    1                  -18.40                    -75.33                    -82.46   
    2                   24.29                     11.09                     71.72   
    3                 -170.49                   -161.30                     45.04   
    4                   59.01                   -224.95                   -163.23   
    
       dif_discount_prev_month  dif_discount_prev_2_month  \
    0                    33.70                     135.73   
    1                    18.40                      75.03   
    2                  -149.83                    -151.01   
    3                   169.89                     161.60   
    4                   -58.41                     223.97   
    
       dif_discount_prev_3_month  dif_periodica_prev_month  \
    0                      38.21                      0.00   
    1                      82.46                      0.00   
    2                    -197.26                      8.54   
    3                     -53.42                      0.30   
    4                     144.46                      0.00   
    
       dif_periodica_prev_2_month  dif_periodica_prev_3_month  \
    0                       -0.17                       -2.92   
    1                        0.00                        0.00   
    2                        8.54                        8.54   
    3                        1.00                        1.00   
    4                        0.00                        0.00   
    
       dif_aperiodica_prev_month  dif_aperiodica_prev_2_month  \
    0                       0.00                          0.0   
    1                       0.00                          0.0   
    2                     134.08                        120.0   
    3                       0.00                          0.0   
    4                       0.00                          0.0   
    
       dif_aperiodica_prev_3_month  dif_ajuste_prev_month  \
    0                          0.0                    0.0   
    1                          0.0                    0.0   
    2                        120.0                    0.0   
    3                         60.5                    0.0   
    4                          0.0                    0.0   
    
       dif_ajuste_prev_2_month  dif_ajuste_prev_3_month  \
    0                      0.0                      0.0   
    1                      0.0                      0.0   
    2                      0.0                      0.0   
    3                      0.0                      0.0   
    4                      0.0                      0.0   
    
       service_mobile_pending_install  service_fix_pending_install  \
    0                               0                            0   
    1                               0                            0   
    2                               0                            0   
    3                               0                            0   
    4                               0                            0   
    
       service_mobile_cancelled  service_fix_cancelled  \
    0                         0                      0   
    1                         0                      0   
    2                         0                      0   
    3                         0                      0   
    4                         0                      0   
    
       service_mobile_pending_install_3month  service_fix_pending_install_3month  \
    0                                      0                                   0   
    1                                      0                                   0   
    2                                      0                                   0   
    3                                      0                                   0   
    4                                      0                                   0   
    
       service_mobile_cancelled_3month  service_fix_cancelled_3month  \
    0                                0                             0   
    1                                0                             0   
    2                                0                             0   
    3                                0                             0   
    4                                0                             0   
    
       service_mobile_pending_install_6month  service_fix_pending_install_6month  \
    0                                      0                                   0   
    1                                      0                                   0   
    2                                      0                                   0   
    3                                      0                                   0   
    4                                      0                                   0   
    
       service_mobile_cancelled_6month  service_fix_cancelled_6month  \
    0                                2                             1   
    1                                0                             0   
    2                                0                             0   
    3                                0                             0   
    4                                0                             0   
    
       NUM_DAYS_ACT  order_mobile_from_new_alta  MIN_DAYS_PERM_CURR  \
    0           147                           0                 212   
    1           186                           0                 179   
    2            39                           0                 105   
    3           341                           0                  22   
    4             3                           0                 245   
    
       PREV_FINISHED_PERM  NUM_DAYS_LINE_TYPE_MAIN_POST_ACT  \
    0                   0                               147   
    1                   2                               186   
    2                   0                               253   
    3                   0                               341   
    4                   2                                 3   
    
       Import_Rest_quota_disp  LINE_TYPE_FIXE_POST_ACT_LAST_DT_D  \
    0                     NaN                                 -5   
    1                     NaN                                 -7   
    2                     NaN                                 -2   
    3                     NaN                                -12   
    4                54.54625                                 -1   
    
       MAX_DAYS_PERM_CURR  MAX_PENALTY_AMOUNT_CURR  MIN_PENALTY_AMOUNT_CURR  \
    0                 234                    120.0                    60.00   
    1                 179                     60.0                    60.00   
    2                 105                     60.0                    60.00   
    3                  24                    120.0                    90.15   
    4                 363                    144.0                    80.00   
    
       MM_GROUP_MOB_PORT  NUM_CALL_INTR_CURR  NUM_CALL_NATR_CURR  \
    0                  0                   0                 289   
    1                  0                   0                 131   
    2                  2                   0                  99   
    3                  0                   0                 357   
    4                  3                   0                 396   
    
       NUM_CALL_OWNN_CURR  NUM_CALL_SERV_FIXE_CURR  NUM_CALL_SERV_MOBI_CURR  \
    0                  13                        0                      302   
    1                 189                        0                      320   
    2                 230                        0                      329   
    3                  79                        0                      436   
    4                 446                        0                      842   
    
       NUM_CALL_SERV_UNKN_CURR  NUM_CALL_TYPE_IN_CURR  NUM_CALL_TYPE_IN_INTR_CURR  \
    0                        0                    169                           0   
    1                        0                    138                           0   
    2                        0                    175                           0   
    3                        0                    232                           0   
    4                        0                    379                           0   
    
       NUM_CALL_TYPE_IN_NATR_CURR  NUM_CALL_TYPE_IN_OWNN_CURR  \
    0                         163                           6   
    1                          55                          83   
    2                          59                         116   
    3                         203                          29   
    4                         169                         210   
    
       NUM_CALL_TYPE_IN_SERV_FIXE_CURR  NUM_CALL_TYPE_IN_SERV_MOBI_CURR  \
    0                                0                              169   
    1                                0                              138   
    2                                0                              175   
    3                                0                              232   
    4                                0                              379   
    
       NUM_CALL_TYPE_IN_SERV_UNKN_CURR  NUM_CALL_TYPE_IN_WEEKEND_CURR  \
    0                                0                             40   
    1                                0                             34   
    2                                0                             41   
    3                                0                             54   
    4                                0                             77   
    
       NUM_CALL_TYPE_IN_WEEK_CURR  NUM_CALL_TYPE_OUT_CURR  \
    0                         129                     133   
    1                         104                     182   
    2                         134                     154   
    3                         178                     204   
    4                         302                     463   
    
       NUM_CALL_TYPE_OUT_INTR_CURR  NUM_CALL_TYPE_OUT_NATR_CURR  \
    0                            0                          126   
    1                            0                           76   
    2                            0                           40   
    3                            0                          154   
    4                            0                          227   
    
       NUM_CALL_TYPE_OUT_OWNN_CURR  NUM_CALL_TYPE_OUT_SERV_FIXE_CURR  \
    0                            7                                 0   
    1                          106                                 0   
    2                          114                                 0   
    3                           50                                 0   
    4                          236                                 0   
    
       NUM_CALL_TYPE_OUT_SERV_MOBI_CURR  NUM_CALL_TYPE_OUT_SERV_UNKN_CURR  \
    0                               133                                 0   
    1                               182                                 0   
    2                               154                                 0   
    3                               204                                 0   
    4                               463                                 0   
    
       NUM_CALL_TYPE_OUT_WEEKEND_CURR  NUM_CALL_TYPE_OUT_WEEK_CURR  \
    0                              49                           84   
    1                              53                          129   
    2                              44                          110   
    3                              70                          134   
    4                             105                          358   
    
       NUM_CUSTOMER_BUNDLE_2P  NUM_CUSTOMER_BUNDLE_FMC  NUM_CUSTOMER_BUNDLE_MO  \
    0                       0                        1                       0   
    1                       0                        1                       0   
    2                       0                        1                       0   
    3                       0                        1                       0   
    4                       0                        1                       0   
    
       NUM_DAYS_LINE_TYPE_FIXE_POST_ACT  NUM_DISC_ACTI  NUM_DISC_CURR  \
    0                               147              0             34   
    1                               186              0             22   
    2                                39              1             32   
    3                               341              0             21   
    4                                 3             17             54   
    
       NUM_DUR_OBJ  NUM_FIX_PORT  NUM_FIX_PORT_LAST_1_MONTH  \
    0            0             0                          0   
    1            0             0                          0   
    2            0             0                          0   
    3            0             0                          0   
    4            0             0                          0   
    
       NUM_FIX_PORT_LAST_3_MONTHS  NUM_FIX_PORT_LAST_6_MONTHS  NUM_INTR_CURR  \
    0                           0                           0              0   
    1                           0                           0              2   
    2                           0                           0              0   
    3                           0                           0              0   
    4                           0                           0              0   
    
       NUM_LINES_POST_ACTI  NUM_LINES_POST_CURR  NUM_LINES_TOTA  \
    0                    0                    4               0   
    1                    0                    4               0   
    2                    0                    4               0   
    3                    0                    4               0   
    4                    5                    6               6   
    
       NUM_LINE_TYPE_ADDI_CURR  NUM_LINE_TYPE_ADDI_POST_ACTI  \
    0                        1                             0   
    1                        0                             0   
    2                        0                             0   
    3                        1                             0   
    4                        2                             2   
    
       NUM_LINE_TYPE_ADDI_POST_CURR  NUM_LINE_TYPE_ADDI_TOTA  \
    0                             1                        0   
    1                             0                        0   
    2                             0                        0   
    3                             1                        0   
    4                             2                        3   
    
       NUM_LINE_TYPE_FIXE_CURR  NUM_LINE_TYPE_FIXE_POST_ACTI  \
    0                        2                             0   
    1                        2                             0   
    2                        2                             0   
    3                        2                             0   
    4                        2                             1   
    
       NUM_LINE_TYPE_FIXE_TOTA  NUM_LINE_TYPE_MAIN_CURR  \
    0                        0                        3   
    1                        0                        4   
    2                        0                        4   
    3                        0                        3   
    4                        1                        4   
    
       NUM_LINE_TYPE_MAIN_POST_ACTI  NUM_LINE_TYPE_MAIN_POST_CURR  \
    0                             0                             1   
    1                             0                             2   
    2                             0                             2   
    3                             0                             1   
    4                             2                             2   
    
       NUM_LINE_TYPE_MAIN_TOTA  NUM_MOB_PORT  NUM_MOB_PORT_LAST_1_MONTH  \
    0                        0             2                          0   
    1                        0             2                          0   
    2                        0             2                          0   
    3                        0             2                          0   
    4                        2             2                          0   
    
       NUM_MOB_PORT_LAST_3_MONTHS  NUM_MOB_PORT_LAST_6_MONTHS  \
    0                           0                           2   
    1                           0                           0   
    2                           0                           0   
    3                           0                           0   
    4                           0                           0   
    
       NUM_MOB_PORT_REQS_LAST_1_MONTH  NUM_MOB_PORT_REQS_LAST_3_MONTHS  \
    0                               0                                0   
    1                               0                                0   
    2                               0                                0   
    3                               0                                0   
    4                               4                                4   
    
       NUM_MOB_PORT_REQS_LAST_6_MONTHS  NUM_MOB_PORT_TRANS_CURR  NUM_NATR_CURR  \
    0                                2                        0            310   
    1                                0                        0            408   
    2                                0                        0            172   
    3                                0                        0            238   
    4                                4                        0            917   
    
       NUM_NETW_OSPN_CURR  NUM_NETW_RAAS_CURR  NUM_NETW_ROAM_CURR  \
    0                 298                 185                   0   
    1                 327                 107                   2   
    2                  90                   0                   0   
    3                 229                  21                   0   
    4                 881                  88                   0   
    
       NUM_NETW_TMEN_CURR  NUM_NETW_VODA_CURR  NUM_OWNN_CURR  NUM_PERM_CURR  \
    0                  12                   0            190              3   
    1                  81                   0            286              6   
    2                  82                   0            101              1   
    3                   9                   0            164              2   
    4                  36                   0            704              4   
    
       NUM_PORT_OPER_DONO_MASMOVIL_GRP_LAST_THREE_MONTHS  \
    0                                                  0   
    1                                                  0   
    2                                                  0   
    3                                                  0   
    4                                                  0   
    
       NUM_PORT_REQS_OPER_DONO_MASMOVIL_GRP_LAST_THREE_MONTHS  NUM_PREV_OBJ  \
    0                                                  0                  0   
    1                                                  0                  0   
    2                                                  0                  0   
    3                                                  0                  2   
    4                                                  4                  0   
    
       NUM_RAAS_CURR  NUM_SECS_SERV_FIXE_CURR  NUM_SECS_SERV_MOBI_CURR  \
    0            185                        0                    50325   
    1            107                        0                    79089   
    2              0                        0                    51388   
    3             21                        0                    34393   
    4             88                        0                    92313   
    
       NUM_SECS_TYPE_IN_CURR  NUM_SECS_TYPE_IN_SERV_FIXE_CURR  \
    0                  32546                                0   
    1                  32562                                0   
    2                  25411                                0   
    3                  10762                                0   
    4                  36376                                0   
    
       NUM_SECS_TYPE_IN_SERV_MOBI_CURR  NUM_SECS_TYPE_IN_WEEKEND_CURR  \
    0                            32546                           3494   
    1                            32562                           6998   
    2                            25411                           7401   
    3                            10762                           4718   
    4                            36376                           5087   
    
       NUM_SECS_TYPE_IN_WEEK_CURR  NUM_SECS_TYPE_OUT_CURR  \
    0                       29052                   17779   
    1                       25564                   46527   
    2                       18010                   25977   
    3                        6044                   23631   
    4                       31289                   55937   
    
       NUM_SECS_TYPE_OUT_SERV_FIXE_CURR  NUM_SECS_TYPE_OUT_SERV_MOBI_CURR  \
    0                                 0                             17779   
    1                                 0                             46527   
    2                                 0                             25977   
    3                                 0                             23631   
    4                                 0                             55937   
    
       NUM_SECS_TYPE_OUT_WEEKEND_CURR  NUM_SECS_TYPE_OUT_WEEK_CURR  \
    0                            2038                        15741   
    1                           10228                        36299   
    2                            5952                        20025   
    3                            8264                        15367   
    4                            9819                        46118   
    
       ORDER_FIX_FROM_NEW_ALTA  cust_days_since_last_fix_port  \
    0                        0                            152   
    1                        0                            654   
    2                        0                            558   
    3                        0                            341   
    4                        0                              0   
    
       cust_days_since_last_mob_port  cust_max_days_between_fix_port  \
    0                            148                            <NA>   
    1                            653                               4   
    2                            586                              32   
    3                            342                            <NA>   
    4                            368                            <NA>   
    
       cust_max_days_between_mob_port  cust_max_months_between_fix_port  \
    0                             361                              <NA>   
    1                             543                                 0   
    2                            1000                                 1   
    3                             709                              <NA>   
    4                             812                              <NA>   
    
       cust_max_months_between_mob_port  cust_min_days_between_fix_port  \
    0                                12                            <NA>   
    1                                18                               4   
    2                                33                              32   
    3                                23                            <NA>   
    4                                26                            <NA>   
    
       cust_min_days_between_mob_port  cust_n_fix_port  cust_n_fix_recent_port  \
    0                               0                1                       1   
    1                               0                2                       2   
    2                               0                1                       1   
    3                               0                1                       1   
    4                               2                0                       0   
    
       order_mobile_from_migra_pre_to_post  pago_final_0  consumo_0  aperiodica_0  \
    0                                    0       27.9999    98.3143           0.0   
    1                                    0       50.0449   272.0501           0.0   
    2                                    0       42.0000   169.3614         120.0   
    3                                    0       90.2900   183.7272           0.0   
    4                                    0      103.3928   368.5390           0.0   
    
       periodica_0  discount_0  ajuste_0  NUM_GB_OWNN_CURR  NUM_GB_2G_CURR  \
    0     162.1334   -232.4478       0.0          1.754356        0.000848   
    1     129.5957   -351.6009       0.0          2.481070        0.000196   
    2      93.2966   -340.6580       0.0          0.515946        0.000000   
    3     203.4705   -296.9077       0.0          0.737249        0.000899   
    4     197.4279   -462.5741       0.0         40.055186        0.000277   
    
       NUM_GB_3G_CURR  NUM_GB_4G_CURR  NUM_GB_5G_CURR  NUM_SESS_CURR  \
    0        0.304749       50.741414        0.000000          12255   
    1        0.095219        4.979776        0.013595           6136   
    2        0.013818        0.745551        0.000000           2788   
    3        0.039029        1.749069        0.000000           4807   
    4        0.429209       50.262257        1.957807          12573   
    
       NUM_SECS_CURR  PERC_SECS_TYPE_IN_CURR  PERC_SECS_TYPE_OUT_CURR  \
    0          50325               64.671634                35.328366   
    1          79089               41.171339                58.828661   
    2          51388               49.449288                50.550712   
    3          34393               31.291251                68.708749   
    4          92313               39.405068                60.594932   
    
       PERC_SECS_OWNN_CURR  PERC_SECS_NATR_CURR  PERC_SECS_SERV_MOBI_CURR  \
    0             1.639344            98.360656                     100.0   
    1            53.341173            46.658827                     100.0   
    2            76.449755            23.550245                     100.0   
    3            15.040851            84.959149                     100.0   
    4            54.540531            45.459469                     100.0   
    
       PERC_SECS_TYPE_IN_OWNN_CURR  PERC_SECS_TYPE_OUT_OWNN_CURR  \
    0                    66.303030                     33.696970   
    1                    55.429398                     44.570602   
    2                    52.504709                     47.495291   
    3                    12.777885                     87.222115   
    4                    40.855645                     59.144355   
    
       PERC_SECS_TYPE_IN_NATR_CURR  PERC_SECS_TYPE_OUT_NATR_CURR  \
    0                    64.644444                     35.355556   
    1                    24.871281                     75.128719   
    2                    39.530656                     60.469344   
    3                    34.568789                     65.431211   
    4                    37.664721                     62.335279   
    
       NUM_PLAT_GMM_CURR  NUM_PLAT_OMV_CURR  NUM_NETW_OWNN_CURR  NUM_CALL_CURR  \
    0                685                  0                 190            302   
    1                810                  0                 286            320   
    2                273                  0                 101            329   
    3                423                  0                 164            436   
    4               1759                  0                 704            842   
    
       PERC_CALL_TYPE_IN_CURR  PERC_CALL_TYPE_OUT_CURR  PERC_CALL_OWNN_CURR  \
    0               55.960265                44.039735             4.304636   
    1               43.125000                56.875000            59.062500   
    2               53.191489                46.808511            69.908815   
    3               53.211009                46.788991            18.119266   
    4               45.011876                54.988124            52.969121   
    
       PERC_CALL_NATR_CURR  NUM_CALL_WEEK_CURR  NUM_CALL_WEEKEND_CURR  \
    0            95.695364                 213                     89   
    1            40.937500                 233                     87   
    2            30.091185                 244                     85   
    3            81.880734                 312                    124   
    4            47.030879                 660                    182   
    
       NUM_SECS_WEEK_CURR  NUM_SECS_WEEKEND_CURR  NUM_CALL_WEEK  NUM_CALL_WEEKEND  \
    0               44793                   5532            148                69   
    1               61863                  17226            180                72   
    2               38035                  13353            202                75   
    3               21411                  12982            237                91   
    4               77407                  14906            565               147   
    
       NUM_DAYS_LINE_TYPE_FIXE_POST_DEA  
    0                              <NA>  
    1                              <NA>  
    2                              <NA>  
    3                              <NA>  
    4                                 3  
    INFO - Starting feature computation
    INFO - Train computation from 2022-07-01 to 2022-12-01. Target for 2023-02-01 00:00:00
    INFO - Test computation from 2022-08-01 00:00:00 to 2023-01-01 00:00:00. Target for 2023-03-01 00:00:00
    /tmp/ipykernel_482/2080027226.py:180: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    INFO - Removing 7999 previous churned users from test set
    INFO - Unique customers in train: 50856
    INFO - Unique customers in test: 43326
    INFO - Starting features and target computation
    INFO - Initial number of features passed: 177
    INFO - Starting computation
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    INFO - Final number of features computed: 358
    INFO - Length train data: 50856
    INFO - Length test data: 41347
    INFO - Computation done!
    INFO - Features: ['pago_final_0', 'consumo_0', 'aperiodica_0', 'periodica_0', 'discount_0', 'ajuste_0', 'NUM_GB_OWNN_CURR', 'NUM_GB_2G_CURR', 'NUM_GB_3G_CURR', 'NUM_GB_4G_CURR', 'NUM_GB_5G_CURR', 'NUM_SESS_CURR', 'NUM_SECS_CURR', 'PERC_SECS_TYPE_IN_CURR', 'PERC_SECS_TYPE_OUT_CURR', 'PERC_SECS_OWNN_CURR', 'PERC_SECS_NATR_CURR', 'PERC_SECS_SERV_MOBI_CURR', 'PERC_SECS_TYPE_IN_OWNN_CURR', 'PERC_SECS_TYPE_OUT_OWNN_CURR', 'PERC_SECS_TYPE_IN_NATR_CURR', 'PERC_SECS_TYPE_OUT_NATR_CURR', 'NUM_PLAT_GMM_CURR', 'NUM_PLAT_OMV_CURR', 'NUM_NETW_OWNN_CURR', 'NUM_CALL_CURR', 'PERC_CALL_TYPE_IN_CURR', 'PERC_CALL_TYPE_OUT_CURR', 'PERC_CALL_OWNN_CURR', 'PERC_CALL_NATR_CURR', 'NUM_CALL_WEEK_CURR', 'NUM_CALL_WEEKEND_CURR', 'NUM_SECS_WEEK_CURR', 'NUM_SECS_WEEKEND_CURR', 'NUM_CALL_WEEK', 'NUM_CALL_WEEKEND', 'NUM_DAYS_ACT', 'order_mobile_from_new_alta', 'MIN_DAYS_PERM_CURR', 'PREV_FINISHED_PERM', 'NUM_DAYS_LINE_TYPE_MAIN_POST_ACT', 'Import_Rest_quota_disp', 'LINE_TYPE_FIXE_POST_ACT_LAST_DT_D', 'MAX_DAYS_PERM_CURR', 'MAX_PENALTY_AMOUNT_CURR', 'MIN_PENALTY_AMOUNT_CURR', 'MM_GROUP_MOB_PORT', 'NUM_CALL_INTR_CURR', 'NUM_CALL_NATR_CURR', 'NUM_CALL_OWNN_CURR', 'NUM_CALL_SERV_FIXE_CURR', 'NUM_CALL_SERV_MOBI_CURR', 'NUM_CALL_SERV_UNKN_CURR', 'NUM_CALL_TYPE_IN_CURR', 'NUM_CALL_TYPE_IN_INTR_CURR', 'NUM_CALL_TYPE_IN_NATR_CURR', 'NUM_CALL_TYPE_IN_OWNN_CURR', 'NUM_CALL_TYPE_IN_SERV_FIXE_CURR', 'NUM_CALL_TYPE_IN_SERV_MOBI_CURR', 'NUM_CALL_TYPE_IN_SERV_UNKN_CURR', 'NUM_CALL_TYPE_IN_WEEKEND_CURR', 'NUM_CALL_TYPE_IN_WEEK_CURR', 'NUM_CALL_TYPE_OUT_CURR', 'NUM_CALL_TYPE_OUT_INTR_CURR', 'NUM_CALL_TYPE_OUT_NATR_CURR', 'NUM_CALL_TYPE_OUT_OWNN_CURR', 'NUM_CALL_TYPE_OUT_SERV_FIXE_CURR', 'NUM_CALL_TYPE_OUT_SERV_MOBI_CURR', 'NUM_CALL_TYPE_OUT_SERV_UNKN_CURR', 'NUM_CALL_TYPE_OUT_WEEKEND_CURR', 'NUM_CALL_TYPE_OUT_WEEK_CURR', 'NUM_CUSTOMER_BUNDLE_2P', 'NUM_CUSTOMER_BUNDLE_FMC', 'NUM_CUSTOMER_BUNDLE_MO', 'NUM_DAYS_LINE_TYPE_FIXE_POST_ACT', 'NUM_DISC_ACTI', 'NUM_DISC_CURR', 'NUM_DUR_OBJ', 'NUM_FIX_PORT', 'NUM_FIX_PORT_LAST_1_MONTH', 'NUM_FIX_PORT_LAST_3_MONTHS', 'NUM_FIX_PORT_LAST_6_MONTHS', 'NUM_INTR_CURR', 'NUM_LINES_POST_ACTI', 'NUM_LINES_POST_CURR', 'NUM_LINES_TOTA', 'NUM_LINE_TYPE_ADDI_CURR', 'NUM_LINE_TYPE_ADDI_POST_ACTI', 'NUM_LINE_TYPE_ADDI_POST_CURR', 'NUM_LINE_TYPE_ADDI_TOTA', 'NUM_LINE_TYPE_FIXE_CURR', 'NUM_LINE_TYPE_FIXE_POST_ACTI', 'NUM_LINE_TYPE_FIXE_TOTA', 'NUM_LINE_TYPE_MAIN_CURR', 'NUM_LINE_TYPE_MAIN_POST_ACTI', 'NUM_LINE_TYPE_MAIN_POST_CURR', 'NUM_LINE_TYPE_MAIN_TOTA', 'NUM_MOB_PORT', 'NUM_MOB_PORT_LAST_1_MONTH', 'NUM_MOB_PORT_LAST_3_MONTHS', 'NUM_MOB_PORT_LAST_6_MONTHS', 'NUM_MOB_PORT_REQS_LAST_1_MONTH', 'NUM_MOB_PORT_REQS_LAST_3_MONTHS', 'NUM_MOB_PORT_REQS_LAST_6_MONTHS', 'NUM_MOB_PORT_TRANS_CURR', 'NUM_NATR_CURR', 'NUM_NETW_OSPN_CURR', 'NUM_NETW_RAAS_CURR', 'NUM_NETW_ROAM_CURR', 'NUM_NETW_TMEN_CURR', 'NUM_NETW_VODA_CURR', 'NUM_OWNN_CURR', 'NUM_PERM_CURR', 'NUM_PORT_OPER_DONO_MASMOVIL_GRP_LAST_THREE_MONTHS', 'NUM_PORT_REQS_OPER_DONO_MASMOVIL_GRP_LAST_THREE_MONTHS', 'NUM_PREV_OBJ', 'NUM_RAAS_CURR', 'NUM_SECS_SERV_FIXE_CURR', 'NUM_SECS_SERV_MOBI_CURR', 'NUM_SECS_TYPE_IN_CURR', 'NUM_SECS_TYPE_IN_SERV_FIXE_CURR', 'NUM_SECS_TYPE_IN_SERV_MOBI_CURR', 'NUM_SECS_TYPE_IN_WEEKEND_CURR', 'NUM_SECS_TYPE_IN_WEEK_CURR', 'NUM_SECS_TYPE_OUT_CURR', 'NUM_SECS_TYPE_OUT_SERV_FIXE_CURR', 'NUM_SECS_TYPE_OUT_SERV_MOBI_CURR', 'NUM_SECS_TYPE_OUT_WEEKEND_CURR', 'NUM_SECS_TYPE_OUT_WEEK_CURR', 'ORDER_FIX_FROM_NEW_ALTA', 'cust_days_since_last_fix_port', 'cust_days_since_last_mob_port', 'cust_max_days_between_fix_port', 'cust_max_days_between_mob_port', 'cust_max_months_between_fix_port', 'cust_max_months_between_mob_port', 'cust_min_days_between_fix_port', 'cust_min_days_between_mob_port', 'cust_n_fix_port', 'cust_n_fix_recent_port', 'order_mobile_from_migra_pre_to_post', 'dif_pago_final_prev_month', 'dif_pago_final_prev_2_month', 'dif_pago_final_prev_3_month', 'dif_consumo_prev_month', 'dif_consumo_prev_2_month', 'dif_consumo_prev_3_month', 'dif_discount_prev_month', 'dif_discount_prev_2_month', 'dif_discount_prev_3_month', 'dif_periodica_prev_month', 'dif_periodica_prev_2_month', 'dif_periodica_prev_3_month', 'dif_aperiodica_prev_month', 'dif_aperiodica_prev_2_month', 'dif_aperiodica_prev_3_month', 'dif_ajuste_prev_month', 'dif_ajuste_prev_2_month', 'dif_ajuste_prev_3_month', 'service_mobile_pending_install', 'service_fix_pending_install', 'service_mobile_cancelled', 'service_fix_cancelled', 'service_mobile_pending_install_3month', 'service_fix_pending_install_3month', 'service_mobile_cancelled_3month', 'service_fix_cancelled_3month', 'service_mobile_pending_install_6month', 'service_fix_pending_install_6month', 'service_mobile_cancelled_6month', 'service_fix_cancelled_6month', 'pago_final_0_prev_month', 'pago_final_0_avg_3_months', 'pago_final_0_avg_6_months', 'pago_final_0_std_3_months', 'pago_final_0_std_6_months', 'consumo_0_prev_month', 'consumo_0_avg_3_months', 'consumo_0_avg_6_months', 'consumo_0_std_3_months', 'consumo_0_std_6_months', 'aperiodica_0_prev_month', 'aperiodica_0_avg_3_months', 'aperiodica_0_avg_6_months', 'aperiodica_0_std_3_months', 'aperiodica_0_std_6_months', 'periodica_0_prev_month', 'periodica_0_avg_3_months', 'periodica_0_avg_6_months', 'periodica_0_std_3_months', 'periodica_0_std_6_months', 'discount_0_prev_month', 'discount_0_avg_3_months', 'discount_0_avg_6_months', 'discount_0_std_3_months', 'discount_0_std_6_months', 'ajuste_0_prev_month', 'ajuste_0_avg_3_months', 'ajuste_0_avg_6_months', 'ajuste_0_std_3_months', 'ajuste_0_std_6_months', 'NUM_GB_OWNN_CURR_prev_month', 'NUM_GB_OWNN_CURR_avg_3_months', 'NUM_GB_OWNN_CURR_avg_6_months', 'NUM_GB_OWNN_CURR_std_3_months', 'NUM_GB_OWNN_CURR_std_6_months', 'NUM_GB_2G_CURR_prev_month', 'NUM_GB_2G_CURR_avg_3_months', 'NUM_GB_2G_CURR_avg_6_months', 'NUM_GB_2G_CURR_std_3_months', 'NUM_GB_2G_CURR_std_6_months', 'NUM_GB_3G_CURR_prev_month', 'NUM_GB_3G_CURR_avg_3_months', 'NUM_GB_3G_CURR_avg_6_months', 'NUM_GB_3G_CURR_std_3_months', 'NUM_GB_3G_CURR_std_6_months', 'NUM_GB_4G_CURR_prev_month', 'NUM_GB_4G_CURR_avg_3_months', 'NUM_GB_4G_CURR_avg_6_months', 'NUM_GB_4G_CURR_std_3_months', 'NUM_GB_4G_CURR_std_6_months', 'NUM_GB_5G_CURR_prev_month', 'NUM_GB_5G_CURR_avg_3_months', 'NUM_GB_5G_CURR_avg_6_months', 'NUM_GB_5G_CURR_std_3_months', 'NUM_GB_5G_CURR_std_6_months', 'NUM_SESS_CURR_prev_month', 'NUM_SESS_CURR_avg_3_months', 'NUM_SESS_CURR_avg_6_months', 'NUM_SESS_CURR_std_3_months', 'NUM_SESS_CURR_std_6_months', 'NUM_SECS_CURR_prev_month', 'NUM_SECS_CURR_avg_3_months', 'NUM_SECS_CURR_avg_6_months', 'NUM_SECS_CURR_std_3_months', 'NUM_SECS_CURR_std_6_months', 'PERC_SECS_TYPE_IN_CURR_prev_month', 'PERC_SECS_TYPE_IN_CURR_avg_3_months', 'PERC_SECS_TYPE_IN_CURR_avg_6_months', 'PERC_SECS_TYPE_IN_CURR_std_3_months', 'PERC_SECS_TYPE_IN_CURR_std_6_months', 'PERC_SECS_TYPE_OUT_CURR_prev_month', 'PERC_SECS_TYPE_OUT_CURR_avg_3_months', 'PERC_SECS_TYPE_OUT_CURR_avg_6_months', 'PERC_SECS_TYPE_OUT_CURR_std_3_months', 'PERC_SECS_TYPE_OUT_CURR_std_6_months', 'PERC_SECS_OWNN_CURR_prev_month', 'PERC_SECS_OWNN_CURR_avg_3_months', 'PERC_SECS_OWNN_CURR_avg_6_months', 'PERC_SECS_OWNN_CURR_std_3_months', 'PERC_SECS_OWNN_CURR_std_6_months', 'PERC_SECS_NATR_CURR_prev_month', 'PERC_SECS_NATR_CURR_avg_3_months', 'PERC_SECS_NATR_CURR_avg_6_months', 'PERC_SECS_NATR_CURR_std_3_months', 'PERC_SECS_NATR_CURR_std_6_months', 'PERC_SECS_SERV_MOBI_CURR_prev_month', 'PERC_SECS_SERV_MOBI_CURR_avg_3_months', 'PERC_SECS_SERV_MOBI_CURR_avg_6_months', 'PERC_SECS_SERV_MOBI_CURR_std_3_months', 'PERC_SECS_SERV_MOBI_CURR_std_6_months', 'PERC_SECS_TYPE_IN_OWNN_CURR_prev_month', 'PERC_SECS_TYPE_IN_OWNN_CURR_avg_3_months', 'PERC_SECS_TYPE_IN_OWNN_CURR_avg_6_months', 'PERC_SECS_TYPE_IN_OWNN_CURR_std_3_months', 'PERC_SECS_TYPE_IN_OWNN_CURR_std_6_months', 'PERC_SECS_TYPE_OUT_OWNN_CURR_prev_month', 'PERC_SECS_TYPE_OUT_OWNN_CURR_avg_3_months', 'PERC_SECS_TYPE_OUT_OWNN_CURR_avg_6_months', 'PERC_SECS_TYPE_OUT_OWNN_CURR_std_3_months', 'PERC_SECS_TYPE_OUT_OWNN_CURR_std_6_months', 'PERC_SECS_TYPE_IN_NATR_CURR_prev_month', 'PERC_SECS_TYPE_IN_NATR_CURR_avg_3_months', 'PERC_SECS_TYPE_IN_NATR_CURR_avg_6_months', 'PERC_SECS_TYPE_IN_NATR_CURR_std_3_months', 'PERC_SECS_TYPE_IN_NATR_CURR_std_6_months', 'PERC_SECS_TYPE_OUT_NATR_CURR_prev_month', 'PERC_SECS_TYPE_OUT_NATR_CURR_avg_3_months', 'PERC_SECS_TYPE_OUT_NATR_CURR_avg_6_months', 'PERC_SECS_TYPE_OUT_NATR_CURR_std_3_months', 'PERC_SECS_TYPE_OUT_NATR_CURR_std_6_months', 'NUM_PLAT_GMM_CURR_prev_month', 'NUM_PLAT_GMM_CURR_avg_3_months', 'NUM_PLAT_GMM_CURR_avg_6_months', 'NUM_PLAT_GMM_CURR_std_3_months', 'NUM_PLAT_GMM_CURR_std_6_months', 'NUM_PLAT_OMV_CURR_prev_month', 'NUM_PLAT_OMV_CURR_avg_3_months', 'NUM_PLAT_OMV_CURR_avg_6_months', 'NUM_PLAT_OMV_CURR_std_3_months', 'NUM_PLAT_OMV_CURR_std_6_months', 'NUM_NETW_OWNN_CURR_prev_month', 'NUM_NETW_OWNN_CURR_avg_3_months', 'NUM_NETW_OWNN_CURR_avg_6_months', 'NUM_NETW_OWNN_CURR_std_3_months', 'NUM_NETW_OWNN_CURR_std_6_months', 'NUM_CALL_CURR_prev_month', 'NUM_CALL_CURR_avg_3_months', 'NUM_CALL_CURR_avg_6_months', 'NUM_CALL_CURR_std_3_months', 'NUM_CALL_CURR_std_6_months', 'PERC_CALL_TYPE_IN_CURR_prev_month', 'PERC_CALL_TYPE_IN_CURR_avg_3_months', 'PERC_CALL_TYPE_IN_CURR_avg_6_months', 'PERC_CALL_TYPE_IN_CURR_std_3_months', 'PERC_CALL_TYPE_IN_CURR_std_6_months', 'PERC_CALL_TYPE_OUT_CURR_prev_month', 'PERC_CALL_TYPE_OUT_CURR_avg_3_months', 'PERC_CALL_TYPE_OUT_CURR_avg_6_months', 'PERC_CALL_TYPE_OUT_CURR_std_3_months', 'PERC_CALL_TYPE_OUT_CURR_std_6_months', 'PERC_CALL_OWNN_CURR_prev_month', 'PERC_CALL_OWNN_CURR_avg_3_months', 'PERC_CALL_OWNN_CURR_avg_6_months', 'PERC_CALL_OWNN_CURR_std_3_months', 'PERC_CALL_OWNN_CURR_std_6_months', 'PERC_CALL_NATR_CURR_prev_month', 'PERC_CALL_NATR_CURR_avg_3_months', 'PERC_CALL_NATR_CURR_avg_6_months', 'PERC_CALL_NATR_CURR_std_3_months', 'PERC_CALL_NATR_CURR_std_6_months', 'NUM_CALL_WEEK_CURR_prev_month', 'NUM_CALL_WEEK_CURR_avg_3_months', 'NUM_CALL_WEEK_CURR_avg_6_months', 'NUM_CALL_WEEK_CURR_std_3_months', 'NUM_CALL_WEEK_CURR_std_6_months', 'NUM_CALL_WEEKEND_CURR_prev_month', 'NUM_CALL_WEEKEND_CURR_avg_3_months', 'NUM_CALL_WEEKEND_CURR_avg_6_months', 'NUM_CALL_WEEKEND_CURR_std_3_months', 'NUM_CALL_WEEKEND_CURR_std_6_months', 'NUM_SECS_WEEK_CURR_prev_month', 'NUM_SECS_WEEK_CURR_avg_3_months', 'NUM_SECS_WEEK_CURR_avg_6_months', 'NUM_SECS_WEEK_CURR_std_3_months', 'NUM_SECS_WEEK_CURR_std_6_months', 'NUM_SECS_WEEKEND_CURR_prev_month', 'NUM_SECS_WEEKEND_CURR_avg_3_months', 'NUM_SECS_WEEKEND_CURR_avg_6_months', 'NUM_SECS_WEEKEND_CURR_std_3_months', 'NUM_SECS_WEEKEND_CURR_std_6_months', 'NUM_CALL_WEEK_prev_month', 'NUM_CALL_WEEK_avg_3_months', 'NUM_CALL_WEEK_avg_6_months', 'NUM_CALL_WEEK_std_3_months', 'NUM_CALL_WEEK_std_6_months', 'NUM_CALL_WEEKEND_prev_month', 'NUM_CALL_WEEKEND_avg_3_months', 'NUM_CALL_WEEKEND_avg_6_months', 'NUM_CALL_WEEKEND_std_3_months', 'NUM_CALL_WEEKEND_std_6_months', 'WHICH_MONTH_CHURNED']
    INFO - Target: NUM_DAYS_LINE_TYPE_FIXE_POST_DEA
    INFO - Completed feature computation!
    INFO - Features saved on src/features
    INFO - Targets saved on src/target
    INFO - Starting Modeling
    INFO - Building model pipeline
    INFO - Training model


    [LightGBM] [Info] Number of positive: 1523, number of negative: 47773
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.224948 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 68460
    [LightGBM] [Info] Number of data points in the train set: 49296, number of used features: 351
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.030895 -> initscore=-3.445779
    [LightGBM] [Info] Start training from score -3.445779
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf


    INFO - Completed model training!
    INFO - Started evaluation for Pipeline(steps=[('lightgbm',
                     LGBMClassifier(colsample_bytree=0.64, learning_rate=0.005,
                                    num_leaves=12, random_state=500, reg_alpha=0,
                                    reg_lambda=1, subsample=0.7))])


    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf


    INFO - Generating plots



    
![png](target_train_files/target_train_5_5.png)
    


    INFO - Precision in the first decile: 0.12
    /tmp/ipykernel_482/2046921147.py:175: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
    INFO - Uplift by decile:
    INFO - decile
    0    3.258897
    1    1.110391
    2    1.067158
    3    0.914569
    4    0.814285
    5    0.802512
    6    0.670169
    7    0.445159
    8    0.557436
    9    0.369577
    Name: y_true, dtype: float64
    INFO - Completed evaluation!
    INFO - Feature importance
    INFO -                          Feature  Coefficient
    351          WHICH_MONTH_CHURNED          192
    112                NUM_PERM_CURR           28
    160  service_fix_pending_install           25
    151   dif_periodica_prev_2_month           23
    38            MIN_DAYS_PERM_CURR           21
    0                   pago_final_0           21
    77                   NUM_DUR_OBJ           15
    150     dif_periodica_prev_month           14
    43            MAX_DAYS_PERM_CURR           13
    195      discount_0_std_6_months           12


Now for last 23 month available


```python
def calculate_uplift(target, y_pred_proba):
    """
    Calculate the uplift for each decile.

    Args:
    - y_true (array-like): True labels.
    - y_pred_proba (array-like): Predicted probabilities.

    Returns:
    - pd.Series: Uplift for each decile.
    """
    data = pd.DataFrame({"y_true": target, "y_pred_proba": y_pred_proba})
    data_sorted = data.sort_values(by="y_pred_proba", ascending=False)
    try:
        data_sorted["decile"] = pd.qcut(data_sorted["y_pred_proba"], q=10, labels=list(reversed(range(10))))
    except ValueError:
        # Handle fewer bins than desired
        bins = pd.qcut(data_sorted["y_pred_proba"], q=137, duplicates='drop').categories
        unique_bins = len(bins)
        labels = list(reversed(range(unique_bins-1)))  # Create one less label
        data_sorted["decile"] = pd.qcut(data_sorted["y_pred_proba"], q=unique_bins, labels=labels, duplicates='drop')
    decile_churn_rate = data_sorted.groupby("decile", observed=True)["y_true"].mean()

    overall_churn_rate = data["y_true"].mean()
    uplift = decile_churn_rate / overall_churn_rate

    # return by ascending deciles
    return uplift.sort_index(ascending=False)

def evaluate_metrics_lgb(y_true, y_pred, n_trees, lr, max_depth, path_smooth, train_or_test="test"):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    uplift = calculate_uplift(y_true, y_pred)

    print(
        f"LGBM with n_trees {n_trees} n of trees and lr {lr}  and depth {max_depth} and reg {path_smooth} on {train_or_test}: Roc AUC {roc_auc:.4f} and PR AUC {pr_auc:.4f}"
    )

    print(
        f"LGBM with n_trees {n_trees} n of trees and lr {lr}  and depth {max_depth} and reg {path_smooth} on {train_or_test} {uplift[0]} Uplift on the first decile"
    )
    print(uplift)
```


```python
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
```

    INFO - Started querying data
    INFO - Data succesfully retrieved! Length: 1019032
    INFO - Starting cleaning data
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Dropping column Import_Rest_quota_disp
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Dropping column NUM_FIX_PORT
    INFO - Dropping column NUM_FIX_PORT_LAST_1_MONTH
    INFO - Dropping column NUM_FIX_PORT_LAST_3_MONTHS
    INFO - Dropping column NUM_FIX_PORT_LAST_6_MONTHS
    INFO - Should fillna
    INFO - Dropping column NUM_MOB_PORT
    INFO - Dropping column NUM_MOB_PORT_LAST_1_MONTH
    INFO - Dropping column NUM_MOB_PORT_LAST_3_MONTHS
    INFO - Dropping column NUM_MOB_PORT_LAST_6_MONTHS
    INFO - Dropping column NUM_MOB_PORT_REQS_LAST_1_MONTH
    INFO - Dropping column NUM_MOB_PORT_REQS_LAST_3_MONTHS
    INFO - Dropping column NUM_MOB_PORT_REQS_LAST_6_MONTHS
    INFO - Dropping column NUM_MOB_PORT_TRANS_CURR
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Dropping column NUM_PORT_OPER_DONO_MASMOVIL_GRP_LAST_THREE_MONTHS
    INFO - Dropping column NUM_PORT_REQS_OPER_DONO_MASMOVIL_GRP_LAST_THREE_MONTHS
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Dropping column cust_max_days_between_fix_port
    INFO - Dropping column cust_max_days_between_mob_port
    INFO - Dropping column cust_max_months_between_fix_port
    INFO - Dropping column cust_max_months_between_mob_port
    INFO - Dropping column cust_min_days_between_fix_port
    INFO - Dropping column cust_min_days_between_mob_port
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Should fillna
    INFO - Completed cleaning data!
    INFO -   customer_id MONTH  YEAR  dif_pago_final_prev_month  \
    0     6369010    08  2022                     -23.46   
    1     7477889    08  2022                      44.97   
    2     7319839    04  2022                       0.00   
    3     7108377    05  2022                       4.63   
    4     7169549    03  2022                      -0.60   
    
       dif_pago_final_prev_2_month  dif_pago_final_prev_3_month  \
    0                       -23.19                       -23.19   
    1                        45.30                        45.30   
    2                         0.00                         3.57   
    3                         4.03                         4.33   
    4                         0.00                        -0.60   
    
       dif_consumo_prev_month  dif_consumo_prev_2_month  dif_consumo_prev_3_month  \
    0                    5.48                    -22.01                     34.58   
    1                  153.53                    153.53                    153.53   
    2                   78.69                     43.91                     20.74   
    3                   38.16                      0.06                      7.92   
    4                 -190.37                   -128.25                      9.21   
    
       dif_discount_prev_month  dif_discount_prev_2_month  \
    0                    -3.95                      21.37   
    1                  -215.76                    -308.88   
    2                   -78.69                     -43.91   
    3                   -34.28                       3.22   
    4                   189.77                     128.25   
    
       dif_discount_prev_3_month  dif_periodica_prev_month  \
    0                     -35.22                     -2.58   
    1                    -308.88                    107.04   
    2                     -17.84                      0.00   
    3                      -4.34                      0.75   
    4                      -9.81                      0.00   
    
       dif_periodica_prev_2_month  dif_periodica_prev_3_month  \
    0                       -0.14                       -0.14   
    1                      110.49                      110.49   
    2                        0.00                        0.67   
    3                        0.75                        0.75   
    4                        0.00                        0.00   
    
       dif_aperiodica_prev_month  dif_aperiodica_prev_2_month  \
    0                     -22.41                       -22.41   
    1                       0.15                        90.15   
    2                       0.00                         0.00   
    3                       0.00                         0.00   
    4                       0.00                         0.00   
    
       dif_aperiodica_prev_3_month  dif_ajuste_prev_month  \
    0                       -22.41                    0.0   
    1                        90.15                    0.0   
    2                         0.00                    0.0   
    3                         0.00                    0.0   
    4                         0.00                    0.0   
    
       dif_ajuste_prev_2_month  dif_ajuste_prev_3_month  \
    0                      0.0                      0.0   
    1                      0.0                      0.0   
    2                      0.0                      0.0   
    3                      0.0                      0.0   
    4                      0.0                      0.0   
    
       service_mobile_pending_install  service_fix_pending_install  \
    0                               0                            0   
    1                               0                            0   
    2                               0                            0   
    3                               0                            0   
    4                               0                            0   
    
       service_mobile_cancelled  service_fix_cancelled  \
    0                         0                      0   
    1                         0                      0   
    2                         0                      0   
    3                         0                      0   
    4                         0                      0   
    
       service_mobile_pending_install_3month  service_fix_pending_install_3month  \
    0                                      0                                   0   
    1                                      0                                   0   
    2                                      0                                   0   
    3                                      0                                   0   
    4                                      0                                   0   
    
       service_mobile_cancelled_3month  service_fix_cancelled_3month  \
    0                                0                             0   
    1                                0                             0   
    2                                0                             0   
    3                                0                             0   
    4                                0                             0   
    
       service_mobile_pending_install_6month  service_fix_pending_install_6month  \
    0                                      0                                   0   
    1                                      0                                   0   
    2                                      0                                   0   
    3                                      0                                   0   
    4                                      0                                   0   
    
       service_mobile_cancelled_6month  service_fix_cancelled_6month  \
    0                                0                             0   
    1                                0                             0   
    2                                0                             0   
    3                                0                             0   
    4                                0                             0   
    
       NUM_DAYS_ACT  order_mobile_from_new_alta  MIN_DAYS_PERM_CURR  \
    0            49                           0                  84   
    1            37                           0                 222   
    2           131                           0                 229   
    3           376                           0                 -19   
    4           265                           0                  94   
    
       PREV_FINISHED_PERM  NUM_DAYS_LINE_TYPE_MAIN_POST_ACT  \
    0                   0                               205   
    1                   0                                37   
    2                   0                               131   
    3                   0                               382   
    4                   0                               265   
    
       Import_Rest_quota_disp  LINE_TYPE_FIXE_POST_ACT_LAST_DT_D  \
    0                     NaN                                 -2   
    1                     NaN                                 -2   
    2                     NaN                                 -5   
    3                     NaN                                -13   
    4                     NaN                                 -9   
    
       MAX_DAYS_PERM_CURR  MAX_PENALTY_AMOUNT_CURR  MIN_PENALTY_AMOUNT_CURR  \
    0                 125                     72.0                    60.00   
    1                 222                      0.0                     0.00   
    2                 229                    120.0                    90.15   
    3                 294                    120.0                    60.00   
    4                 100                    120.0                    60.00   
    
       MM_GROUP_MOB_PORT  NUM_CALL_INTR_CURR  NUM_CALL_NATR_CURR  \
    0                  0                   0                 321   
    1                  0                   0                 127   
    2                  1                   0                 294   
    3                  1                   0                 302   
    4                  1                   0                 392   
    
       NUM_CALL_OWNN_CURR  NUM_CALL_SERV_FIXE_CURR  NUM_CALL_SERV_MOBI_CURR  \
    0                   2                        0                      323   
    1                 195                        0                      322   
    2                  67                        0                      361   
    3                 203                        0                      505   
    4                 307                        0                      699   
    
       NUM_CALL_SERV_UNKN_CURR  NUM_CALL_TYPE_IN_CURR  NUM_CALL_TYPE_IN_INTR_CURR  \
    0                        0                    171                           0   
    1                        0                    162                           0   
    2                        0                    164                           0   
    3                        0                    257                           0   
    4                        0                    285                           0   
    
       NUM_CALL_TYPE_IN_NATR_CURR  NUM_CALL_TYPE_IN_OWNN_CURR  \
    0                         171                           0   
    1                          65                          97   
    2                         133                          31   
    3                         151                         106   
    4                         123                         162   
    
       NUM_CALL_TYPE_IN_SERV_FIXE_CURR  NUM_CALL_TYPE_IN_SERV_MOBI_CURR  \
    0                                0                              171   
    1                                0                              162   
    2                                0                              164   
    3                                0                              257   
    4                                0                              285   
    
       NUM_CALL_TYPE_IN_SERV_UNKN_CURR  NUM_CALL_TYPE_IN_WEEKEND_CURR  \
    0                                0                             35   
    1                                0                             46   
    2                                0                             45   
    3                                0                             35   
    4                                0                             49   
    
       NUM_CALL_TYPE_IN_WEEK_CURR  NUM_CALL_TYPE_OUT_CURR  \
    0                         136                     152   
    1                         116                     160   
    2                         119                     197   
    3                         222                     248   
    4                         236                     414   
    
       NUM_CALL_TYPE_OUT_INTR_CURR  NUM_CALL_TYPE_OUT_NATR_CURR  \
    0                            0                          150   
    1                            0                           62   
    2                            0                          161   
    3                            0                          151   
    4                            0                          269   
    
       NUM_CALL_TYPE_OUT_OWNN_CURR  NUM_CALL_TYPE_OUT_SERV_FIXE_CURR  \
    0                            2                                 0   
    1                           98                                 0   
    2                           36                                 0   
    3                           97                                 0   
    4                          145                                 0   
    
       NUM_CALL_TYPE_OUT_SERV_MOBI_CURR  NUM_CALL_TYPE_OUT_SERV_UNKN_CURR  \
    0                               152                                 0   
    1                               160                                 0   
    2                               197                                 0   
    3                               248                                 0   
    4                               414                                 0   
    
       NUM_CALL_TYPE_OUT_WEEKEND_CURR  NUM_CALL_TYPE_OUT_WEEK_CURR  \
    0                              38                          114   
    1                              41                          119   
    2                              51                          146   
    3                              60                          188   
    4                              86                          328   
    
       NUM_CUSTOMER_BUNDLE_2P  NUM_CUSTOMER_BUNDLE_FMC  NUM_CUSTOMER_BUNDLE_MO  \
    0                       0                        1                       0   
    1                       0                        1                       0   
    2                       0                        1                       0   
    3                       0                        1                       0   
    4                       0                        1                       0   
    
       NUM_DAYS_LINE_TYPE_FIXE_POST_ACT  NUM_DISC_ACTI  NUM_DISC_CURR  \
    0                                49              0             23   
    1                                37              0             14   
    2                               134              0              9   
    3                               382              0             22   
    4                               265              0             12   
    
       NUM_DUR_OBJ  NUM_FIX_PORT  NUM_FIX_PORT_LAST_1_MONTH  \
    0            1             0                          0   
    1            0             0                          0   
    2            0             0                          0   
    3            0             0                          0   
    4            0             0                          0   
    
       NUM_FIX_PORT_LAST_3_MONTHS  NUM_FIX_PORT_LAST_6_MONTHS  NUM_INTR_CURR  \
    0                           0                           0              2   
    1                           0                           0              0   
    2                           0                           0              5   
    3                           0                           0              1   
    4                           0                           0              0   
    
       NUM_LINES_POST_ACTI  NUM_LINES_POST_CURR  NUM_LINES_TOTA  \
    0                    0                    5               0   
    1                    0                    4               0   
    2                    0                    4               0   
    3                    0                    4               0   
    4                    0                    4               0   
    
       NUM_LINE_TYPE_ADDI_CURR  NUM_LINE_TYPE_ADDI_POST_ACTI  \
    0                        0                             0   
    1                        1                             0   
    2                        0                             0   
    3                        1                             0   
    4                        1                             0   
    
       NUM_LINE_TYPE_ADDI_POST_CURR  NUM_LINE_TYPE_ADDI_TOTA  \
    0                             0                        0   
    1                             1                        0   
    2                             0                        0   
    3                             1                        0   
    4                             1                        0   
    
       NUM_LINE_TYPE_FIXE_CURR  NUM_LINE_TYPE_FIXE_POST_ACTI  \
    0                        2                             0   
    1                        2                             0   
    2                        2                             0   
    3                        2                             0   
    4                        2                             0   
    
       NUM_LINE_TYPE_FIXE_TOTA  NUM_LINE_TYPE_MAIN_CURR  \
    0                        0                        5   
    1                        0                        3   
    2                        0                        4   
    3                        0                        3   
    4                        0                        3   
    
       NUM_LINE_TYPE_MAIN_POST_ACTI  NUM_LINE_TYPE_MAIN_POST_CURR  \
    0                             0                             3   
    1                             0                             1   
    2                             0                             2   
    3                             0                             1   
    4                             0                             1   
    
       NUM_LINE_TYPE_MAIN_TOTA  NUM_MOB_PORT  NUM_MOB_PORT_LAST_1_MONTH  \
    0                        0             2                          0   
    1                        0             2                          0   
    2                        0             2                          0   
    3                        0             2                          0   
    4                        0             2                          0   
    
       NUM_MOB_PORT_LAST_3_MONTHS  NUM_MOB_PORT_LAST_6_MONTHS  \
    0                           0                           0   
    1                           2                           2   
    2                           0                           2   
    3                           0                           0   
    4                           0                           0   
    
       NUM_MOB_PORT_REQS_LAST_1_MONTH  NUM_MOB_PORT_REQS_LAST_3_MONTHS  \
    0                               0                                0   
    1                               0                                2   
    2                               0                                0   
    3                               0                                0   
    4                               0                                0   
    
       NUM_MOB_PORT_REQS_LAST_6_MONTHS  NUM_MOB_PORT_TRANS_CURR  NUM_NATR_CURR  \
    0                                0                        0            255   
    1                                2                        0            368   
    2                                3                        0            748   
    3                                0                        0            534   
    4                                0                        0            148   
    
       NUM_NETW_OSPN_CURR  NUM_NETW_RAAS_CURR  NUM_NETW_ROAM_CURR  \
    0                 145                  29                   2   
    1                 242                  80                   0   
    2                 555                 143                   5   
    3                 509                 117                   1   
    4                 146                  10                   0   
    
       NUM_NETW_TMEN_CURR  NUM_NETW_VODA_CURR  NUM_OWNN_CURR  NUM_PERM_CURR  \
    0                 110                   0             16              2   
    1                 126                   0            198              1   
    2                 193                   0            373              2   
    3                  25                   0            400              3   
    4                   2                   0            167              3   
    
       NUM_PORT_OPER_DONO_MASMOVIL_GRP_LAST_THREE_MONTHS  \
    0                                                  0   
    1                                                  0   
    2                                                  0   
    3                                                  0   
    4                                                  0   
    
       NUM_PORT_REQS_OPER_DONO_MASMOVIL_GRP_LAST_THREE_MONTHS  NUM_PREV_OBJ  \
    0                                                  0                  0   
    1                                                  0                  0   
    2                                                  0                  0   
    3                                                  0                  1   
    4                                                  0                  0   
    
       NUM_RAAS_CURR  NUM_SECS_SERV_FIXE_CURR  NUM_SECS_SERV_MOBI_CURR  \
    0             29                        0                    22002   
    1             80                        0                    48447   
    2            143                        0                    69186   
    3            117                        0                    34403   
    4             10                        0                   215452   
    
       NUM_SECS_TYPE_IN_CURR  NUM_SECS_TYPE_IN_SERV_FIXE_CURR  \
    0                  11181                                0   
    1                  30374                                0   
    2                  33864                                0   
    3                  14857                                0   
    4                 100335                                0   
    
       NUM_SECS_TYPE_IN_SERV_MOBI_CURR  NUM_SECS_TYPE_IN_WEEKEND_CURR  \
    0                            11181                           2035   
    1                            30374                           8453   
    2                            33864                           7554   
    3                            14857                           1694   
    4                           100335                          16038   
    
       NUM_SECS_TYPE_IN_WEEK_CURR  NUM_SECS_TYPE_OUT_CURR  \
    0                        9146                   10821   
    1                       21921                   18073   
    2                       26310                   35322   
    3                       13163                   19546   
    4                       84297                  115117   
    
       NUM_SECS_TYPE_OUT_SERV_FIXE_CURR  NUM_SECS_TYPE_OUT_SERV_MOBI_CURR  \
    0                                 0                             10821   
    1                                 0                             18073   
    2                                 0                             35322   
    3                                 0                             19546   
    4                                 0                            115117   
    
       NUM_SECS_TYPE_OUT_WEEKEND_CURR  NUM_SECS_TYPE_OUT_WEEK_CURR  \
    0                            2903                         7918   
    1                            3455                        14618   
    2                            6291                        29031   
    3                            2953                        16593   
    4                           23680                        91437   
    
       ORDER_FIX_FROM_NEW_ALTA  cust_days_since_last_fix_port  \
    0                        0                            648   
    1                        0                             39   
    2                        0                              0   
    3                        0                            382   
    4                        0                              0   
    
       cust_days_since_last_mob_port  cust_max_days_between_fix_port  \
    0                            647                            <NA>   
    1                             41                             740   
    2                            132                            <NA>   
    3                            377                            <NA>   
    4                            266                            <NA>   
    
       cust_max_days_between_mob_port  cust_max_months_between_fix_port  \
    0                               0                              <NA>   
    1                            1493                                24   
    2                            1310                              <NA>   
    3                             542                              <NA>   
    4                             968                              <NA>   
    
       cust_max_months_between_mob_port  cust_min_days_between_fix_port  \
    0                                 0                            <NA>   
    1                                49                               2   
    2                                43                            <NA>   
    3                                18                            <NA>   
    4                                32                            <NA>   
    
       cust_min_days_between_mob_port  cust_n_fix_port  cust_n_fix_recent_port  \
    0                               0                1                       1   
    1                               0                2                       1   
    2                               3                0                       0   
    3                               6                1                       1   
    4                               0                0                       0   
    
       order_mobile_from_migra_pre_to_post  pago_final_0  consumo_0  aperiodica_0  \
    0                                    0       54.0554    84.7712      -22.4140   
    1                                    0       45.3001   153.5339       90.1518   
    2                                    0       47.0000   239.5209        0.0000   
    3                                    0       51.5833   142.9825        0.0000   
    4                                    0       36.5000   772.3869        0.0000   
    
       periodica_0  discount_0  ajuste_0  NUM_GB_OWNN_CURR  NUM_GB_2G_CURR  \
    0     148.9344   -157.2362       0.0          0.038480        0.000755   
    1     110.4937   -308.8793       0.0          4.140498        0.000142   
    2      78.7059   -271.2268       0.0          3.066705        0.032002   
    3     196.7528   -288.1520       0.0         16.944626        0.001087   
    4      80.7719   -816.6588       0.0          0.471723        0.000021   
    
       NUM_GB_3G_CURR  NUM_GB_4G_CURR  NUM_GB_5G_CURR  NUM_SESS_CURR  \
    0        0.149002       18.769457        0.000000           5199   
    1        0.238579       11.656624        0.598905           7538   
    2        1.154262       29.411557        0.000000          13205   
    3        0.207145       22.172518        0.000000          10485   
    4        0.073669        3.195340        0.139645           4230   
    
       NUM_SECS_CURR  PERC_SECS_TYPE_IN_CURR  PERC_SECS_TYPE_OUT_CURR  \
    0          22002               50.818107                49.181893   
    1          48447               62.695317                37.304683   
    2          69186               48.946319                51.053681   
    3          34403               43.185187                56.814813   
    4         215452               46.569538                53.430462   
    
       PERC_SECS_OWNN_CURR  PERC_SECS_NATR_CURR  PERC_SECS_SERV_MOBI_CURR  \
    0             0.000000           100.000000                     100.0   
    1            62.497162            37.502838                     100.0   
    2            16.220045            83.779955                     100.0   
    3            32.877947            67.122053                     100.0   
    4            46.448397            53.551603                     100.0   
    
       PERC_SECS_TYPE_IN_OWNN_CURR  PERC_SECS_TYPE_OUT_OWNN_CURR  \
    0                          NaN                           NaN   
    1                    69.700773                     30.299227   
    2                    48.921761                     51.078239   
    3                    54.053576                     45.946424   
    4                    63.362112                     36.637888   
    
       PERC_SECS_TYPE_IN_NATR_CURR  PERC_SECS_TYPE_OUT_NATR_CURR  \
    0                    50.818107                     49.181893   
    1                    51.020970                     48.979030   
    2                    48.951073                     51.048927   
    3                    37.861597                     62.138403   
    4                    32.004368                     67.995632   
    
       NUM_PLAT_GMM_CURR  NUM_PLAT_OMV_CURR  NUM_NETW_OWNN_CURR  NUM_CALL_CURR  \
    0                302                  0                  16            323   
    1                679                  0                 198            322   
    2               1269                  0                 373            361   
    3               1052                  0                 400            505   
    4                373                  0                 167            699   
    
       PERC_CALL_TYPE_IN_CURR  PERC_CALL_TYPE_OUT_CURR  PERC_CALL_OWNN_CURR  \
    0               52.941176                47.058824             0.619195   
    1               50.310559                49.689441            60.559006   
    2               45.429363                54.570637            18.559557   
    3               50.891089                49.108911            40.198020   
    4               40.772532                59.227468            43.919886   
    
       PERC_CALL_NATR_CURR  NUM_CALL_WEEK_CURR  NUM_CALL_WEEKEND_CURR  \
    0            99.380805                 250                     73   
    1            39.440994                 235                     87   
    2            81.440443                 265                     96   
    3            59.801980                 410                     95   
    4            56.080114                 564                    135   
    
       NUM_SECS_WEEK_CURR  NUM_SECS_WEEKEND_CURR  NUM_CALL_WEEK  NUM_CALL_WEEKEND  \
    0               17064                   4938            184                68   
    1               36539                  11908            167                71   
    2               55341                  13845            212                83   
    3               29756                   4647            340                83   
    4              175734                  39718            344                86   
    
       NUM_DAYS_LINE_TYPE_FIXE_POST_DEA  
    0                              <NA>  
    1                              <NA>  
    2                              <NA>  
    3                              <NA>  
    4                              <NA>  
    INFO - Starting feature computation
    INFO - Train computation from 2023-01-01 to 2023-06-01. Target for 2023-08-01 00:00:00
    INFO - Test computation from 2023-02-01 00:00:00 to 2023-07-01 00:00:00. Target for 2023-09-01 00:00:00
    /tmp/ipykernel_482/2080027226.py:180: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    INFO - Removing 9650 previous churned users from test set
    INFO - Unique customers in train: 53376
    INFO - Unique customers in test: 43629
    INFO - Starting features and target computation
    INFO - Initial number of features passed: 177
    INFO - Starting computation
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    INFO - Final number of features computed: 358
    INFO - Length train data: 53376
    INFO - Length test data: 40760
    INFO - Computation done!
    INFO - Features: ['pago_final_0', 'consumo_0', 'aperiodica_0', 'periodica_0', 'discount_0', 'ajuste_0', 'NUM_GB_OWNN_CURR', 'NUM_GB_2G_CURR', 'NUM_GB_3G_CURR', 'NUM_GB_4G_CURR', 'NUM_GB_5G_CURR', 'NUM_SESS_CURR', 'NUM_SECS_CURR', 'PERC_SECS_TYPE_IN_CURR', 'PERC_SECS_TYPE_OUT_CURR', 'PERC_SECS_OWNN_CURR', 'PERC_SECS_NATR_CURR', 'PERC_SECS_SERV_MOBI_CURR', 'PERC_SECS_TYPE_IN_OWNN_CURR', 'PERC_SECS_TYPE_OUT_OWNN_CURR', 'PERC_SECS_TYPE_IN_NATR_CURR', 'PERC_SECS_TYPE_OUT_NATR_CURR', 'NUM_PLAT_GMM_CURR', 'NUM_PLAT_OMV_CURR', 'NUM_NETW_OWNN_CURR', 'NUM_CALL_CURR', 'PERC_CALL_TYPE_IN_CURR', 'PERC_CALL_TYPE_OUT_CURR', 'PERC_CALL_OWNN_CURR', 'PERC_CALL_NATR_CURR', 'NUM_CALL_WEEK_CURR', 'NUM_CALL_WEEKEND_CURR', 'NUM_SECS_WEEK_CURR', 'NUM_SECS_WEEKEND_CURR', 'NUM_CALL_WEEK', 'NUM_CALL_WEEKEND', 'NUM_DAYS_ACT', 'order_mobile_from_new_alta', 'MIN_DAYS_PERM_CURR', 'PREV_FINISHED_PERM', 'NUM_DAYS_LINE_TYPE_MAIN_POST_ACT', 'Import_Rest_quota_disp', 'LINE_TYPE_FIXE_POST_ACT_LAST_DT_D', 'MAX_DAYS_PERM_CURR', 'MAX_PENALTY_AMOUNT_CURR', 'MIN_PENALTY_AMOUNT_CURR', 'MM_GROUP_MOB_PORT', 'NUM_CALL_INTR_CURR', 'NUM_CALL_NATR_CURR', 'NUM_CALL_OWNN_CURR', 'NUM_CALL_SERV_FIXE_CURR', 'NUM_CALL_SERV_MOBI_CURR', 'NUM_CALL_SERV_UNKN_CURR', 'NUM_CALL_TYPE_IN_CURR', 'NUM_CALL_TYPE_IN_INTR_CURR', 'NUM_CALL_TYPE_IN_NATR_CURR', 'NUM_CALL_TYPE_IN_OWNN_CURR', 'NUM_CALL_TYPE_IN_SERV_FIXE_CURR', 'NUM_CALL_TYPE_IN_SERV_MOBI_CURR', 'NUM_CALL_TYPE_IN_SERV_UNKN_CURR', 'NUM_CALL_TYPE_IN_WEEKEND_CURR', 'NUM_CALL_TYPE_IN_WEEK_CURR', 'NUM_CALL_TYPE_OUT_CURR', 'NUM_CALL_TYPE_OUT_INTR_CURR', 'NUM_CALL_TYPE_OUT_NATR_CURR', 'NUM_CALL_TYPE_OUT_OWNN_CURR', 'NUM_CALL_TYPE_OUT_SERV_FIXE_CURR', 'NUM_CALL_TYPE_OUT_SERV_MOBI_CURR', 'NUM_CALL_TYPE_OUT_SERV_UNKN_CURR', 'NUM_CALL_TYPE_OUT_WEEKEND_CURR', 'NUM_CALL_TYPE_OUT_WEEK_CURR', 'NUM_CUSTOMER_BUNDLE_2P', 'NUM_CUSTOMER_BUNDLE_FMC', 'NUM_CUSTOMER_BUNDLE_MO', 'NUM_DAYS_LINE_TYPE_FIXE_POST_ACT', 'NUM_DISC_ACTI', 'NUM_DISC_CURR', 'NUM_DUR_OBJ', 'NUM_FIX_PORT', 'NUM_FIX_PORT_LAST_1_MONTH', 'NUM_FIX_PORT_LAST_3_MONTHS', 'NUM_FIX_PORT_LAST_6_MONTHS', 'NUM_INTR_CURR', 'NUM_LINES_POST_ACTI', 'NUM_LINES_POST_CURR', 'NUM_LINES_TOTA', 'NUM_LINE_TYPE_ADDI_CURR', 'NUM_LINE_TYPE_ADDI_POST_ACTI', 'NUM_LINE_TYPE_ADDI_POST_CURR', 'NUM_LINE_TYPE_ADDI_TOTA', 'NUM_LINE_TYPE_FIXE_CURR', 'NUM_LINE_TYPE_FIXE_POST_ACTI', 'NUM_LINE_TYPE_FIXE_TOTA', 'NUM_LINE_TYPE_MAIN_CURR', 'NUM_LINE_TYPE_MAIN_POST_ACTI', 'NUM_LINE_TYPE_MAIN_POST_CURR', 'NUM_LINE_TYPE_MAIN_TOTA', 'NUM_MOB_PORT', 'NUM_MOB_PORT_LAST_1_MONTH', 'NUM_MOB_PORT_LAST_3_MONTHS', 'NUM_MOB_PORT_LAST_6_MONTHS', 'NUM_MOB_PORT_REQS_LAST_1_MONTH', 'NUM_MOB_PORT_REQS_LAST_3_MONTHS', 'NUM_MOB_PORT_REQS_LAST_6_MONTHS', 'NUM_MOB_PORT_TRANS_CURR', 'NUM_NATR_CURR', 'NUM_NETW_OSPN_CURR', 'NUM_NETW_RAAS_CURR', 'NUM_NETW_ROAM_CURR', 'NUM_NETW_TMEN_CURR', 'NUM_NETW_VODA_CURR', 'NUM_OWNN_CURR', 'NUM_PERM_CURR', 'NUM_PORT_OPER_DONO_MASMOVIL_GRP_LAST_THREE_MONTHS', 'NUM_PORT_REQS_OPER_DONO_MASMOVIL_GRP_LAST_THREE_MONTHS', 'NUM_PREV_OBJ', 'NUM_RAAS_CURR', 'NUM_SECS_SERV_FIXE_CURR', 'NUM_SECS_SERV_MOBI_CURR', 'NUM_SECS_TYPE_IN_CURR', 'NUM_SECS_TYPE_IN_SERV_FIXE_CURR', 'NUM_SECS_TYPE_IN_SERV_MOBI_CURR', 'NUM_SECS_TYPE_IN_WEEKEND_CURR', 'NUM_SECS_TYPE_IN_WEEK_CURR', 'NUM_SECS_TYPE_OUT_CURR', 'NUM_SECS_TYPE_OUT_SERV_FIXE_CURR', 'NUM_SECS_TYPE_OUT_SERV_MOBI_CURR', 'NUM_SECS_TYPE_OUT_WEEKEND_CURR', 'NUM_SECS_TYPE_OUT_WEEK_CURR', 'ORDER_FIX_FROM_NEW_ALTA', 'cust_days_since_last_fix_port', 'cust_days_since_last_mob_port', 'cust_max_days_between_fix_port', 'cust_max_days_between_mob_port', 'cust_max_months_between_fix_port', 'cust_max_months_between_mob_port', 'cust_min_days_between_fix_port', 'cust_min_days_between_mob_port', 'cust_n_fix_port', 'cust_n_fix_recent_port', 'order_mobile_from_migra_pre_to_post', 'dif_pago_final_prev_month', 'dif_pago_final_prev_2_month', 'dif_pago_final_prev_3_month', 'dif_consumo_prev_month', 'dif_consumo_prev_2_month', 'dif_consumo_prev_3_month', 'dif_discount_prev_month', 'dif_discount_prev_2_month', 'dif_discount_prev_3_month', 'dif_periodica_prev_month', 'dif_periodica_prev_2_month', 'dif_periodica_prev_3_month', 'dif_aperiodica_prev_month', 'dif_aperiodica_prev_2_month', 'dif_aperiodica_prev_3_month', 'dif_ajuste_prev_month', 'dif_ajuste_prev_2_month', 'dif_ajuste_prev_3_month', 'service_mobile_pending_install', 'service_fix_pending_install', 'service_mobile_cancelled', 'service_fix_cancelled', 'service_mobile_pending_install_3month', 'service_fix_pending_install_3month', 'service_mobile_cancelled_3month', 'service_fix_cancelled_3month', 'service_mobile_pending_install_6month', 'service_fix_pending_install_6month', 'service_mobile_cancelled_6month', 'service_fix_cancelled_6month', 'pago_final_0_prev_month', 'pago_final_0_avg_3_months', 'pago_final_0_avg_6_months', 'pago_final_0_std_3_months', 'pago_final_0_std_6_months', 'consumo_0_prev_month', 'consumo_0_avg_3_months', 'consumo_0_avg_6_months', 'consumo_0_std_3_months', 'consumo_0_std_6_months', 'aperiodica_0_prev_month', 'aperiodica_0_avg_3_months', 'aperiodica_0_avg_6_months', 'aperiodica_0_std_3_months', 'aperiodica_0_std_6_months', 'periodica_0_prev_month', 'periodica_0_avg_3_months', 'periodica_0_avg_6_months', 'periodica_0_std_3_months', 'periodica_0_std_6_months', 'discount_0_prev_month', 'discount_0_avg_3_months', 'discount_0_avg_6_months', 'discount_0_std_3_months', 'discount_0_std_6_months', 'ajuste_0_prev_month', 'ajuste_0_avg_3_months', 'ajuste_0_avg_6_months', 'ajuste_0_std_3_months', 'ajuste_0_std_6_months', 'NUM_GB_OWNN_CURR_prev_month', 'NUM_GB_OWNN_CURR_avg_3_months', 'NUM_GB_OWNN_CURR_avg_6_months', 'NUM_GB_OWNN_CURR_std_3_months', 'NUM_GB_OWNN_CURR_std_6_months', 'NUM_GB_2G_CURR_prev_month', 'NUM_GB_2G_CURR_avg_3_months', 'NUM_GB_2G_CURR_avg_6_months', 'NUM_GB_2G_CURR_std_3_months', 'NUM_GB_2G_CURR_std_6_months', 'NUM_GB_3G_CURR_prev_month', 'NUM_GB_3G_CURR_avg_3_months', 'NUM_GB_3G_CURR_avg_6_months', 'NUM_GB_3G_CURR_std_3_months', 'NUM_GB_3G_CURR_std_6_months', 'NUM_GB_4G_CURR_prev_month', 'NUM_GB_4G_CURR_avg_3_months', 'NUM_GB_4G_CURR_avg_6_months', 'NUM_GB_4G_CURR_std_3_months', 'NUM_GB_4G_CURR_std_6_months', 'NUM_GB_5G_CURR_prev_month', 'NUM_GB_5G_CURR_avg_3_months', 'NUM_GB_5G_CURR_avg_6_months', 'NUM_GB_5G_CURR_std_3_months', 'NUM_GB_5G_CURR_std_6_months', 'NUM_SESS_CURR_prev_month', 'NUM_SESS_CURR_avg_3_months', 'NUM_SESS_CURR_avg_6_months', 'NUM_SESS_CURR_std_3_months', 'NUM_SESS_CURR_std_6_months', 'NUM_SECS_CURR_prev_month', 'NUM_SECS_CURR_avg_3_months', 'NUM_SECS_CURR_avg_6_months', 'NUM_SECS_CURR_std_3_months', 'NUM_SECS_CURR_std_6_months', 'PERC_SECS_TYPE_IN_CURR_prev_month', 'PERC_SECS_TYPE_IN_CURR_avg_3_months', 'PERC_SECS_TYPE_IN_CURR_avg_6_months', 'PERC_SECS_TYPE_IN_CURR_std_3_months', 'PERC_SECS_TYPE_IN_CURR_std_6_months', 'PERC_SECS_TYPE_OUT_CURR_prev_month', 'PERC_SECS_TYPE_OUT_CURR_avg_3_months', 'PERC_SECS_TYPE_OUT_CURR_avg_6_months', 'PERC_SECS_TYPE_OUT_CURR_std_3_months', 'PERC_SECS_TYPE_OUT_CURR_std_6_months', 'PERC_SECS_OWNN_CURR_prev_month', 'PERC_SECS_OWNN_CURR_avg_3_months', 'PERC_SECS_OWNN_CURR_avg_6_months', 'PERC_SECS_OWNN_CURR_std_3_months', 'PERC_SECS_OWNN_CURR_std_6_months', 'PERC_SECS_NATR_CURR_prev_month', 'PERC_SECS_NATR_CURR_avg_3_months', 'PERC_SECS_NATR_CURR_avg_6_months', 'PERC_SECS_NATR_CURR_std_3_months', 'PERC_SECS_NATR_CURR_std_6_months', 'PERC_SECS_SERV_MOBI_CURR_prev_month', 'PERC_SECS_SERV_MOBI_CURR_avg_3_months', 'PERC_SECS_SERV_MOBI_CURR_avg_6_months', 'PERC_SECS_SERV_MOBI_CURR_std_3_months', 'PERC_SECS_SERV_MOBI_CURR_std_6_months', 'PERC_SECS_TYPE_IN_OWNN_CURR_prev_month', 'PERC_SECS_TYPE_IN_OWNN_CURR_avg_3_months', 'PERC_SECS_TYPE_IN_OWNN_CURR_avg_6_months', 'PERC_SECS_TYPE_IN_OWNN_CURR_std_3_months', 'PERC_SECS_TYPE_IN_OWNN_CURR_std_6_months', 'PERC_SECS_TYPE_OUT_OWNN_CURR_prev_month', 'PERC_SECS_TYPE_OUT_OWNN_CURR_avg_3_months', 'PERC_SECS_TYPE_OUT_OWNN_CURR_avg_6_months', 'PERC_SECS_TYPE_OUT_OWNN_CURR_std_3_months', 'PERC_SECS_TYPE_OUT_OWNN_CURR_std_6_months', 'PERC_SECS_TYPE_IN_NATR_CURR_prev_month', 'PERC_SECS_TYPE_IN_NATR_CURR_avg_3_months', 'PERC_SECS_TYPE_IN_NATR_CURR_avg_6_months', 'PERC_SECS_TYPE_IN_NATR_CURR_std_3_months', 'PERC_SECS_TYPE_IN_NATR_CURR_std_6_months', 'PERC_SECS_TYPE_OUT_NATR_CURR_prev_month', 'PERC_SECS_TYPE_OUT_NATR_CURR_avg_3_months', 'PERC_SECS_TYPE_OUT_NATR_CURR_avg_6_months', 'PERC_SECS_TYPE_OUT_NATR_CURR_std_3_months', 'PERC_SECS_TYPE_OUT_NATR_CURR_std_6_months', 'NUM_PLAT_GMM_CURR_prev_month', 'NUM_PLAT_GMM_CURR_avg_3_months', 'NUM_PLAT_GMM_CURR_avg_6_months', 'NUM_PLAT_GMM_CURR_std_3_months', 'NUM_PLAT_GMM_CURR_std_6_months', 'NUM_PLAT_OMV_CURR_prev_month', 'NUM_PLAT_OMV_CURR_avg_3_months', 'NUM_PLAT_OMV_CURR_avg_6_months', 'NUM_PLAT_OMV_CURR_std_3_months', 'NUM_PLAT_OMV_CURR_std_6_months', 'NUM_NETW_OWNN_CURR_prev_month', 'NUM_NETW_OWNN_CURR_avg_3_months', 'NUM_NETW_OWNN_CURR_avg_6_months', 'NUM_NETW_OWNN_CURR_std_3_months', 'NUM_NETW_OWNN_CURR_std_6_months', 'NUM_CALL_CURR_prev_month', 'NUM_CALL_CURR_avg_3_months', 'NUM_CALL_CURR_avg_6_months', 'NUM_CALL_CURR_std_3_months', 'NUM_CALL_CURR_std_6_months', 'PERC_CALL_TYPE_IN_CURR_prev_month', 'PERC_CALL_TYPE_IN_CURR_avg_3_months', 'PERC_CALL_TYPE_IN_CURR_avg_6_months', 'PERC_CALL_TYPE_IN_CURR_std_3_months', 'PERC_CALL_TYPE_IN_CURR_std_6_months', 'PERC_CALL_TYPE_OUT_CURR_prev_month', 'PERC_CALL_TYPE_OUT_CURR_avg_3_months', 'PERC_CALL_TYPE_OUT_CURR_avg_6_months', 'PERC_CALL_TYPE_OUT_CURR_std_3_months', 'PERC_CALL_TYPE_OUT_CURR_std_6_months', 'PERC_CALL_OWNN_CURR_prev_month', 'PERC_CALL_OWNN_CURR_avg_3_months', 'PERC_CALL_OWNN_CURR_avg_6_months', 'PERC_CALL_OWNN_CURR_std_3_months', 'PERC_CALL_OWNN_CURR_std_6_months', 'PERC_CALL_NATR_CURR_prev_month', 'PERC_CALL_NATR_CURR_avg_3_months', 'PERC_CALL_NATR_CURR_avg_6_months', 'PERC_CALL_NATR_CURR_std_3_months', 'PERC_CALL_NATR_CURR_std_6_months', 'NUM_CALL_WEEK_CURR_prev_month', 'NUM_CALL_WEEK_CURR_avg_3_months', 'NUM_CALL_WEEK_CURR_avg_6_months', 'NUM_CALL_WEEK_CURR_std_3_months', 'NUM_CALL_WEEK_CURR_std_6_months', 'NUM_CALL_WEEKEND_CURR_prev_month', 'NUM_CALL_WEEKEND_CURR_avg_3_months', 'NUM_CALL_WEEKEND_CURR_avg_6_months', 'NUM_CALL_WEEKEND_CURR_std_3_months', 'NUM_CALL_WEEKEND_CURR_std_6_months', 'NUM_SECS_WEEK_CURR_prev_month', 'NUM_SECS_WEEK_CURR_avg_3_months', 'NUM_SECS_WEEK_CURR_avg_6_months', 'NUM_SECS_WEEK_CURR_std_3_months', 'NUM_SECS_WEEK_CURR_std_6_months', 'NUM_SECS_WEEKEND_CURR_prev_month', 'NUM_SECS_WEEKEND_CURR_avg_3_months', 'NUM_SECS_WEEKEND_CURR_avg_6_months', 'NUM_SECS_WEEKEND_CURR_std_3_months', 'NUM_SECS_WEEKEND_CURR_std_6_months', 'NUM_CALL_WEEK_prev_month', 'NUM_CALL_WEEK_avg_3_months', 'NUM_CALL_WEEK_avg_6_months', 'NUM_CALL_WEEK_std_3_months', 'NUM_CALL_WEEK_std_6_months', 'NUM_CALL_WEEKEND_prev_month', 'NUM_CALL_WEEKEND_avg_3_months', 'NUM_CALL_WEEKEND_avg_6_months', 'NUM_CALL_WEEKEND_std_3_months', 'NUM_CALL_WEEKEND_std_6_months', 'WHICH_MONTH_CHURNED']
    INFO - Target: NUM_DAYS_LINE_TYPE_FIXE_POST_DEA
    INFO - Completed feature computation!
    INFO - Features saved on src/features
    INFO - Targets saved on src/target
    INFO - Starting Modeling
    INFO - Building model pipeline
    INFO - Training model


    [LightGBM] [Info] Number of positive: 1892, number of negative: 48757
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.246058 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 68429
    [LightGBM] [Info] Number of data points in the train set: 50649, number of used features: 351
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.037355 -> initscore=-3.249214
    [LightGBM] [Info] Start training from score -3.249214
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf


    INFO - Completed model training!
    INFO - Started evaluation for Pipeline(steps=[('lightgbm',
                     LGBMClassifier(colsample_bytree=0.64, learning_rate=0.005,
                                    num_leaves=12, random_state=500, reg_alpha=0,
                                    reg_lambda=1, subsample=0.7))])


    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf


    INFO - Generating plots



    
![png](target_train_files/target_train_8_5.png)
    


    INFO - Precision in the first decile: 0.15



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb Cell 8 line 1
          <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=3'>4</a> features, target, features_test, target_test = feature_computation(
          <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=4'>5</a>     clean_data,
          <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=5'>6</a>     train_from,
       (...)
         <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=9'>10</a>     save_target_path=save_target_path,
         <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=10'>11</a> )
         <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=11'>12</a> model = modeling(features, target, lightgbm_params, logger)
    ---> <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=12'>13</a> model_metrics, precision_decile, uplift_by_decile, feature_importance = evaluation(
         <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=13'>14</a>     model, features_test, target_test, logger, save_curves_path
         <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=14'>15</a> )


    /home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb Cell 8 line 1
        <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=113'>114</a> logger.info(f"Precision in the first decile: {precision_decile:.2f}")
        <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=115'>116</a> # Calculate Uplift for Each Decile
    --> <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=116'>117</a> uplift_by_decile = calculate_uplift(target_test, preds)
        <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=117'>118</a> logger.info("Uplift by decile:")
        <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=118'>119</a> logger.info(uplift_by_decile)


    /home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb Cell 8 line 1
        <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=169'>170</a> data = pd.DataFrame({"y_true": target, "y_pred_proba": y_pred_proba})
        <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=170'>171</a> data_sorted = data.sort_values(by="y_pred_proba", ascending=False)
    --> <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=171'>172</a> data_sorted["decile"] = pd.qcut(
        <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=172'>173</a>     data_sorted["y_pred_proba"], q=10, labels=list(reversed(range(10)))
        <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=173'>174</a> )
        <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=174'>175</a> decile_churn_rate = data_sorted.groupby("decile")["y_true"].mean()
        <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X40sdnNjb2RlLXJlbW90ZQ%3D%3D?line=176'>177</a> overall_churn_rate = data["y_true"].mean()


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-YCVm4i4L-py3.11/lib/python3.11/site-packages/pandas/core/reshape/tile.py:379, in qcut(x, q, labels, retbins, precision, duplicates)
        376 x_np = x_np[~np.isnan(x_np)]
        377 bins = np.quantile(x_np, quantiles)
    --> 379 fac, bins = _bins_to_cuts(
        380     x,
        381     bins,
        382     labels=labels,
        383     precision=precision,
        384     include_lowest=True,
        385     dtype=dtype,
        386     duplicates=duplicates,
        387 )
        389 return _postprocess_for_cut(fac, bins, retbins, dtype, original)


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-YCVm4i4L-py3.11/lib/python3.11/site-packages/pandas/core/reshape/tile.py:421, in _bins_to_cuts(x, bins, right, labels, precision, include_lowest, dtype, duplicates, ordered)
        419 if len(unique_bins) < len(bins) and len(bins) != 2:
        420     if duplicates == "raise":
    --> 421         raise ValueError(
        422             f"Bin edges must be unique: {repr(bins)}.\n"
        423             f"You can drop duplicate edges by setting the 'duplicates' kwarg"
        424         )
        425     bins = unique_bins
        427 side: Literal["left", "right"] = "left" if right else "right"


    ValueError: Bin edges must be unique: array([0.02507509, 0.02524017, 0.02524017, 0.02543771, 0.02560515,
           0.02597802, 0.02609761, 0.02617076, 0.02631692, 0.02654081,
           0.0446491 ]).
    You can drop duplicate edges by setting the 'duplicates' kwarg



```python
model_metrics, precision_decile, uplift_by_decile, feature_importance = evaluation(
    model, features_test, target_test, logger, save_curves_path)
```

    INFO - Started evaluation for Pipeline(steps=[('lightgbm',
                     LGBMClassifier(colsample_bytree=0.64, learning_rate=0.005,
                                    num_leaves=12, random_state=500, reg_alpha=0,
                                    reg_lambda=1, subsample=0.7))])
    INFO - Generating plots



    
![png](target_train_files/target_train_9_1.png)
    


    INFO - Precision in the first decile: 0.15



    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb Cell 10 line 1
        <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X51sdnNjb2RlLXJlbW90ZQ%3D%3D?line=171'>172</a> try:
    --> <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X51sdnNjb2RlLXJlbW90ZQ%3D%3D?line=172'>173</a>     data_sorted["decile"] = pd.qcut(data_sorted["y_pred_proba"], q=10, labels=list(reversed(range(10))))
        <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X51sdnNjb2RlLXJlbW90ZQ%3D%3D?line=173'>174</a> except ValueError:
        <a href='vscode-notebook-cell://wsl%2Bubuntu/home/dan1dr/zrive-ds-4q24-churn/src/eda/target_train.ipynb#X51sdnNjb2RlLXJlbW90ZQ%3D%3D?line=174'>175</a>     # Handle fewer bins than desired


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-YCVm4i4L-py3.11/lib/python3.11/site-packages/pandas/core/reshape/tile.py:379, in qcut(x, q, labels, retbins, precision, duplicates)
        377 bins = np.quantile(x_np, quantiles)
    --> 379 fac, bins = _bins_to_cuts(
        380     x,
        381     bins,
        382     labels=labels,
        383     precision=precision,
        384     include_lowest=True,
        385     dtype=dtype,
        386     duplicates=duplicates,
        387 )
        389 return _postprocess_for_cut(fac, bins, retbins, dtype, original)


    File ~/.cache/pypoetry/virtualenvs/zrive-ds-YCVm4i4L-py3.11/lib/python3.11/site-packages/pandas/core/reshape/tile.py:421, in _bins_to_cuts(x, bins, right, labels, precision, include_lowest, dtype, duplicates, ordered)
        420 if duplicates == "raise":
    --> 421     raise ValueError(
        422         f"Bin edges must be unique: {repr(bins)}.\n"
        423         f"You can drop duplicate edges by setting the 'duplicates' kwarg"
        424     )
        425 bins = unique_bins


    ValueError: Bin edges must be unique: array([0.02507509, 0.02524017, 0.02524017, 0.02543771, 0.02560515,
           0.02597802, 0.02609761, 0.02617076, 0.02631692, 0.02654081,
           0.0446491 ]).
    You can drop duplicate edges by setting the 'duplicates' kwarg

    
    During handling of the above exception, another exception occurred:


    AttributeError                            Traceback (most recent call last)

    /tmp/ipykernel_482/3728012994.py in ?()
    ----> 1 model_metrics, precision_decile, uplift_by_decile, feature_importance = evaluation(
          2     model, features_test, target_test, logger, save_curves_path)


    /tmp/ipykernel_482/1257825612.py in ?(model, features_test, target_test, logger, save_curves_path)
        113     precision_decile = calculate_precision_first_decile(target_test, preds)
        114     logger.info(f"Precision in the first decile: {precision_decile:.2f}")
        115 
        116     # Calculate Uplift for Each Decile
    --> 117     uplift_by_decile = calculate_uplift(target_test, preds)
        118     logger.info("Uplift by decile:")
        119     logger.info(uplift_by_decile)
        120 


    /tmp/ipykernel_482/1257825612.py in ?(target, y_pred_proba)
        172     try:
        173         data_sorted["decile"] = pd.qcut(data_sorted["y_pred_proba"], q=10, labels=list(reversed(range(10))))
        174     except ValueError:
        175         # Handle fewer bins than desired
    --> 176         bins = pd.qcut(data_sorted["y_pred_proba"], q=137, duplicates='drop').categories
        177         unique_bins = len(bins)
        178         labels = list(reversed(range(unique_bins-1)))  # Create one less label
        179         data_sorted["decile"] = pd.qcut(data_sorted["y_pred_proba"], q=unique_bins, labels=labels, duplicates='drop')


    ~/.cache/pypoetry/virtualenvs/zrive-ds-YCVm4i4L-py3.11/lib/python3.11/site-packages/pandas/core/generic.py in ?(self, name)
       6200             and name not in self._accessors
       6201             and self._info_axis._can_hold_identifiers_and_holds_name(name)
       6202         ):
       6203             return self[name]
    -> 6204         return object.__getattribute__(self, name)
    

    AttributeError: 'Series' object has no attribute 'categories'



```python
import sys
import os
import pandas as pd
import numpy as np
import configparser
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import lightgbm as lgb
import lightgbm as lgb
from plotnine import ggplot, aes, geom_boxplot, labs, scale_color_manual, geom_point
```




    (40110,)




```python
lr = 0.1
n_estimators = 100
max_depth = 10
path_smooth = 0
metric = "binary"
```


```python
model = lgb.LGBMClassifier(
                    learning_rate=lightgbm_params["learning_rate"],
                    n_estimators=lightgbm_params["n_estimators"],
                    num_leaves=lightgbm_params["num_leaves"],
                    colsample_bytree=lightgbm_params["colsample_bytree"],
                    subsample=lightgbm_params["subsample"],
                    reg_alpha=lightgbm_params["reg_alpha"],
                    reg_lambda=lightgbm_params["reg_lambda"],
                    random_state=lightgbm_params["random_state"],
                
)

model.fit(features, target,
    eval_set=[(features, target), (features_test, target_test)],
        eval_metric = 'logloss',
)
         
evals_result = model.evals_result_
```

    [LightGBM] [Info] Number of positive: 1892, number of negative: 48757
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.178656 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 68429
    [LightGBM] [Info] Number of data points in the train set: 50649, number of used features: 351
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.037355 -> initscore=-3.249214
    [LightGBM] [Info] Start training from score -3.249214
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf



```python
def compute_learning_rate(evals_result, set_) -> pd.DataFrame:
    """
    Calculate the normalized learning rates for the model

    Parameters:
    - set_ (str): The dataset type ('training' or 'valid_0').
    - all_results (dict): Dictionary containing training results for each execution date.

    Returns:
    - pd.DataFrame: DataFrame with normalized learning rates, including execution dates and tree categories.
    """

    train_logloss = evals_result[set_]['binary_logloss']
    df = pd.DataFrame({'binary_logloss': train_logloss, 'set': set_})

    # Combine dataframes
    df['n_trees'] = range(len(df))

    # Calculate the % diff respect to first tree
    df["first_tree_logloss"] = df["binary_logloss"].iloc[0]
    df["normalized_learning_rate"] = (df["binary_logloss"] - df["first_tree_logloss"]) / df["first_tree_logloss"]
    df = df.drop(columns="first_tree_logloss")

    return df

learning_rates_train = compute_learning_rate(evals_result, 'training')
learning_rates_test = compute_learning_rate(evals_result, 'valid_1')
```


```python
learning_rates_train = compute_learning_rate(evals_result, 'training')
learning_rates_test = compute_learning_rate(evals_result, 'valid_1')

#learning_rates_train = learning_rates_train[learning_rates_train['n_trees'] <= 75]
#learning_rates_test= learning_rates_test[learning_rates_test['n_trees'] <= 75]


combined_data = pd.concat([learning_rates_train, learning_rates_test])

combined_data["n_trees_cat"] = pd.Categorical(
    combined_data["n_trees"],
    categories=sorted(combined_data["n_trees"].unique()),
)


plot = (
    ggplot(combined_data, aes(x='n_trees_cat', y='normalized_learning_rate', color='set')) +
    geom_point() +
    labs(title='Comparison of Normalized Learning Rates',
         x='Number of Trees',
         y='Normalized Learning Rate') +
    scale_color_manual(values=["red", "blue"])
)

print(plot)
```


    
![png](target_train_files/target_train_14_0.png)
    


    



```python
model = lgb.LGBMClassifier(
                    learning_rate=0.001,
                    n_estimators=lightgbm_params["n_estimators"],
                    num_leaves=20,
                    colsample_bytree=lightgbm_params["colsample_bytree"],
                    subsample=lightgbm_params["subsample"],
                    reg_alpha=lightgbm_params["reg_alpha"],
                    reg_lambda=lightgbm_params["reg_lambda"],
                    random_state=lightgbm_params["random_state"],
                    max_depth=5,
                    path_smooth=1,
                
)

model.fit(features, target,
    eval_set=[(features, target), (features_test, target_test)],
        eval_metric = 'logloss',
)
         
evals_result = model.evals_result_
```

    [LightGBM] [Info] Number of positive: 1892, number of negative: 48757
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.211799 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 68429
    [LightGBM] [Info] Number of data points in the train set: 50649, number of used features: 351
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.037355 -> initscore=-3.249214
    [LightGBM] [Info] Start training from score -3.249214
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf



```python
learning_rates_train = compute_learning_rate(evals_result, 'training')
learning_rates_test = compute_learning_rate(evals_result, 'valid_1')

#learning_rates_train = learning_rates_train[learning_rates_train['n_trees'] <= 75]
#learning_rates_test= learning_rates_test[learning_rates_test['n_trees'] <= 75]


combined_data = pd.concat([learning_rates_train, learning_rates_test])

combined_data["n_trees_cat"] = pd.Categorical(
    combined_data["n_trees"],
    categories=sorted(combined_data["n_trees"].unique()),
)


plot = (
    ggplot(combined_data, aes(x='n_trees_cat', y='normalized_learning_rate', color='set')) +
    geom_point() +
    labs(title='Comparison of Normalized Learning Rates',
         x='Number of Trees',
         y='Normalized Learning Rate') +
    scale_color_manual(values=["red", "blue"])
)

print(plot)
```


    
![png](target_train_files/target_train_16_0.png)
    


    



```python

```


```python
get_initial_params()
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
```

    INFO - Starting feature computation
    INFO - Train computation from 2022-12-01 to 2023-05-01. Target for 2023-07-01 00:00:00
    INFO - Test computation from 2023-01-01 00:00:00 to 2023-06-01 00:00:00. Target for 2023-08-01 00:00:00
    /tmp/ipykernel_482/2080027226.py:180: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    INFO - Removing 9564 previous churned users from test set
    INFO - Unique customers in train: 53193
    INFO - Unique customers in test: 43812
    INFO - Starting features and target computation
    INFO - Initial number of features passed: 177
    INFO - Starting computation
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:208: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:210: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:211: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:212: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    /tmp/ipykernel_482/2080027226.py:213: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
    INFO - Final number of features computed: 358
    INFO - Length train data: 53193
    INFO - Length test data: 40805
    INFO - Computation done!
    INFO - Features: ['pago_final_0', 'consumo_0', 'aperiodica_0', 'periodica_0', 'discount_0', 'ajuste_0', 'NUM_GB_OWNN_CURR', 'NUM_GB_2G_CURR', 'NUM_GB_3G_CURR', 'NUM_GB_4G_CURR', 'NUM_GB_5G_CURR', 'NUM_SESS_CURR', 'NUM_SECS_CURR', 'PERC_SECS_TYPE_IN_CURR', 'PERC_SECS_TYPE_OUT_CURR', 'PERC_SECS_OWNN_CURR', 'PERC_SECS_NATR_CURR', 'PERC_SECS_SERV_MOBI_CURR', 'PERC_SECS_TYPE_IN_OWNN_CURR', 'PERC_SECS_TYPE_OUT_OWNN_CURR', 'PERC_SECS_TYPE_IN_NATR_CURR', 'PERC_SECS_TYPE_OUT_NATR_CURR', 'NUM_PLAT_GMM_CURR', 'NUM_PLAT_OMV_CURR', 'NUM_NETW_OWNN_CURR', 'NUM_CALL_CURR', 'PERC_CALL_TYPE_IN_CURR', 'PERC_CALL_TYPE_OUT_CURR', 'PERC_CALL_OWNN_CURR', 'PERC_CALL_NATR_CURR', 'NUM_CALL_WEEK_CURR', 'NUM_CALL_WEEKEND_CURR', 'NUM_SECS_WEEK_CURR', 'NUM_SECS_WEEKEND_CURR', 'NUM_CALL_WEEK', 'NUM_CALL_WEEKEND', 'NUM_DAYS_ACT', 'order_mobile_from_new_alta', 'MIN_DAYS_PERM_CURR', 'PREV_FINISHED_PERM', 'NUM_DAYS_LINE_TYPE_MAIN_POST_ACT', 'Import_Rest_quota_disp', 'LINE_TYPE_FIXE_POST_ACT_LAST_DT_D', 'MAX_DAYS_PERM_CURR', 'MAX_PENALTY_AMOUNT_CURR', 'MIN_PENALTY_AMOUNT_CURR', 'MM_GROUP_MOB_PORT', 'NUM_CALL_INTR_CURR', 'NUM_CALL_NATR_CURR', 'NUM_CALL_OWNN_CURR', 'NUM_CALL_SERV_FIXE_CURR', 'NUM_CALL_SERV_MOBI_CURR', 'NUM_CALL_SERV_UNKN_CURR', 'NUM_CALL_TYPE_IN_CURR', 'NUM_CALL_TYPE_IN_INTR_CURR', 'NUM_CALL_TYPE_IN_NATR_CURR', 'NUM_CALL_TYPE_IN_OWNN_CURR', 'NUM_CALL_TYPE_IN_SERV_FIXE_CURR', 'NUM_CALL_TYPE_IN_SERV_MOBI_CURR', 'NUM_CALL_TYPE_IN_SERV_UNKN_CURR', 'NUM_CALL_TYPE_IN_WEEKEND_CURR', 'NUM_CALL_TYPE_IN_WEEK_CURR', 'NUM_CALL_TYPE_OUT_CURR', 'NUM_CALL_TYPE_OUT_INTR_CURR', 'NUM_CALL_TYPE_OUT_NATR_CURR', 'NUM_CALL_TYPE_OUT_OWNN_CURR', 'NUM_CALL_TYPE_OUT_SERV_FIXE_CURR', 'NUM_CALL_TYPE_OUT_SERV_MOBI_CURR', 'NUM_CALL_TYPE_OUT_SERV_UNKN_CURR', 'NUM_CALL_TYPE_OUT_WEEKEND_CURR', 'NUM_CALL_TYPE_OUT_WEEK_CURR', 'NUM_CUSTOMER_BUNDLE_2P', 'NUM_CUSTOMER_BUNDLE_FMC', 'NUM_CUSTOMER_BUNDLE_MO', 'NUM_DAYS_LINE_TYPE_FIXE_POST_ACT', 'NUM_DISC_ACTI', 'NUM_DISC_CURR', 'NUM_DUR_OBJ', 'NUM_FIX_PORT', 'NUM_FIX_PORT_LAST_1_MONTH', 'NUM_FIX_PORT_LAST_3_MONTHS', 'NUM_FIX_PORT_LAST_6_MONTHS', 'NUM_INTR_CURR', 'NUM_LINES_POST_ACTI', 'NUM_LINES_POST_CURR', 'NUM_LINES_TOTA', 'NUM_LINE_TYPE_ADDI_CURR', 'NUM_LINE_TYPE_ADDI_POST_ACTI', 'NUM_LINE_TYPE_ADDI_POST_CURR', 'NUM_LINE_TYPE_ADDI_TOTA', 'NUM_LINE_TYPE_FIXE_CURR', 'NUM_LINE_TYPE_FIXE_POST_ACTI', 'NUM_LINE_TYPE_FIXE_TOTA', 'NUM_LINE_TYPE_MAIN_CURR', 'NUM_LINE_TYPE_MAIN_POST_ACTI', 'NUM_LINE_TYPE_MAIN_POST_CURR', 'NUM_LINE_TYPE_MAIN_TOTA', 'NUM_MOB_PORT', 'NUM_MOB_PORT_LAST_1_MONTH', 'NUM_MOB_PORT_LAST_3_MONTHS', 'NUM_MOB_PORT_LAST_6_MONTHS', 'NUM_MOB_PORT_REQS_LAST_1_MONTH', 'NUM_MOB_PORT_REQS_LAST_3_MONTHS', 'NUM_MOB_PORT_REQS_LAST_6_MONTHS', 'NUM_MOB_PORT_TRANS_CURR', 'NUM_NATR_CURR', 'NUM_NETW_OSPN_CURR', 'NUM_NETW_RAAS_CURR', 'NUM_NETW_ROAM_CURR', 'NUM_NETW_TMEN_CURR', 'NUM_NETW_VODA_CURR', 'NUM_OWNN_CURR', 'NUM_PERM_CURR', 'NUM_PORT_OPER_DONO_MASMOVIL_GRP_LAST_THREE_MONTHS', 'NUM_PORT_REQS_OPER_DONO_MASMOVIL_GRP_LAST_THREE_MONTHS', 'NUM_PREV_OBJ', 'NUM_RAAS_CURR', 'NUM_SECS_SERV_FIXE_CURR', 'NUM_SECS_SERV_MOBI_CURR', 'NUM_SECS_TYPE_IN_CURR', 'NUM_SECS_TYPE_IN_SERV_FIXE_CURR', 'NUM_SECS_TYPE_IN_SERV_MOBI_CURR', 'NUM_SECS_TYPE_IN_WEEKEND_CURR', 'NUM_SECS_TYPE_IN_WEEK_CURR', 'NUM_SECS_TYPE_OUT_CURR', 'NUM_SECS_TYPE_OUT_SERV_FIXE_CURR', 'NUM_SECS_TYPE_OUT_SERV_MOBI_CURR', 'NUM_SECS_TYPE_OUT_WEEKEND_CURR', 'NUM_SECS_TYPE_OUT_WEEK_CURR', 'ORDER_FIX_FROM_NEW_ALTA', 'cust_days_since_last_fix_port', 'cust_days_since_last_mob_port', 'cust_max_days_between_fix_port', 'cust_max_days_between_mob_port', 'cust_max_months_between_fix_port', 'cust_max_months_between_mob_port', 'cust_min_days_between_fix_port', 'cust_min_days_between_mob_port', 'cust_n_fix_port', 'cust_n_fix_recent_port', 'order_mobile_from_migra_pre_to_post', 'dif_pago_final_prev_month', 'dif_pago_final_prev_2_month', 'dif_pago_final_prev_3_month', 'dif_consumo_prev_month', 'dif_consumo_prev_2_month', 'dif_consumo_prev_3_month', 'dif_discount_prev_month', 'dif_discount_prev_2_month', 'dif_discount_prev_3_month', 'dif_periodica_prev_month', 'dif_periodica_prev_2_month', 'dif_periodica_prev_3_month', 'dif_aperiodica_prev_month', 'dif_aperiodica_prev_2_month', 'dif_aperiodica_prev_3_month', 'dif_ajuste_prev_month', 'dif_ajuste_prev_2_month', 'dif_ajuste_prev_3_month', 'service_mobile_pending_install', 'service_fix_pending_install', 'service_mobile_cancelled', 'service_fix_cancelled', 'service_mobile_pending_install_3month', 'service_fix_pending_install_3month', 'service_mobile_cancelled_3month', 'service_fix_cancelled_3month', 'service_mobile_pending_install_6month', 'service_fix_pending_install_6month', 'service_mobile_cancelled_6month', 'service_fix_cancelled_6month', 'pago_final_0_prev_month', 'pago_final_0_avg_3_months', 'pago_final_0_avg_6_months', 'pago_final_0_std_3_months', 'pago_final_0_std_6_months', 'consumo_0_prev_month', 'consumo_0_avg_3_months', 'consumo_0_avg_6_months', 'consumo_0_std_3_months', 'consumo_0_std_6_months', 'aperiodica_0_prev_month', 'aperiodica_0_avg_3_months', 'aperiodica_0_avg_6_months', 'aperiodica_0_std_3_months', 'aperiodica_0_std_6_months', 'periodica_0_prev_month', 'periodica_0_avg_3_months', 'periodica_0_avg_6_months', 'periodica_0_std_3_months', 'periodica_0_std_6_months', 'discount_0_prev_month', 'discount_0_avg_3_months', 'discount_0_avg_6_months', 'discount_0_std_3_months', 'discount_0_std_6_months', 'ajuste_0_prev_month', 'ajuste_0_avg_3_months', 'ajuste_0_avg_6_months', 'ajuste_0_std_3_months', 'ajuste_0_std_6_months', 'NUM_GB_OWNN_CURR_prev_month', 'NUM_GB_OWNN_CURR_avg_3_months', 'NUM_GB_OWNN_CURR_avg_6_months', 'NUM_GB_OWNN_CURR_std_3_months', 'NUM_GB_OWNN_CURR_std_6_months', 'NUM_GB_2G_CURR_prev_month', 'NUM_GB_2G_CURR_avg_3_months', 'NUM_GB_2G_CURR_avg_6_months', 'NUM_GB_2G_CURR_std_3_months', 'NUM_GB_2G_CURR_std_6_months', 'NUM_GB_3G_CURR_prev_month', 'NUM_GB_3G_CURR_avg_3_months', 'NUM_GB_3G_CURR_avg_6_months', 'NUM_GB_3G_CURR_std_3_months', 'NUM_GB_3G_CURR_std_6_months', 'NUM_GB_4G_CURR_prev_month', 'NUM_GB_4G_CURR_avg_3_months', 'NUM_GB_4G_CURR_avg_6_months', 'NUM_GB_4G_CURR_std_3_months', 'NUM_GB_4G_CURR_std_6_months', 'NUM_GB_5G_CURR_prev_month', 'NUM_GB_5G_CURR_avg_3_months', 'NUM_GB_5G_CURR_avg_6_months', 'NUM_GB_5G_CURR_std_3_months', 'NUM_GB_5G_CURR_std_6_months', 'NUM_SESS_CURR_prev_month', 'NUM_SESS_CURR_avg_3_months', 'NUM_SESS_CURR_avg_6_months', 'NUM_SESS_CURR_std_3_months', 'NUM_SESS_CURR_std_6_months', 'NUM_SECS_CURR_prev_month', 'NUM_SECS_CURR_avg_3_months', 'NUM_SECS_CURR_avg_6_months', 'NUM_SECS_CURR_std_3_months', 'NUM_SECS_CURR_std_6_months', 'PERC_SECS_TYPE_IN_CURR_prev_month', 'PERC_SECS_TYPE_IN_CURR_avg_3_months', 'PERC_SECS_TYPE_IN_CURR_avg_6_months', 'PERC_SECS_TYPE_IN_CURR_std_3_months', 'PERC_SECS_TYPE_IN_CURR_std_6_months', 'PERC_SECS_TYPE_OUT_CURR_prev_month', 'PERC_SECS_TYPE_OUT_CURR_avg_3_months', 'PERC_SECS_TYPE_OUT_CURR_avg_6_months', 'PERC_SECS_TYPE_OUT_CURR_std_3_months', 'PERC_SECS_TYPE_OUT_CURR_std_6_months', 'PERC_SECS_OWNN_CURR_prev_month', 'PERC_SECS_OWNN_CURR_avg_3_months', 'PERC_SECS_OWNN_CURR_avg_6_months', 'PERC_SECS_OWNN_CURR_std_3_months', 'PERC_SECS_OWNN_CURR_std_6_months', 'PERC_SECS_NATR_CURR_prev_month', 'PERC_SECS_NATR_CURR_avg_3_months', 'PERC_SECS_NATR_CURR_avg_6_months', 'PERC_SECS_NATR_CURR_std_3_months', 'PERC_SECS_NATR_CURR_std_6_months', 'PERC_SECS_SERV_MOBI_CURR_prev_month', 'PERC_SECS_SERV_MOBI_CURR_avg_3_months', 'PERC_SECS_SERV_MOBI_CURR_avg_6_months', 'PERC_SECS_SERV_MOBI_CURR_std_3_months', 'PERC_SECS_SERV_MOBI_CURR_std_6_months', 'PERC_SECS_TYPE_IN_OWNN_CURR_prev_month', 'PERC_SECS_TYPE_IN_OWNN_CURR_avg_3_months', 'PERC_SECS_TYPE_IN_OWNN_CURR_avg_6_months', 'PERC_SECS_TYPE_IN_OWNN_CURR_std_3_months', 'PERC_SECS_TYPE_IN_OWNN_CURR_std_6_months', 'PERC_SECS_TYPE_OUT_OWNN_CURR_prev_month', 'PERC_SECS_TYPE_OUT_OWNN_CURR_avg_3_months', 'PERC_SECS_TYPE_OUT_OWNN_CURR_avg_6_months', 'PERC_SECS_TYPE_OUT_OWNN_CURR_std_3_months', 'PERC_SECS_TYPE_OUT_OWNN_CURR_std_6_months', 'PERC_SECS_TYPE_IN_NATR_CURR_prev_month', 'PERC_SECS_TYPE_IN_NATR_CURR_avg_3_months', 'PERC_SECS_TYPE_IN_NATR_CURR_avg_6_months', 'PERC_SECS_TYPE_IN_NATR_CURR_std_3_months', 'PERC_SECS_TYPE_IN_NATR_CURR_std_6_months', 'PERC_SECS_TYPE_OUT_NATR_CURR_prev_month', 'PERC_SECS_TYPE_OUT_NATR_CURR_avg_3_months', 'PERC_SECS_TYPE_OUT_NATR_CURR_avg_6_months', 'PERC_SECS_TYPE_OUT_NATR_CURR_std_3_months', 'PERC_SECS_TYPE_OUT_NATR_CURR_std_6_months', 'NUM_PLAT_GMM_CURR_prev_month', 'NUM_PLAT_GMM_CURR_avg_3_months', 'NUM_PLAT_GMM_CURR_avg_6_months', 'NUM_PLAT_GMM_CURR_std_3_months', 'NUM_PLAT_GMM_CURR_std_6_months', 'NUM_PLAT_OMV_CURR_prev_month', 'NUM_PLAT_OMV_CURR_avg_3_months', 'NUM_PLAT_OMV_CURR_avg_6_months', 'NUM_PLAT_OMV_CURR_std_3_months', 'NUM_PLAT_OMV_CURR_std_6_months', 'NUM_NETW_OWNN_CURR_prev_month', 'NUM_NETW_OWNN_CURR_avg_3_months', 'NUM_NETW_OWNN_CURR_avg_6_months', 'NUM_NETW_OWNN_CURR_std_3_months', 'NUM_NETW_OWNN_CURR_std_6_months', 'NUM_CALL_CURR_prev_month', 'NUM_CALL_CURR_avg_3_months', 'NUM_CALL_CURR_avg_6_months', 'NUM_CALL_CURR_std_3_months', 'NUM_CALL_CURR_std_6_months', 'PERC_CALL_TYPE_IN_CURR_prev_month', 'PERC_CALL_TYPE_IN_CURR_avg_3_months', 'PERC_CALL_TYPE_IN_CURR_avg_6_months', 'PERC_CALL_TYPE_IN_CURR_std_3_months', 'PERC_CALL_TYPE_IN_CURR_std_6_months', 'PERC_CALL_TYPE_OUT_CURR_prev_month', 'PERC_CALL_TYPE_OUT_CURR_avg_3_months', 'PERC_CALL_TYPE_OUT_CURR_avg_6_months', 'PERC_CALL_TYPE_OUT_CURR_std_3_months', 'PERC_CALL_TYPE_OUT_CURR_std_6_months', 'PERC_CALL_OWNN_CURR_prev_month', 'PERC_CALL_OWNN_CURR_avg_3_months', 'PERC_CALL_OWNN_CURR_avg_6_months', 'PERC_CALL_OWNN_CURR_std_3_months', 'PERC_CALL_OWNN_CURR_std_6_months', 'PERC_CALL_NATR_CURR_prev_month', 'PERC_CALL_NATR_CURR_avg_3_months', 'PERC_CALL_NATR_CURR_avg_6_months', 'PERC_CALL_NATR_CURR_std_3_months', 'PERC_CALL_NATR_CURR_std_6_months', 'NUM_CALL_WEEK_CURR_prev_month', 'NUM_CALL_WEEK_CURR_avg_3_months', 'NUM_CALL_WEEK_CURR_avg_6_months', 'NUM_CALL_WEEK_CURR_std_3_months', 'NUM_CALL_WEEK_CURR_std_6_months', 'NUM_CALL_WEEKEND_CURR_prev_month', 'NUM_CALL_WEEKEND_CURR_avg_3_months', 'NUM_CALL_WEEKEND_CURR_avg_6_months', 'NUM_CALL_WEEKEND_CURR_std_3_months', 'NUM_CALL_WEEKEND_CURR_std_6_months', 'NUM_SECS_WEEK_CURR_prev_month', 'NUM_SECS_WEEK_CURR_avg_3_months', 'NUM_SECS_WEEK_CURR_avg_6_months', 'NUM_SECS_WEEK_CURR_std_3_months', 'NUM_SECS_WEEK_CURR_std_6_months', 'NUM_SECS_WEEKEND_CURR_prev_month', 'NUM_SECS_WEEKEND_CURR_avg_3_months', 'NUM_SECS_WEEKEND_CURR_avg_6_months', 'NUM_SECS_WEEKEND_CURR_std_3_months', 'NUM_SECS_WEEKEND_CURR_std_6_months', 'NUM_CALL_WEEK_prev_month', 'NUM_CALL_WEEK_avg_3_months', 'NUM_CALL_WEEK_avg_6_months', 'NUM_CALL_WEEK_std_3_months', 'NUM_CALL_WEEK_std_6_months', 'NUM_CALL_WEEKEND_prev_month', 'NUM_CALL_WEEKEND_avg_3_months', 'NUM_CALL_WEEKEND_avg_6_months', 'NUM_CALL_WEEKEND_std_3_months', 'NUM_CALL_WEEKEND_std_6_months', 'WHICH_MONTH_CHURNED']
    INFO - Target: NUM_DAYS_LINE_TYPE_FIXE_POST_DEA
    INFO - Completed feature computation!
    INFO - Features saved on src/features
    INFO - Targets saved on src/target
    INFO - Starting Modeling
    INFO - Building model pipeline
    INFO - Training model


    [LightGBM] [Info] Number of positive: 1856, number of negative: 48568
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.269378 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 68333
    [LightGBM] [Info] Number of data points in the train set: 50424, number of used features: 351
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.036808 -> initscore=-3.264541
    [LightGBM] [Info] Start training from score -3.264541
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf


    INFO - Completed model training!
    INFO - Started evaluation for Pipeline(steps=[('lightgbm',
                     LGBMClassifier(colsample_bytree=0.64, learning_rate=0.005,
                                    num_leaves=12, random_state=500, reg_alpha=0,
                                    reg_lambda=1, subsample=0.7))])
    INFO - Generating plots



    
![png](target_train_files/target_train_18_3.png)
    


    INFO - Precision in the first decile: 0.14
    INFO - Uplift by decile:
    INFO - decile
    0    3.947955
    1    1.108694
    2    0.792832
    3    0.867608
    4    0.710678
    5    0.696174
    6    0.562108
    7    0.497531
    8    0.532183
    9    0.301928
    Name: y_true, dtype: float64
    INFO - Completed evaluation!
    INFO - Feature importance
    INFO -                                 Feature  Coefficient
    351                 WHICH_MONTH_CHURNED          192
    150            dif_periodica_prev_month           44
    112                       NUM_PERM_CURR           36
    3                           periodica_0           24
    38                   MIN_DAYS_PERM_CURR           20
    74     NUM_DAYS_LINE_TYPE_FIXE_POST_ACT           20
    164  service_fix_pending_install_3month           20
    183           aperiodica_0_avg_6_months           19
    185           aperiodica_0_std_6_months           13
    43                   MAX_DAYS_PERM_CURR           12



```python

```
