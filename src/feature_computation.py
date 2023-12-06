import pandas as pd
import numpy as np


def feature_computation(
    clean_data: pd.DataFrame, train_from: str, train_to: str, logger
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
        DataFrame: Pandas DataFrame representing the target variable for train set.
        DataFrame: Pandas DataFrame with computed features for model testing.
        DataFrame: Pandas DataFrame representing the target variable for test set.
    """
    logger.info("Starting feature computation")
    # user-info cols to aggregate data later on
    users_cols = ["customer_id", "MONTH", "YEAR"]

    # pre-cooked features
    diff_cols = [
        "dif_pago_final_prev_month",
        "dif_pago_final_prev_2_month",
        "dif_pago_final_prev_3_month",
        "dif_consumo_prev_month",
        "dif_consumo_prev_2_month",
        "dif_consumo_prev_3_month",
        "dif_discount_prev_month",
        "dif_discount_prev_2_month",
        "dif_discount_prev_3_month",
    ]
    # to-be-cooked features
    transform_cols = ["pago_final_0"]

    # target
    target_col = ["NUM_DAYS_LINE_TYPE_FIXE_POST_DEA"]

    # TO-DO: Catch exceptions
    # TO-DO: Potential unit tests validating same length for features/targets
    # TO-DO: Test should be 1 month in advance, not 2!
    # TO-DO: Instead of defining the cols every time import them somewhere else (they're need in data_cleaning also)
    # Isolate the feature computation from the target comput. in 2 diff functions

    # Convert the train_from and train_to to datetime
    train_from_dt = pd.to_datetime(train_from)
    train_to_dt = pd.to_datetime(train_to)

    # Filter train and test data before feature computation
    test_from_dt = train_from_dt + pd.DateOffset(months=2)
    test_to_dt = train_to_dt + pd.DateOffset(months=2)
    target_train_month = train_to_dt + pd.DateOffset(months=2)
    target_test_month = target_train_month + pd.DateOffset(months=2)

    # Code suggested
    """train_df, train_df_target, test_df, test_df_target= split_train_test(train_from,train_to)
    train_df_features = compute_features(train_df)
    test_df_features = compute_features(test_df)"""

    logger.info(
        f"Train computation from {train_from_dt} to {train_to_dt}. Target for {target_train_month}"
    )
    logger.info(
        f"Test computation from {test_from_dt} to {test_to_dt}. Target for {target_test_month}"
    )

    # Create date col to mix month and year
    clean_data["date"] = pd.to_datetime(
        clean_data["YEAR"].astype(str) + "-" + clean_data["MONTH"].astype(str) + "-01"
    )

    # Filter compute_data for the specific cols and date intervals. Also sort i.
    compute_data = clean_data[["date"] + users_cols + transform_cols + diff_cols]
    compute_data = compute_data[
        (compute_data["date"] >= train_from_dt) & (compute_data["date"] <= test_to_dt)
    ]

    # Perform feature computations for the combined dataset. Assigns nans if needed.
    compute_data = compute_data.sort_values(by=["customer_id", "date"])
    compute_data["pago_final_prev_month"] = compute_data.groupby("customer_id")[
        "pago_final_0"
    ].shift(1)
    compute_data["pago_final_prev_month"] = compute_data[
        "pago_final_prev_month"
    ].fillna(0)
    compute_data["pago_final_avg_3_months"] = compute_data.groupby("customer_id")[
        "pago_final_0"
    ].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    logger.info("Features computed")

    # Split the combined dataset into training and testing sets
    features_train = compute_data[compute_data["date"] == train_to_dt]
    features_test = compute_data[(compute_data["date"] == test_to_dt)]

    # Select only the most recent month's data per customer
    # final_features_train = features_train.groupby('customer_id').tail(1)
    # final_features_test = features_test.groupby('customer_id').tail(1)

    # Extract the target for the training and testing sets
    target_train = clean_data[clean_data["date"] == target_train_month][
        ["customer_id"] + target_col
    ]
    target_test = clean_data[clean_data["date"] == target_test_month][
        ["customer_id"] + target_col
    ]

    for target_df in [target_train, target_test]:
        for col in target_col:
            target_df[col].fillna(0, inplace=True)
            target_df[col] = np.where(target_df[col] > 0, 1, 0)

    # Now we need to join it with customer_id from features df
    # Check: i'm using an inner join because there are some edge cases to clarify (e.g. customer_id = 1322985)
    features_and_target_train = features_train.merge(
        target_train, on="customer_id", how="inner"
    )
    features_and_target_test = features_test.merge(
        target_test, on="customer_id", how="inner"
    )

    # Split train and test features + target (squeeze into 1D array)
    features = features_and_target_train.drop(
        columns=target_col + users_cols + ["date"]
    )
    target = features_and_target_train[target_col].squeeze()
    features_test = features_and_target_test.drop(
        columns=target_col + users_cols + ["date"]
    )
    target_test = features_and_target_test[target_col].squeeze()

    logger.info(f"Features: {features.columns.tolist()}")
    logger.info(f"Target: {target.name}")
    logger.info("Completed feature computation!")

    return features, target, features_test, target_test
