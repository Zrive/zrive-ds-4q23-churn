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
    transform_cols = [
        "pago_final_0",
        "consumo_0",
        "aperiodica_0",
        "discount_0",
        "ajuste_0",
    ]

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
    test_from_dt = train_from_dt + pd.DateOffset(months=1)
    test_to_dt = train_to_dt + pd.DateOffset(months=1)
    target_train_month = train_to_dt + pd.DateOffset(months=2)
    target_test_month = test_to_dt + pd.DateOffset(months=2)

    logger.info(
        f"Train computation from {train_from_dt} to {train_to_dt}. Target for {target_train_month}"
    )
    logger.info(
        f"Test computation from {test_from_dt} to {test_to_dt}. Target for {target_test_month}"
    )

    compute_ready_data = clean_data[
        users_cols + transform_cols + diff_cols + target_col
    ].copy()
    train_df, test_df = split_train_test(
        compute_ready_data, train_from_dt, train_to_dt, test_from_dt, test_to_dt
    )

    logger.info("Starting features and target computation")

    train_df_features = compute_features(train_df, target_col)
    test_df_features = compute_features(test_df, target_col)
    train_df_target = compute_target(compute_ready_data, target_col, target_train_month)
    test_df_target = compute_target(compute_ready_data, target_col, target_test_month)

    logger.info("Features computed")

    # Select only the most recent month's data per customer
    # final_features_train = features_train.groupby('customer_id').tail(1)
    # final_features_test = features_test.groupby('customer_id').tail(1)

    # As there are customer that leave between the month we use for training and the target month
    # We have to join the features and the targets and drop those that don't have target
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

    # Split train and test features + target (squeeze into 1D array)
    features = features_and_target_train.drop(
        columns=target_col + users_cols + ["date"]
    )
    features_test = features_and_target_test.drop(
        columns=target_col + users_cols + ["date"]
    )
    target = features_and_target_train[target_col].squeeze()
    target_test = features_and_target_test[target_col].squeeze()

    logger.info(f"Features: {features.columns.tolist()}")
    logger.info(f"Target: {target.name}")
    logger.info("Completed feature computation!")

    return features, target, features_test, target_test


def split_train_test(
    df: pd.DataFrame,
    train_from: pd.Series,
    train_to: pd.Series,
    test_from: pd.Series,
    test_to: pd.Series,
) -> (pd.DataFrame, pd.DataFrame):
    """
    Split data into train and test.
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
    df["date"] = pd.to_datetime(
        df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01"
    )

    # Filter compute_data for the specific date intervals.
    df = df[(df["date"] >= train_from) & (df["date"] <= test_to)]

    features_train = df[df["date"] == train_to]
    features_test = df[(df["date"] == test_to)]

    return features_train, features_test


def compute_features(df: pd.DataFrame, target_col: list[str]) -> pd.DataFrame:
    """
    Compute the features and adds them to the df.
    Args:
        df: The clean dataset with the columns we want to use as features.
        target_col: Name of the target column.

    Returns:
        DataFrame: Pandas DataFrame with new computed variables.
    """
    df = df.drop(columns=target_col)

    df = df.sort_values(by=["customer_id", "date"])
    df["pago_final_prev_month"] = df.groupby("customer_id")["pago_final_0"].shift(1)
    df["pago_final_prev_month"] = df["pago_final_prev_month"].fillna(0)
    df["pago_final_avg_3_months"] = compute_x_months_avg(df, "pago_final_0", 3)
    df["pago_final_avg_6_months"] = compute_x_months_avg(df, "pago_final_0", 6)
    df["consumo_avg_3_months"] = compute_x_months_avg(df, "consumo_0", 3)
    df["consumo_avg_6_months"] = compute_x_months_avg(df, "consumo_0", 6)
    df["aperiodica_avg_3_months"] = compute_x_months_avg(df, "aperiodica_0", 3)
    df["aperiodica_avg_6_months"] = compute_x_months_avg(df, "aperiodica_0", 6)
    df["discount_avg_3_months"] = compute_x_months_avg(df, "discount_0", 3)
    df["discount_avg_6_months"] = compute_x_months_avg(df, "discount_0", 6)
    df["ajuste_avg_3_months"] = compute_x_months_avg(df, "ajuste_0", 3)
    df["ajuste_avg_6_months"] = compute_x_months_avg(df, "ajuste_0", 6)

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
    return df.groupby("customer_id")[col_name].transform(
        lambda x: x.rolling(window=months, min_periods=1).mean()
    )


def compute_target(
    df: pd.DataFrame,
    target_col: str,
    target_month: pd.Series,
) -> pd.DataFrame:
    """
    Compute the target column for a df.
    Args:
        df: The clean dataset with all the data.
        target_col: Name of the target column.
        target_month: The date where the target has to be computed.

    Returns:
        DataFrame: Pandas DataFrame with the customer_id and the target computed.
    """

    drop_churn_between_month = target_month + pd.DateOffset(months=-1)

    target_df = df[
        (df["date"] == target_month) | (df["date"] == drop_churn_between_month)
    ][["customer_id"] + target_col + ["date"]]

    for col in target_col:
        target_df[col].fillna(0, inplace=True)
        target_df[col] = np.where(
            ((target_df[col] > 0) & (target_df["date"] == target_month)),
            1,
            target_df[col],
        )
        target_df[col] = np.where(
            ((target_df[col] > 0) & (target_df["date"] == drop_churn_between_month)),
            2,
            target_df[col],
        )

    target_df = target_df[
        (target_df["NUM_DAYS_LINE_TYPE_FIXE_POST_DEA"] != 2)
        & (target_df["date"] != drop_churn_between_month)
    ][["customer_id"] + target_col]

    target_df[target_col] = target_df[target_col].astype("int")

    return target_df
