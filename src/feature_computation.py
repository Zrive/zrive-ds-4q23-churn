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
        Series: Pandas Series representing the target variable for train set.
        DataFrame: Pandas DataFrame with computed features for model testing.
        Series: Pandas Series representing the target variable for test set.
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
        "service_mobile_pending_install",
        "service_fix_pending_install",
        "service_mobile_cancelled",
        "service_fix_cancelled",
        "service_mobile_pending_install_3month",
        "service_fix_pending_install_3month",
        "service_mobile_cancelled_3month",
        "service_fix_cancelled_3month",
        "service_mobile_pending_install_6month",
        "service_fix_pending_install_6month",
        "service_mobile_cancelled_6month",
        "service_fix_cancelled_6month",
    ]

    # to-be-cooked features
    transform_cols = [
        "pago_final_0",
        "consumo_0",
        "aperiodica_0",
        "discount_0",
        "ajuste_0",
        "NUM_GB_OWNN_CURR",
        "NUM_GB_2G_CURR",
        "NUM_GB_3G_CURR",
        "NUM_GB_4G_CURR",
        "NUM_GB_5G_CURR",
        "NUM_SESS_CURR",
        "NUM_SECS_CURR",
        "NUM_CALL_CURR",
        "NUM_CALL_WEEK_CURR",
        "NUM_CALL_WEEKEND_CURR",
        "NUM_SECS_WEEK_CURR",
        "NUM_SECS_WEEKEND_CURR",
        "NUM_CALL_WEEK",
        "NUM_CALL_WEEKEND",
    ]

    # direct-to-model features
    keep_cols = [
        "NUM_DAYS_ACT",
        "order_mobile_from_new_alta",
        "MIN_DAYS_PERM_CURR",
        "PREV_FINISHED_PERM",
    ]

    # target
    target_col = ["NUM_DAYS_LINE_TYPE_FIXE_POST_DEA"]

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
        f"Train computation from {train_from_dt} to {train_to_dt}. Target for {target_train_month}"
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
    
    # Need to remove users that churned previously to train_to/test_to
    previous_churned_users_train = train_df[(train_df['date'] <= train_to_dt) & (train_df[target_col[0]] > 0)]['customer_id'].unique()
    previous_churned_users_test = test_df[(test_df['date'] <= test_to_dt) & (test_df[target_col[0]] > 0)]['customer_id'].unique()
    train_df = train_df[~train_df['customer_id'].isin(previous_churned_users_train)]
    test_df = test_df[~test_df['customer_id'].isin(previous_churned_users_test)]

    logger.info(f"Removing {len(previous_churned_users_train)} previous churned users from train set")
    logger.info(f"Removing {len(previous_churned_users_test)} previous churned users from test set")

    logger.info("Starting features and target computation")

    train_df_features = compute_features(train_df, target_col, train_to_dt)
    test_df_features = compute_features(test_df, target_col, test_to_dt)
    train_df_target = compute_target(
        compute_ready_data, target_col, target_train_month, False
    )
    test_df_target = compute_target(
        compute_ready_data, target_col, target_test_month, False
    )

    logger.info("Features computed")

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

    train_df = df[(df["date"] >= train_from) & (df["date"] <= train_to)]
    test_df = df[(df["date"] >= test_from) & (df["date"] <= test_to)]

    return train_df, test_df


def compute_features(
    df: pd.DataFrame, target_col: list[str], train_to_dt: pd.Timestamp
) -> pd.DataFrame:
    """
    Compute the features and adds them to the df.
    Args:
        df: The clean dataset with the columns we want to use as features.
        target_col: Name of the target column.

    Returns:
        DataFrame: Pandas DataFrame with new computed variables.
    """

    # TO-DO: The rolling function is propagated backwards for each single month.
    # we just need it for the last one (but for that we need past data also). Didn't find
    # any option to do it with pandas
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
    df["NUM_GB_OWNN_CURR_avg_3_months"] = compute_x_months_avg(
        df, "NUM_GB_OWNN_CURR", 3
    )
    df["NUM_GB_OWNN_CURR_avg_6_months"] = compute_x_months_avg(
        df, "NUM_GB_OWNN_CURR", 6
    )
    df["NUM_SECS_CURR_avg_3_months"] = compute_x_months_avg(df, "NUM_SECS_CURR", 3)
    df["NUM_SECS_CURR_avg_6_months"] = compute_x_months_avg(df, "NUM_SECS_CURR", 6)
    df["NUM_CALL_WEEK_CURR_avg_3_months"] = compute_x_months_avg(
        df, "NUM_CALL_WEEK_CURR", 3
    )
    df["NUM_CALL_WEEK_CURR_avg_6_months"] = compute_x_months_avg(
        df, "NUM_CALL_WEEK_CURR", 6
    )
    df["NUM_CALL_WEEKEND_CURR_avg_3_months"] = compute_x_months_avg(
        df, "NUM_CALL_WEEKEND_CURR", 3
    )
    df["NUM_CALL_WEEKEND_CURR_avg_6_months"] = compute_x_months_avg(
        df, "NUM_CALL_WEEKEND_CURR", 6
    )
    df["NUM_SECS_WEEK_CURR_avg_3_months"] = compute_x_months_avg(
        df, "NUM_SECS_WEEK_CURR", 3
    )
    df["NUM_SECS_WEEK_CURR_avg_6_months"] = compute_x_months_avg(
        df, "NUM_SECS_WEEK_CURR", 6
    )
    df["NUM_SECS_WEEKEND_CURR_avg_3_months"] = compute_x_months_avg(
        df, "NUM_SECS_WEEKEND_CURR", 3
    )
    df["NUM_SECS_WEEKEND_CURR_avg_6_months"] = compute_x_months_avg(
        df, "NUM_SECS_WEEKEND_CURR", 6
    )

    # Filter only the computation backwards from the last month
    df = df[df["date"] == train_to_dt]

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

    drop_churn_between_month = target_month - pd.DateOffset(months=1)

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

    if keep_gap_month_churns:
        # Convert all values that are 2 (gap month churns) into 1 (real churns)
        target_df[col] = np.where(target_df[col] == 2, 1, target_df[col])

    # Exclude the records that are still marked as 2 (gap month churns) if keep_gap_month_churns is False
    target_df = target_df[
        (target_df["NUM_DAYS_LINE_TYPE_FIXE_POST_DEA"] != 2)
        & (target_df["date"] != drop_churn_between_month)
    ][["customer_id"] + target_col]

    target_df[target_col] = target_df[target_col].astype("int")

    return target_df
