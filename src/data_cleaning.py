import pandas as pd


def data_cleaning(raw_df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Cleans raw data by handling missing values, removing duplicates, correcting errors, and performing type conversions for data quality and consistency.
    Returns:
        DataFrame: Pandas DataFrame with cleaned and preprocessed data.
    """
    logger.info("Starting cleaning data")

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

    filter_df = raw_df[users_cols + diff_cols + keep_cols + transform_cols + target_col]
    clean_df = filter_df.dropna(how="all")

    columns_to_drop = []
    for col in clean_df.columns:
        number_nulls = clean_df[col].isnull().sum()
        if number_nulls > len(clean_df) // 2 and col != target_col[0]:
            columns_to_drop.append(col)
            logger.info(f"Dropping column {col}")
        elif number_nulls > 0:
            clean_df[col] = clean_df[col].fillna(value=0)

    clean_df = clean_df.drop(columns=columns_to_drop)

    logger.info("Completed cleaning data!")
    logger.info(clean_df.head())
    return clean_df
