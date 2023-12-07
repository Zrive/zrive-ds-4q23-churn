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

    filter_df = raw_df[users_cols + diff_cols + transform_cols + target_col]
    clean_df = filter_df.dropna(how="all")

    logger.info("Completed cleaning data!")
    logger.info(clean_df.head())
    return clean_df
