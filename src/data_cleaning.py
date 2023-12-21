import pandas as pd
from column_config import (
    users_cols,
    diff_cols,
    keep_cols,
    transform_cols,
    target_col,
)


def column_to_float(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Change type of the column values to float
    Args:
        df: original dataframe we want to change the type
        col_name: name of the column we want to change the type
    Returns:
        DataFrame: Only one column with float type.
    """
    return df[col_name].astype(float)


def data_cleaning(raw_df: pd.DataFrame, logger) -> pd.DataFrame:
    """
    Cleans raw data by handling missing values, removing duplicates, correcting errors, and performing type conversions for data quality and consistency.
    Returns:
        DataFrame: Pandas DataFrame with cleaned and preprocessed data.
    """
    logger.info("Starting cleaning data")

    filter_df = raw_df[users_cols + diff_cols + keep_cols + transform_cols + target_col]
    clean_df = filter_df.dropna(how="all")

    columns_to_drop = []
    for col in clean_df.columns:
        number_nulls = clean_df[col].isnull().sum()
        if number_nulls > len(clean_df) // 2 and col != target_col[0]:
            columns_to_drop.append(col)
            logger.info(f"Dropping column {col}")
        elif number_nulls > 0:
            # clean_df[col] = clean_df[col].fillna(value=0)
            logger.info("Should fillna")

    # clean_df = clean_df.drop(columns=columns_to_drop)
    clean_df["MAX_PENALTY_AMOUNT_CURR"] = column_to_float(
        clean_df, "MAX_PENALTY_AMOUNT_CURR"
    )
    clean_df["MIN_PENALTY_AMOUNT_CURR"] = column_to_float(
        clean_df, "MIN_PENALTY_AMOUNT_CURR"
    )

    logger.info("Completed cleaning data!")
    logger.info(clean_df.head())
    return clean_df
