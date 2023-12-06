import pandas as pd
from db_connectors.bigquery_service import BigqueryService


def data_gathering(query: str, logger) -> pd.DataFrame:
    """
    Gathers raw data from DB and create a ready-to-use format.

    Returns:
        DataFrame: Pandas DataFrame with collected raw data
    """
    logger.info("Started querying data")
    bq_client = BigqueryService()

    result_df = bq_client.query_to_df(query)
    logger.info(f"Data succesfully retrieved! Length: {len(result_df)}")

    return result_df
