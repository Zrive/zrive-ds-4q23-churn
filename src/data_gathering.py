import pandas as pd
from db_connectors.bigquery_service import BigqueryService


def data_gathering() -> pd.DataFrame:
    """
    Gathers raw data from DB and create a ready-to-use format.

    Returns:
        DataFrame: Pandas DataFrame with collected raw data
    """

    bq_client = BigqueryService()

    query = """
            SELECT *
            FROM `mm-bi-catedras-upm.ESTIMACION_CHURN.multibrand_monthly_customer_base_mp2023_1`
            LIMIT 5
            """

    return bq_client.query_to_df(query)
