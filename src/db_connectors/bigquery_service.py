import json
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud import bigquery_storage_v1


class BigqueryService:
    """
    A class to represent the Bigquery Client

    Methods:
        query_to_df(query):
            returns Dataframe from sql query
    """

    def __init__(self, path="config/mm-bi-catedras-upm.json") -> None:
        """
        Constructs the Bigquery Client with the json credentials
        """
        self.path = path

        with open(path) as credentials_json:
            credentials = json.load(credentials_json)

        bq_credentials = service_account.Credentials.from_service_account_info(
            credentials
        )

        self.client = bigquery.Client(credentials=bq_credentials)
        self.bqstorageclient = bigquery_storage_v1.BigQueryReadClient(
            credentials=bq_credentials
        )

        # TODO: Catch file errors and BigQuery exceptions

    def query_to_df(self, query: str) -> pd.DataFrame:
        """
        Executes SQL query and convert it to a Dataframe

        Returns:
            Pandas Dataframe with sql results
        """

        query_job = self.client.query(query)

        # Use BigQuery Storage API
        return query_job.result().to_dataframe(
            create_bqstorage_client=self.bqstorageclient
        )
