import pandas as pd
import pytest

from src.data_gathering import data_gathering
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_data_gathering():
    sample_query = """
    SELECT "1" as col_1, "2" as col_2, "3" as col_3
    FROM `mm-bi-catedras-upm.ESTIMACION_CHURN.multibrand_monthly_customer_base_mp2022`
    LIMIT 3
    """

    expected_result = pd.DataFrame(
        {"col_1": ["1", "1", "1"], "col_2": ["2", "2", "2"], "col_3": ["3", "3", "3"]}
    )

    pd.testing.assert_frame_equal(data_gathering(sample_query, logger), expected_result)
