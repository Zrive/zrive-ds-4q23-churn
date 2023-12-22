import pandas as pd
from src.feature_computation import (
    split_train_test,
    compute_features,
    compute_target,
    compute_x_months_avg,
    compute_x_months_std,
)
from unittest.mock import patch


def test_split_train():
    train_from = pd.to_datetime("2023-08-01")
    train_to = pd.to_datetime("2023-09-01")
    test_from = pd.to_datetime("2023-11-01")
    test_to = pd.to_datetime("2023-12-01")

    data = pd.DataFrame(
        {
            "col_1": [1, 2, 3, 4, 5, 6],
            "MONTH": ["07", "08", "09", "10", "11", "12"],
            "YEAR": ["2023", "2023", "2023", "2023", "2023", "2023"],
        }
    )

    expected_result_train = pd.DataFrame(
        {
            "col_1": [2, 3],
            "MONTH": ["08", "09"],
            "YEAR": ["2023", "2023"],
            "date": [train_from, train_to],
        }
    )

    expected_result_test = pd.DataFrame(
        {
            "col_1": [5, 6],
            "MONTH": ["11", "12"],
            "YEAR": ["2023", "2023"],
            "date": [test_from, test_to],
        }
    )

    train_df_res, test_df_res = split_train_test(
        data, train_from, train_to, test_from, test_to
    )

    assert_df_cols = ["col_1", "MONTH", "YEAR", "date"]

    pd.testing.assert_frame_equal(
        train_df_res[assert_df_cols].reset_index(drop=True),
        expected_result_train.reset_index(drop=True),
    )

    pd.testing.assert_frame_equal(
        test_df_res[assert_df_cols].reset_index(drop=True),
        expected_result_test.reset_index(drop=True),
    )


def test_compute_target():
    target_col = ["NUM_DAYS_LINE_TYPE_FIXE_POST_DEA"]
    target_month = pd.to_datetime("2023-12-01")

    data = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "NUM_DAYS_LINE_TYPE_FIXE_POST_DEA": [0, 7, 4],
            "date": [
                target_month,
                target_month - pd.DateOffset(months=1),
                target_month,
            ],
        }
    )

    expected_result = pd.DataFrame(
        {"customer_id": [1, 3], "NUM_DAYS_LINE_TYPE_FIXE_POST_DEA": [0, 1]}
    )

    target_df = compute_target(data, target_col, target_month)

    pd.testing.assert_frame_equal(
        target_df.reset_index(drop=True), expected_result.reset_index(drop=True)
    )


def test_compute_target_not_drop_churnbwetween_months():
    target_col = ["NUM_DAYS_LINE_TYPE_FIXE_POST_DEA"]
    target_month = pd.to_datetime("2023-12-01")

    data = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "NUM_DAYS_LINE_TYPE_FIXE_POST_DEA": [0, 7, 4],
            "date": [
                target_month,
                target_month - pd.DateOffset(months=1),
                target_month,
            ],
        }
    )

    expected_result = pd.DataFrame(
        {"customer_id": [1, 2, 3], "NUM_DAYS_LINE_TYPE_FIXE_POST_DEA": [0, 1, 1]}
    )

    target_df = compute_target(
        data, target_col, target_month, keep_gap_month_churns=True
    )

    pd.testing.assert_frame_equal(
        target_df.reset_index(drop=True), expected_result.reset_index(drop=True)
    )
