import pandas as pd

def feature_computation(clean_data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Selects and computes significant features for churn prediction from cleaned data, including creating derived variables and initial transformations.

    Returns:
        DataFrame: Pandas DataFrame with computed features for model training.
    """

    features = clean_data[["pago_final_0", "IS_CUST_SEGM_RESI", "ORDER_MOBILE_FROM_PORTABILITY"]]
    target = clean_data[["NUM_DAYS_LINE_TYPE_FIXE_POST_DEA"]]
    print(features)
    target = target.clip(upper=1)
    target = target.fillna(0)
    print(target)

    return features, target
