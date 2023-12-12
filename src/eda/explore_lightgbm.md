```python
import sys
import os
import pandas as pd
import numpy as np
import configparser
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns

project_root = "/mnt/c/Users/tonbo/Documents/ZriveDS/zrive-ds-4q24-churn"


# Define the project root path
current_wd = os.getcwd()

# Change the working directory if necessary
if current_wd != project_root:
    print(f"Changing working directory from {current_wd} to {project_root}")
    os.chdir(project_root)
else:
    print("Already in the correct path")

# Add 'src' directory to sys.path
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    print(f"Adding {src_path} to sys.path")
    sys.path.insert(0, src_path)

# Import the modules
from db_connectors.bigquery_service import BigqueryService
from data_gathering import data_gathering
from utils.logger import get_logger
from data_cleaning import data_cleaning
from feature_computation import feature_computation

logger = get_logger(__name__)
```

    Changing working directory from /mnt/c/Users/tonbo/Documents/ZriveDS/zrive-ds-4q24-churn/src/eda to /mnt/c/Users/tonbo/Documents/ZriveDS/zrive-ds-4q24-churn
    Adding /mnt/c/Users/tonbo/Documents/ZriveDS/zrive-ds-4q24-churn/src to sys.path



```python
config = configparser.ConfigParser()
config.read("src/params.ini")
```




    ['src/params.ini']




```python
train_from = config.get("PARAMS", "train_from")
train_to = config.get("PARAMS", "train_to")

# Load Logistic Regression parameters
logistic_regression_params = {
    "penalty": config.get("LOGISTIC_REGRESSION", "penalty", fallback="l2"),
    "C": config.getfloat("LOGISTIC_REGRESSION", "C", fallback=1.0),
    "solver": config.get("LOGISTIC_REGRESSION", "solver", fallback="saga"),
    "max_iter": config.getint("LOGISTIC_REGRESSION", "max_iter", fallback=10000),
}

# Load LightGBM parameters
lightgbm_params = {
    "boosting_type": config.get("LIGHTGBM", "boosting_type", fallback="gbdt"),
    "num_leaves": config.getint("LIGHTGBM", "num_leaves", fallback=31),
    "max_depth": config.getint("LIGHTGBM", "max_depth", fallback=-1),
    "learning_rate": config.getfloat("LIGHTGBM", "learning_rate", fallback=0.1),
    "n_estimators": config.getint("LIGHTGBM", "n_estimators", fallback=100),
}
```


```python
query = """
WITH selectable_customer AS (
    SELECT customer_id
    FROM `mm-bi-catedras-upm.ESTIMACION_CHURN.multibrand_monthly_customer_base_mp2022`
    GROUP BY customer_id
), 

customer_selected AS (
    SELECT customer_id AS selected_customer
    FROM   selectable_customer
    WHERE  RAND() < 0.25
)

SELECT 
    customer_id,
    MONTH,
    YEAR,
    NUM_DAYS_ACT,
    order_mobile_from_new_alta,
    service_mobile_pending_install,
    service_fix_pending_install,
    service_mobile_cancelled,
    service_fix_cancelled,
    service_mobile_pending_install_3month,
    service_fix_pending_install_3month,
    service_mobile_cancelled_3month,
    service_fix_cancelled_3month,
    service_mobile_pending_install_6month,
    service_fix_pending_install_6month,
    service_mobile_cancelled_6month,
    service_fix_cancelled_6month,
    pago_final_0,
    dif_pago_final_prev_month,
    dif_pago_final_prev_2_month,
    dif_pago_final_prev_3_month,
    periodica_0,
    dif_periodica_prev_month,
    dif_periodica_prev_2_month,
    dif_periodica_prev_3_month,
    consumo_0,
    dif_consumo_prev_month,
    dif_consumo_prev_2_month,
    dif_consumo_prev_3_month,
    aperiodica_0,
    dif_aperiodica_prev_month,
    dif_aperiodica_prev_2_month,
    dif_aperiodica_prev_3_month,
    discount_0,
    dif_discount_prev_month,
    dif_discount_prev_2_month,
    dif_discount_prev_3_month,
    ajuste_0,
    dif_ajuste_prev_month,
    dif_ajuste_prev_2_month,
    dif_ajuste_prev_3_month,
    Tota_Compra_disp,
    Curr_Compra_disp,
    Curr_Compra_Finanz_disp,
    Curr_Finanz_disp,
    Month_purchase_disp,
    Modelo_disp,
    Import_Rest_quota_disp,
    pvp_total_disp,
    pvp_total_disp_movil,
    Curr_cancel_disp,
    Tota_cancel_disp,
    MIN_DAYS_PERM_CURR,
    MIN_PENALTY_AMOUNT_CURR,
    PREV_FINISHED_PERM,
    NUM_GB_OWNN_CURR,
    NUM_GB_2G_CURR,
    NUM_GB_3G_CURR,
    NUM_GB_4G_CURR,
    NUM_GB_5G_CURR,
    NUM_SESS_CURR,
    NUM_SECS_CURR,
    NUM_CALL_CURR,
    NUM_CALL_WEEK_CURR,
    NUM_CALL_WEEKEND_CURR,
    NUM_SECS_WEEK_CURR,
    NUM_SECS_WEEKEND_CURR,
    NUM_CALL_WEEK,
    NUM_CALL_WEEKEND,
    NUM_DAYS_LINE_TYPE_FIXE_POST_DEA
FROM `mm-bi-catedras-upm.ESTIMACION_CHURN.multibrand_monthly_customer_base_mp2022`
INNER JOIN customer_selected
ON customer_id = selected_customer
WHERE IS_CUST_SEGM_RESI > 0
AND IS_CUST_BILL_POST_CURR = TRUE
AND CUST_BUNDLE_CURR = 'FMC'
AND NUM_IMPAGOS = 0
AND pago_final_0 IS NOT NULL
"""
```


```python
raw_data = data_gathering(query, logger)
clean_data = data_cleaning(raw_data, logger)
```

    INFO - Started querying data
    INFO - Data succesfully retrieved! Length: 716237
    INFO - Starting cleaning data
    INFO - Completed cleaning data!
    INFO -   customer_id MONTH  YEAR  dif_pago_final_prev_month  \
    0      359543    12  2022                      -1.32   
    1     4124818    12  2022                       0.07   
    2      279557    12  2022                       0.00   
    3      199882    12  2022                     -57.76   
    4     1417539    09  2022                       3.99   
    
       dif_pago_final_prev_2_month  dif_pago_final_prev_3_month  \
    0                        -0.60                         0.00   
    1                        -4.53                        -9.12   
    2                         0.30                        -0.30   
    3                       -53.80                       -52.73   
    4                         4.04                         5.00   
    
       dif_consumo_prev_month  dif_consumo_prev_2_month  dif_consumo_prev_3_month  \
    0                    9.93                    -27.75                   -195.06   
    1                  -46.11                    -23.63                    -94.70   
    2                    0.01                    -12.77                     -3.53   
    3                  -27.61                    -28.53                     85.03   
    4                   28.74                     25.70                     25.43   
    
       dif_discount_prev_month  ...  NUM_SESS_CURR  NUM_SECS_CURR  NUM_CALL_CURR  \
    0                   -11.25  ...           7311          37937            608   
    1                    46.18  ...           2263          34619            208   
    2                    -0.01  ...           4667           8209            108   
    3                    10.52  ...           8242          71757            426   
    4                   -32.70  ...           7634          20117            153   
    
       NUM_CALL_WEEK_CURR  NUM_CALL_WEEKEND_CURR  NUM_SECS_WEEK_CURR  \
    0                 423                    185               30177   
    1                 154                     54               26222   
    2                  75                     33                6304   
    3                 318                    108               55416   
    4                 143                     10               18150   
    
       NUM_SECS_WEEKEND_CURR  NUM_CALL_WEEK  NUM_CALL_WEEKEND  \
    0                   7760            318               132   
    1                   8397            110                43   
    2                   1905             68                32   
    3                  16341            260                87   
    4                   1967            118                 9   
    
       NUM_DAYS_LINE_TYPE_FIXE_POST_DEA  
    0                                 0  
    1                                 0  
    2                                 0  
    3                                 0  
    4                                 0  
    
    [5 rows x 48 columns]



```python
features, target, features_test, target_test = feature_computation(
    clean_data, train_from, train_to, logger
)
```

    INFO - Starting feature computation
    INFO - Train computation from 2022-01-01 00:00:00 to 2022-06-01 00:00:00. Target for 2022-08-01 00:00:00
    INFO - Test computation from 2022-02-01 00:00:00 to 2022-07-01 00:00:00. Target for 2022-09-01 00:00:00
    INFO - Removing 23617 previous churned users from train set
    INFO - Removing 26871 previous churned users from test set
    INFO - Starting features and target computation
    INFO - Features computed
    INFO - Features: ['pago_final_0', 'consumo_0', 'aperiodica_0', 'discount_0', 'ajuste_0', 'NUM_GB_OWNN_CURR', 'NUM_GB_2G_CURR', 'NUM_GB_3G_CURR', 'NUM_GB_4G_CURR', 'NUM_GB_5G_CURR', 'NUM_SESS_CURR', 'NUM_SECS_CURR', 'NUM_CALL_CURR', 'NUM_CALL_WEEK_CURR', 'NUM_CALL_WEEKEND_CURR', 'NUM_SECS_WEEK_CURR', 'NUM_SECS_WEEKEND_CURR', 'NUM_CALL_WEEK', 'NUM_CALL_WEEKEND', 'NUM_DAYS_ACT', 'order_mobile_from_new_alta', 'MIN_DAYS_PERM_CURR', 'PREV_FINISHED_PERM', 'dif_pago_final_prev_month', 'dif_pago_final_prev_2_month', 'dif_pago_final_prev_3_month', 'dif_consumo_prev_month', 'dif_consumo_prev_2_month', 'dif_consumo_prev_3_month', 'dif_discount_prev_month', 'dif_discount_prev_2_month', 'dif_discount_prev_3_month', 'service_mobile_pending_install', 'service_fix_pending_install', 'service_mobile_cancelled', 'service_fix_cancelled', 'service_mobile_pending_install_3month', 'service_fix_pending_install_3month', 'service_mobile_cancelled_3month', 'service_fix_cancelled_3month', 'service_mobile_pending_install_6month', 'service_fix_pending_install_6month', 'service_mobile_cancelled_6month', 'service_fix_cancelled_6month', 'pago_final_prev_month', 'pago_final_avg_3_months', 'pago_final_avg_6_months', 'consumo_avg_3_months', 'consumo_avg_6_months', 'aperiodica_avg_3_months', 'aperiodica_avg_6_months', 'discount_avg_3_months', 'discount_avg_6_months', 'ajuste_avg_3_months', 'ajuste_avg_6_months', 'NUM_GB_OWNN_CURR_avg_3_months', 'NUM_GB_OWNN_CURR_avg_6_months', 'NUM_SECS_CURR_avg_3_months', 'NUM_SECS_CURR_avg_6_months', 'NUM_CALL_WEEK_CURR_avg_3_months', 'NUM_CALL_WEEK_CURR_avg_6_months', 'NUM_CALL_WEEKEND_CURR_avg_3_months', 'NUM_CALL_WEEKEND_CURR_avg_6_months', 'NUM_SECS_WEEK_CURR_avg_3_months', 'NUM_SECS_WEEK_CURR_avg_6_months', 'NUM_SECS_WEEKEND_CURR_avg_3_months', 'NUM_SECS_WEEKEND_CURR_avg_6_months']
    INFO - Target: NUM_DAYS_LINE_TYPE_FIXE_POST_DEA
    INFO - Completed feature computation!



```python
def assess_NA(data: pd.DataFrame):
    """
    Returns a pd.DataFrame denoting the total number of NA
    values and the percentage of NA values in each column.
    """
    # pd.Datadenoting features and the sum of their null values
    nulls = data.isnull().sum().reset_index().rename(columns={0: "count"})
    nulls["percent"] = nulls["count"] * 100 / len(data)

    return nulls
```


```python
null_clean = assess_NA(clean_data)
```


```python
null_clean.sort_values(by=["count"], ascending=False).head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>count</th>
      <th>percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>customer_id</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MONTH</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>MIN_DAYS_PERM_CURR</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>PREV_FINISHED_PERM</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>pago_final_0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>consumo_0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>aperiodica_0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>discount_0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32</th>
      <td>ajuste_0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>NUM_GB_OWNN_CURR</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>34</th>
      <td>NUM_GB_2G_CURR</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>NUM_GB_3G_CURR</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>NUM_GB_4G_CURR</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>37</th>
      <td>NUM_GB_5G_CURR</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>NUM_SESS_CURR</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
(target == 1).sum()
```




    1037




```python
(target == 0).sum()
```




    34604




```python
def calculate_uplift(target, y_pred_proba):
    """
    Calculate the uplift for each decile.

    Args:
    - y_true (array-like): True labels.
    - y_pred_proba (array-like): Predicted probabilities.

    Returns:
    - pd.Series: Uplift for each decile.
    """
    data = pd.DataFrame({"y_true": target, "y_pred_proba": y_pred_proba})
    data_sorted = data.sort_values(by="y_pred_proba", ascending=False)
    data_sorted["decile"] = pd.qcut(
        data_sorted["y_pred_proba"], q=10, labels=list(reversed(range(10)))
    )
    decile_churn_rate = data_sorted.groupby("decile", observed=True)["y_true"].mean()

    overall_churn_rate = data["y_true"].mean()
    uplift = decile_churn_rate / overall_churn_rate

    # return by ascending deciles
    return uplift.sort_index(ascending=False)
```


```python
def evaluate_metrics_lgb(y_true, y_pred, n_trees, lr, train_or_test="test"):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    uplift = calculate_uplift(y_true, y_pred)

    print(uplift)

    print(
        f"Lightgbm model with {n_trees} n of trees and {lr} learning rate on {train_or_test}: Roc AUC {roc_auc:.4f} and PR AUC {pr_auc:.4f}"
    )

    print(
        f"Lightgbm model with {n_trees} n of trees and {lr} learning rate on {train_or_test}: {uplift[0]} Uplift on the first decile"
    )
```


```python
n_trees = [100, 150, 200, 250, 300]
lr = [0.1, 0.05, 0.01]
```


```python
for n in n_trees:
    for r in lr:
        pipeline = Pipeline(
            [
                (
                    "lightgbm",
                    lgb.LGBMClassifier(
                        boosting_type="gbdt",
                        max_depth=-1,
                        learning_rate=r,
                        n_estimators=n,
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        )

        pipeline.fit(features, target)

        preds_train = pipeline.predict_proba(features)[:, 1]
        preds = pipeline.predict_proba(features_test)[:, 1]

        # evaluate_metrics_lgb(target, preds_train, n, r, "train")
        evaluate_metrics_lgb(target_test, preds, n, r, "test")
```

    decile
    0    1.562191
    1    1.395112
    2    1.152847
    3    1.119431
    4    1.027852
    5    0.943998
    6    0.760211
    7    0.827042
    8    0.768565
    9    0.442760
    Name: y_true, dtype: float64
    Lightgbm model with 100 n of trees and 0.1 learning rate on test: Roc AUC 0.5904 and PR AUC 0.0516
    Lightgbm model with 100 n of trees and 0.1 learning rate on test: 1.5621911265311381 Uplift on the first decile
    decile
    0    1.687501
    1    1.403466
    2    1.136139
    3    1.086015
    4    0.977713
    5    1.035891
    6    0.793627
    7    0.718441
    8    0.735149
    9    0.426052
    Name: y_true, dtype: float64
    Lightgbm model with 100 n of trees and 0.05 learning rate on test: Roc AUC 0.6004 and PR AUC 0.0543
    Lightgbm model with 100 n of trees and 0.05 learning rate on test: 1.6875005751833685 Uplift on the first decile
    decile
    0    1.720916
    1    1.453590
    2    1.129859
    3    1.202603
    4    1.076673
    5    0.793384
    6    0.785273
    7    0.852887
    8    0.609883
    9    0.378720
    Name: y_true, dtype: float64
    Lightgbm model with 100 n of trees and 0.01 learning rate on test: Roc AUC 0.6098 and PR AUC 0.0553
    Lightgbm model with 100 n of trees and 0.01 learning rate on test: 1.7209164281572966 Uplift on the first decile
    decile
    0    1.520421
    1    1.386758
    2    1.269802
    3    0.918936
    4    1.153200
    5    0.818688
    6    0.835396
    7    0.918936
    8    0.743503
    9    0.434406
    Name: y_true, dtype: float64
    Lightgbm model with 150 n of trees and 0.1 learning rate on test: Roc AUC 0.5881 and PR AUC 0.0510
    Lightgbm model with 150 n of trees and 0.1 learning rate on test: 1.520421310313728 Uplift on the first decile
    decile
    0    1.712562
    1    1.228033
    2    1.253094
    3    1.136139
    4    0.818939
    5    1.169555
    6    0.626547
    7    0.835396
    8    0.768565
    9    0.451114
    Name: y_true, dtype: float64
    Lightgbm model with 150 n of trees and 0.05 learning rate on test: Roc AUC 0.5945 and PR AUC 0.0535
    Lightgbm model with 150 n of trees and 0.05 learning rate on test: 1.7125624649138145 Uplift on the first decile
    decile
    0    1.695855
    1    1.478651
    2    1.303218
    3    0.953227
    4    1.160491
    5    0.860458
    6    0.701733
    7    0.852365
    8    0.618004
    9    0.375928
    Name: y_true, dtype: float64
    Lightgbm model with 150 n of trees and 0.01 learning rate on test: Roc AUC 0.6106 and PR AUC 0.0558
    Lightgbm model with 150 n of trees and 0.01 learning rate on test: 1.6958545384268506 Uplift on the first decile
    decile
    0    1.629023
    1    1.294864
    2    1.127785
    3    1.094369
    4    1.061278
    5    0.868812
    6    0.818688
    7    0.918936
    8    0.743503
    9    0.442760
    Name: y_true, dtype: float64
    Lightgbm model with 200 n of trees and 0.1 learning rate on test: Roc AUC 0.5863 and PR AUC 0.0507
    Lightgbm model with 200 n of trees and 0.1 learning rate on test: 1.6290228324789944 Uplift on the first decile
    decile
    0    1.662439
    1    1.261448
    2    1.278156
    3    1.102723
    4    0.852365
    5    0.994122
    6    0.776919
    7    0.877166
    8    0.726795
    9    0.467822
    Name: y_true, dtype: float64
    Lightgbm model with 200 n of trees and 0.05 learning rate on test: Roc AUC 0.5921 and PR AUC 0.0533
    Lightgbm model with 200 n of trees and 0.05 learning rate on test: 1.6624386854529223 Uplift on the first decile
    decile
    0    1.754332
    1    1.461944
    2    1.194617
    3    1.077661
    4    1.002782
    5    0.885520
    6    0.810334
    7    0.793627
    8    0.677500
    9    0.342094
    Name: y_true, dtype: float64
    Lightgbm model with 200 n of trees and 0.01 learning rate on test: Roc AUC 0.6109 and PR AUC 0.0557
    Lightgbm model with 200 n of trees and 0.01 learning rate on test: 1.7543322811312247 Uplift on the first decile
    decile
    0    1.612315
    1    1.219679
    2    1.060953
    3    1.144493
    4    1.136487
    5    0.910582
    6    0.768565
    7    0.927290
    8    0.760211
    9    0.459468
    Name: y_true, dtype: float64
    Lightgbm model with 250 n of trees and 0.1 learning rate on test: Roc AUC 0.5825 and PR AUC 0.0504
    Lightgbm model with 250 n of trees and 0.1 learning rate on test: 1.6123149059920303 Uplift on the first decile
    decile
    0    1.662439
    1    1.194617
    2    1.269802
    3    1.094369
    4    1.036209
    5    0.860458
    6    0.810334
    7    0.910582
    8    0.685025
    9    0.476176
    Name: y_true, dtype: float64
    Lightgbm model with 250 n of trees and 0.05 learning rate on test: Roc AUC 0.5899 and PR AUC 0.0533
    Lightgbm model with 250 n of trees and 0.05 learning rate on test: 1.6624386854529223 Uplift on the first decile
    decile
    0    1.745978
    1    1.395112
    2    1.336634
    3    0.969060
    4    1.086348
    5    0.827042
    6    0.726795
    7    0.835396
    8    0.751857
    9    0.325805
    Name: y_true, dtype: float64
    Lightgbm model with 250 n of trees and 0.01 learning rate on test: Roc AUC 0.6093 and PR AUC 0.0553
    Lightgbm model with 250 n of trees and 0.01 learning rate on test: 1.7459783178877426 Uplift on the first decile
    decile
    0    1.645731
    1    1.127785
    2    1.261448
    3    1.027537
    4    1.036209
    5    0.977414
    6    0.768565
    7    0.969060
    8    0.693379
    9    0.492884
    Name: y_true, dtype: float64
    Lightgbm model with 300 n of trees and 0.1 learning rate on test: Roc AUC 0.5811 and PR AUC 0.0500
    Lightgbm model with 300 n of trees and 0.1 learning rate on test: 1.6457307589659584 Uplift on the first decile
    decile
    0    1.687501
    1    1.169555
    2    1.169555
    3    1.077661
    4    1.136487
    5    0.793627
    6    0.952352
    7    0.843750
    8    0.701733
    9    0.467822
    Name: y_true, dtype: float64
    Lightgbm model with 300 n of trees and 0.05 learning rate on test: Roc AUC 0.5874 and PR AUC 0.0526
    Lightgbm model with 300 n of trees and 0.05 learning rate on test: 1.6875005751833685 Uplift on the first decile
    decile
    0    1.796102
    1    1.336634
    2    1.236387
    3    1.060953
    4    1.019495
    5    0.877166
    6    0.793627
    7    0.768565
    8    0.726795
    9    0.384282
    Name: y_true, dtype: float64
    Lightgbm model with 300 n of trees and 0.01 learning rate on test: Roc AUC 0.6072 and PR AUC 0.0552
    Lightgbm model with 300 n of trees and 0.01 learning rate on test: 1.7961020973486346 Uplift on the first decile



```python
best_model_lr = 0.01
best_model_n_trees = 150
pipeline = Pipeline(
    [
        (
            "lightgbm",
            lgb.LGBMClassifier(
                boosting_type="gbdt",
                max_depth=-1,
                learning_rate=best_model_lr,
                n_estimators=best_model_n_trees,
                random_state=42,
                verbose=-1,
            ),
        ),
    ]
)

pipeline.fit(features, target)

preds_train = pipeline.predict_proba(features)[:, 1]
preds = pipeline.predict_proba(features_test)[:, 1]
preds_test = pipeline.predict(features_test)

evaluate_metrics_lgb(target, preds_train, best_model_n_trees, best_model_lr, "train")
evaluate_metrics_lgb(target_test, preds, best_model_n_trees, best_model_lr, "test")
```

    decile
    0    6.335761
    1    1.301869
    2    0.771478
    3    0.405253
    4    0.385523
    5    0.376095
    6    0.154296
    7    0.154296
    8    0.086816
    9    0.028914
    Name: y_true, dtype: float64
    Lightgbm model with 150 n of trees and 0.01 learning rate on train: Roc AUC 0.8755 and PR AUC 0.4006
    Lightgbm model with 150 n of trees and 0.01 learning rate on train: 6.335761179782395 Uplift on the first decile
    decile
    0    1.695855
    1    1.478651
    2    1.303218
    3    0.953227
    4    1.160491
    5    0.860458
    6    0.701733
    7    0.852365
    8    0.618004
    9    0.375928
    Name: y_true, dtype: float64
    Lightgbm model with 150 n of trees and 0.01 learning rate on test: Roc AUC 0.6106 and PR AUC 0.0558
    Lightgbm model with 150 n of trees and 0.01 learning rate on test: 1.6958545384268506 Uplift on the first decile



```python
data = pd.DataFrame({"y_true": target_test, "y_pred_proba": preds, "y_pred": preds_test})
data_sorted = data.sort_values(by="y_true", ascending=False)
```


```python
feature_names = features.columns
# Extract coefficients
lr_model = pipeline.named_steps["lightgbm"]
coefficients = lr_model.feature_importances_  # for Logistic Regression

# Create a DataFrame for easy visualization
feature_importance = pd.DataFrame(
    {"Feature": feature_names, "Coefficient": coefficients}
)
```


```python
cols_no_importance = feature_importance[feature_importance["Coefficient"] < 30]["Feature"]
```


```python
order_columns = feature_importance.sort_values("Coefficient", ascending=False)[
    "Feature"
]
sns.barplot(
    feature_importance,
    x="Coefficient",
    y="Feature",
    order=order_columns,
)
```




    <Axes: xlabel='Coefficient', ylabel='Feature'>




    
![png](explore_lightgbm_files/explore_lightgbm_19_1.png)
    



```python
features_dropped = features.drop(columns=cols_no_importance)
features_dropped_test = features_test.drop(columns=cols_no_importance)
```


```python
features_dropped
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pago_final_0</th>
      <th>consumo_0</th>
      <th>aperiodica_0</th>
      <th>discount_0</th>
      <th>NUM_GB_OWNN_CURR</th>
      <th>NUM_GB_2G_CURR</th>
      <th>NUM_GB_3G_CURR</th>
      <th>NUM_GB_4G_CURR</th>
      <th>NUM_GB_5G_CURR</th>
      <th>NUM_SESS_CURR</th>
      <th>...</th>
      <th>discount_avg_6_months</th>
      <th>NUM_GB_OWNN_CURR_avg_3_months</th>
      <th>NUM_GB_OWNN_CURR_avg_6_months</th>
      <th>NUM_CALL_WEEK_CURR_avg_3_months</th>
      <th>NUM_CALL_WEEK_CURR_avg_6_months</th>
      <th>NUM_CALL_WEEKEND_CURR_avg_3_months</th>
      <th>NUM_CALL_WEEKEND_CURR_avg_6_months</th>
      <th>NUM_SECS_WEEK_CURR_avg_3_months</th>
      <th>NUM_SECS_WEEKEND_CURR_avg_3_months</th>
      <th>NUM_SECS_WEEKEND_CURR_avg_6_months</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>55.0292</td>
      <td>110.1659</td>
      <td>0.0</td>
      <td>-191.7339</td>
      <td>0.031088</td>
      <td>0.000096</td>
      <td>0.011931</td>
      <td>4.133809</td>
      <td>0.000000</td>
      <td>1954</td>
      <td>...</td>
      <td>-165.862500</td>
      <td>0.185588</td>
      <td>0.195415</td>
      <td>152.000000</td>
      <td>120.500000</td>
      <td>37.333333</td>
      <td>26.333333</td>
      <td>30289.666667</td>
      <td>8900.000000</td>
      <td>5586.500000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85.0900</td>
      <td>7.8844</td>
      <td>120.0</td>
      <td>-156.0896</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>...</td>
      <td>-51.789683</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>15.000000</td>
      <td>12.833333</td>
      <td>1.333333</td>
      <td>2.833333</td>
      <td>764.666667</td>
      <td>6.666667</td>
      <td>86.333333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>67.8450</td>
      <td>376.4855</td>
      <td>0.0</td>
      <td>-457.0280</td>
      <td>4.333472</td>
      <td>0.000539</td>
      <td>0.204715</td>
      <td>7.033735</td>
      <td>8.274004</td>
      <td>26972</td>
      <td>...</td>
      <td>-366.458383</td>
      <td>6.438764</td>
      <td>6.748296</td>
      <td>327.000000</td>
      <td>263.333333</td>
      <td>128.000000</td>
      <td>115.333333</td>
      <td>52375.333333</td>
      <td>17220.333333</td>
      <td>16678.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.9000</td>
      <td>223.8407</td>
      <td>120.0</td>
      <td>-419.6658</td>
      <td>1.849722</td>
      <td>0.000366</td>
      <td>0.108648</td>
      <td>22.732534</td>
      <td>0.000000</td>
      <td>8232</td>
      <td>...</td>
      <td>-257.401883</td>
      <td>3.739307</td>
      <td>3.968482</td>
      <td>263.666667</td>
      <td>249.000000</td>
      <td>56.000000</td>
      <td>57.166667</td>
      <td>37379.333333</td>
      <td>7876.000000</td>
      <td>9317.166667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55.8141</td>
      <td>100.6445</td>
      <td>0.0</td>
      <td>-178.7333</td>
      <td>2.864490</td>
      <td>0.000440</td>
      <td>0.084532</td>
      <td>44.816553</td>
      <td>0.454133</td>
      <td>8060</td>
      <td>...</td>
      <td>-168.873950</td>
      <td>7.038711</td>
      <td>5.319472</td>
      <td>147.333333</td>
      <td>143.500000</td>
      <td>30.000000</td>
      <td>27.000000</td>
      <td>17026.333333</td>
      <td>6618.333333</td>
      <td>5485.666667</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>47629</th>
      <td>70.2885</td>
      <td>343.8690</td>
      <td>165.0</td>
      <td>-607.1275</td>
      <td>6.262426</td>
      <td>0.000000</td>
      <td>0.018946</td>
      <td>7.890733</td>
      <td>0.168042</td>
      <td>5424</td>
      <td>...</td>
      <td>-451.316433</td>
      <td>6.287275</td>
      <td>6.783493</td>
      <td>265.000000</td>
      <td>267.000000</td>
      <td>26.000000</td>
      <td>25.500000</td>
      <td>43875.333333</td>
      <td>2777.666667</td>
      <td>2742.166667</td>
    </tr>
    <tr>
      <th>47630</th>
      <td>30.3000</td>
      <td>12.1267</td>
      <td>0.0</td>
      <td>-40.0349</td>
      <td>0.493641</td>
      <td>0.000046</td>
      <td>0.018045</td>
      <td>1.302436</td>
      <td>0.000000</td>
      <td>3082</td>
      <td>...</td>
      <td>-52.518833</td>
      <td>0.622566</td>
      <td>0.616213</td>
      <td>18.333333</td>
      <td>25.333333</td>
      <td>10.666667</td>
      <td>12.000000</td>
      <td>2199.000000</td>
      <td>1255.333333</td>
      <td>1431.500000</td>
    </tr>
    <tr>
      <th>47631</th>
      <td>115.9758</td>
      <td>544.6884</td>
      <td>0.0</td>
      <td>-641.1017</td>
      <td>15.880268</td>
      <td>0.008036</td>
      <td>1.420019</td>
      <td>67.363741</td>
      <td>0.000000</td>
      <td>28610</td>
      <td>...</td>
      <td>-617.292100</td>
      <td>7.577509</td>
      <td>6.416394</td>
      <td>675.333333</td>
      <td>683.666667</td>
      <td>177.666667</td>
      <td>184.333333</td>
      <td>87544.333333</td>
      <td>20386.333333</td>
      <td>20566.833333</td>
    </tr>
    <tr>
      <th>47632</th>
      <td>116.1201</td>
      <td>424.2206</td>
      <td>0.0</td>
      <td>-509.4223</td>
      <td>21.188541</td>
      <td>0.012646</td>
      <td>0.850269</td>
      <td>51.251579</td>
      <td>0.000000</td>
      <td>28121</td>
      <td>...</td>
      <td>-391.108983</td>
      <td>16.194390</td>
      <td>13.489955</td>
      <td>508.333333</td>
      <td>405.333333</td>
      <td>149.666667</td>
      <td>127.833333</td>
      <td>69688.333333</td>
      <td>19579.000000</td>
      <td>17083.333333</td>
    </tr>
    <tr>
      <th>47633</th>
      <td>88.4158</td>
      <td>78.3903</td>
      <td>0.0</td>
      <td>-115.8796</td>
      <td>1.378261</td>
      <td>0.000000</td>
      <td>0.011571</td>
      <td>1.543061</td>
      <td>0.000000</td>
      <td>2101</td>
      <td>...</td>
      <td>-110.095683</td>
      <td>1.177657</td>
      <td>1.033008</td>
      <td>94.000000</td>
      <td>73.166667</td>
      <td>25.666667</td>
      <td>18.333333</td>
      <td>15025.333333</td>
      <td>3834.666667</td>
      <td>3766.000000</td>
    </tr>
  </tbody>
</table>
<p>46964 rows × 48 columns</p>
</div>




```python
best_model_lr = 0.01
best_model_n_trees = 150
pipeline = Pipeline(
    [
        (
            "lightgbm",
            lgb.LGBMClassifier(
                boosting_type="gbdt",
                max_depth=-1,
                learning_rate=best_model_lr,
                n_estimators=best_model_n_trees,
                random_state=42,
                verbose=-1,
            ),
        ),
    ]
)

pipeline.fit(features_dropped, target)

preds_train = pipeline.predict_proba(features_dropped)[:, 1]
preds = pipeline.predict_proba(features_dropped_test)[:, 1]

evaluate_metrics_lgb(target, preds_train, best_model_n_trees, best_model_lr, "train")
evaluate_metrics_lgb(target_test, preds, best_model_n_trees, best_model_lr, "test")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /mnt/c/Users/tonbo/Documents/ZriveDS/zrive-ds-4q24-churn/src/eda/explore_lightgbm.ipynb Cell 23 line 1
          <a href='vscode-notebook-cell://wsl%2Bubuntu-22.04/mnt/c/Users/tonbo/Documents/ZriveDS/zrive-ds-4q24-churn/src/eda/explore_lightgbm.ipynb#X26sdnNjb2RlLXJlbW90ZQ%3D%3D?line=1'>2</a> best_model_n_trees = 150
          <a href='vscode-notebook-cell://wsl%2Bubuntu-22.04/mnt/c/Users/tonbo/Documents/ZriveDS/zrive-ds-4q24-churn/src/eda/explore_lightgbm.ipynb#X26sdnNjb2RlLXJlbW90ZQ%3D%3D?line=2'>3</a> pipeline = Pipeline(
          <a href='vscode-notebook-cell://wsl%2Bubuntu-22.04/mnt/c/Users/tonbo/Documents/ZriveDS/zrive-ds-4q24-churn/src/eda/explore_lightgbm.ipynb#X26sdnNjb2RlLXJlbW90ZQ%3D%3D?line=3'>4</a>     [
          <a href='vscode-notebook-cell://wsl%2Bubuntu-22.04/mnt/c/Users/tonbo/Documents/ZriveDS/zrive-ds-4q24-churn/src/eda/explore_lightgbm.ipynb#X26sdnNjb2RlLXJlbW90ZQ%3D%3D?line=4'>5</a>         (
       (...)
         <a href='vscode-notebook-cell://wsl%2Bubuntu-22.04/mnt/c/Users/tonbo/Documents/ZriveDS/zrive-ds-4q24-churn/src/eda/explore_lightgbm.ipynb#X26sdnNjb2RlLXJlbW90ZQ%3D%3D?line=15'>16</a>     ]
         <a href='vscode-notebook-cell://wsl%2Bubuntu-22.04/mnt/c/Users/tonbo/Documents/ZriveDS/zrive-ds-4q24-churn/src/eda/explore_lightgbm.ipynb#X26sdnNjb2RlLXJlbW90ZQ%3D%3D?line=16'>17</a> )
    ---> <a href='vscode-notebook-cell://wsl%2Bubuntu-22.04/mnt/c/Users/tonbo/Documents/ZriveDS/zrive-ds-4q24-churn/src/eda/explore_lightgbm.ipynb#X26sdnNjb2RlLXJlbW90ZQ%3D%3D?line=18'>19</a> pipeline.fit(features_dropped, target)
         <a href='vscode-notebook-cell://wsl%2Bubuntu-22.04/mnt/c/Users/tonbo/Documents/ZriveDS/zrive-ds-4q24-churn/src/eda/explore_lightgbm.ipynb#X26sdnNjb2RlLXJlbW90ZQ%3D%3D?line=20'>21</a> preds_train = pipeline.predict_proba(features_dropped)[:, 1]
         <a href='vscode-notebook-cell://wsl%2Bubuntu-22.04/mnt/c/Users/tonbo/Documents/ZriveDS/zrive-ds-4q24-churn/src/eda/explore_lightgbm.ipynb#X26sdnNjb2RlLXJlbW90ZQ%3D%3D?line=21'>22</a> preds = pipeline.predict_proba(features_dropped_test)[:, 1]


    NameError: name 'features_dropped' is not defined



```python
feature_names = features_dropped.columns
# Extract coefficients
lr_model = pipeline.named_steps["lightgbm"]
coefficients = lr_model.feature_importances_  # for Logistic Regression

# Create a DataFrame for easy visualization
feature_importance = pd.DataFrame(
    {"Feature": feature_names, "Coefficient": coefficients}
)
```


```python
order_columns = feature_importance.sort_values("Coefficient", ascending=False)[
    "Feature"
]
sns.barplot(
    feature_importance,
    x="Coefficient",
    y="Feature",
    order=order_columns,
)
```




    <Axes: xlabel='Coefficient', ylabel='Feature'>




    
![png](explore_lightgbm_files/explore_lightgbm_24_1.png)
    


Let's see what results we where getting with LogisticRegression model


```python
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "logistic_regression",
            LogisticRegression(
                penalty=logistic_regression_params["penalty"],
                C=logistic_regression_params["C"],
                solver=logistic_regression_params["solver"],
                max_iter=logistic_regression_params["max_iter"],
                random_state=42,
            ),
        ),
    ]
)

pipeline.fit(features, target)
preds_train = pipeline.predict_proba(features)[:, 1]
preds_test = pipeline.predict_proba(features_test)[:, 1]

evaluate_metrics_lgb(target, preds_train, 0, 0, "train")
evaluate_metrics_lgb(target_test, preds_test, 0, 0, "test")
```

    decile
    0    2.536233
    1    1.369373
    2    1.292225
    3    1.060782
    4    0.993278
    5    0.916130
    6    0.646113
    7    0.405026
    8    0.453243
    9    0.327786
    Name: y_true, dtype: float64
    Lightgbm model with 0 n of trees and 0 learning rate on train: Roc AUC 0.6692 and PR AUC 0.0875
    Lightgbm model with 0 n of trees and 0 learning rate on train: 2.536233166336027 Uplift on the first decile
    decile
    0    1.745978
    1    1.578899
    2    1.060953
    3    1.186263
    4    0.802226
    5    0.877166
    6    0.827042
    7    0.501238
    8    0.685025
    9    0.735149
    Name: y_true, dtype: float64
    Lightgbm model with 0 n of trees and 0 learning rate on test: Roc AUC 0.6023 and PR AUC 0.0577
    Lightgbm model with 0 n of trees and 0 learning rate on test: 1.7459783178877426 Uplift on the first decile



```python
from sklearn.model_selection import GridSearchCV
```


```python
param_grid = {
    "lightgbm__learning_rate": [1, 0.5, 0.25, 0.01],
    "lightgbm__n_estimators": [70, 100, 150, 200, 250],
}
grid_params = {
    "learning_rate": [0.005, 0.01],
    "n_estimators": [70, 100, 150],
    "num_leaves": [
        6,
        8,
        12,
        16,
    ],  # large num_leaves helps improve accuracy but might lead to over-fitting
    # "boosting_type": ["gbdt", "dart"],  # for better accuracy -> try dart
    "objective": ["binary"],
    # "max_bin": [
    #     255,
    #     510,
    # ],  # large max_bin helps improve accuracy but might slow down training progress
    "random_state": [500],
    "colsample_bytree": [0.64, 0.65, 0.66],
    "subsample": [0.7, 0.75],
    "reg_alpha": [0, 1, 1.2],
    "reg_lambda": [0, 1, 1.2, 1.4],
}
```


```python
def custom_scorer(estimator, X, y):
    # Calculate validation score (F1 score)
    y_pred = estimator.predict_proba(X)[:, 1]
    data = pd.DataFrame({"y_true": y, "y_pred_proba": y_pred})
    data_sorted = data.sort_values(by="y_pred_proba", ascending=False)
    decile_cutoff = int(len(data_sorted) * 0.1)
    first_decile = data_sorted.head(decile_cutoff)
    true_positives = first_decile["y_true"].sum()
    return true_positives / decile_cutoff
```


```python
import itertools
```


```python
keys, values = zip(*grid_params.items())

permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
```


```python
for i in permutations_dicts:
    print(i)
    break
```

    {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}



```python
def accuracy_first_decile(target, preds):
    # Calculate validation score (F1 score)
    data = pd.DataFrame({"y_true": target, "y_pred_proba": preds})
    data_sorted = data.sort_values(by="y_pred_proba", ascending=False)
    decile_cutoff = int(len(data_sorted) * 0.1)
    first_decile = data_sorted.head(decile_cutoff)
    true_positives = first_decile["y_true"].sum()
    return true_positives / decile_cutoff, data
```


```python
def own_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    paramgrid: dict[str, list[int]],
    custom_scorer,
    model: str = "lightgbm",
):
    best_accuracy = 0
    best_accuracy_params = {}

    keys, values = zip(*paramgrid.items())

    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for params in permutations_dicts:
        logger.info(f"Training model with params {params}")
        pipeline = Pipeline(
            [
                (
                    "lightgbm",
                    lgb.LGBMClassifier(
                        learning_rate=params["learning_rate"],
                        n_estimators=params["n_estimators"],
                        num_leaves=params["num_leaves"],
                        colsample_bytree=params["colsample_bytree"],
                        subsample=params["subsample"],
                        reg_alpha=params["reg_alpha"],
                        reg_lambda=params["reg_lambda"],
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict_proba(X_test)[:, 1]

        accuracy = custom_scorer(y_test, preds)

        logger.info(f"Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_accuracy_params = params

    return best_accuracy, best_accuracy_params
```


```python
best_accuracy, best_accuracy_params = own_grid_search(
    features, target, features_test, target_test, grid_params, accuracy_first_decile
)
```

    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05907560453014998
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.057545148454239366
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.05938169574533211
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05907560453014998
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.057545148454239366
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.05938169574533211
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05938169574533211
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05938169574533211
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05938169574533211
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05938169574533211
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06917661463116008
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06917661463116008
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06887052341597796
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06887052341597796
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06887052341597796
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06887052341597796
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06887052341597796
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06887052341597796
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06856443220079583
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06887052341597796
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06887052341597796
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06856443220079583
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06887052341597796
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06887052341597796
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06795224977043159
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.005, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05938169574533211
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05938169574533211
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05938169574533211
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05938169574533211
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05938169574533211
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.05938169574533211
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.05999387817569636
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06825834098561372
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 70, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06397306397306397
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 100, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.05968778696051423
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 6, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06091215182124273
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06060606060606061
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.061218243036424855
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.060299969390878484
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.062136516681971225
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06152433425160698
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 8, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0618304254667891
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06274869911233548
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06336088154269973
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06366697275788184
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06458524640342822
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 12, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06489133761861035
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.06244260789715335
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.06519742883379247
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06734006734006734
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06703397612488522
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.64, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06672788490970309
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06856443220079583
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06887052341597796
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06856443220079583
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06887052341597796
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.65, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06856443220079583
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06887052341597796
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.7, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 0}
    INFO - Accuracy: 0.0630547903275176
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 0, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 0}
    INFO - Accuracy: 0.0655035200489746
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1}
    INFO - Accuracy: 0.06580961126415671
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06856443220079583
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06611570247933884
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 0}
    INFO - Accuracy: 0.0642791551882461
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1}
    INFO - Accuracy: 0.06887052341597796
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.2}
    INFO - Accuracy: 0.06764615855524947
    INFO - Training model with params {'learning_rate': 0.01, 'n_estimators': 150, 'num_leaves': 16, 'objective': 'binary', 'random_state': 500, 'colsample_bytree': 0.66, 'subsample': 0.75, 'reg_alpha': 1.2, 'reg_lambda': 1.4}
    INFO - Accuracy: 0.06642179369452096



```python
best_accuracy
```




    0.06917661463116008




```python
best_accuracy_params
```




    {'learning_rate': 0.005,
     'n_estimators': 100,
     'num_leaves': 12,
     'objective': 'binary',
     'random_state': 500,
     'colsample_bytree': 0.64,
     'subsample': 0.7,
     'reg_alpha': 0,
     'reg_lambda': 1}




```python
pipeline = Pipeline(
    [
        (
            "lightgbm",
            lgb.LGBMClassifier(
                learning_rate=best_accuracy_params["learning_rate"],
                n_estimators=best_accuracy_params["n_estimators"],
                num_leaves=best_accuracy_params["num_leaves"],
                colsample_bytree=best_accuracy_params["colsample_bytree"],
                subsample=best_accuracy_params["subsample"],
                reg_alpha=best_accuracy_params["reg_alpha"],
                reg_lambda=best_accuracy_params["reg_lambda"],
                random_state=500,
                verbose=-1,
            ),
        ),
    ]
)

pipeline.fit(features, target)

preds_train = pipeline.predict_proba(features)[:, 1]
preds = pipeline.predict_proba(features_test)[:, 1]
preds_no = pipeline.predict(features_test)

evaluate_metrics_lgb(target, preds_train, best_model_n_trees, best_model_lr, "train")
evaluate_metrics_lgb(target_test, preds, best_model_n_trees, best_model_lr, "test")
acc, data= accuracy_first_decile(target_test, preds)
print(acc)
```

    decile
    0    3.707254
    1    1.545992
    2    1.196126
    3    0.865000
    4    0.745266
    5    0.643764
    6    0.491817
    7    0.446860
    8    0.247048
    9    0.121512
    Name: y_true, dtype: float64
    Lightgbm model with 70 n of trees and 0.01 learning rate on train: Roc AUC 0.7449 and PR AUC 0.1350
    Lightgbm model with 70 n of trees and 0.01 learning rate on train: 3.7072540712730113 Uplift on the first decile
    decile
    0    1.812810
    1    1.395539
    2    1.170271
    3    1.176828
    4    1.052922
    5    0.857615
    6    0.822622
    7    0.745969
    8    0.600373
    9    0.382113
    Name: y_true, dtype: float64
    Lightgbm model with 70 n of trees and 0.01 learning rate on test: Roc AUC 0.6182 and PR AUC 0.0591
    Lightgbm model with 70 n of trees and 0.01 learning rate on test: 1.8128100238355989 Uplift on the first decile
    0.06642179369452096



```python
data_sorted = data.sort_values(by="y_pred_proba", ascending=False)
```


```python
data_sorted
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_true</th>
      <th>y_pred_proba</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22027</th>
      <td>0.0</td>
      <td>0.169420</td>
    </tr>
    <tr>
      <th>7108</th>
      <td>1.0</td>
      <td>0.167308</td>
    </tr>
    <tr>
      <th>13144</th>
      <td>0.0</td>
      <td>0.166744</td>
    </tr>
    <tr>
      <th>22430</th>
      <td>0.0</td>
      <td>0.166744</td>
    </tr>
    <tr>
      <th>31307</th>
      <td>0.0</td>
      <td>0.166354</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8033</th>
      <td>0.0</td>
      <td>0.021574</td>
    </tr>
    <tr>
      <th>24847</th>
      <td>0.0</td>
      <td>0.021574</td>
    </tr>
    <tr>
      <th>32301</th>
      <td>0.0</td>
      <td>0.021574</td>
    </tr>
    <tr>
      <th>8042</th>
      <td>0.0</td>
      <td>0.021574</td>
    </tr>
    <tr>
      <th>21327</th>
      <td>0.0</td>
      <td>0.021574</td>
    </tr>
  </tbody>
</table>
<p>32679 rows × 2 columns</p>
</div>




```python
decile_cutoff = int(len(data_sorted) * 0.1)
first_decile = data_sorted.head(decile_cutoff)
```


```python
data_sorted["y_true"].sum()
```




    1197.0




```python
print(first_decile["y_true"].sum(), len(first_decile["y_true"]))
```

    217.0 3267



```python
data_sorted = data.sort_values(by="y_pred_proba", ascending=False)
data_sorted["decile"] = pd.qcut(
    data_sorted["y_pred_proba"], q=10, labels=list(reversed(range(10)))
)
decile_churn_rate = data_sorted.groupby("decile", observed=True)["y_true"].sum()
```


```python
decile_churn_rate.plot(kind="bar")
```




    <Axes: xlabel='decile'>




    
![png](explore_lightgbm_files/explore_lightgbm_45_1.png)
    



```python
best_model_lr = 0.01
best_model_n_trees = 150
pipeline = Pipeline(
    [
        (
            "lightgbm",
            lgb.LGBMClassifier(
                random_state=42,
                verbose=-1,
            ),
        ),
    ]
)

search = GridSearchCV(pipeline, grid_params, n_jobs=-1, scoring=custom_scorer)

search.fit(features, target)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
```


```python
best_model_lr = 0.01
best_model_n_trees = 70
pipeline = Pipeline(
    [
        (
            "lightgbm",
            lgb.LGBMClassifier(
                boosting_type="gbdt",
                max_depth=-1,
                learning_rate=best_model_lr,
                n_estimators=best_model_n_trees,
                random_state=42,
                verbose=-1,
            ),
        ),
    ]
)

pipeline.fit(features, target)

preds_train = pipeline.predict_proba(features)[:, 1]
preds = pipeline.predict_proba(features_test)[:, 1]
preds_test = pipeline.predict(features_test)

evaluate_metrics_lgb(target, preds_train, best_model_n_trees, best_model_lr, "train")
evaluate_metrics_lgb(target_test, preds, best_model_n_trees, best_model_lr, "test")
print(custom_scorer(pipeline, features_test, target_test))
```

    decile
    0    5.342484
    1    1.224721
    2    0.853250
    3    0.678925
    4    0.486067
    5    0.452355
    6    0.350813
    7    0.244977
    8    0.302665
    9    0.126606
    Name: y_true, dtype: float64
    Lightgbm model with 70 n of trees and 0.01 learning rate on train: Roc AUC 0.8116 and PR AUC 0.2907
    Lightgbm model with 70 n of trees and 0.01 learning rate on train: 5.342483551901745 Uplift on the first decile
    decile
    0    1.704209
    1    1.479104
    2    1.202603
    3    1.161912
    4    1.010830
    5    0.814694
    6    0.870934
    7    0.660957
    8    0.643748
    9    0.484929
    Name: y_true, dtype: float64
    Lightgbm model with 70 n of trees and 0.01 learning rate on test: Roc AUC 0.6102 and PR AUC 0.0563
    Lightgbm model with 70 n of trees and 0.01 learning rate on test: 1.7042085016703326 Uplift on the first decile
    0.06244260789715335



```python
preds_test.sum()
```




    0.0



Let's test without dropping the churned users in the train months + drop month


```python
features, target, features_test, target_test = feature_computation(
    clean_data, train_from, train_to, logger, keep_gap_month_churns=True
)
```

    INFO - Starting feature computation
    INFO - Train computation from 2022-01-01 00:00:00 to 2022-06-01 00:00:00. Target for 2022-08-01 00:00:00
    INFO - Test computation from 2022-02-01 00:00:00 to 2022-07-01 00:00:00. Target for 2022-09-01 00:00:00
    INFO - Removing 23428 previous churned users from train set
    INFO - Removing 26672 previous churned users from test set
    INFO - Unique customers in train: 38325
    INFO - Unique customers in test: 35213
    INFO - Starting features and target computation
    INFO - Length train data: 36477
    INFO - Length test data: 33520
    INFO - Features computed
    INFO - Features: ['pago_final_0', 'consumo_0', 'aperiodica_0', 'discount_0', 'ajuste_0', 'NUM_GB_OWNN_CURR', 'NUM_GB_2G_CURR', 'NUM_GB_3G_CURR', 'NUM_GB_4G_CURR', 'NUM_GB_5G_CURR', 'NUM_SESS_CURR', 'NUM_SECS_CURR', 'NUM_CALL_CURR', 'NUM_CALL_WEEK_CURR', 'NUM_CALL_WEEKEND_CURR', 'NUM_SECS_WEEK_CURR', 'NUM_SECS_WEEKEND_CURR', 'NUM_CALL_WEEK', 'NUM_CALL_WEEKEND', 'NUM_DAYS_ACT', 'order_mobile_from_new_alta', 'MIN_DAYS_PERM_CURR', 'PREV_FINISHED_PERM', 'dif_pago_final_prev_month', 'dif_pago_final_prev_2_month', 'dif_pago_final_prev_3_month', 'dif_consumo_prev_month', 'dif_consumo_prev_2_month', 'dif_consumo_prev_3_month', 'dif_discount_prev_month', 'dif_discount_prev_2_month', 'dif_discount_prev_3_month', 'service_mobile_pending_install', 'service_fix_pending_install', 'service_mobile_cancelled', 'service_fix_cancelled', 'service_mobile_pending_install_3month', 'service_fix_pending_install_3month', 'service_mobile_cancelled_3month', 'service_fix_cancelled_3month', 'service_mobile_pending_install_6month', 'service_fix_pending_install_6month', 'service_mobile_cancelled_6month', 'service_fix_cancelled_6month', 'pago_final_prev_month', 'pago_final_avg_3_months', 'pago_final_avg_6_months', 'consumo_avg_3_months', 'consumo_avg_6_months', 'aperiodica_avg_3_months', 'aperiodica_avg_6_months', 'discount_avg_3_months', 'discount_avg_6_months', 'ajuste_avg_3_months', 'ajuste_avg_6_months', 'NUM_GB_OWNN_CURR_avg_3_months', 'NUM_GB_OWNN_CURR_avg_6_months', 'NUM_SECS_CURR_avg_3_months', 'NUM_SECS_CURR_avg_6_months', 'NUM_CALL_WEEK_CURR_avg_3_months', 'NUM_CALL_WEEK_CURR_avg_6_months', 'NUM_CALL_WEEKEND_CURR_avg_3_months', 'NUM_CALL_WEEKEND_CURR_avg_6_months', 'NUM_SECS_WEEK_CURR_avg_3_months', 'NUM_SECS_WEEK_CURR_avg_6_months', 'NUM_SECS_WEEKEND_CURR_avg_3_months', 'NUM_SECS_WEEKEND_CURR_avg_6_months']
    INFO - Target: NUM_DAYS_LINE_TYPE_FIXE_POST_DEA
    INFO - Completed feature computation!



```python
best_accuracy_params = {
    "learning_rate": 0.005,
    "n_estimators": 100,
    "num_leaves": 12,
    "objective": "binary",
    "random_state": 500,
    "colsample_bytree": 0.64,
    "subsample": 0.7,
    "reg_alpha": 0,
    "reg_lambda": 1,
}
```


```python
pipeline = Pipeline(
    [
        (
            "lightgbm",
            lgb.LGBMClassifier(
                learning_rate=best_accuracy_params["learning_rate"],
                n_estimators=best_accuracy_params["n_estimators"],
                num_leaves=best_accuracy_params["num_leaves"],
                colsample_bytree=best_accuracy_params["colsample_bytree"],
                subsample=best_accuracy_params["subsample"],
                reg_alpha=best_accuracy_params["reg_alpha"],
                reg_lambda=best_accuracy_params["reg_lambda"],
                random_state=500,
                verbose=-1,
            ),
        ),
    ]
)

pipeline.fit(features, target)

preds_train = pipeline.predict_proba(features)[:, 1]
preds = pipeline.predict_proba(features_test)[:, 1]
preds_no = pipeline.predict(features_test)

evaluate_metrics_lgb(target, preds_train, best_accuracy_params["n_estimators"], best_accuracy_params["learning_rate"], "train")
evaluate_metrics_lgb(target_test, preds, best_accuracy_params["n_estimators"], best_accuracy_params["learning_rate"], "test")
acc, data = accuracy_first_decile(target_test, preds)
print(acc)
```

    decile
    0    3.465998
    1    1.538333
    2    1.120826
    3    1.058075
    4    0.903569
    5    0.606220
    6    0.466263
    7    0.454943
    8    0.265959
    9    0.123412
    Name: y_true, dtype: float64
    Lightgbm model with 100 n of trees and 0.005 learning rate on train: Roc AUC 0.7362 and PR AUC 0.1380
    Lightgbm model with 100 n of trees and 0.005 learning rate on train: 3.4659977818213874 Uplift on the first decile
    decile
    0    1.761269
    1    1.485810
    2    1.078755
    3    1.083173
    4    1.002885
    5    0.975446
    6    0.966374
    7    0.820906
    8    0.510264
    9    0.319636
    Name: y_true, dtype: float64
    Lightgbm model with 100 n of trees and 0.005 learning rate on test: Roc AUC 0.6143 and PR AUC 0.0555
    Lightgbm model with 100 n of trees and 0.005 learning rate on test: 1.7612687813021703 Uplift on the first decile
    0.06392002423508028



```python
type(pipeline["lightgbm"]).__name__
```




    'LGBMClassifier'




```python

```
