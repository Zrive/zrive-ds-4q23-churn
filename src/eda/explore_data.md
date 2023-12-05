# Load libraries and data

Enter your project root:


```python
project_root = '/home/dan1dr/zrive-ds-4q24-churn'
```


```python
import sys
import os
import pandas as pd
import numpy as np
import configparser
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

# Define the project root path
current_wd = os.getcwd()

# Change the working directory if necessary
if current_wd != project_root:
    print(f"Changing working directory from {current_wd} to {project_root}")
    os.chdir(project_root)
else:
    print("Already in the correct path")

# Add 'src' directory to sys.path
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    print(f"Adding {src_path} to sys.path")
    sys.path.insert(0, src_path)

# Import the modules
from db_connectors.bigquery_service import BigqueryService
from data_gathering import data_gathering
from utils.logger import get_logger
logger = get_logger(__name__)
```

    Changing working directory from /home/dan1dr/zrive-ds-4q24-churn/src/eda to /home/dan1dr/zrive-ds-4q24-churn
    Adding /home/dan1dr/zrive-ds-4q24-churn/src to sys.path


### Extract the data from BigQuery

Define the query:


```python
query_sql_22 = """WITH selectable_customer as (
  SELECT customer_id
  FROM `mm-bi-catedras-upm.ESTIMACION_CHURN.multibrand_monthly_customer_base_mp2022`
  WHERE IS_CUST_SEGM_RESI > 0 
  AND IS_CUST_BILL_POST_CURR = TRUE
  AND CUST_BUNDLE_CURR = 'FMC'
  AND NUM_IMPAGOS = 0
  AND pago_final_0 IS NOT NULL
  GROUP BY customer_id
),

customer_selected as (
  SELECT customer_id as selected_customer
  FROM selectable_customer
  WHERE RAND() < 0.1
)

SELECT 
customer_id, MONTH, YEAR, pago_final_0, dif_pago_final_prev_month, dif_pago_final_prev_2_month, dif_pago_final_prev_3_month, periodica_0, dif_periodica_prev_month, dif_periodica_prev_2_month, 
dif_periodica_prev_3_month, consumo_0, dif_consumo_prev_month, dif_consumo_prev_2_month, dif_consumo_prev_3_month, aperiodica_0, dif_aperiodica_prev_month, 
dif_aperiodica_prev_2_month, dif_aperiodica_prev_3_month, discount_0, dif_discount_prev_month, dif_discount_prev_2_month, dif_discount_prev_3_month, ajuste_0, 
dif_ajuste_prev_month, dif_ajuste_prev_2_month, dif_ajuste_prev_3_month, Tota_Compra_disp, Curr_Compra_disp, Curr_Compra_Finanz_disp, Curr_Finanz_disp, Month_purchase_disp, Modelo_disp, Import_Rest_quota_disp, pvp_total_disp, pvp_total_disp_movil, Curr_cancel_disp, Tota_cancel_disp
NUM_GB_OWNN_CURR, NUM_GB_2G_CURR, NUM_GB_3G_CURR, NUM_GB_4G_CURR, NUM_GB_5G_CURR, NUM_SESS_CURR, NUM_SECS_CURR, NUM_CALL_CURR, NUM_CALL_WEEK_CURR, NUM_CALL_WEEKEND_CURR, 
NUM_SECS_WEEK_CURR, NUM_SECS_WEEKEND_CURR, NUM_CALL_WEEK, NUM_CALL_WEEKEND, NUM_DAYS_LINE_TYPE_FIXE_POST_DEA
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
query_sql_23 = """SELECT *
FROM `mm-bi-catedras-upm.ESTIMACION_CHURN.multibrand_monthly_customer_base_mp2022`
  """
```


```python
sample = data_gathering(query_sql_22)
logger.info(f"Extraction completed - data_2022")
#data_2023 = data_gathering(query_sql_23)
#logging.info(f"Extraction completed - data_2023")
```

    INFO - Extraction completed - data_2022


### Save the data


```python
save_path = os.path.join(project_root, 'data')
sample.to_parquet(os.path.join(save_path, 'subsample_users.parquet'))
```

### Load the data


```python
save_path = '/home/dan1dr/zrive-ds-4q24-churn/data'
read_path = os.path.join(save_path, 'subsample_users.parquet')
sample = pd.read_parquet(read_path)
```

## Explore data


```python
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
```


```python
sample.head()
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
      <th>customer_id</th>
      <th>MONTH</th>
      <th>YEAR</th>
      <th>pago_final_0</th>
      <th>dif_pago_final_prev_month</th>
      <th>dif_pago_final_prev_2_month</th>
      <th>dif_pago_final_prev_3_month</th>
      <th>periodica_0</th>
      <th>dif_periodica_prev_month</th>
      <th>dif_periodica_prev_2_month</th>
      <th>dif_periodica_prev_3_month</th>
      <th>consumo_0</th>
      <th>dif_consumo_prev_month</th>
      <th>dif_consumo_prev_2_month</th>
      <th>dif_consumo_prev_3_month</th>
      <th>aperiodica_0</th>
      <th>dif_aperiodica_prev_month</th>
      <th>dif_aperiodica_prev_2_month</th>
      <th>dif_aperiodica_prev_3_month</th>
      <th>discount_0</th>
      <th>dif_discount_prev_month</th>
      <th>dif_discount_prev_2_month</th>
      <th>dif_discount_prev_3_month</th>
      <th>ajuste_0</th>
      <th>dif_ajuste_prev_month</th>
      <th>dif_ajuste_prev_2_month</th>
      <th>dif_ajuste_prev_3_month</th>
      <th>Tota_Compra_disp</th>
      <th>Curr_Compra_disp</th>
      <th>Curr_Compra_Finanz_disp</th>
      <th>Curr_Finanz_disp</th>
      <th>Month_purchase_disp</th>
      <th>Modelo_disp</th>
      <th>Import_Rest_quota_disp</th>
      <th>pvp_total_disp</th>
      <th>pvp_total_disp_movil</th>
      <th>Curr_cancel_disp</th>
      <th>NUM_GB_OWNN_CURR</th>
      <th>NUM_GB_2G_CURR</th>
      <th>NUM_GB_3G_CURR</th>
      <th>NUM_GB_4G_CURR</th>
      <th>NUM_GB_5G_CURR</th>
      <th>NUM_SESS_CURR</th>
      <th>NUM_SECS_CURR</th>
      <th>NUM_CALL_CURR</th>
      <th>NUM_CALL_WEEK_CURR</th>
      <th>NUM_CALL_WEEKEND_CURR</th>
      <th>NUM_SECS_WEEK_CURR</th>
      <th>NUM_SECS_WEEKEND_CURR</th>
      <th>NUM_CALL_WEEK</th>
      <th>NUM_CALL_WEEKEND</th>
      <th>NUM_DAYS_LINE_TYPE_FIXE_POST_DEA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4142337</td>
      <td>12</td>
      <td>2022</td>
      <td>87.1499</td>
      <td>-0.92</td>
      <td>5.00</td>
      <td>13.15</td>
      <td>123.2162</td>
      <td>0.0</td>
      <td>0.82</td>
      <td>2.23</td>
      <td>440.7130</td>
      <td>-83.33</td>
      <td>109.00</td>
      <td>-288.48</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>-476.7793</td>
      <td>82.40</td>
      <td>-104.83</td>
      <td>299.39</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>758.676250</td>
      <td>2355.36</td>
      <td>2355.36</td>
      <td>0</td>
      <td>0</td>
      <td>0.000379</td>
      <td>1.478475</td>
      <td>34.539025</td>
      <td>8.552260</td>
      <td>12871</td>
      <td>88554</td>
      <td>820</td>
      <td>624</td>
      <td>196</td>
      <td>69564</td>
      <td>18990</td>
      <td>471</td>
      <td>140</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>1</th>
      <td>620962</td>
      <td>10</td>
      <td>2022</td>
      <td>88.0450</td>
      <td>22.35</td>
      <td>4.19</td>
      <td>15.41</td>
      <td>174.9015</td>
      <td>0.0</td>
      <td>-2.74</td>
      <td>-6.01</td>
      <td>244.8217</td>
      <td>-165.29</td>
      <td>-60.96</td>
      <td>-95.95</td>
      <td>0.0</td>
      <td>23.85</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>-331.6782</td>
      <td>163.79</td>
      <td>67.89</td>
      <td>117.37</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>173.550000</td>
      <td>1433.05</td>
      <td>1433.05</td>
      <td>0</td>
      <td>0</td>
      <td>0.003215</td>
      <td>0.943611</td>
      <td>14.486598</td>
      <td>0.000000</td>
      <td>11221</td>
      <td>104166</td>
      <td>564</td>
      <td>408</td>
      <td>156</td>
      <td>68144</td>
      <td>36022</td>
      <td>300</td>
      <td>113</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>2</th>
      <td>127593</td>
      <td>01</td>
      <td>2022</td>
      <td>177.5206</td>
      <td>0.89</td>
      <td>22.48</td>
      <td>2.95</td>
      <td>254.5857</td>
      <td>0.0</td>
      <td>-1.35</td>
      <td>-5.09</td>
      <td>69.6490</td>
      <td>33.88</td>
      <td>29.52</td>
      <td>7.54</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>-0.73</td>
      <td>0.0</td>
      <td>-146.7141</td>
      <td>-32.99</td>
      <td>-4.96</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>1</td>
      <td>54.546250</td>
      <td>1436.23</td>
      <td>1436.23</td>
      <td>0</td>
      <td>0</td>
      <td>0.000995</td>
      <td>0.084578</td>
      <td>24.819340</td>
      <td>0.000000</td>
      <td>4390</td>
      <td>9911</td>
      <td>154</td>
      <td>109</td>
      <td>45</td>
      <td>7230</td>
      <td>2681</td>
      <td>100</td>
      <td>34</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464466</td>
      <td>07</td>
      <td>2022</td>
      <td>106.6281</td>
      <td>7.58</td>
      <td>7.51</td>
      <td>11.08</td>
      <td>218.4490</td>
      <td>-2.3</td>
      <td>-2.30</td>
      <td>-1.69</td>
      <td>123.9285</td>
      <td>-39.25</td>
      <td>-67.05</td>
      <td>-155.04</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>-235.7494</td>
      <td>49.13</td>
      <td>76.86</td>
      <td>167.81</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>212.394583</td>
      <td>1572.89</td>
      <td>1434.05</td>
      <td>0</td>
      <td>0</td>
      <td>0.014278</td>
      <td>2.380983</td>
      <td>23.064308</td>
      <td>1.185673</td>
      <td>15480</td>
      <td>34602</td>
      <td>476</td>
      <td>337</td>
      <td>139</td>
      <td>23975</td>
      <td>10627</td>
      <td>308</td>
      <td>115</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>4</th>
      <td>311788</td>
      <td>11</td>
      <td>2022</td>
      <td>159.0900</td>
      <td>-0.60</td>
      <td>-1.50</td>
      <td>3.00</td>
      <td>217.1160</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.52</td>
      <td>125.9174</td>
      <td>-29.83</td>
      <td>-58.43</td>
      <td>-20.02</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>-183.9434</td>
      <td>29.23</td>
      <td>56.93</td>
      <td>22.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>17</td>
      <td>1</td>
      <td>40.495000</td>
      <td>972.89</td>
      <td>972.89</td>
      <td>0</td>
      <td>0</td>
      <td>0.002118</td>
      <td>0.373219</td>
      <td>137.773213</td>
      <td>0.000000</td>
      <td>25080</td>
      <td>37916</td>
      <td>265</td>
      <td>194</td>
      <td>71</td>
      <td>25642</td>
      <td>12274</td>
      <td>168</td>
      <td>51</td>
      <td>&lt;NA&gt;</td>
    </tr>
  </tbody>
</table>
</div>




```python
sample.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 283442 entries, 0 to 283441
    Data columns (total 52 columns):
     #   Column                            Non-Null Count   Dtype  
    ---  ------                            --------------   -----  
     0   customer_id                       283442 non-null  object 
     1   MONTH                             283442 non-null  object 
     2   YEAR                              283442 non-null  object 
     3   pago_final_0                      283442 non-null  float64
     4   dif_pago_final_prev_month         283442 non-null  float64
     5   dif_pago_final_prev_2_month       283442 non-null  float64
     6   dif_pago_final_prev_3_month       283442 non-null  float64
     7   periodica_0                       283442 non-null  float64
     8   dif_periodica_prev_month          283442 non-null  float64
     9   dif_periodica_prev_2_month        283442 non-null  float64
     10  dif_periodica_prev_3_month        283442 non-null  float64
     11  consumo_0                         283442 non-null  float64
     12  dif_consumo_prev_month            283442 non-null  float64
     13  dif_consumo_prev_2_month          283442 non-null  float64
     14  dif_consumo_prev_3_month          283442 non-null  float64
     15  aperiodica_0                      283442 non-null  float64
     16  dif_aperiodica_prev_month         283442 non-null  float64
     17  dif_aperiodica_prev_2_month       283442 non-null  float64
     18  dif_aperiodica_prev_3_month       283442 non-null  float64
     19  discount_0                        283442 non-null  float64
     20  dif_discount_prev_month           283442 non-null  float64
     21  dif_discount_prev_2_month         283442 non-null  float64
     22  dif_discount_prev_3_month         283442 non-null  float64
     23  ajuste_0                          283442 non-null  float64
     24  dif_ajuste_prev_month             283442 non-null  float64
     25  dif_ajuste_prev_2_month           283442 non-null  float64
     26  dif_ajuste_prev_3_month           283442 non-null  float64
     27  Tota_Compra_disp                  108075 non-null  Int64  
     28  Curr_Compra_disp                  108075 non-null  Int64  
     29  Curr_Compra_Finanz_disp           108075 non-null  Int64  
     30  Curr_Finanz_disp                  108075 non-null  Int64  
     31  Month_purchase_disp               100321 non-null  Int64  
     32  Modelo_disp                       100321 non-null  Int64  
     33  Import_Rest_quota_disp            57766 non-null   float64
     34  pvp_total_disp                    104372 non-null  float64
     35  pvp_total_disp_movil              100321 non-null  float64
     36  Curr_cancel_disp                  108075 non-null  Int64  
     37  NUM_GB_OWNN_CURR                  108075 non-null  Int64  
     38  NUM_GB_2G_CURR                    266848 non-null  float64
     39  NUM_GB_3G_CURR                    266848 non-null  float64
     40  NUM_GB_4G_CURR                    266848 non-null  float64
     41  NUM_GB_5G_CURR                    266848 non-null  float64
     42  NUM_SESS_CURR                     266848 non-null  Int64  
     43  NUM_SECS_CURR                     278186 non-null  Int64  
     44  NUM_CALL_CURR                     278186 non-null  Int64  
     45  NUM_CALL_WEEK_CURR                278186 non-null  Int64  
     46  NUM_CALL_WEEKEND_CURR             278186 non-null  Int64  
     47  NUM_SECS_WEEK_CURR                278186 non-null  Int64  
     48  NUM_SECS_WEEKEND_CURR             278186 non-null  Int64  
     49  NUM_CALL_WEEK                     283442 non-null  Int64  
     50  NUM_CALL_WEEKEND                  283442 non-null  Int64  
     51  NUM_DAYS_LINE_TYPE_FIXE_POST_DEA  17340 non-null   Int64  
    dtypes: Int64(18), float64(31), object(3)
    memory usage: 117.3+ MB



```python
sample.columns.tolist()
```




    ['customer_id',
     'MONTH',
     'YEAR',
     'pago_final_0',
     'dif_pago_final_prev_month',
     'dif_pago_final_prev_2_month',
     'dif_pago_final_prev_3_month',
     'periodica_0',
     'dif_periodica_prev_month',
     'dif_periodica_prev_2_month',
     'dif_periodica_prev_3_month',
     'consumo_0',
     'dif_consumo_prev_month',
     'dif_consumo_prev_2_month',
     'dif_consumo_prev_3_month',
     'aperiodica_0',
     'dif_aperiodica_prev_month',
     'dif_aperiodica_prev_2_month',
     'dif_aperiodica_prev_3_month',
     'discount_0',
     'dif_discount_prev_month',
     'dif_discount_prev_2_month',
     'dif_discount_prev_3_month',
     'ajuste_0',
     'dif_ajuste_prev_month',
     'dif_ajuste_prev_2_month',
     'dif_ajuste_prev_3_month',
     'Tota_Compra_disp',
     'Curr_Compra_disp',
     'Curr_Compra_Finanz_disp',
     'Curr_Finanz_disp',
     'Month_purchase_disp',
     'Modelo_disp',
     'Import_Rest_quota_disp',
     'pvp_total_disp',
     'pvp_total_disp_movil',
     'Curr_cancel_disp',
     'NUM_GB_OWNN_CURR',
     'NUM_GB_2G_CURR',
     'NUM_GB_3G_CURR',
     'NUM_GB_4G_CURR',
     'NUM_GB_5G_CURR',
     'NUM_SESS_CURR',
     'NUM_SECS_CURR',
     'NUM_CALL_CURR',
     'NUM_CALL_WEEK_CURR',
     'NUM_CALL_WEEKEND_CURR',
     'NUM_SECS_WEEK_CURR',
     'NUM_SECS_WEEKEND_CURR',
     'NUM_CALL_WEEK',
     'NUM_CALL_WEEKEND',
     'NUM_DAYS_LINE_TYPE_FIXE_POST_DEA']




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

nulls = assess_NA(sample)
```


```python
nulls.head(20)
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
      <th>2</th>
      <td>YEAR</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pago_final_0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dif_pago_final_prev_month</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>dif_pago_final_prev_2_month</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>dif_pago_final_prev_3_month</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>periodica_0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>dif_periodica_prev_month</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>dif_periodica_prev_2_month</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>dif_periodica_prev_3_month</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>consumo_0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>dif_consumo_prev_month</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>dif_consumo_prev_2_month</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>dif_consumo_prev_3_month</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>aperiodica_0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>dif_aperiodica_prev_month</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>dif_aperiodica_prev_2_month</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>dif_aperiodica_prev_3_month</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>discount_0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
na_info_sorted = nulls.sort_values(by='percent', ascending=False)
na_info_sorted.head(20)

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
      <th>51</th>
      <td>NUM_DAYS_LINE_TYPE_FIXE_POST_DEA</td>
      <td>266102</td>
      <td>93.882346</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Import_Rest_quota_disp</td>
      <td>225676</td>
      <td>79.619816</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Month_purchase_disp</td>
      <td>183121</td>
      <td>64.606163</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Modelo_disp</td>
      <td>183121</td>
      <td>64.606163</td>
    </tr>
    <tr>
      <th>35</th>
      <td>pvp_total_disp_movil</td>
      <td>183121</td>
      <td>64.606163</td>
    </tr>
    <tr>
      <th>34</th>
      <td>pvp_total_disp</td>
      <td>179070</td>
      <td>63.176946</td>
    </tr>
    <tr>
      <th>37</th>
      <td>NUM_GB_OWNN_CURR</td>
      <td>175367</td>
      <td>61.870506</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Curr_Compra_disp</td>
      <td>175367</td>
      <td>61.870506</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Curr_Compra_Finanz_disp</td>
      <td>175367</td>
      <td>61.870506</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Curr_Finanz_disp</td>
      <td>175367</td>
      <td>61.870506</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Curr_cancel_disp</td>
      <td>175367</td>
      <td>61.870506</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Tota_Compra_disp</td>
      <td>175367</td>
      <td>61.870506</td>
    </tr>
    <tr>
      <th>40</th>
      <td>NUM_GB_4G_CURR</td>
      <td>16594</td>
      <td>5.854461</td>
    </tr>
    <tr>
      <th>38</th>
      <td>NUM_GB_2G_CURR</td>
      <td>16594</td>
      <td>5.854461</td>
    </tr>
    <tr>
      <th>41</th>
      <td>NUM_GB_5G_CURR</td>
      <td>16594</td>
      <td>5.854461</td>
    </tr>
    <tr>
      <th>42</th>
      <td>NUM_SESS_CURR</td>
      <td>16594</td>
      <td>5.854461</td>
    </tr>
    <tr>
      <th>39</th>
      <td>NUM_GB_3G_CURR</td>
      <td>16594</td>
      <td>5.854461</td>
    </tr>
    <tr>
      <th>43</th>
      <td>NUM_SECS_CURR</td>
      <td>5256</td>
      <td>1.854348</td>
    </tr>
    <tr>
      <th>44</th>
      <td>NUM_CALL_CURR</td>
      <td>5256</td>
      <td>1.854348</td>
    </tr>
    <tr>
      <th>45</th>
      <td>NUM_CALL_WEEK_CURR</td>
      <td>5256</td>
      <td>1.854348</td>
    </tr>
  </tbody>
</table>
</div>




```python
sample['customer_id'].nunique()
```




    26312



* 26k unique users in this subsample
* 7% of positive class

### Prepare data for first model

1. First selection of cols (let's include this in ``data_cleaning()``)
2. Clean, normalize, drop, etc (idem)
3. Create feature target (`feature_computation()`)
4. Agreggate data if needed to reflect the past user behavior (idem)


We said we will use for the moment the current precooked metrics for payments, discounts, and consumpitons, which are given by difference month on month. Additionally, let's try to add one col `pago_final_0` and compute the rolling avg for the last 3 months e.g

*IMPORTANT INFO*. First we will use a logistic regression for training our first model, so we should keep an eye on NANs and possible outliers. Later on, when moving into lightGBM this will be less important as it handle those cases natively (same for scaling).

** There is an user that from one month to other, data for the required filter becomes null. Inspect*** -> customer_id = 1322985


```python
# user-info cols to aggregate data later on
users_cols = ['customer_id', 'MONTH', 'YEAR']

# pre-cooked features
diff_cols = ['dif_pago_final_prev_month', 
                   'dif_pago_final_prev_2_month', 
                   'dif_pago_final_prev_3_month', 
                   'dif_consumo_prev_month', 
                   'dif_consumo_prev_2_month', 
                   'dif_consumo_prev_3_month', 
                   'dif_discount_prev_month', 
                   'dif_discount_prev_2_month', 
                   'dif_discount_prev_3_month']

# to-be-cooked features
transform_cols = ['pago_final_0']

# target
target_col = ['NUM_DAYS_LINE_TYPE_FIXE_POST_DEA']
```


```python
sample = sample[users_cols + diff_cols + transform_cols + target_col ]
assess_NA(sample)
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
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MONTH</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>YEAR</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dif_pago_final_prev_month</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dif_pago_final_prev_2_month</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>dif_pago_final_prev_3_month</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>dif_consumo_prev_month</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>dif_consumo_prev_2_month</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>dif_consumo_prev_3_month</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>dif_discount_prev_month</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>dif_discount_prev_2_month</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>dif_discount_prev_3_month</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>pago_final_0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NUM_DAYS_LINE_TYPE_FIXE_POST_DEA</td>
      <td>266102</td>
      <td>93.882346</td>
    </tr>
  </tbody>
</table>
</div>




```python
for col in sample.drop(columns=users_cols):
    min_value = sample[col].min()
    max_value = sample[col].max()
    mean_value = sample[col].mean()
    median_value = sample[col].median()
    perc_zero_count = (sample[col] == 0).sum() / len(sample[col])  # Count the number of zero values

    print(f"'{col}': min {min_value} | max {max_value} | mean: {mean_value} | median {median_value} | perc_zero_counts {perc_zero_count}")
```

    'dif_pago_final_prev_month': min -6113.03 | max 6114.23 | mean: 0.7766547300682327 | median 0.0 | perc_zero_counts 0.29130827470875875
    'dif_pago_final_prev_2_month': min -6114.23 | max 6119.03 | mean: 1.5550578601618674 | median 0.0 | perc_zero_counts 0.24261048115663875
    'dif_pago_final_prev_3_month': min -6135.93 | max 6118.53 | mean: 2.3307775841265586 | median 0.0 | perc_zero_counts 0.21863026650955045
    'dif_consumo_prev_month': min -29185.55 | max 29188.48 | mean: 0.7267077920703351 | median 0.0 | perc_zero_counts 0.014704948455063116
    'dif_consumo_prev_2_month': min -29178.13 | max 29136.18 | mean: 2.5177857903909793 | median 0.0 | perc_zero_counts 0.013974640314420587
    'dif_consumo_prev_3_month': min -29188.69 | max 29147.3 | mean: 4.621394676865111 | median 0.0 | perc_zero_counts 0.013731204267539744
    'dif_discount_prev_month': min -23113.58 | max 23138.6 | mean: -1.6016290458012585 | median 0.3 | perc_zero_counts 0.011483830907205
    'dif_discount_prev_2_month': min -23101.98 | max 23140.09 | mean: -4.316179394726258 | median 0.13 | perc_zero_counts 0.00958926341191496
    'dif_discount_prev_3_month': min -23095.88 | max 23108.38 | mean: -7.322025740715914 | median 0.0 | perc_zero_counts 0.008287409769900014
    'pago_final_0': min -6003.943899999999 | max 6224.953800000007 | mean: 65.39301011000488 | median 60.366750000000025 | perc_zero_counts 0.0027518857473486638
    'NUM_DAYS_LINE_TYPE_FIXE_POST_DEA': min 0 | max 30 | mean: 11.665513264129181 | median 8.0 | perc_zero_counts 0.002010993430754793



```python
sample.groupby('MONTH').apply(lambda x: x[x['NUM_DAYS_LINE_TYPE_FIXE_POST_DEA'] > 0]['customer_id'].nunique())
```




    MONTH
    01     934
    02     960
    03    1015
    04     809
    05     772
    06    5916
    07    2724
    08     626
    09    1009
    10     700
    11     653
    12     652
    dtype: int64




```python
grouped = sample.groupby('MONTH')

churns = grouped.apply(lambda x: x[x['NUM_DAYS_LINE_TYPE_FIXE_POST_DEA'] > 0]['customer_id'].nunique())
unique_customers = grouped['customer_id'].nunique()
result = pd.DataFrame({
    'churns': churns,
    'unique_customers': unique_customers
}).reset_index()
result['ratio_positive_class'] = result['churns'] / result['unique_customers'] * 100
result
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
      <th>MONTH</th>
      <th>churns</th>
      <th>unique_customers</th>
      <th>ratio_positive_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01</td>
      <td>934</td>
      <td>22752</td>
      <td>4.105134</td>
    </tr>
    <tr>
      <th>1</th>
      <td>02</td>
      <td>960</td>
      <td>22986</td>
      <td>4.176455</td>
    </tr>
    <tr>
      <th>2</th>
      <td>03</td>
      <td>1015</td>
      <td>23152</td>
      <td>4.384070</td>
    </tr>
    <tr>
      <th>3</th>
      <td>04</td>
      <td>809</td>
      <td>23258</td>
      <td>3.478373</td>
    </tr>
    <tr>
      <th>4</th>
      <td>05</td>
      <td>772</td>
      <td>23413</td>
      <td>3.297313</td>
    </tr>
    <tr>
      <th>5</th>
      <td>06</td>
      <td>5916</td>
      <td>23586</td>
      <td>25.082676</td>
    </tr>
    <tr>
      <th>6</th>
      <td>07</td>
      <td>2724</td>
      <td>23696</td>
      <td>11.495611</td>
    </tr>
    <tr>
      <th>7</th>
      <td>08</td>
      <td>626</td>
      <td>23821</td>
      <td>2.627933</td>
    </tr>
    <tr>
      <th>8</th>
      <td>09</td>
      <td>1009</td>
      <td>23917</td>
      <td>4.218757</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>700</td>
      <td>24093</td>
      <td>2.905408</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>653</td>
      <td>24316</td>
      <td>2.685475</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>652</td>
      <td>24452</td>
      <td>2.666449</td>
    </tr>
  </tbody>
</table>
</div>



First view makes sense, some considerations:
- Heavily unbalanced (3-4% of positive class, but we will even see months with 2.5%). Lots churn ocurred in Jun and July (keep an eye on seasonality too).
- We understand negative payments as refundings; negative consumption as reversals or corrections; negative discounts as penalizations or reversal of previous 
discounts.
- We should decided what to do with customers that joined this month! as for the moment, we will keep them and assign 0 for payments_prev_month and for the avg_3_months it will imputed the current payment.

Let's build our first basic model with the most basic feature engineering


```python
def get_initial_params():
    """
    Loads all relevant parameters into a global variable. These parameters are then accessible to other functions in the script.
    Each function can consume the parameters it requires for its operation.
    This approach ensures centralized management and consistency of parameters across different functions.

    Returns:
        None: This function does not return a value but populates a global variable
        with necessary parameters.
    """
    global train_from, train_to, logistic_regression_params

    config = configparser.ConfigParser()
    config.read("src/params.ini")

    train_from = config.get("PARAMS", "train_from")
    train_to = config.get("PARAMS", "train_to")

    # Load Logistic Regression parameters
    logistic_regression_params = {
        'penalty': config.get('LOGISTIC_REGRESSION', 'penalty', fallback='l2'),
        'C': config.getfloat('LOGISTIC_REGRESSION', 'C', fallback=1.0),
        'solver': config.get('LOGISTIC_REGRESSION', 'solver', fallback='saga'),
        'max_iter': config.getint('LOGISTIC_REGRESSION', 'max_iter', fallback=10000)
    }
```


```python
def data_cleaning(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw data by handling missing values, removing duplicates, correcting errors, and performing type conversions for data quality and consistency.
    Returns:
        DataFrame: Pandas DataFrame with cleaned and preprocessed data.
    """
    logger.info("Starting cleaning data")

    filter_df = raw_df[users_cols + diff_cols + transform_cols + target_col]
    clean_df = filter_df.dropna(how="all")

    logger.info("Completed cleaning data!")
    return clean_df
```


```python
def feature_computation(clean_data: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    """
    Split data into train and test features set, aggregate the data into historical behavior for those cols needed.
    It also joins it with already calculated features, and extract the needed target from 2 months ahead.
    Args:
        clean_data: The cleaned dataset with customer, month, and payment information.
        train_from: The starting date of the training period.
        train_to: The ending date of the training period.
    
    Returns:
        DataFrame: Pandas DataFrame with computed features for model training.
        DataFrame: Pandas DataFrame representing the target variable for train set.
        DataFrame: Pandas DataFrame with computed features for model testing.
        DataFrame: Pandas DataFrame representing the target variable for test set.
    """
    logger.info("Starting feature computation")

    #TO-DO: Catch exceptions
    # TO-DO: Potential unit tests validating same length for features/targets

    # Convert the train_from and train_to to datetime
    train_from_dt = pd.to_datetime(train_from)
    train_to_dt = pd.to_datetime(train_to)
    
    # Filter train and test data before feature computation
    test_from_dt = train_from_dt + pd.DateOffset(months=2)
    test_to_dt = train_to_dt + pd.DateOffset(months=2)
    target_train_month = train_to_dt + pd.DateOffset(months=2)
    target_test_month = target_train_month + pd.DateOffset(months=2)

    logger.info(f"Train computation from {train_from_dt} to {train_to_dt}. Target for {target_train_month}")
    logger.info(f"Test computation from {test_from_dt} to {test_to_dt}. Target for {target_test_month}")

    # Create date col to mix month and year
    clean_data['date'] = pd.to_datetime(clean_data['YEAR'].astype(str) + '-' + clean_data['MONTH'].astype(str) + '-01')

    # Filter compute_data for the specific cols and date intervals. Also sort i.
    compute_data = clean_data[['date'] + users_cols + transform_cols + diff_cols]
    compute_data = compute_data[(compute_data['date'] >= train_from_dt) & (compute_data['date'] <= test_to_dt)]

    # Perform feature computations for the combined dataset. Assigns nans if needed.
    compute_data = compute_data.sort_values(by=['customer_id', 'date'])
    compute_data['pago_final_prev_month'] = compute_data.groupby('customer_id')['pago_final_0'].shift(1)
    compute_data['pago_final_prev_month'] = compute_data['pago_final_prev_month'].fillna(0)
    compute_data['pago_final_avg_3_months'] = compute_data.groupby('customer_id')['pago_final_0'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    logger.info("Features computed")
 
    # Split the combined dataset into training and testing sets
    features_train = compute_data[compute_data['date'] == train_to_dt]
    features_test = compute_data[(compute_data['date'] == test_to_dt)]

    # Select only the most recent month's data per customer
    #final_features_train = features_train.groupby('customer_id').tail(1)
    #final_features_test = features_test.groupby('customer_id').tail(1)

    # Extract the target for the training and testing sets
    target_train = clean_data[clean_data['date'] == target_train_month][['customer_id'] + target_col]
    target_test = clean_data[clean_data['date'] == target_test_month][['customer_id'] + target_col]

    for target_df in [target_train, target_test]:
        for col in target_col:
            target_df[col].fillna(0, inplace=True)
            target_df[col] = np.where(target_df[col] > 0, 1, 0)

    # Now we need to join it with customer_id from features df
    # Check: i'm using an inner join because there are some edge cases to clarify (e.g. customer_id = 1322985)
    features_and_target_train = features_train.merge(target_train, on='customer_id', how='inner')
    features_and_target_test = features_test.merge(target_test, on='customer_id', how='inner')

    #Split train and test features + target (squeeze into 1D array)
    features = features_and_target_train.drop(columns=target_col + users_cols + ['date'])
    target = features_and_target_train[target_col].squeeze()
    features_test = features_and_target_test.drop(columns=target_col + users_cols + ['date'])
    target_test = features_and_target_test[target_col].squeeze()

    logger.info(f"Features: {features.columns.tolist()}")
    logger.info(f"Target: {target.name}")
    logger.info("Completed feature computation!")

    return features, target, features_test, target_test

```


```python
def modeling(features: pd.DataFrame, target: pd.Series) -> Pipeline:
    """
    Prepares a machine learning pipeline that scales features and trains a logistic regression model 
    with processed data to predict churn.

    Args:
        features (pd.DataFrame): The features to train the model on.
        target (pd.Series): The target variable to predict.

    Returns:
        Pipeline: A pipeline with a standard scaler and a trained logistic regression model.
    """

    logger.info("Starting Modeling")
    
    # Directly use the logistic_regression_params global variable
    logger.info("Building model pipeline")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic_regression', LogisticRegression(
            penalty=logistic_regression_params['penalty'],
            C=logistic_regression_params['C'],
            solver=logistic_regression_params['solver'],
            max_iter=logistic_regression_params['max_iter'],
            random_state=42
        ))
    ])

    logger.info("Training model")
    model = pipeline.fit(features, target)

    logger.info("Completed model training!")

    return model
```


```python
def evaluation(model: LogisticRegression, features_test: pd.DataFrame, target_test: pd.Series) -> dict[str, str]:
    """
    Assesses trained model's performance using a test dataset and computes metrics like accuracy, precision, recall, and ROC-AUC.

    Returns:
        dict: Dictionary with key performance metrics of the model.
    """
    logger.info(f"Started evaluation for {model}")
    preds = model.predict_proba(features_test)[:, 1]    
    
    # Plotting
    logger.info("Generating plots")

    generate_evaluation_curves(model, preds, target_test, )
    
    preds = model.predict_proba(features_test)
    precision, recall, _ = precision_recall_curve(target_test, preds[:, 1])
    model_metrics = {
        "Precision Curve": precision,
        "Recall Curve": recall,
        #"ROC AUC": roc_auc
    }

    # Calculate Precision in the First Decile
    precision_decile = calculate_precision_first_decile(target_test, model.predict_proba(features_test)[:, 1])
    logger.info(f"Precision in the first decile: {precision_decile:.2f}")

    # Calculate Uplift for Each Decile
    uplift_by_decile = calculate_uplift(target_test, model.predict_proba(features_test)[:, 1])
    logger.info("Uplift by decile:")
    logger.info(uplift_by_decile)

    logger.info("Completed evaluation!")
    return model_metrics, precision_decile, uplift_by_decile
```


```python
def generate_evaluation_curves(
    model: str, y_pred, y_test, save_curves_path: str = None
):
    """
    Generate ROC and Precision-Recall curves for a binary classification model
    and save them in a single figure.

    Parameters:
    - model_name (str): Name of the model for labeling the curves.
    - y_pred (array-like): Predicted probabilities or scores.
    - y_test (array-like): True labels.
    - save_curves_path (str, optional): Directory to save the generated figure.
    If None, the figure will not be saved.

    Returns:
    - None
    """

    # Create a timestamp for unique filenames
    #timestamp = datetime.now().strftime("%Y_%m_%d")
    model_type = type(model[-1]).__name__  # Assuming 'model' is your pipeline

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f}) - {model_type}")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend(loc="lower right")

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"PR (AUC = {pr_auc:.2f}) - {model_type}")
    plt.xlim([-0.005, 1.0])  # Adjusted to start slightly before 0 for a clearer view
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")

    plt.tight_layout()

    if save_curves_path:
        # Define the filename with a timestamp
        figure_filename = f"Evaluation_Curves_{timestamp}.png"
        figure_path = os.path.join(save_curves_path, figure_filename)

        plt.savefig(figure_path)

    plt.show()
```


```python
def calculate_precision_first_decile(target, y_pred_proba):
    """
    Calculate the precision in the first decile of predictions.

    Args:
    - y_true (array-like): True labels.
    - y_pred_proba (array-like): Predicted probabilities.

    Returns:
    - precision_decile (float): Precision in the first decile.
    """
    data = pd.DataFrame({'y_true': target, 'y_pred_proba': y_pred_proba})
    data_sorted = data.sort_values(by='y_pred_proba', ascending=False)
    decile_cutoff = int(len(data_sorted) * 0.1)
    first_decile = data_sorted.head(decile_cutoff)
    true_positives = first_decile['y_true'].sum()
    precision_decile = true_positives / decile_cutoff

    return precision_decile
```


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
    data = pd.DataFrame({'y_true': target, 'y_pred_proba': y_pred_proba})
    data_sorted = data.sort_values(by='y_pred_proba', ascending=False)
    data_sorted['decile'] = pd.qcut(data_sorted['y_pred_proba'], 10, labels=False)
    decile_churn_rate = data_sorted.groupby('decile')['y_true'].mean()
    overall_churn_rate = data['y_true'].mean()
    uplift = decile_churn_rate / overall_churn_rate

    return uplift
```


```python
get_initial_params()
logger.info(f"Train from {train_from} to {train_to}")
clean_data = data_cleaning(sample)
features, target, features_test, target_test = feature_computation(clean_data)
model = modeling(features, target)
model_metrics, precision_decile, uplift_by_decile = evaluation(model, features, target)
```

    INFO - Train from 2022-01-01 to 2022-06-01
    INFO - Starting cleaning data
    INFO - Completed cleaning data!
    INFO - Starting feature computation
    INFO - Train computation from 2022-01-01 00:00:00 to 2022-06-01 00:00:00. Target for 2022-08-01 00:00:00
    INFO - Test computation from 2022-03-01 00:00:00 to 2022-08-01 00:00:00. Target for 2022-10-01 00:00:00
    INFO - Features computed
    INFO - Features: ['pago_final_0', 'dif_pago_final_prev_month', 'dif_pago_final_prev_2_month', 'dif_pago_final_prev_3_month', 'dif_consumo_prev_month', 'dif_consumo_prev_2_month', 'dif_consumo_prev_3_month', 'dif_discount_prev_month', 'dif_discount_prev_2_month', 'dif_discount_prev_3_month', 'pago_final_prev_month', 'pago_final_avg_3_months']
    INFO - Target: NUM_DAYS_LINE_TYPE_FIXE_POST_DEA
    INFO - Completed feature computation!
    INFO - Starting Modeling
    INFO - Building model pipeline
    INFO - Training model
    INFO - Completed model training!
    INFO - Started evaluation for Pipeline(steps=[('scaler', StandardScaler()),
                    ('logistic_regression',
                     LogisticRegression(max_iter=10000, random_state=42,
                                        solver='saga'))])
    INFO - Generating plots



    
![png](explore_data_files/explore_data_41_1.png)
    


    INFO - Precision in the first decile: 0.04
    INFO - Uplift by decile:
    INFO - decile
    0    0.553644
    1    0.687860
    2    0.754969
    3    0.755293
    4    0.738192
    5    1.006625
    6    1.258821
    7    1.392498
    8    1.140842
    9    1.711263
    Name: y_true, dtype: float64
    INFO - Completed evaluation!



```python

def get_feature_importance_logistic_regression(model, feature_names):
    """
    Get feature importance for Logistic Regression model.

    Args:
    - model: Trained Logistic Regression model.
    - feature_names (list or array-like): List of feature names.

    Returns:
    - pd.DataFrame: DataFrame containing feature names and their corresponding coefficients.
    """
    feature_names = features.columns
    # Extract coefficients
    lr_model = model.named_steps['logistic_regression']
    coefficients = lr_model.coef_[0]  # for Logistic Regression

    # Create a DataFrame for easy visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    # Sort by absolute value of coefficients in descending order
    feature_importance = feature_importance.reindex(feature_importance.Coefficient.abs().sort_values(ascending=False).index)

    return feature_importance

# Example usage
# Assuming 'feature_names' is a list of your feature names
feature_importance_lr = get_feature_importance_logistic_regression(model, features)
feature_importance_lr.head(10)
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
      <th>Feature</th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>dif_discount_prev_2_month</td>
      <td>-0.403543</td>
    </tr>
    <tr>
      <th>5</th>
      <td>dif_consumo_prev_2_month</td>
      <td>-0.364464</td>
    </tr>
    <tr>
      <th>7</th>
      <td>dif_discount_prev_month</td>
      <td>0.248656</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dif_consumo_prev_month</td>
      <td>0.141422</td>
    </tr>
    <tr>
      <th>9</th>
      <td>dif_discount_prev_3_month</td>
      <td>-0.117019</td>
    </tr>
    <tr>
      <th>0</th>
      <td>pago_final_0</td>
      <td>0.093403</td>
    </tr>
    <tr>
      <th>10</th>
      <td>pago_final_prev_month</td>
      <td>0.088020</td>
    </tr>
    <tr>
      <th>11</th>
      <td>pago_final_avg_3_months</td>
      <td>0.084572</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dif_pago_final_prev_month</td>
      <td>0.052332</td>
    </tr>
    <tr>
      <th>6</th>
      <td>dif_consumo_prev_3_month</td>
      <td>0.018606</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
