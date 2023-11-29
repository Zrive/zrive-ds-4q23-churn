# Load libraries and data


```python
import sys
import os
import pandas as pd

# Get project root and change wd
current_wd = os.getcwd()
project_root = os.path.dirname(os.path.dirname(current_wd))
os.chdir(project_root)

src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Load the needed class and functions
from db_connectors.bigquery_service import BigqueryService
from data_gathering import data_gathering
```

Define the query


```python
query_sql = """SELECT *
FROM `mm-bi-catedras-upm.ESTIMACION_CHURN.multibrand_monthly_customer_base_mp2022`
WHERE IS_CUST_SEGM_RESI > 0 
  AND IS_CUST_BILL_POST_CURR = TRUE
  AND CUST_BUNDLE_CURR = 'FMC'
  AND NUM_IMPAGOS = 0
  LIMIT 10000
  """
```


```python
sample = data_gathering(query_sql)
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
      <th>brand_ds</th>
      <th>customer_id</th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>EOP</th>
      <th>IS_CUST_SEGM_RESI</th>
      <th>IS_CUST_SEGM_SOHO</th>
      <th>IS_CUST_SEGM_BUSI</th>
      <th>CHANNEL_LAST_MOBILE</th>
      <th>CHANNEL_FIRST_MOBILE</th>
      <th>...</th>
      <th>AVG_NUM_DAYS_BETWEEN_FIX_PORT</th>
      <th>MIN_NUM_DAYS_BETWEEN_FIX_PORT</th>
      <th>NUM_PORT_OPER_DONO_PREM_TWO_YEAR_AGO</th>
      <th>NUM_MOB_PORT_TRANS_CURR</th>
      <th>MM_GROUP_MOB_PORT</th>
      <th>NUM_MONTHS_SINCE_LAST_MOB_PORT_REQ</th>
      <th>MAX_NUM_DAYS_BETWEEN_MOB_PORT_REQS</th>
      <th>NUM_MONTHS_SINCE_LAST_MOB_PORT</th>
      <th>MAX_NUM_MONTHS_BETWEEN_MOB_PORT_REQS</th>
      <th>NUM_PORT_OPER_RECE_YOIGO_TWO_YEAR_AGO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>EUSKALTEL</td>
      <td>441555</td>
      <td>2022</td>
      <td>12</td>
      <td>202212</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>EXCLUSIVO</td>
      <td>EXCLUSIVO</td>
      <td>...</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>1</th>
      <td>EUSKALTEL</td>
      <td>568658</td>
      <td>2022</td>
      <td>12</td>
      <td>202212</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>CARTERA</td>
      <td>CARTERA</td>
      <td>...</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EUSKALTEL</td>
      <td>2331514</td>
      <td>2022</td>
      <td>12</td>
      <td>202212</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>PLAT. FIDELIZACION/RETENCION</td>
      <td>PLAT. FIDELIZACION/RETENCION</td>
      <td>...</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EUSKALTEL</td>
      <td>359543</td>
      <td>2022</td>
      <td>12</td>
      <td>202212</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>CARTERA</td>
      <td>CARTERA</td>
      <td>...</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EUSKALTEL</td>
      <td>1498138</td>
      <td>2022</td>
      <td>12</td>
      <td>202212</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>PLAT. FIDELIZACION/RETENCION</td>
      <td>PLAT. FIDELIZACION/RETENCION</td>
      <td>...</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 430 columns</p>
</div>




```python

```
