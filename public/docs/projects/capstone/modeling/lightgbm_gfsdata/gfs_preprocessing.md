# Exploring Model For Pollen Count

In this notebook, we will perform data preprocessing for the pollen count data. We will also apply feature selection methods on the dataset. We will try several regression models to compare and discuss the most reasonable and appropriate model.


```python
import pickle
import datetime
import pandas as pd
import numpy as np

# visulization
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go


import math

import warnings
warnings.simplefilter(/docs/projects/capstone/Modeling/lightgbm_gfsdata/action='ignore', category=FutureWarning)
warnings.filterwarnings(/docs/projects/capstone/Modeling/lightgbm_gfsdata/"ignore")
```

## Preprocessing Steps
We load two datasets: gfs dataset and the grass count dataset:


```python
weather_df = pd.read_csv(/docs/projects/capstone/Modeling/lightgbm_gfsdata/'all_data.csv')
grass_df = pd.read_csv(/docs/projects/capstone/Modeling/lightgbm_gfsdata/'../preprocessing/melbourne_grass_preprocessed.csv')
```


```python
weather_df
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
      <th>coord</th>
      <th>t_mean_9am</th>
      <th>t_min_9am</th>
      <th>t_max_9am</th>
      <th>t_sd_9am</th>
      <th>t_2m_mean_9am</th>
      <th>t_2m_min_9am</th>
      <th>t_2m_max_9am</th>
      <th>t_2m_sd_9am</th>
      <th>msl_mean_9am</th>
      <th>...</th>
      <th>u_10m_sd_4pm</th>
      <th>v_10m_mean_4pm</th>
      <th>v_10m_min_4pm</th>
      <th>v_10m_max_4pm</th>
      <th>v_10m_sd_4pm</th>
      <th>pwat_mean_4pm</th>
      <th>pwat_min_4pm</th>
      <th>pwat_max_4pm</th>
      <th>pwat_sd_4pm</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(/docs/projects/capstone/Modeling/lightgbm_gfsdata/145,-38)</td>
      <td>285.8750</td>
      <td>280.10000</td>
      <td>290.8</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>0.499166</td>
      <td>3.300000</td>
      <td>1.499999</td>
      <td>6.000000</td>
      <td>1.961293</td>
      <td>13.875000</td>
      <td>13.000000</td>
      <td>14.900000</td>
      <td>0.880814</td>
      <td>2000-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(/docs/projects/capstone/Modeling/lightgbm_gfsdata/145,-38)</td>
      <td>285.8750</td>
      <td>280.10000</td>
      <td>290.8</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>1.175797</td>
      <td>-2.450000</td>
      <td>-8.000000</td>
      <td>2.900000</td>
      <td>5.402777</td>
      <td>16.925000</td>
      <td>13.600000</td>
      <td>20.000000</td>
      <td>2.780138</td>
      <td>2000-01-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(/docs/projects/capstone/Modeling/lightgbm_gfsdata/145,-38)</td>
      <td>285.8750</td>
      <td>280.10000</td>
      <td>290.8</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>2.538208</td>
      <td>3.100000</td>
      <td>1.099999</td>
      <td>5.700000</td>
      <td>1.971464</td>
      <td>22.625000</td>
      <td>18.800000</td>
      <td>24.700000</td>
      <td>2.617091</td>
      <td>2000-01-03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(/docs/projects/capstone/Modeling/lightgbm_gfsdata/145,-38)</td>
      <td>285.8750</td>
      <td>280.10000</td>
      <td>290.8</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>2.053453</td>
      <td>4.850000</td>
      <td>3.500000</td>
      <td>6.200000</td>
      <td>1.161895</td>
      <td>16.400000</td>
      <td>12.300000</td>
      <td>20.200000</td>
      <td>3.799123</td>
      <td>2000-01-04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(/docs/projects/capstone/Modeling/lightgbm_gfsdata/145,-38)</td>
      <td>285.8750</td>
      <td>280.10000</td>
      <td>290.8</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>0.618467</td>
      <td>3.975000</td>
      <td>2.800000</td>
      <td>5.600000</td>
      <td>1.286792</td>
      <td>14.550000</td>
      <td>13.400000</td>
      <td>15.100000</td>
      <td>0.776745</td>
      <td>2000-01-05</td>
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
      <th>8296</th>
      <td>(/docs/projects/capstone/Modeling/lightgbm_gfsdata/145,-38)</td>
      <td>285.1931</td>
      <td>285.12766</td>
      <td>285.3</td>
      <td>0.077648</td>
      <td>283.9077</td>
      <td>282.96112</td>
      <td>285.07898</td>
      <td>0.876952</td>
      <td>103015.92</td>
      <td>...</td>
      <td>1.610581</td>
      <td>-1.089071</td>
      <td>-7.484739</td>
      <td>3.261987</td>
      <td>5.142459</td>
      <td>7.042216</td>
      <td>6.314592</td>
      <td>7.748879</td>
      <td>0.727160</td>
      <td>2022-09-13</td>
    </tr>
    <tr>
      <th>8297</th>
      <td>(/docs/projects/capstone/Modeling/lightgbm_gfsdata/145,-38)</td>
      <td>285.1931</td>
      <td>285.12766</td>
      <td>285.3</td>
      <td>0.077648</td>
      <td>283.9077</td>
      <td>282.96112</td>
      <td>285.07898</td>
      <td>0.876952</td>
      <td>103015.92</td>
      <td>...</td>
      <td>0.222384</td>
      <td>-7.259369</td>
      <td>-8.018267</td>
      <td>-5.899118</td>
      <td>0.970635</td>
      <td>19.331320</td>
      <td>11.912160</td>
      <td>24.669758</td>
      <td>5.546390</td>
      <td>2022-09-14</td>
    </tr>
    <tr>
      <th>8298</th>
      <td>(/docs/projects/capstone/Modeling/lightgbm_gfsdata/145,-38)</td>
      <td>285.1931</td>
      <td>285.12766</td>
      <td>285.3</td>
      <td>0.077648</td>
      <td>283.9077</td>
      <td>282.96112</td>
      <td>285.07898</td>
      <td>0.876952</td>
      <td>103015.92</td>
      <td>...</td>
      <td>0.674031</td>
      <td>-5.383536</td>
      <td>-6.902886</td>
      <td>-4.472268</td>
      <td>1.053194</td>
      <td>15.732620</td>
      <td>13.822749</td>
      <td>16.694824</td>
      <td>1.317087</td>
      <td>2022-09-15</td>
    </tr>
    <tr>
      <th>8299</th>
      <td>(/docs/projects/capstone/Modeling/lightgbm_gfsdata/145,-38)</td>
      <td>285.1931</td>
      <td>285.12766</td>
      <td>285.3</td>
      <td>0.077648</td>
      <td>283.9077</td>
      <td>282.96112</td>
      <td>285.07898</td>
      <td>0.876952</td>
      <td>103015.92</td>
      <td>...</td>
      <td>0.380408</td>
      <td>-7.232979</td>
      <td>-10.783916</td>
      <td>-5.370007</td>
      <td>2.497160</td>
      <td>17.352787</td>
      <td>14.354100</td>
      <td>20.170527</td>
      <td>2.413311</td>
      <td>2022-09-16</td>
    </tr>
    <tr>
      <th>8300</th>
      <td>(/docs/projects/capstone/Modeling/lightgbm_gfsdata/145,-38)</td>
      <td>285.1931</td>
      <td>285.12766</td>
      <td>285.3</td>
      <td>0.077648</td>
      <td>283.9077</td>
      <td>282.96112</td>
      <td>285.07898</td>
      <td>0.876952</td>
      <td>103015.92</td>
      <td>...</td>
      <td>2.184013</td>
      <td>-3.693820</td>
      <td>-7.139402</td>
      <td>-2.338088</td>
      <td>2.308063</td>
      <td>15.472172</td>
      <td>14.678462</td>
      <td>16.852072</td>
      <td>0.961116</td>
      <td>2022-09-17</td>
    </tr>
  </tbody>
</table>
<p>8301 rows × 58 columns</p>
</div>




```python
grass_df['Count Date'] = grass_df['Count Date'].apply(/docs/projects/capstone/Modeling/lightgbm_gfsdata/lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))
weather_df['date'] = weather_df['date'].apply(/docs/projects/capstone/Modeling/lightgbm_gfsdata/lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))
```


```python
df = pd.merge(/docs/projects/capstone/Modeling/lightgbm_gfsdata/grass_df, weather_df, left_on='Count Date', right_on='date', how='left').drop('Count Date', axis=1)
df = pd.merge(grass_df, weather_df, left_on='Count Date', right_on='date', how='left').drop(/docs/projects/capstone/Modeling/lightgbm_gfsdata/'Count Date', axis=1)
```


```python
# check missing values
/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/f/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/i/docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/n/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/v/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/u/docs/projects/capstone/Modeling/lightgbm_gfsdata/e/docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/u/docs/projects/capstone/Modeling/lightgbm_gfsdata/m/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata//docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/f/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/i/docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/n/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/v/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/u/docs/projects/capstone/Modeling/lightgbm_gfsdata/e/docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/u/docs/projects/capstone/Modeling/lightgbm_gfsdata/m/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/```




    174812




```python
# dul with missing values
/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/f/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/=/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/f/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/n/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/df = df.drop(/docs/projects/capstone/Modeling/lightgbm_gfsdata/'coord', axis=1)
```


```python
df
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
      <th>grass_count</th>
      <th>t_mean_9am</th>
      <th>t_min_9am</th>
      <th>t_max_9am</th>
      <th>t_sd_9am</th>
      <th>t_2m_mean_9am</th>
      <th>t_2m_min_9am</th>
      <th>t_2m_max_9am</th>
      <th>t_2m_sd_9am</th>
      <th>msl_mean_9am</th>
      <th>...</th>
      <th>u_10m_sd_4pm</th>
      <th>v_10m_mean_4pm</th>
      <th>v_10m_min_4pm</th>
      <th>v_10m_max_4pm</th>
      <th>v_10m_sd_4pm</th>
      <th>pwat_mean_4pm</th>
      <th>pwat_min_4pm</th>
      <th>pwat_max_4pm</th>
      <th>pwat_sd_4pm</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3014</th>
      <td>0.0</td>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.80000</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>0.499166</td>
      <td>3.300000</td>
      <td>1.499999</td>
      <td>6.000000</td>
      <td>1.961293</td>
      <td>13.875000</td>
      <td>13.000000</td>
      <td>14.900000</td>
      <td>0.880814</td>
      <td>2000-01-01</td>
    </tr>
    <tr>
      <th>3015</th>
      <td>0.0</td>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.80000</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>1.175797</td>
      <td>-2.450000</td>
      <td>-8.000000</td>
      <td>2.900000</td>
      <td>5.402777</td>
      <td>16.925000</td>
      <td>13.600000</td>
      <td>20.000000</td>
      <td>2.780138</td>
      <td>2000-01-02</td>
    </tr>
    <tr>
      <th>3016</th>
      <td>0.0</td>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.80000</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>2.538208</td>
      <td>3.100000</td>
      <td>1.099999</td>
      <td>5.700000</td>
      <td>1.971464</td>
      <td>22.625000</td>
      <td>18.800000</td>
      <td>24.700000</td>
      <td>2.617091</td>
      <td>2000-01-03</td>
    </tr>
    <tr>
      <th>3017</th>
      <td>0.0</td>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.80000</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>2.053453</td>
      <td>4.850000</td>
      <td>3.500000</td>
      <td>6.200000</td>
      <td>1.161895</td>
      <td>16.400000</td>
      <td>12.300000</td>
      <td>20.200000</td>
      <td>3.799123</td>
      <td>2000-01-04</td>
    </tr>
    <tr>
      <th>3018</th>
      <td>0.0</td>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.80000</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>0.618467</td>
      <td>3.975000</td>
      <td>2.800000</td>
      <td>5.600000</td>
      <td>1.286792</td>
      <td>14.550000</td>
      <td>13.400000</td>
      <td>15.100000</td>
      <td>0.776745</td>
      <td>2000-01-05</td>
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
      <th>10716</th>
      <td>1.0</td>
      <td>290.56924</td>
      <td>290.1858</td>
      <td>290.96576</td>
      <td>0.372835</td>
      <td>290.5599</td>
      <td>286.59177</td>
      <td>294.92386</td>
      <td>3.661287</td>
      <td>101139.20</td>
      <td>...</td>
      <td>5.365728</td>
      <td>3.833712</td>
      <td>2.327358</td>
      <td>6.158801</td>
      <td>1.793364</td>
      <td>22.789146</td>
      <td>15.239947</td>
      <td>33.476690</td>
      <td>8.651606</td>
      <td>2021-01-27</td>
    </tr>
    <tr>
      <th>10717</th>
      <td>7.0</td>
      <td>290.56924</td>
      <td>290.1858</td>
      <td>290.96576</td>
      <td>0.372835</td>
      <td>290.5599</td>
      <td>286.59177</td>
      <td>294.92386</td>
      <td>3.661287</td>
      <td>101139.20</td>
      <td>...</td>
      <td>2.745395</td>
      <td>0.180323</td>
      <td>-2.282188</td>
      <td>1.797918</td>
      <td>1.736471</td>
      <td>42.359985</td>
      <td>39.000000</td>
      <td>46.439950</td>
      <td>3.152500</td>
      <td>2021-01-28</td>
    </tr>
    <tr>
      <th>10718</th>
      <td>2.0</td>
      <td>290.56924</td>
      <td>290.1858</td>
      <td>290.96576</td>
      <td>0.372835</td>
      <td>290.5599</td>
      <td>286.59177</td>
      <td>294.92386</td>
      <td>3.661287</td>
      <td>101139.20</td>
      <td>...</td>
      <td>3.541771</td>
      <td>-1.670463</td>
      <td>-5.191489</td>
      <td>1.783296</td>
      <td>3.729120</td>
      <td>31.447510</td>
      <td>19.669306</td>
      <td>41.039948</td>
      <td>9.968616</td>
      <td>2021-01-29</td>
    </tr>
    <tr>
      <th>10719</th>
      <td>0.0</td>
      <td>290.56924</td>
      <td>290.1858</td>
      <td>290.96576</td>
      <td>0.372835</td>
      <td>290.5599</td>
      <td>286.59177</td>
      <td>294.92386</td>
      <td>3.661287</td>
      <td>101139.20</td>
      <td>...</td>
      <td>2.336742</td>
      <td>5.019487</td>
      <td>3.665413</td>
      <td>7.490205</td>
      <td>1.705057</td>
      <td>14.177647</td>
      <td>12.470641</td>
      <td>17.039948</td>
      <td>2.022703</td>
      <td>2021-01-30</td>
    </tr>
    <tr>
      <th>10720</th>
      <td>0.0</td>
      <td>290.56924</td>
      <td>290.1858</td>
      <td>290.96576</td>
      <td>0.372835</td>
      <td>290.5599</td>
      <td>286.59177</td>
      <td>294.92386</td>
      <td>3.661287</td>
      <td>101139.20</td>
      <td>...</td>
      <td>2.376949</td>
      <td>2.417457</td>
      <td>-0.668796</td>
      <td>6.751799</td>
      <td>3.680807</td>
      <td>16.241596</td>
      <td>13.200000</td>
      <td>22.600000</td>
      <td>4.324917</td>
      <td>2021-01-31</td>
    </tr>
  </tbody>
</table>
<p>7707 rows × 58 columns</p>
</div>




```python
# sort the whole dataframe by date (/docs/projects/capstone/Modeling/lightgbm_gfsdata/in time order)
df = df.sort_values(/docs/projects/capstone/Modeling/lightgbm_gfsdata/by=['date'])

# add a new column "year" to count each year's data
df['year'] = df['date'].dt.to_period(/docs/projects/capstone/Modeling/lightgbm_gfsdata/"Y").astype(str)
df['year'] = df['date'].dt.to_period("Y").astype(/docs/projects/capstone/Modeling/lightgbm_gfsdata/str)

# for training
df['train'] = df['date'].dt.to_period(/docs/projects/capstone/Modeling/lightgbm_gfsdata/"M").astype(str)
df['train'] = df['date'].dt.to_period("M").astype(/docs/projects/capstone/Modeling/lightgbm_gfsdata/str)

# add a new column "month" to count each year's data
df['month'] = df['date'].dt.strftime(/docs/projects/capstone/Modeling/lightgbm_gfsdata/'%m').astype(str)
df['month'] = df['date'].dt./docs/projects/capstone/Modeling/lightgbm_gfsdata/strftime('%m').astype(/docs/projects/capstone/Modeling/lightgbm_gfsdata/str)
# df['season'] = df['month'].map(/docs/projects/capstone/Modeling/lightgbm_gfsdata/{'12':1, '01':1, '02':1, '03':2, '04': 2, '05': 2, '06':3, '07':3, '08':3, '09':4, '10':4, '11':4})

# show the counts of each year
df['year'].value_counts(/docs/projects/capstone/Modeling/lightgbm_gfsdata/ascending = False)
```




    2008    371
    2000    366
    2012    366
    2020    366
    2004    366
    2016    366
    2019    365
    2018    365
    2017    365
    2015    365
    2014    365
    2013    365
    2011    365
    2001    365
    2010    365
    2009    365
    2007    365
    2006    365
    2005    365
    2003    365
    2002    365
    2021     31
    Name: year, dtype: int64




```python
df = df[df['year'] != '2021']
df
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
      <th>grass_count</th>
      <th>t_mean_9am</th>
      <th>t_min_9am</th>
      <th>t_max_9am</th>
      <th>t_sd_9am</th>
      <th>t_2m_mean_9am</th>
      <th>t_2m_min_9am</th>
      <th>t_2m_max_9am</th>
      <th>t_2m_sd_9am</th>
      <th>msl_mean_9am</th>
      <th>...</th>
      <th>v_10m_max_4pm</th>
      <th>v_10m_sd_4pm</th>
      <th>pwat_mean_4pm</th>
      <th>pwat_min_4pm</th>
      <th>pwat_max_4pm</th>
      <th>pwat_sd_4pm</th>
      <th>date</th>
      <th>year</th>
      <th>train</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3014</th>
      <td>0.0</td>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.80000</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>6.000000</td>
      <td>1.961293</td>
      <td>13.875000</td>
      <td>13.000000</td>
      <td>14.900000</td>
      <td>0.880814</td>
      <td>2000-01-01</td>
      <td>2000</td>
      <td>2000-01</td>
      <td>01</td>
    </tr>
    <tr>
      <th>3015</th>
      <td>0.0</td>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.80000</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>2.900000</td>
      <td>5.402777</td>
      <td>16.925000</td>
      <td>13.600000</td>
      <td>20.000000</td>
      <td>2.780138</td>
      <td>2000-01-02</td>
      <td>2000</td>
      <td>2000-01</td>
      <td>01</td>
    </tr>
    <tr>
      <th>3016</th>
      <td>0.0</td>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.80000</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>5.700000</td>
      <td>1.971464</td>
      <td>22.625000</td>
      <td>18.800000</td>
      <td>24.700000</td>
      <td>2.617091</td>
      <td>2000-01-03</td>
      <td>2000</td>
      <td>2000-01</td>
      <td>01</td>
    </tr>
    <tr>
      <th>3017</th>
      <td>0.0</td>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.80000</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>6.200000</td>
      <td>1.161895</td>
      <td>16.400000</td>
      <td>12.300000</td>
      <td>20.200000</td>
      <td>3.799123</td>
      <td>2000-01-04</td>
      <td>2000</td>
      <td>2000-01</td>
      <td>01</td>
    </tr>
    <tr>
      <th>3018</th>
      <td>0.0</td>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.80000</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>5.600000</td>
      <td>1.286792</td>
      <td>14.550000</td>
      <td>13.400000</td>
      <td>15.100000</td>
      <td>0.776745</td>
      <td>2000-01-05</td>
      <td>2000</td>
      <td>2000-01</td>
      <td>01</td>
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
      <th>10685</th>
      <td>119.0</td>
      <td>290.56924</td>
      <td>290.1858</td>
      <td>290.96576</td>
      <td>0.372835</td>
      <td>290.5599</td>
      <td>286.59177</td>
      <td>294.92386</td>
      <td>3.661287</td>
      <td>101139.20</td>
      <td>...</td>
      <td>8.915293</td>
      <td>2.105036</td>
      <td>19.927906</td>
      <td>11.900000</td>
      <td>29.617905</td>
      <td>9.217530</td>
      <td>2020-12-27</td>
      <td>2020</td>
      <td>2020-12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>10686</th>
      <td>52.0</td>
      <td>290.56924</td>
      <td>290.1858</td>
      <td>290.96576</td>
      <td>0.372835</td>
      <td>290.5599</td>
      <td>286.59177</td>
      <td>294.92386</td>
      <td>3.661287</td>
      <td>101139.20</td>
      <td>...</td>
      <td>5.712373</td>
      <td>1.995186</td>
      <td>15.316968</td>
      <td>14.800000</td>
      <td>15.933380</td>
      <td>0.585012</td>
      <td>2020-12-28</td>
      <td>2020</td>
      <td>2020-12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>10687</th>
      <td>2.0</td>
      <td>290.56924</td>
      <td>290.1858</td>
      <td>290.96576</td>
      <td>0.372835</td>
      <td>290.5599</td>
      <td>286.59177</td>
      <td>294.92386</td>
      <td>3.661287</td>
      <td>101139.20</td>
      <td>...</td>
      <td>4.508652</td>
      <td>0.843708</td>
      <td>20.322964</td>
      <td>17.733380</td>
      <td>24.500000</td>
      <td>3.043544</td>
      <td>2020-12-29</td>
      <td>2020</td>
      <td>2020-12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>10688</th>
      <td>9.0</td>
      <td>290.56924</td>
      <td>290.1858</td>
      <td>290.96576</td>
      <td>0.372835</td>
      <td>290.5599</td>
      <td>286.59177</td>
      <td>294.92386</td>
      <td>3.661287</td>
      <td>101139.20</td>
      <td>...</td>
      <td>6.708228</td>
      <td>0.871834</td>
      <td>23.850000</td>
      <td>21.100000</td>
      <td>26.000000</td>
      <td>2.350177</td>
      <td>2020-12-30</td>
      <td>2020</td>
      <td>2020-12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>10689</th>
      <td>14.0</td>
      <td>290.56924</td>
      <td>290.1858</td>
      <td>290.96576</td>
      <td>0.372835</td>
      <td>290.5599</td>
      <td>286.59177</td>
      <td>294.92386</td>
      <td>3.661287</td>
      <td>101139.20</td>
      <td>...</td>
      <td>6.183415</td>
      <td>2.657620</td>
      <td>28.945654</td>
      <td>27.019995</td>
      <td>31.662620</td>
      <td>1.950055</td>
      <td>2020-12-31</td>
      <td>2020</td>
      <td>2020-12</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
<p>7676 rows × 61 columns</p>
</div>




```python
df['day'] = df['date'].apply(/docs/projects/capstone/Modeling/lightgbm_gfsdata/lambda x: x.strftime("%m-%d"))
```


```python
day_list = list(/docs/projects/capstone/Modeling/lightgbm_gfsdata/pd.unique(df['day']))
pollen_day_list = day_list[274:]
```


```python
np.mean(/docs/projects/capstone/Modeling/lightgbm_gfsdata/df[df['day'] == '10-01']['grass_count'])
```




    5.904761904761905




```python
df['seasonal_avg_count'] = 0
avg_list = []
for day in pollen_day_list:
    avg = np.mean(/docs/projects/capstone/Modeling/lightgbm_gfsdata/df[df['day'] == day]['grass_count'])
    df['seasonal_avg_count'][df['day'] == day] = avg
    /docs/projects/capstone/Modeling/lightgbm_gfsdata/avg_list.append(/docs/projects/capstone/Modeling/lightgbm_gfsdata/avg)

```


```python
df.to_csv(/docs/projects/capstone/Modeling/lightgbm_gfsdata/'gfs_preprocessed_new.csv', index=False)
```


```python
/docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/e/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/n/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/e/docs/projects/capstone/Modeling/lightgbm_gfsdata/n/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/c/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/u/docs/projects/capstone/Modeling/lightgbm_gfsdata/n/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/=/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/D/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/F/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/m/docs/projects/capstone/Modeling/lightgbm_gfsdata/e/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/seasonal_pollen_count.index = pollen_day_list
seasonal_pollen_count['avg'] = avg_list
```


```python
seasonal_pollen_count.plot(/docs/projects/capstone/Modeling/lightgbm_gfsdata/figsize=(10,4))
/docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/g/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/i/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/```


    
![png](/docs/projects/capstone/Modeling/lightgbm_gfsdata/gfs_preprocessing_files/gfs_preprocessing_18_0.png)
    



```python
df
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
      <th>grass_count</th>
      <th>t_mean_9am</th>
      <th>t_min_9am</th>
      <th>t_max_9am</th>
      <th>t_sd_9am</th>
      <th>t_2m_mean_9am</th>
      <th>t_2m_min_9am</th>
      <th>t_2m_max_9am</th>
      <th>t_2m_sd_9am</th>
      <th>msl_mean_9am</th>
      <th>...</th>
      <th>pwat_mean_4pm</th>
      <th>pwat_min_4pm</th>
      <th>pwat_max_4pm</th>
      <th>pwat_sd_4pm</th>
      <th>date</th>
      <th>year</th>
      <th>train</th>
      <th>month</th>
      <th>day</th>
      <th>seasonal_avg_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3014</th>
      <td>0.0</td>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.80000</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>13.875000</td>
      <td>13.000000</td>
      <td>14.900000</td>
      <td>0.880814</td>
      <td>2000-01-01</td>
      <td>2000</td>
      <td>2000-01</td>
      <td>01</td>
      <td>01-01</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3015</th>
      <td>0.0</td>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.80000</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>16.925000</td>
      <td>13.600000</td>
      <td>20.000000</td>
      <td>2.780138</td>
      <td>2000-01-02</td>
      <td>2000</td>
      <td>2000-01</td>
      <td>01</td>
      <td>01-02</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3016</th>
      <td>0.0</td>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.80000</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>22.625000</td>
      <td>18.800000</td>
      <td>24.700000</td>
      <td>2.617091</td>
      <td>2000-01-03</td>
      <td>2000</td>
      <td>2000-01</td>
      <td>01</td>
      <td>01-03</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3017</th>
      <td>0.0</td>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.80000</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>16.400000</td>
      <td>12.300000</td>
      <td>20.200000</td>
      <td>3.799123</td>
      <td>2000-01-04</td>
      <td>2000</td>
      <td>2000-01</td>
      <td>01</td>
      <td>01-04</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3018</th>
      <td>0.0</td>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.80000</td>
      <td>4.780080</td>
      <td>285.7500</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>14.550000</td>
      <td>13.400000</td>
      <td>15.100000</td>
      <td>0.776745</td>
      <td>2000-01-05</td>
      <td>2000</td>
      <td>2000-01</td>
      <td>01</td>
      <td>01-05</td>
      <td>0.000000</td>
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
      <th>10685</th>
      <td>119.0</td>
      <td>290.56924</td>
      <td>290.1858</td>
      <td>290.96576</td>
      <td>0.372835</td>
      <td>290.5599</td>
      <td>286.59177</td>
      <td>294.92386</td>
      <td>3.661287</td>
      <td>101139.20</td>
      <td>...</td>
      <td>19.927906</td>
      <td>11.900000</td>
      <td>29.617905</td>
      <td>9.217530</td>
      <td>2020-12-27</td>
      <td>2020</td>
      <td>2020-12</td>
      <td>12</td>
      <td>12-27</td>
      <td>24.047619</td>
    </tr>
    <tr>
      <th>10686</th>
      <td>52.0</td>
      <td>290.56924</td>
      <td>290.1858</td>
      <td>290.96576</td>
      <td>0.372835</td>
      <td>290.5599</td>
      <td>286.59177</td>
      <td>294.92386</td>
      <td>3.661287</td>
      <td>101139.20</td>
      <td>...</td>
      <td>15.316968</td>
      <td>14.800000</td>
      <td>15.933380</td>
      <td>0.585012</td>
      <td>2020-12-28</td>
      <td>2020</td>
      <td>2020-12</td>
      <td>12</td>
      <td>12-28</td>
      <td>31.142857</td>
    </tr>
    <tr>
      <th>10687</th>
      <td>2.0</td>
      <td>290.56924</td>
      <td>290.1858</td>
      <td>290.96576</td>
      <td>0.372835</td>
      <td>290.5599</td>
      <td>286.59177</td>
      <td>294.92386</td>
      <td>3.661287</td>
      <td>101139.20</td>
      <td>...</td>
      <td>20.322964</td>
      <td>17.733380</td>
      <td>24.500000</td>
      <td>3.043544</td>
      <td>2020-12-29</td>
      <td>2020</td>
      <td>2020-12</td>
      <td>12</td>
      <td>12-29</td>
      <td>39.761905</td>
    </tr>
    <tr>
      <th>10688</th>
      <td>9.0</td>
      <td>290.56924</td>
      <td>290.1858</td>
      <td>290.96576</td>
      <td>0.372835</td>
      <td>290.5599</td>
      <td>286.59177</td>
      <td>294.92386</td>
      <td>3.661287</td>
      <td>101139.20</td>
      <td>...</td>
      <td>23.850000</td>
      <td>21.100000</td>
      <td>26.000000</td>
      <td>2.350177</td>
      <td>2020-12-30</td>
      <td>2020</td>
      <td>2020-12</td>
      <td>12</td>
      <td>12-30</td>
      <td>14.285714</td>
    </tr>
    <tr>
      <th>10689</th>
      <td>14.0</td>
      <td>290.56924</td>
      <td>290.1858</td>
      <td>290.96576</td>
      <td>0.372835</td>
      <td>290.5599</td>
      <td>286.59177</td>
      <td>294.92386</td>
      <td>3.661287</td>
      <td>101139.20</td>
      <td>...</td>
      <td>28.945654</td>
      <td>27.019995</td>
      <td>31.662620</td>
      <td>1.950055</td>
      <td>2020-12-31</td>
      <td>2020</td>
      <td>2020-12</td>
      <td>12</td>
      <td>12-31</td>
      <td>17.571429</td>
    </tr>
  </tbody>
</table>
<p>7676 rows × 63 columns</p>
</div>


