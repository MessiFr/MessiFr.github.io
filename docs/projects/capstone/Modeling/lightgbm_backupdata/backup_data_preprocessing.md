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

# model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor

#metrics 
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, recall_score


import math


import warnings
warnings.simplefilter(/docs/projects/capstone/Modeling/lightgbm_backupdata/action='ignore', category=FutureWarning)
warnings.filterwarnings(/docs/projects/capstone/Modeling/lightgbm_backupdata/"ignore")
```

## Preprocessing Steps
First we load the backup datasets: weather dataset and the grass count dataset:


```python
weather_dict = pickle.load(/docs/projects/capstone/Modeling/lightgbm_backupdata/open('../weather_v2.pkl', "rb"))
grass_df = pd.read_csv(/docs/projects/capstone/Modeling/lightgbm_backupdata/'../preprocessing/melbourne_grass_preprocessed.csv')
```


```python
weather_dict[1]
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
      <th>av_abl_ht</th>
      <th>accum_prcp</th>
      <th>av_lwsfcdown</th>
      <th>av_mslp</th>
      <th>av_qsair_scrn</th>
      <th>av_swsfcdown</th>
      <th>av_temp_scrn</th>
      <th>av_uwnd10m</th>
      <th>av_vwnd10m</th>
      <th>av_wndgust10m</th>
      <th>...</th>
      <th>thermal_time_1D</th>
      <th>thermal_time_10D</th>
      <th>thermal_time_30D</th>
      <th>thermal_time_90D</th>
      <th>thermal_time_180D</th>
      <th>soil_mois_1D</th>
      <th>soil_mois_10D</th>
      <th>soil_mois_30D</th>
      <th>soil_mois_90D</th>
      <th>soil_mois_180D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-02</th>
      <td>753.565</td>
      <td>0.000156</td>
      <td>293.490000</td>
      <td>101769.3600</td>
      <td>0.006113</td>
      <td>411.645000</td>
      <td>289.752500</td>
      <td>-0.85500</td>
      <td>4.34000</td>
      <td>8.49500</td>
      <td>...</td>
      <td>5.375000</td>
      <td>6.375000</td>
      <td>6.375000</td>
      <td>6.375000</td>
      <td>6.375000</td>
      <td>2343.648438</td>
      <td>2721.484375</td>
      <td>2721.484375</td>
      <td>2721.484375</td>
      <td>2721.484375</td>
    </tr>
    <tr>
      <th>2000-01-03</th>
      <td>882.295</td>
      <td>0.011341</td>
      <td>340.160000</td>
      <td>101239.1200</td>
      <td>0.007275</td>
      <td>333.087500</td>
      <td>294.993125</td>
      <td>0.78000</td>
      <td>-1.27000</td>
      <td>8.58000</td>
      <td>...</td>
      <td>6.000000</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>2341.230469</td>
      <td>4969.046875</td>
      <td>4969.046875</td>
      <td>4969.046875</td>
      <td>4969.046875</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>642.775</td>
      <td>0.241001</td>
      <td>354.760625</td>
      <td>101026.8000</td>
      <td>0.008047</td>
      <td>274.170000</td>
      <td>289.227500</td>
      <td>3.05000</td>
      <td>4.83500</td>
      <td>12.92000</td>
      <td>...</td>
      <td>4.125000</td>
      <td>17.375000</td>
      <td>17.375000</td>
      <td>17.375000</td>
      <td>17.375000</td>
      <td>2340.359375</td>
      <td>7215.773438</td>
      <td>7215.773438</td>
      <td>7215.773438</td>
      <td>7215.773438</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>850.585</td>
      <td>0.021159</td>
      <td>332.816250</td>
      <td>101789.0000</td>
      <td>0.006387</td>
      <td>309.916250</td>
      <td>287.412500</td>
      <td>3.65500</td>
      <td>7.55500</td>
      <td>16.02000</td>
      <td>...</td>
      <td>2.750000</td>
      <td>20.250000</td>
      <td>20.250000</td>
      <td>20.250000</td>
      <td>20.250000</td>
      <td>2339.445312</td>
      <td>9461.621094</td>
      <td>9461.621094</td>
      <td>9461.621094</td>
      <td>9461.621094</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>772.685</td>
      <td>0.006436</td>
      <td>339.945000</td>
      <td>101948.0000</td>
      <td>0.007227</td>
      <td>344.033125</td>
      <td>289.284375</td>
      <td>0.78500</td>
      <td>5.96500</td>
      <td>10.74500</td>
      <td>...</td>
      <td>2.500000</td>
      <td>23.000000</td>
      <td>23.000000</td>
      <td>23.000000</td>
      <td>23.000000</td>
      <td>2338.539062</td>
      <td>11706.601562</td>
      <td>11706.601562</td>
      <td>11706.601562</td>
      <td>11706.601562</td>
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
      <th>2021-11-15</th>
      <td>1553.725</td>
      <td>0.221102</td>
      <td>330.088350</td>
      <td>100551.4900</td>
      <td>0.004862</td>
      <td>205.552563</td>
      <td>284.695424</td>
      <td>6.17675</td>
      <td>-1.08775</td>
      <td>14.75175</td>
      <td>...</td>
      <td>12.093847</td>
      <td>186.943945</td>
      <td>580.543951</td>
      <td>1585.193548</td>
      <td>2362.679663</td>
      <td>1045.839682</td>
      <td>11343.954672</td>
      <td>31693.474692</td>
      <td>92312.464760</td>
      <td>379875.052086</td>
    </tr>
    <tr>
      <th>2021-11-16</th>
      <td>1083.965</td>
      <td>1.337187</td>
      <td>318.977125</td>
      <td>101237.7225</td>
      <td>0.005006</td>
      <td>190.507200</td>
      <td>283.395150</td>
      <td>4.48575</td>
      <td>0.85275</td>
      <td>10.67300</td>
      <td>...</td>
      <td>10.062468</td>
      <td>173.893874</td>
      <td>561.493819</td>
      <td>1573.693480</td>
      <td>2362.054488</td>
      <td>1025.873758</td>
      <td>11336.536873</td>
      <td>31776.326890</td>
      <td>92366.421961</td>
      <td>373246.079276</td>
    </tr>
    <tr>
      <th>2021-11-17</th>
      <td>1128.380</td>
      <td>1.268011</td>
      <td>320.695600</td>
      <td>101640.2200</td>
      <td>0.005241</td>
      <td>250.670000</td>
      <td>286.692800</td>
      <td>1.83000</td>
      <td>-0.44400</td>
      <td>7.31600</td>
      <td>...</td>
      <td>17.999969</td>
      <td>173.859443</td>
      <td>563.559455</td>
      <td>1581.559089</td>
      <td>2375.895118</td>
      <td>1008.494997</td>
      <td>11319.998743</td>
      <td>31802.448759</td>
      <td>92394.588830</td>
      <td>366604.576183</td>
    </tr>
    <tr>
      <th>2021-11-18</th>
      <td>979.960</td>
      <td>0.101604</td>
      <td>306.938000</td>
      <td>101763.1200</td>
      <td>0.004780</td>
      <td>233.195602</td>
      <td>285.242799</td>
      <td>3.24800</td>
      <td>2.11200</td>
      <td>10.89600</td>
      <td>...</td>
      <td>11.300018</td>
      <td>155.684588</td>
      <td>561.284564</td>
      <td>1580.934207</td>
      <td>2381.395312</td>
      <td>1012.830002</td>
      <td>11331.521254</td>
      <td>31817.711264</td>
      <td>92418.116333</td>
      <td>359958.878697</td>
    </tr>
    <tr>
      <th>2021-11-19</th>
      <td>1204.930</td>
      <td>0.001189</td>
      <td>302.354999</td>
      <td>102333.3500</td>
      <td>0.004275</td>
      <td>324.467502</td>
      <td>285.771202</td>
      <td>-0.51000</td>
      <td>3.14200</td>
      <td>8.37700</td>
      <td>...</td>
      <td>16.650009</td>
      <td>161.634463</td>
      <td>568.634433</td>
      <td>1593.984241</td>
      <td>2403.545352</td>
      <td>1017.665000</td>
      <td>11328.056264</td>
      <td>31800.556268</td>
      <td>92416.361337</td>
      <td>353290.311187</td>
    </tr>
  </tbody>
</table>
<p>7993 rows × 171 columns</p>
</div>




```python
grass_df
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
      <th>Count Date</th>
      <th>grass_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1991-10-01</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1991-10-02</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1991-10-03</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1991-10-04</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1991-10-05</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10711</th>
      <td>2021-01-27</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10712</th>
      <td>2021-01-28</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>10713</th>
      <td>2021-01-29</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>10714</th>
      <td>2021-01-30</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10715</th>
      <td>2021-01-31</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>10716 rows × 2 columns</p>
</div>




```python
grass_df['Count Date'] = grass_df['Count Date'].apply(/docs/projects/capstone/Modeling/lightgbm_backupdata/lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))
```


```python
weather_dict[1]['date'] = weather_dict[1].index
df = pd.merge(/docs/projects/capstone/Modeling/lightgbm_backupdata/grass_df, weather_dict[1], left_on='Count Date', right_on='date', how='left').drop('Count Date', axis=1)
df = pd.merge(grass_df, weather_dict[1], left_on='Count Date', right_on='date', how='left').drop(/docs/projects/capstone/Modeling/lightgbm_backupdata/'Count Date', axis=1)
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
      <th>av_abl_ht</th>
      <th>accum_prcp</th>
      <th>av_lwsfcdown</th>
      <th>av_mslp</th>
      <th>av_qsair_scrn</th>
      <th>av_swsfcdown</th>
      <th>av_temp_scrn</th>
      <th>av_uwnd10m</th>
      <th>av_vwnd10m</th>
      <th>...</th>
      <th>thermal_time_10D</th>
      <th>thermal_time_30D</th>
      <th>thermal_time_90D</th>
      <th>thermal_time_180D</th>
      <th>soil_mois_1D</th>
      <th>soil_mois_10D</th>
      <th>soil_mois_30D</th>
      <th>soil_mois_90D</th>
      <th>soil_mois_180D</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaT</td>
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
      <th>10711</th>
      <td>1.0</td>
      <td>834.791667</td>
      <td>0.075095</td>
      <td>352.086666</td>
      <td>100960.166667</td>
      <td>0.007849</td>
      <td>130.549167</td>
      <td>289.445415</td>
      <td>-0.145833</td>
      <td>4.887499</td>
      <td>...</td>
      <td>75.999756</td>
      <td>184.374908</td>
      <td>184.374908</td>
      <td>787.493103</td>
      <td>7547.602417</td>
      <td>76221.471649</td>
      <td>229415.330170</td>
      <td>692386.868195</td>
      <td>961095.985832</td>
      <td>2021-01-27</td>
    </tr>
    <tr>
      <th>10712</th>
      <td>7.0</td>
      <td>963.125000</td>
      <td>0.000219</td>
      <td>335.512920</td>
      <td>101483.541667</td>
      <td>0.007803</td>
      <td>334.555417</td>
      <td>293.186250</td>
      <td>-2.487500</td>
      <td>2.991666</td>
      <td>...</td>
      <td>76.049896</td>
      <td>186.924896</td>
      <td>186.924896</td>
      <td>789.674408</td>
      <td>7535.679962</td>
      <td>76149.324097</td>
      <td>229251.895264</td>
      <td>692180.215698</td>
      <td>967718.756729</td>
      <td>2021-01-28</td>
    </tr>
    <tr>
      <th>10713</th>
      <td>2.0</td>
      <td>896.600000</td>
      <td>0.484703</td>
      <td>395.593999</td>
      <td>101408.280000</td>
      <td>0.009936</td>
      <td>132.476798</td>
      <td>295.356401</td>
      <td>-2.292000</td>
      <td>-0.148000</td>
      <td>...</td>
      <td>74.274933</td>
      <td>185.274902</td>
      <td>185.274902</td>
      <td>790.161865</td>
      <td>7836.774933</td>
      <td>76074.451599</td>
      <td>229086.352783</td>
      <td>691967.418121</td>
      <td>974334.436188</td>
      <td>2021-01-29</td>
    </tr>
    <tr>
      <th>10714</th>
      <td>0.0</td>
      <td>632.791667</td>
      <td>0.407535</td>
      <td>395.608746</td>
      <td>100911.333333</td>
      <td>0.011993</td>
      <td>86.142084</td>
      <td>292.612085</td>
      <td>0.550000</td>
      <td>-1.445833</td>
      <td>...</td>
      <td>71.924927</td>
      <td>182.549927</td>
      <td>182.549927</td>
      <td>789.127441</td>
      <td>7542.737488</td>
      <td>76027.771698</td>
      <td>228949.192780</td>
      <td>691782.330627</td>
      <td>980975.300159</td>
      <td>2021-01-30</td>
    </tr>
    <tr>
      <th>10715</th>
      <td>0.0</td>
      <td>748.458333</td>
      <td>0.001671</td>
      <td>359.496667</td>
      <td>101510.916667</td>
      <td>0.008391</td>
      <td>275.843329</td>
      <td>291.180419</td>
      <td>0.341666</td>
      <td>3.920833</td>
      <td>...</td>
      <td>67.650024</td>
      <td>185.149872</td>
      <td>185.149872</td>
      <td>791.405579</td>
      <td>7537.892426</td>
      <td>76006.474243</td>
      <td>228821.920197</td>
      <td>691599.008057</td>
      <td>987616.377117</td>
      <td>2021-01-31</td>
    </tr>
  </tbody>
</table>
<p>10716 rows × 173 columns</p>
</div>




```python
# check missing values
/docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/f/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/i/docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/n/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/v/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/u/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/u/docs/projects/capstone/Modeling/lightgbm_backupdata/m/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata//docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/f/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/i/docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/n/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/v/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/u/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/u/docs/projects/capstone/Modeling/lightgbm_backupdata/m/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata/
# dual with missing values
/docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/f/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/=/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/f/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/p/docs/projects/capstone/Modeling/lightgbm_backupdata/n/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata/```


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
      <th>av_abl_ht</th>
      <th>accum_prcp</th>
      <th>av_lwsfcdown</th>
      <th>av_mslp</th>
      <th>av_qsair_scrn</th>
      <th>av_swsfcdown</th>
      <th>av_temp_scrn</th>
      <th>av_uwnd10m</th>
      <th>av_vwnd10m</th>
      <th>...</th>
      <th>thermal_time_10D</th>
      <th>thermal_time_30D</th>
      <th>thermal_time_90D</th>
      <th>thermal_time_180D</th>
      <th>soil_mois_1D</th>
      <th>soil_mois_10D</th>
      <th>soil_mois_30D</th>
      <th>soil_mois_90D</th>
      <th>soil_mois_180D</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3015</th>
      <td>0.0</td>
      <td>753.565000</td>
      <td>0.000156</td>
      <td>293.490000</td>
      <td>101769.360000</td>
      <td>0.006113</td>
      <td>411.645000</td>
      <td>289.752500</td>
      <td>-0.855000</td>
      <td>4.340000</td>
      <td>...</td>
      <td>6.375000</td>
      <td>6.375000</td>
      <td>6.375000</td>
      <td>6.375000</td>
      <td>2343.648438</td>
      <td>2721.484375</td>
      <td>2721.484375</td>
      <td>2721.484375</td>
      <td>2721.484375</td>
      <td>2000-01-02</td>
    </tr>
    <tr>
      <th>3016</th>
      <td>0.0</td>
      <td>882.295000</td>
      <td>0.011341</td>
      <td>340.160000</td>
      <td>101239.120000</td>
      <td>0.007275</td>
      <td>333.087500</td>
      <td>294.993125</td>
      <td>0.780000</td>
      <td>-1.270000</td>
      <td>...</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>2341.230469</td>
      <td>4969.046875</td>
      <td>4969.046875</td>
      <td>4969.046875</td>
      <td>4969.046875</td>
      <td>2000-01-03</td>
    </tr>
    <tr>
      <th>3017</th>
      <td>0.0</td>
      <td>642.775000</td>
      <td>0.241001</td>
      <td>354.760625</td>
      <td>101026.800000</td>
      <td>0.008047</td>
      <td>274.170000</td>
      <td>289.227500</td>
      <td>3.050000</td>
      <td>4.835000</td>
      <td>...</td>
      <td>17.375000</td>
      <td>17.375000</td>
      <td>17.375000</td>
      <td>17.375000</td>
      <td>2340.359375</td>
      <td>7215.773438</td>
      <td>7215.773438</td>
      <td>7215.773438</td>
      <td>7215.773438</td>
      <td>2000-01-04</td>
    </tr>
    <tr>
      <th>3018</th>
      <td>0.0</td>
      <td>850.585000</td>
      <td>0.021159</td>
      <td>332.816250</td>
      <td>101789.000000</td>
      <td>0.006387</td>
      <td>309.916250</td>
      <td>287.412500</td>
      <td>3.655000</td>
      <td>7.555000</td>
      <td>...</td>
      <td>20.250000</td>
      <td>20.250000</td>
      <td>20.250000</td>
      <td>20.250000</td>
      <td>2339.445312</td>
      <td>9461.621094</td>
      <td>9461.621094</td>
      <td>9461.621094</td>
      <td>9461.621094</td>
      <td>2000-01-05</td>
    </tr>
    <tr>
      <th>3019</th>
      <td>0.0</td>
      <td>772.685000</td>
      <td>0.006436</td>
      <td>339.945000</td>
      <td>101948.000000</td>
      <td>0.007227</td>
      <td>344.033125</td>
      <td>289.284375</td>
      <td>0.785000</td>
      <td>5.965000</td>
      <td>...</td>
      <td>23.000000</td>
      <td>23.000000</td>
      <td>23.000000</td>
      <td>23.000000</td>
      <td>2338.539062</td>
      <td>11706.601562</td>
      <td>11706.601562</td>
      <td>11706.601562</td>
      <td>11706.601562</td>
      <td>2000-01-06</td>
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
      <th>10711</th>
      <td>1.0</td>
      <td>834.791667</td>
      <td>0.075095</td>
      <td>352.086666</td>
      <td>100960.166667</td>
      <td>0.007849</td>
      <td>130.549167</td>
      <td>289.445415</td>
      <td>-0.145833</td>
      <td>4.887499</td>
      <td>...</td>
      <td>75.999756</td>
      <td>184.374908</td>
      <td>184.374908</td>
      <td>787.493103</td>
      <td>7547.602417</td>
      <td>76221.471649</td>
      <td>229415.330170</td>
      <td>692386.868195</td>
      <td>961095.985832</td>
      <td>2021-01-27</td>
    </tr>
    <tr>
      <th>10712</th>
      <td>7.0</td>
      <td>963.125000</td>
      <td>0.000219</td>
      <td>335.512920</td>
      <td>101483.541667</td>
      <td>0.007803</td>
      <td>334.555417</td>
      <td>293.186250</td>
      <td>-2.487500</td>
      <td>2.991666</td>
      <td>...</td>
      <td>76.049896</td>
      <td>186.924896</td>
      <td>186.924896</td>
      <td>789.674408</td>
      <td>7535.679962</td>
      <td>76149.324097</td>
      <td>229251.895264</td>
      <td>692180.215698</td>
      <td>967718.756729</td>
      <td>2021-01-28</td>
    </tr>
    <tr>
      <th>10713</th>
      <td>2.0</td>
      <td>896.600000</td>
      <td>0.484703</td>
      <td>395.593999</td>
      <td>101408.280000</td>
      <td>0.009936</td>
      <td>132.476798</td>
      <td>295.356401</td>
      <td>-2.292000</td>
      <td>-0.148000</td>
      <td>...</td>
      <td>74.274933</td>
      <td>185.274902</td>
      <td>185.274902</td>
      <td>790.161865</td>
      <td>7836.774933</td>
      <td>76074.451599</td>
      <td>229086.352783</td>
      <td>691967.418121</td>
      <td>974334.436188</td>
      <td>2021-01-29</td>
    </tr>
    <tr>
      <th>10714</th>
      <td>0.0</td>
      <td>632.791667</td>
      <td>0.407535</td>
      <td>395.608746</td>
      <td>100911.333333</td>
      <td>0.011993</td>
      <td>86.142084</td>
      <td>292.612085</td>
      <td>0.550000</td>
      <td>-1.445833</td>
      <td>...</td>
      <td>71.924927</td>
      <td>182.549927</td>
      <td>182.549927</td>
      <td>789.127441</td>
      <td>7542.737488</td>
      <td>76027.771698</td>
      <td>228949.192780</td>
      <td>691782.330627</td>
      <td>980975.300159</td>
      <td>2021-01-30</td>
    </tr>
    <tr>
      <th>10715</th>
      <td>0.0</td>
      <td>748.458333</td>
      <td>0.001671</td>
      <td>359.496667</td>
      <td>101510.916667</td>
      <td>0.008391</td>
      <td>275.843329</td>
      <td>291.180419</td>
      <td>0.341666</td>
      <td>3.920833</td>
      <td>...</td>
      <td>67.650024</td>
      <td>185.149872</td>
      <td>185.149872</td>
      <td>791.405579</td>
      <td>7537.892426</td>
      <td>76006.474243</td>
      <td>228821.920197</td>
      <td>691599.008057</td>
      <td>987616.377117</td>
      <td>2021-01-31</td>
    </tr>
  </tbody>
</table>
<p>7695 rows × 173 columns</p>
</div>




```python
# sort the whole dataframe by date (/docs/projects/capstone/Modeling/lightgbm_backupdata/in time order)
df = df.sort_values(/docs/projects/capstone/Modeling/lightgbm_backupdata/by=['date'])

# add a new column "year" to count each year's data
df['year'] = df['date'].dt.to_period(/docs/projects/capstone/Modeling/lightgbm_backupdata/"Y").astype(str)
df['year'] = df['date'].dt.to_period("Y").astype(/docs/projects/capstone/Modeling/lightgbm_backupdata/str)

# for training
df['train'] = df['date'].dt.to_period(/docs/projects/capstone/Modeling/lightgbm_backupdata/"M").astype(str)
df['train'] = df['date'].dt.to_period("M").astype(/docs/projects/capstone/Modeling/lightgbm_backupdata/str)

# add a new column "month" to count each year's data
# df['month'] = df['date'].dt.strftime(/docs/projects/capstone/Modeling/lightgbm_backupdata/'%m').astype(str)
# df['month'] = df['date'].dt./docs/projects/capstone/Modeling/lightgbm_backupdata/strftime('%m').astype(/docs/projects/capstone/Modeling/lightgbm_backupdata/str)

# show the counts of each year
df['year'].value_counts(/docs/projects/capstone/Modeling/lightgbm_backupdata/ascending = False)
```




    2012    366
    2004    366
    2016    366
    2008    366
    2000    365
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
    2020    360
    2021     31
    Name: year, dtype: int64




```python
df = df[df['year'] != '2021']
```

### 1) Plot of grass pollen count for each year
In this part, we will show the grass pollen count for each year in general.


```python
group1 = df.groupby(/docs/projects/capstone/Modeling/lightgbm_backupdata/'year')
/docs/projects/capstone/Modeling/lightgbm_backupdata/f/docs/projects/capstone/Modeling/lightgbm_backupdata/i/docs/projects/capstone/Modeling/lightgbm_backupdata/g/docs/projects/capstone/Modeling/lightgbm_backupdata/1/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/=/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/g/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/F/docs/projects/capstone/Modeling/lightgbm_backupdata/i/docs/projects/capstone/Modeling/lightgbm_backupdata/g/docs/projects/capstone/Modeling/lightgbm_backupdata/u/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata/for group_name, single_df in group1:  
    # by year (/docs/projects/capstone/Modeling/lightgbm_backupdata/continues)
    fig1.add_trace(/docs/projects/capstone/Modeling/lightgbm_backupdata/go.Scatter(x=single_df['date'], y=single_df['grass_count'], name = group_name))
fig1.update_layout(/docs/projects/capstone/Modeling/lightgbm_backupdata/height=600, width=1000)

```



### 2) Distribution of grass pollen count 
In this part, we can visualize the target variable: grass pollen count. We can see most of the pollen days are low between 0 to 10.


```python
# plt.figure(/docs/projects/capstone/Modeling/lightgbm_backupdata/figsize=(8, 6), dpi=80)
# sns.distplot(df['grass_count'],
#              color='#918BC3',
#              label="grass_count",
#              hist_kws={'alpha': .7},
#              kde_kws={'linewidth': 3})
# plt.grid(/docs/projects/capstone/Modeling/lightgbm_backupdata/True)
# plt.title(/docs/projects/capstone/Modeling/lightgbm_backupdata/'Density Plot of grass count', fontsize=14)
/docs/projects/capstone/Modeling/lightgbm_backupdata/#/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/p/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/g/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/n/docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata//docs/projects/capstone/Modeling/lightgbm_backupdata/#/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/p/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/h/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/w/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata/```


```python
plt.figure(/docs/projects/capstone/Modeling/lightgbm_backupdata/figsize=(6, 4))
# kernel density estimation (/docs/projects/capstone/Modeling/lightgbm_backupdata/KDE) is the application of kernel smoothing for probability density estimation
plt.hist(/docs/projects/capstone/Modeling/lightgbm_backupdata/df['grass_count'], label="grass_count", bins=30)
plt.grid(/docs/projects/capstone/Modeling/lightgbm_backupdata/True)
plt.title(/docs/projects/capstone/Modeling/lightgbm_backupdata/'Histogram of grass count', fontsize=12)
/docs/projects/capstone/Modeling/lightgbm_backupdata/p/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/g/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/n/docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata/```




    <matplotlib.legend.Legend at 0x177c63fa0>




    
![png](/docs/projects/capstone/Modeling/lightgbm_backupdata/backup_data_preprocessing_files/backup_data_preprocessing_16_1.png)
    



```python
df.to_csv(/docs/projects/capstone/Modeling/lightgbm_backupdata/'backup_data_preprocessed.csv', index=False)
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
      <th>av_abl_ht</th>
      <th>accum_prcp</th>
      <th>av_lwsfcdown</th>
      <th>av_mslp</th>
      <th>av_qsair_scrn</th>
      <th>av_swsfcdown</th>
      <th>av_temp_scrn</th>
      <th>av_uwnd10m</th>
      <th>av_vwnd10m</th>
      <th>...</th>
      <th>thermal_time_90D</th>
      <th>thermal_time_180D</th>
      <th>soil_mois_1D</th>
      <th>soil_mois_10D</th>
      <th>soil_mois_30D</th>
      <th>soil_mois_90D</th>
      <th>soil_mois_180D</th>
      <th>date</th>
      <th>year</th>
      <th>train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3015</th>
      <td>0.0</td>
      <td>753.565000</td>
      <td>0.000156</td>
      <td>293.490000</td>
      <td>101769.360000</td>
      <td>0.006113</td>
      <td>411.645000</td>
      <td>289.752500</td>
      <td>-0.855000</td>
      <td>4.340000</td>
      <td>...</td>
      <td>6.375000</td>
      <td>6.375000</td>
      <td>2343.648438</td>
      <td>2721.484375</td>
      <td>2721.484375</td>
      <td>2721.484375</td>
      <td>2721.484375</td>
      <td>2000-01-02</td>
      <td>2000</td>
      <td>2000-01</td>
    </tr>
    <tr>
      <th>3016</th>
      <td>0.0</td>
      <td>882.295000</td>
      <td>0.011341</td>
      <td>340.160000</td>
      <td>101239.120000</td>
      <td>0.007275</td>
      <td>333.087500</td>
      <td>294.993125</td>
      <td>0.780000</td>
      <td>-1.270000</td>
      <td>...</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>2341.230469</td>
      <td>4969.046875</td>
      <td>4969.046875</td>
      <td>4969.046875</td>
      <td>4969.046875</td>
      <td>2000-01-03</td>
      <td>2000</td>
      <td>2000-01</td>
    </tr>
    <tr>
      <th>3017</th>
      <td>0.0</td>
      <td>642.775000</td>
      <td>0.241001</td>
      <td>354.760625</td>
      <td>101026.800000</td>
      <td>0.008047</td>
      <td>274.170000</td>
      <td>289.227500</td>
      <td>3.050000</td>
      <td>4.835000</td>
      <td>...</td>
      <td>17.375000</td>
      <td>17.375000</td>
      <td>2340.359375</td>
      <td>7215.773438</td>
      <td>7215.773438</td>
      <td>7215.773438</td>
      <td>7215.773438</td>
      <td>2000-01-04</td>
      <td>2000</td>
      <td>2000-01</td>
    </tr>
    <tr>
      <th>3018</th>
      <td>0.0</td>
      <td>850.585000</td>
      <td>0.021159</td>
      <td>332.816250</td>
      <td>101789.000000</td>
      <td>0.006387</td>
      <td>309.916250</td>
      <td>287.412500</td>
      <td>3.655000</td>
      <td>7.555000</td>
      <td>...</td>
      <td>20.250000</td>
      <td>20.250000</td>
      <td>2339.445312</td>
      <td>9461.621094</td>
      <td>9461.621094</td>
      <td>9461.621094</td>
      <td>9461.621094</td>
      <td>2000-01-05</td>
      <td>2000</td>
      <td>2000-01</td>
    </tr>
    <tr>
      <th>3019</th>
      <td>0.0</td>
      <td>772.685000</td>
      <td>0.006436</td>
      <td>339.945000</td>
      <td>101948.000000</td>
      <td>0.007227</td>
      <td>344.033125</td>
      <td>289.284375</td>
      <td>0.785000</td>
      <td>5.965000</td>
      <td>...</td>
      <td>23.000000</td>
      <td>23.000000</td>
      <td>2338.539062</td>
      <td>11706.601562</td>
      <td>11706.601562</td>
      <td>11706.601562</td>
      <td>11706.601562</td>
      <td>2000-01-06</td>
      <td>2000</td>
      <td>2000-01</td>
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
      <th>10680</th>
      <td>119.0</td>
      <td>1261.250000</td>
      <td>0.000679</td>
      <td>336.648331</td>
      <td>100953.083333</td>
      <td>0.006370</td>
      <td>382.160006</td>
      <td>297.887499</td>
      <td>0.433333</td>
      <td>-0.908333</td>
      <td>...</td>
      <td>160.599976</td>
      <td>677.124481</td>
      <td>7703.592438</td>
      <td>77705.041809</td>
      <td>232021.048706</td>
      <td>671973.964569</td>
      <td>754149.210911</td>
      <td>2020-12-27</td>
      <td>2020</td>
      <td>2020-12</td>
    </tr>
    <tr>
      <th>10681</th>
      <td>52.0</td>
      <td>1430.958333</td>
      <td>0.188170</td>
      <td>373.324172</td>
      <td>100712.875000</td>
      <td>0.006539</td>
      <td>132.561667</td>
      <td>292.519165</td>
      <td>1.533333</td>
      <td>1.174999</td>
      <td>...</td>
      <td>158.274963</td>
      <td>679.274445</td>
      <td>7701.629913</td>
      <td>77707.999268</td>
      <td>231986.591095</td>
      <td>679675.594482</td>
      <td>760968.803474</td>
      <td>2020-12-28</td>
      <td>2020</td>
      <td>2020-12</td>
    </tr>
    <tr>
      <th>10682</th>
      <td>2.0</td>
      <td>1519.916667</td>
      <td>0.001351</td>
      <td>339.529167</td>
      <td>101467.833333</td>
      <td>0.004866</td>
      <td>224.656248</td>
      <td>288.478748</td>
      <td>0.158333</td>
      <td>3.708333</td>
      <td>...</td>
      <td>157.250061</td>
      <td>679.508850</td>
      <td>7698.877380</td>
      <td>77387.521759</td>
      <td>231623.476013</td>
      <td>687053.616882</td>
      <td>767469.413036</td>
      <td>2020-12-29</td>
      <td>2020</td>
      <td>2020-12</td>
    </tr>
    <tr>
      <th>10683</th>
      <td>9.0</td>
      <td>901.160000</td>
      <td>0.000091</td>
      <td>311.778802</td>
      <td>101782.160000</td>
      <td>0.007105</td>
      <td>372.794401</td>
      <td>291.002400</td>
      <td>-0.416000</td>
      <td>3.292000</td>
      <td>...</td>
      <td>159.399963</td>
      <td>683.230713</td>
      <td>8008.187408</td>
      <td>77700.604279</td>
      <td>231889.883423</td>
      <td>695061.804291</td>
      <td>774604.499500</td>
      <td>2020-12-30</td>
      <td>2020</td>
      <td>2020-12</td>
    </tr>
    <tr>
      <th>10684</th>
      <td>14.0</td>
      <td>808.920000</td>
      <td>0.004603</td>
      <td>353.954401</td>
      <td>101928.720000</td>
      <td>0.008712</td>
      <td>297.748395</td>
      <td>291.171202</td>
      <td>0.072000</td>
      <td>4.372000</td>
      <td>...</td>
      <td>157.824890</td>
      <td>686.483765</td>
      <td>7999.360016</td>
      <td>77683.759460</td>
      <td>231836.275970</td>
      <td>695957.374298</td>
      <td>781405.386192</td>
      <td>2020-12-31</td>
      <td>2020</td>
      <td>2020-12</td>
    </tr>
  </tbody>
</table>
<p>7664 rows × 175 columns</p>
</div>


