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

# model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor, plot_importance

# metrics 
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, recall_score


import math
from prophet import Prophet
# from bayes_opt import BayesianOptimization


import warnings
warnings.simplefilter(/docs/projects/capstone/Modeling/lightgbm_gfsdata/action='ignore', category=FutureWarning)
warnings.filterwarnings(/docs/projects/capstone/Modeling/lightgbm_gfsdata/"ignore")
```


```python
df = pd.read_csv(/docs/projects/capstone/Modeling/lightgbm_gfsdata/'gfs_preprocessed_new.csv')
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
      <th>0</th>
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
      <td>1</td>
      <td>01-01</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
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
      <td>1</td>
      <td>01-02</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
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
      <td>1</td>
      <td>01-03</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
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
      <td>1</td>
      <td>01-04</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
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
      <td>1</td>
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
      <th>7671</th>
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
      <th>7672</th>
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
      <th>7673</th>
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
      <th>7674</th>
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
      <th>7675</th>
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



### Discover the correlation


```python
/docs/projects/capstone/Modeling/lightgbm_gfsdata/c/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/f/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/=/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/f/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/c/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/y/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/```


```python
corr_df = corr_df.drop(/docs/projects/capstone/Modeling/lightgbm_gfsdata/['date',  'train', 'year', 'day'], axis=1)
```


```python
/docs/projects/capstone/Modeling/lightgbm_gfsdata/c/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/c/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/=/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/c/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/f/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/c/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/u/docs/projects/capstone/Modeling/lightgbm_gfsdata/m/docs/projects/capstone/Modeling/lightgbm_gfsdata/n/docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/i/docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata//docs/projects/capstone/Modeling/lightgbm_gfsdata/c/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/=/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/c/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/f/docs/projects/capstone/Modeling/lightgbm_gfsdata/[/docs/projects/capstone/Modeling/lightgbm_gfsdata/c/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/c/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/]/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/c/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/```


```python
# sort the dictionary of correlation
sorted_corr = sorted(/docs/projects/capstone/Modeling/lightgbm_gfsdata/corr['grass_count'].items(), key=lambda item: item[1], reverse = True)

# correlation plot for top 50 feature
# select_corr = sorted_corr[1:21]
plt.figure(/docs/projects/capstone/Modeling/lightgbm_gfsdata/figsize=(25,8))
x = list(/docs/projects/capstone/Modeling/lightgbm_gfsdata/x[0] for x in sorted_corr[1:])
y = list(/docs/projects/capstone/Modeling/lightgbm_gfsdata/round(x[1],3) for x in sorted_corr[1:])
plt.bar(/docs/projects/capstone/Modeling/lightgbm_gfsdata/x, y)
plt.xticks(/docs/projects/capstone/Modeling/lightgbm_gfsdata/rotation=90)
plt.title(/docs/projects/capstone/Modeling/lightgbm_gfsdata/"correlation between each feature and grass pollen", fontsize=14)
plt.ylabel(/docs/projects/capstone/Modeling/lightgbm_gfsdata/"correlation")
plt.xlabel(/docs/projects/capstone/Modeling/lightgbm_gfsdata/"features")
for a, b in zip(/docs/projects/capstone/Modeling/lightgbm_gfsdata/x, y):
    plt.text(/docs/projects/capstone/Modeling/lightgbm_gfsdata/a, b, b, ha='center', va='bottom', fontsize=7)
/docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/h/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/w/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/```


    
![png](/docs/projects/capstone/Modeling/lightgbm_gfsdata/gfs_modelling_files/gfs_modelling_8_0.png)
    


## Modelling

### Lightgbm


```python
# training dataset
train_data = df.loc[(/docs/projects/capstone/Modeling/lightgbm_gfsdata/df['train'] != '2020-12') & (df['train'] != '2020-11')& (df['train'] != '2020-10')]
train_data = df.loc[(df['train'] != '2020-12') & (/docs/projects/capstone/Modeling/lightgbm_gfsdata/df['train'] != '2020-11')& (df['train'] != '2020-10')]
train_data = df.loc[(df['train'] != '2020-12') & (df['train'] != '2020-11')& (/docs/projects/capstone/Modeling/lightgbm_gfsdata/df['train'] != '2020-10')]

# plot
/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/i/docs/projects/capstone/Modeling/lightgbm_gfsdata/n/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/=/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/i/docs/projects/capstone/Modeling/lightgbm_gfsdata/n/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/c/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/y/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/train_data_plot['date'] = train_data_plot['date'].apply(/docs/projects/capstone/Modeling/lightgbm_gfsdata/lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))
train_data_plot.index = train_data_plot['date']


X_train = train_data.drop(/docs/projects/capstone/Modeling/lightgbm_gfsdata/['grass_count', 'year', 'date', 'train', 'day'], axis = 1).reset_index(drop = True)
X_train = train_data.drop(['grass_count', 'year', 'date', 'train', 'day'], axis = 1).reset_index(/docs/projects/capstone/Modeling/lightgbm_gfsdata/drop = True)
y_train = train_data['grass_count']
print(/docs/projects/capstone/Modeling/lightgbm_gfsdata/X_train.shape)
print(/docs/projects/capstone/Modeling/lightgbm_gfsdata/y_train.shape)
```

    (/docs/projects/capstone/Modeling/lightgbm_gfsdata/7584, 58)
    (/docs/projects/capstone/Modeling/lightgbm_gfsdata/7584,)



```python
X_train
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
      <th>t_mean_9am</th>
      <th>t_min_9am</th>
      <th>t_max_9am</th>
      <th>t_sd_9am</th>
      <th>t_2m_mean_9am</th>
      <th>t_2m_min_9am</th>
      <th>t_2m_max_9am</th>
      <th>t_2m_sd_9am</th>
      <th>msl_mean_9am</th>
      <th>msl_min_9am</th>
      <th>...</th>
      <th>v_10m_mean_4pm</th>
      <th>v_10m_min_4pm</th>
      <th>v_10m_max_4pm</th>
      <th>v_10m_sd_4pm</th>
      <th>pwat_mean_4pm</th>
      <th>pwat_min_4pm</th>
      <th>pwat_max_4pm</th>
      <th>pwat_sd_4pm</th>
      <th>month</th>
      <th>seasonal_avg_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.8000</td>
      <td>4.780080</td>
      <td>285.75000</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>101654.000</td>
      <td>...</td>
      <td>3.300000</td>
      <td>1.499999</td>
      <td>6.000000</td>
      <td>1.961293</td>
      <td>13.875</td>
      <td>13.0</td>
      <td>14.9</td>
      <td>0.880814</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.8000</td>
      <td>4.780080</td>
      <td>285.75000</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>101654.000</td>
      <td>...</td>
      <td>-2.450000</td>
      <td>-8.000000</td>
      <td>2.900000</td>
      <td>5.402777</td>
      <td>16.925</td>
      <td>13.6</td>
      <td>20.0</td>
      <td>2.780138</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.8000</td>
      <td>4.780080</td>
      <td>285.75000</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>101654.000</td>
      <td>...</td>
      <td>3.100000</td>
      <td>1.099999</td>
      <td>5.700000</td>
      <td>1.971464</td>
      <td>22.625</td>
      <td>18.8</td>
      <td>24.7</td>
      <td>2.617091</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.8000</td>
      <td>4.780080</td>
      <td>285.75000</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>101654.000</td>
      <td>...</td>
      <td>4.850000</td>
      <td>3.500000</td>
      <td>6.200000</td>
      <td>1.161895</td>
      <td>16.400</td>
      <td>12.3</td>
      <td>20.2</td>
      <td>3.799123</td>
      <td>1</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>285.87500</td>
      <td>280.1000</td>
      <td>290.8000</td>
      <td>4.780080</td>
      <td>285.75000</td>
      <td>280.40000</td>
      <td>290.30000</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>101654.000</td>
      <td>...</td>
      <td>3.975000</td>
      <td>2.800000</td>
      <td>5.600000</td>
      <td>1.286792</td>
      <td>14.550</td>
      <td>13.4</td>
      <td>15.1</td>
      <td>0.776745</td>
      <td>1</td>
      <td>0.0</td>
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
      <th>7579</th>
      <td>285.27008</td>
      <td>285.0634</td>
      <td>285.3754</td>
      <td>0.140418</td>
      <td>286.55835</td>
      <td>284.42596</td>
      <td>288.02914</td>
      <td>1.629595</td>
      <td>100903.83</td>
      <td>100535.305</td>
      <td>...</td>
      <td>3.576732</td>
      <td>1.849349</td>
      <td>6.291419</td>
      <td>1.977929</td>
      <td>11.825</td>
      <td>7.8</td>
      <td>14.5</td>
      <td>3.065262</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7580</th>
      <td>285.27008</td>
      <td>285.0634</td>
      <td>285.3754</td>
      <td>0.140418</td>
      <td>286.55835</td>
      <td>284.42596</td>
      <td>288.02914</td>
      <td>1.629595</td>
      <td>100903.83</td>
      <td>100535.305</td>
      <td>...</td>
      <td>1.996604</td>
      <td>0.418494</td>
      <td>3.904170</td>
      <td>1.756942</td>
      <td>8.675</td>
      <td>7.4</td>
      <td>9.4</td>
      <td>0.921502</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7581</th>
      <td>285.27008</td>
      <td>285.0634</td>
      <td>285.3754</td>
      <td>0.140418</td>
      <td>286.55835</td>
      <td>284.42596</td>
      <td>288.02914</td>
      <td>1.629595</td>
      <td>100903.83</td>
      <td>100535.305</td>
      <td>...</td>
      <td>0.020897</td>
      <td>-2.890449</td>
      <td>3.388625</td>
      <td>2.971636</td>
      <td>8.150</td>
      <td>7.6</td>
      <td>8.9</td>
      <td>0.544671</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7582</th>
      <td>285.27008</td>
      <td>285.0634</td>
      <td>285.3754</td>
      <td>0.140418</td>
      <td>286.55835</td>
      <td>284.42596</td>
      <td>288.02914</td>
      <td>1.629595</td>
      <td>100903.83</td>
      <td>100535.305</td>
      <td>...</td>
      <td>-6.098774</td>
      <td>-9.160632</td>
      <td>-4.398267</td>
      <td>2.248680</td>
      <td>19.575</td>
      <td>12.3</td>
      <td>23.2</td>
      <td>5.115581</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7583</th>
      <td>285.66272</td>
      <td>285.3361</td>
      <td>285.8216</td>
      <td>0.224875</td>
      <td>286.38450</td>
      <td>283.62192</td>
      <td>288.27292</td>
      <td>2.091517</td>
      <td>101010.07</td>
      <td>100646.875</td>
      <td>...</td>
      <td>-2.636031</td>
      <td>-7.947781</td>
      <td>0.049238</td>
      <td>3.654941</td>
      <td>17.250</td>
      <td>10.4</td>
      <td>22.9</td>
      <td>6.496409</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>7584 rows × 58 columns</p>
</div>




```python
# testing dataset
dev_data = df.loc[(/docs/projects/capstone/Modeling/lightgbm_gfsdata/df['train'] == '2020-12') | (df['train'] == '2020-11') | (df['train'] == '2020-10')]
dev_data = df.loc[(df['train'] == '2020-12') | (/docs/projects/capstone/Modeling/lightgbm_gfsdata/df['train'] == '2020-11') | (df['train'] == '2020-10')]
dev_data = df.loc[(df['train'] == '2020-12') | (df['train'] == '2020-11') | (/docs/projects/capstone/Modeling/lightgbm_gfsdata/df['train'] == '2020-10')]

# plot
/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/e/docs/projects/capstone/Modeling/lightgbm_gfsdata/v/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/=/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/e/docs/projects/capstone/Modeling/lightgbm_gfsdata/v/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/c/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/y/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/dev_data_plot['date'] = dev_data_plot['date'].apply(/docs/projects/capstone/Modeling/lightgbm_gfsdata/lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))
dev_data_plot.index = dev_data_plot['date']

X_dev = dev_data.drop(/docs/projects/capstone/Modeling/lightgbm_gfsdata/['grass_count', 'year', 'date', 'train', 'day'], axis = 1).reset_index(drop = True)
X_dev = dev_data.drop(['grass_count', 'year', 'date', 'train', 'day'], axis = 1).reset_index(/docs/projects/capstone/Modeling/lightgbm_gfsdata/drop = True)
y_dev = dev_data['grass_count']
print(/docs/projects/capstone/Modeling/lightgbm_gfsdata/X_dev.shape)
print(/docs/projects/capstone/Modeling/lightgbm_gfsdata/y_dev.shape)
```

    (/docs/projects/capstone/Modeling/lightgbm_gfsdata/92, 58)
    (/docs/projects/capstone/Modeling/lightgbm_gfsdata/92,)



```python
plt.figure(/docs/projects/capstone/Modeling/lightgbm_gfsdata/figsize=(20,6))
/docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/g/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/i/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/plt.plot(/docs/projects/capstone/Modeling/lightgbm_gfsdata/train_data_plot['grass_count'], label='Train')
plt.plot(/docs/projects/capstone/Modeling/lightgbm_gfsdata/dev_data_plot['grass_count'], label='Test')
plt.legend(/docs/projects/capstone/Modeling/lightgbm_gfsdata/loc='best')
plt.title(/docs/projects/capstone/Modeling/lightgbm_gfsdata/'Train dataset and Test dataset')
plt.xlabel(/docs/projects/capstone/Modeling/lightgbm_gfsdata/'date')
plt.ylabel(/docs/projects/capstone/Modeling/lightgbm_gfsdata/'grass pollen count')
/docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/h/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/w/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/```


    
![png](/docs/projects/capstone/Modeling/lightgbm_gfsdata/gfs_modelling_files/gfs_modelling_14_0.png)
    



```python
/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/g/docs/projects/capstone/Modeling/lightgbm_gfsdata/b/docs/projects/capstone/Modeling/lightgbm_gfsdata/m/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/=/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/L/docs/projects/capstone/Modeling/lightgbm_gfsdata/G/docs/projects/capstone/Modeling/lightgbm_gfsdata/B/docs/projects/capstone/Modeling/lightgbm_gfsdata/M/docs/projects/capstone/Modeling/lightgbm_gfsdata/R/docs/projects/capstone/Modeling/lightgbm_gfsdata/e/docs/projects/capstone/Modeling/lightgbm_gfsdata/g/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/e/docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/lgbm.fit(/docs/projects/capstone/Modeling/lightgbm_gfsdata/X_train, y_train)

# Evaluation
print(/docs/projects/capstone/Modeling/lightgbm_gfsdata/'The R2 score of prediction is: {}'.format(lgbm.score(X_dev, y_dev))) 
```

    The R2 score of prediction is: 0.23424839491411098



```python
# predict
y_pred = lgbm.predict(/docs/projects/capstone/Modeling/lightgbm_gfsdata/X_dev)
# check neagative and change to 0
y_pred = np.where(/docs/projects/capstone/Modeling/lightgbm_gfsdata/y_pred<0, 0, y_pred)
# round to integer
/docs/projects/capstone/Modeling/lightgbm_gfsdata/y_pred = np.rint(/docs/projects/capstone/Modeling/lightgbm_gfsdata/y_pred)

# evaluation
print(/docs/projects/capstone/Modeling/lightgbm_gfsdata/'='*25 + 'Evaluation results' + '='*25)
print(/docs/projects/capstone/Modeling/lightgbm_gfsdata/'The MSE of prediction is: {}'.format(mean_squared_error(y_dev, y_pred)))
print(/docs/projects/capstone/Modeling/lightgbm_gfsdata/'The RMSE of prediction is: {}'.format(math.sqrt(mean_squared_error(y_dev, y_pred))))
print(/docs/projects/capstone/Modeling/lightgbm_gfsdata/'The R2 score of prediction is: {}'.format(r2_score(y_dev, y_pred))) 
```

    =========================Evaluation results=========================
    The MSE of prediction is: 2853.6521739130435
    The RMSE of prediction is: 53.41958605149467
    The R2 score of prediction is: 0.23393780726101376



```python
result = pd.DataFrame(/docs/projects/capstone/Modeling/lightgbm_gfsdata/y_pred)
result['test'] = list(/docs/projects/capstone/Modeling/lightgbm_gfsdata/y_dev)
result.index = list(/docs/projects/capstone/Modeling/lightgbm_gfsdata/dev_data['date'])
result = result.rename(/docs/projects/capstone/Modeling/lightgbm_gfsdata/columns = {0: 'baseline'})
result.to_csv(/docs/projects/capstone/Modeling/lightgbm_gfsdata/"result_baseline_gfs.csv")
```


```python
result.plot(/docs/projects/capstone/Modeling/lightgbm_gfsdata/figsize=(15,4))
/docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/g/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/i/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/plt.legend(/docs/projects/capstone/Modeling/lightgbm_gfsdata/['baseline', 'test'])
/docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/h/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/w/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/```


    
![png](/docs/projects/capstone/Modeling/lightgbm_gfsdata/gfs_modelling_files/gfs_modelling_18_0.png)
    



```python
plot_importance(/docs/projects/capstone/Modeling/lightgbm_gfsdata/lgbm, height=.5, max_num_features=20, figsize=(8, 6), title="Top 20 Feature importance ")
```




    <AxesSubplot:title={'center':'Top 20 Feature importance '}, xlabel='Feature importance', ylabel='Features'>




    
![png](/docs/projects/capstone/Modeling/lightgbm_gfsdata/gfs_modelling_files/gfs_modelling_19_1.png)
    



```python
df.columns
```




    Index(['grass_count', 't_mean_9am', 't_min_9am', 't_max_9am', 't_sd_9am',
           't_2m_mean_9am', 't_2m_min_9am', 't_2m_max_9am', 't_2m_sd_9am',
           'msl_mean_9am', 'msl_min_9am', 'msl_max_9am', 'msl_sd_9am',
           'hum_atmos_mean_9am', 'hum_atmos_min_9am', 'hum_atmos_max_9am',
           'hum_atmos_sd_9am', 'u_10m_mean_9am', 'u_10m_min_9am', 'u_10m_max_9am',
           'u_10m_sd_9am', 'v_10m_mean_9am', 'v_10m_min_9am', 'v_10m_max_9am',
           'v_10m_sd_9am', 'pwat_mean_9am', 'pwat_min_9am', 'pwat_max_9am',
           'pwat_sd_9am', 't_mean_4pm', 't_min_4pm', 't_max_4pm', 't_sd_4pm',
           't_2m_mean_4pm', 't_2m_min_4pm', 't_2m_max_4pm', 't_2m_sd_4pm',
           'msl_mean_4pm', 'msl_min_4pm', 'msl_max_4pm', 'msl_sd_4pm',
           'hum_atmos_mean_4pm', 'hum_atmos_min_4pm', 'hum_atmos_max_4pm',
           'hum_atmos_sd_4pm', 'u_10m_mean_4pm', 'u_10m_min_4pm', 'u_10m_max_4pm',
           'u_10m_sd_4pm', 'v_10m_mean_4pm', 'v_10m_min_4pm', 'v_10m_max_4pm',
           'v_10m_sd_4pm', 'pwat_mean_4pm', 'pwat_min_4pm', 'pwat_max_4pm',
           'pwat_sd_4pm', 'date', 'year', 'train', 'month', 'day',
           'seasonal_avg_count'],
          dtype='object')




```python
feature_type = ['t', 't_2m', 'msl', 'hum_atmos', 'v_10m', 'u_10m', 'pwat', 'month', 'seasonal_avg_count']
```


```python
feature_imp_df = pd.DataFrame(/docs/projects/capstone/Modeling/lightgbm_gfsdata/sorted(zip(lgbm.feature_importances_, X_train.columns)), columns=['Value','Feature'])
```


```python
def extract_df(/docs/projects/capstone/Modeling/lightgbm_gfsdata/feature, df, colnames):
    '''
    extract each type of features and grass count to a single dataframe
    '''
    all_list = list(/docs/projects/capstone/Modeling/lightgbm_gfsdata/filter(lambda x: feature in x, colnames))
/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/f/docs/projects/capstone/Modeling/lightgbm_gfsdata/i/docs/projects/capstone/Modeling/lightgbm_gfsdata/n/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/f/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/=/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/D/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/F/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/m/docs/projects/capstone/Modeling/lightgbm_gfsdata/e/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/    for lst in all_list:
        one_df = df[df['Feature'] == lst]
        final_df = pd.concat(/docs/projects/capstone/Modeling/lightgbm_gfsdata/[final_df, one_df], axis=0)

    return final_df
```


```python
/docs/projects/capstone/Modeling/lightgbm_gfsdata/c/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/n/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/m/docs/projects/capstone/Modeling/lightgbm_gfsdata/e/docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/=/docs/projects/capstone/Modeling/lightgbm_gfsdata/ /docs/projects/capstone/Modeling/lightgbm_gfsdata/f/docs/projects/capstone/Modeling/lightgbm_gfsdata/e/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/u/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/e/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/i/docs/projects/capstone/Modeling/lightgbm_gfsdata/m/docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/d/docs/projects/capstone/Modeling/lightgbm_gfsdata/f/docs/projects/capstone/Modeling/lightgbm_gfsdata/[/docs/projects/capstone/Modeling/lightgbm_gfsdata/'/docs/projects/capstone/Modeling/lightgbm_gfsdata/F/docs/projects/capstone/Modeling/lightgbm_gfsdata/e/docs/projects/capstone/Modeling/lightgbm_gfsdata/a/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/u/docs/projects/capstone/Modeling/lightgbm_gfsdata/r/docs/projects/capstone/Modeling/lightgbm_gfsdata/e/docs/projects/capstone/Modeling/lightgbm_gfsdata/'/docs/projects/capstone/Modeling/lightgbm_gfsdata/]/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/_/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/i/docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/avg_imp_dict = {}
for feature in feature_type:
    single_df = extract_df(/docs/projects/capstone/Modeling/lightgbm_gfsdata/feature, feature_imp_df, colnames)
    avg_imp_dict[feature] = round(/docs/projects/capstone/Modeling/lightgbm_gfsdata/single_df['Value'].mean(), 4)
```


```python
feature_avg_imp = pd.DataFrame.from_dict(/docs/projects/capstone/Modeling/lightgbm_gfsdata/avg_imp_dict, orient="index")
feature_avg_imp = feature_avg_imp.rename(/docs/projects/capstone/Modeling/lightgbm_gfsdata/columns = {0: 'Avg value'})
feature_avg_imp = feature_avg_imp.sort_values(/docs/projects/capstone/Modeling/lightgbm_gfsdata/by='Avg value', ascending = False)
```


```python
plt.figure(/docs/projects/capstone/Modeling/lightgbm_gfsdata/figsize=(6,4))
plt.bar(/docs/projects/capstone/Modeling/lightgbm_gfsdata/feature_avg_imp.index.to_list(), feature_avg_imp['Avg value'])
plt.xticks(/docs/projects/capstone/Modeling/lightgbm_gfsdata/rotation=90)
plt.title(/docs/projects/capstone/Modeling/lightgbm_gfsdata/"average feature importance value for each type of features")
plt.ylabel(/docs/projects/capstone/Modeling/lightgbm_gfsdata/"average feature importance value")
plt.xlabel(/docs/projects/capstone/Modeling/lightgbm_gfsdata/"types of features")
for a, b in zip(/docs/projects/capstone/Modeling/lightgbm_gfsdata/feature_avg_imp.index.to_list(), feature_avg_imp['Avg value']):
    plt.text(/docs/projects/capstone/Modeling/lightgbm_gfsdata/a, b, b, ha='center', va='bottom', fontsize=8)
/docs/projects/capstone/Modeling/lightgbm_gfsdata/p/docs/projects/capstone/Modeling/lightgbm_gfsdata/l/docs/projects/capstone/Modeling/lightgbm_gfsdata/t/docs/projects/capstone/Modeling/lightgbm_gfsdata/./docs/projects/capstone/Modeling/lightgbm_gfsdata/s/docs/projects/capstone/Modeling/lightgbm_gfsdata/h/docs/projects/capstone/Modeling/lightgbm_gfsdata/o/docs/projects/capstone/Modeling/lightgbm_gfsdata/w/docs/projects/capstone/Modeling/lightgbm_gfsdata/(/docs/projects/capstone/Modeling/lightgbm_gfsdata/)/docs/projects/capstone/Modeling/lightgbm_gfsdata/
/docs/projects/capstone/Modeling/lightgbm_gfsdata/```


    
![png](/docs/projects/capstone/Modeling/lightgbm_gfsdata/gfs_modelling_files/gfs_modelling_26_0.png)
    



```python

```
