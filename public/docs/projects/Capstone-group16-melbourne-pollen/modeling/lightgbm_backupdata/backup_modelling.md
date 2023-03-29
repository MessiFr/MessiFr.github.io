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
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_regression, f_classif, mutual_info_regression

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor, plot_importance

#metrics 
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, recall_score

import math
import warnings

warnings.simplefilter(/docs/projects/capstone/Modeling/lightgbm_backupdata/action='ignore', category=FutureWarning)
warnings.filterwarnings(/docs/projects/capstone/Modeling/lightgbm_backupdata/"ignore")
```


```python
df = pd.read_csv(/docs/projects/capstone/Modeling/lightgbm_backupdata/'backup_data_preprocessed.csv')
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
      <th>0</th>
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
      <th>1</th>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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
      <th>7659</th>
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
      <th>7660</th>
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
      <th>7661</th>
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
      <th>7662</th>
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
      <th>7663</th>
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




```python
/docs/projects/capstone/Modeling/lightgbm_backupdata/c/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/_/docs/projects/capstone/Modeling/lightgbm_backupdata/c/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/=/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/f/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/c/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/u/docs/projects/capstone/Modeling/lightgbm_backupdata/m/docs/projects/capstone/Modeling/lightgbm_backupdata/n/docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/_/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/i/docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata//docs/projects/capstone/Modeling/lightgbm_backupdata/c/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/=/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/f/docs/projects/capstone/Modeling/lightgbm_backupdata/[/docs/projects/capstone/Modeling/lightgbm_backupdata/c/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/_/docs/projects/capstone/Modeling/lightgbm_backupdata/c/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/]/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/c/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata/# avergae correlation with grass count
# remove last one since it is the correlation with itself = 1
feature_corr = corr['grass_count'][:-1]
```


```python
# sort the dictionary of correlation
sorted_corr = sorted(/docs/projects/capstone/Modeling/lightgbm_backupdata/feature_corr.items(), key=lambda item: item[1], reverse = True)

# correlation plot for top 50 feature
select_corr = sorted_corr[1:31]
plt.figure(/docs/projects/capstone/Modeling/lightgbm_backupdata/figsize=(20,5))
x = list(/docs/projects/capstone/Modeling/lightgbm_backupdata/x[0] for x in select_corr)
y = list(/docs/projects/capstone/Modeling/lightgbm_backupdata/round(x[1],4) for x in select_corr)
plt.bar(/docs/projects/capstone/Modeling/lightgbm_backupdata/x, y)
plt.xticks(/docs/projects/capstone/Modeling/lightgbm_backupdata/rotation=90)
plt.title(/docs/projects/capstone/Modeling/lightgbm_backupdata/"correlation between each type of features and grass pollen", fontsize=14)
plt.ylabel(/docs/projects/capstone/Modeling/lightgbm_backupdata/"correlation")
plt.xlabel(/docs/projects/capstone/Modeling/lightgbm_backupdata/"types of features")
for a, b in zip(/docs/projects/capstone/Modeling/lightgbm_backupdata/x, y):
    plt.text(/docs/projects/capstone/Modeling/lightgbm_backupdata/a, b, b, ha='center', va='bottom', fontsize=7)
/docs/projects/capstone/Modeling/lightgbm_backupdata/p/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/h/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/w/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata/```


    
![png](/docs/projects/capstone/Modeling/lightgbm_backupdata/backup_modelling_files/backup_modelling_5_0.png)
    


## Modelling
### Dataset
- We have 172 variables in total.</br>
- The numerical target variable is 'grass count'.
- We decide to use dataset from 2000 to 2019 as our training dataset.
- And the rest of the data are testing dataset, i.e., data of 2020 and 2021 (/docs/projects/capstone/Modeling/lightgbm_backupdata/2020-10-02 to 2021-01-31)


#### <font color=Blue> Regression Model. LightBM Regressor</font>

<b>LightGBM is a gradient boosting framework that uses tree based learning algorithm.</b>

- Splits: LightGBM offers gradient-based one-side sampling (/docs/projects/capstone/Modeling/lightgbm_backupdata/GOSS) that selects the split using all the instances with large gradients (i.e., large error) and a random sample of instances with small gradients. In order to keep the same data distribution when computing the information gain, /docs/projects/capstone/Modeling/lightgbm_backupdata/GOSS introduces a constant multiplier for the data instances with small gradients. Thus, /docs/projects/capstone/Modeling/lightgbm_backupdata/GOSS achieves a good balance between increasing speed by reducing the number of data instances and keeping the accuracy for learned decision trees. This method is not the default method for LightGBM, so it should be selected explicitly.
- Splits: LightGBM offers gradient-based one-side sampling (GOSS) that selects the split using all the instances with large gradients (/docs/projects/capstone/Modeling/lightgbm_backupdata/i.e., large error) and a random sample of instances with small gradients. In order to keep the same data distribution when computing the information gain, GOSS introduces a constant multiplier for the data instances with small gradients. Thus, GOSS achieves a good balance between increasing speed by reducing the number of data instances and keeping the accuracy for learned decision trees. This method is not the default method for LightGBM, so it should be selected explicitly.

- Leaf growth: LightGBM uses leaf-wise (/docs/projects/capstone/Modeling/lightgbm_backupdata/best-first) tree growth. It chooses to grow the leaf that minimizes the loss, allowing a growth of an imbalanced tree. Because it doesn’t grow level-wise, but leaf-wise, overfitting can happen when data is small. In these cases, it is important to control the tree depth. </br>

<i>ref: https://medium.com/riskified-technology/xgboost-lightgbm-or-catboost-which-boosting-algorithm-should-i-use-e7fda7bb36bc </i>

*** *** 


#### LightGBM Model1: use all features for prediction - Baseline
Now, we need to split the dataset for training and testing.


```python
#training dataset
train_data = df.loc[(/docs/projects/capstone/Modeling/lightgbm_backupdata/df['train'] != '2020-12') & (df['train'] != '2020-10')& (df['train'] != '2020-11')]
train_data = df.loc[(df['train'] != '2020-12') & (/docs/projects/capstone/Modeling/lightgbm_backupdata/df['train'] != '2020-10')& (df['train'] != '2020-11')]
train_data = df.loc[(df['train'] != '2020-12') & (df['train'] != '2020-10')& (/docs/projects/capstone/Modeling/lightgbm_backupdata/df['train'] != '2020-11')]

# plot
/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/i/docs/projects/capstone/Modeling/lightgbm_backupdata/n/docs/projects/capstone/Modeling/lightgbm_backupdata/_/docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/_/docs/projects/capstone/Modeling/lightgbm_backupdata/p/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/=/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/i/docs/projects/capstone/Modeling/lightgbm_backupdata/n/docs/projects/capstone/Modeling/lightgbm_backupdata/_/docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/c/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/p/docs/projects/capstone/Modeling/lightgbm_backupdata/y/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata/train_data_plot['date'] = train_data_plot['date'].apply(/docs/projects/capstone/Modeling/lightgbm_backupdata/lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))
train_data_plot.index = train_data_plot['date']

X_train = train_data.drop(/docs/projects/capstone/Modeling/lightgbm_backupdata/['grass_count', 'year', 'date', 'train'], axis = 1).reset_index(drop = True)
X_train = train_data.drop(['grass_count', 'year', 'date', 'train'], axis = 1).reset_index(/docs/projects/capstone/Modeling/lightgbm_backupdata/drop = True)
y_train = train_data['grass_count']
print(/docs/projects/capstone/Modeling/lightgbm_backupdata/X_train.shape)
print(/docs/projects/capstone/Modeling/lightgbm_backupdata/y_train.shape)
```

    (/docs/projects/capstone/Modeling/lightgbm_backupdata/7573, 171)
    (/docs/projects/capstone/Modeling/lightgbm_backupdata/7573,)



```python
# testing dataset
test_data = df.loc[(/docs/projects/capstone/Modeling/lightgbm_backupdata/df['train'] == '2020-12') | (df['train'] == '2020-10') |(df['train'] == '2020-11')]
test_data = df.loc[(df['train'] == '2020-12') | (/docs/projects/capstone/Modeling/lightgbm_backupdata/df['train'] == '2020-10') |(df['train'] == '2020-11')]
test_data = df.loc[(df['train'] == '2020-12') | (df['train'] == '2020-10') |(/docs/projects/capstone/Modeling/lightgbm_backupdata/df['train'] == '2020-11')]

# plot
/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/_/docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/_/docs/projects/capstone/Modeling/lightgbm_backupdata/p/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/=/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/_/docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/c/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/p/docs/projects/capstone/Modeling/lightgbm_backupdata/y/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata/test_data_plot['date'] = test_data_plot['date'].apply(/docs/projects/capstone/Modeling/lightgbm_backupdata/lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))
test_data_plot.index = test_data_plot['date']

X_test = test_data.drop(/docs/projects/capstone/Modeling/lightgbm_backupdata/['grass_count', 'year', 'date', 'train'], axis = 1).reset_index(drop = True)
X_test = test_data.drop(['grass_count', 'year', 'date', 'train'], axis = 1).reset_index(/docs/projects/capstone/Modeling/lightgbm_backupdata/drop = True)
y_test = test_data['grass_count']
print(/docs/projects/capstone/Modeling/lightgbm_backupdata/X_test.shape)
print(/docs/projects/capstone/Modeling/lightgbm_backupdata/y_test.shape)
```

    (/docs/projects/capstone/Modeling/lightgbm_backupdata/91, 171)
    (/docs/projects/capstone/Modeling/lightgbm_backupdata/91,)



```python
plt.figure(/docs/projects/capstone/Modeling/lightgbm_backupdata/figsize=(20,6))
/docs/projects/capstone/Modeling/lightgbm_backupdata/p/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/g/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/i/docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata/plt.plot(/docs/projects/capstone/Modeling/lightgbm_backupdata/train_data_plot['grass_count'], label='Train')
plt.plot(/docs/projects/capstone/Modeling/lightgbm_backupdata/test_data_plot['grass_count'], label='Test')
plt.legend(/docs/projects/capstone/Modeling/lightgbm_backupdata/loc='best')
plt.title(/docs/projects/capstone/Modeling/lightgbm_backupdata/'Train dataset and Test dataset', fontsize=14)
plt.xticks(/docs/projects/capstone/Modeling/lightgbm_backupdata/fontsize=12)
plt.xlabel(/docs/projects/capstone/Modeling/lightgbm_backupdata/'date')
plt.ylabel(/docs/projects/capstone/Modeling/lightgbm_backupdata/'grass pollen count')
/docs/projects/capstone/Modeling/lightgbm_backupdata/p/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/h/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/w/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata/```


    
![png](/docs/projects/capstone/Modeling/lightgbm_backupdata/backup_modelling_files/backup_modelling_10_0.png)
    


Then, we can apply LGBMRegressor on the dataset without tuning parameters as the baseline model.


```python
# baseine
/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/g/docs/projects/capstone/Modeling/lightgbm_backupdata/b/docs/projects/capstone/Modeling/lightgbm_backupdata/m/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/=/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/L/docs/projects/capstone/Modeling/lightgbm_backupdata/G/docs/projects/capstone/Modeling/lightgbm_backupdata/B/docs/projects/capstone/Modeling/lightgbm_backupdata/M/docs/projects/capstone/Modeling/lightgbm_backupdata/R/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/g/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata/lgbm.fit(/docs/projects/capstone/Modeling/lightgbm_backupdata/X_train, y_train)

# predict
y_pred = lgbm.predict(/docs/projects/capstone/Modeling/lightgbm_backupdata/X_test)
y_pred = np.where(/docs/projects/capstone/Modeling/lightgbm_backupdata/y_pred<0, 0, y_pred)
# round to integer
/docs/projects/capstone/Modeling/lightgbm_backupdata/y_pred = np.rint(/docs/projects/capstone/Modeling/lightgbm_backupdata/y_pred)

# evaluation
print(/docs/projects/capstone/Modeling/lightgbm_backupdata/'='*25 + 'Evaluation results' + '='*25)
print(/docs/projects/capstone/Modeling/lightgbm_backupdata/'The MSE of prediction is: {}'.format(mean_squared_error(y_test, y_pred)))
print(/docs/projects/capstone/Modeling/lightgbm_backupdata/'The RMSE of prediction is: {}'.format(math.sqrt(mean_squared_error(y_test, y_pred))))
print(/docs/projects/capstone/Modeling/lightgbm_backupdata/'The R2 score of prediction is: {}'.format(r2_score(y_test, y_pred)))
```

    =========================Evaluation results=========================
    The MSE of prediction is: 2350.010989010989
    The RMSE of prediction is: 48.47691191702488
    The R2 score of prediction is: 0.371468108704485



```python
result = pd.DataFrame(/docs/projects/capstone/Modeling/lightgbm_backupdata/list(y_test))
result['baseline'] = list(/docs/projects/capstone/Modeling/lightgbm_backupdata/y_pred)
result.index = list(/docs/projects/capstone/Modeling/lightgbm_backupdata/test_data['date'])
result = result.rename(/docs/projects/capstone/Modeling/lightgbm_backupdata/columns = {0: 'test'})
result.to_csv(/docs/projects/capstone/Modeling/lightgbm_backupdata/"result_baseline.csv")
```


```python
plot_importance(/docs/projects/capstone/Modeling/lightgbm_backupdata/lgbm, height=.5, max_num_features=20, figsize=(8, 6), title="Top 20 Feature importance ")
```




    <AxesSubplot:title={'center':'Top 20 Feature importance '}, xlabel='Feature importance', ylabel='Features'>




    
![png](/docs/projects/capstone/Modeling/lightgbm_backupdata/backup_modelling_files/backup_modelling_14_1.png)
    



```python
feature_type = ['av_abl_ht', 'accum_prcp', 'av_lwsfcdown', 'av_mslp', 'av_qsair_scrn', 'av_swsfcdown', 
                'av_temp_scrn', 'av_uwnd10m', 'av_vwnd10m', 'av_wndgust10m', 'dewpt_scrn', 'soil_mois', 
                'soil_temp', 'topt', 'thermal_time', 'Wind', 'precp', 'chilling_temp', 'forcing_temp']
```


```python
feature_imp_df = pd.DataFrame(/docs/projects/capstone/Modeling/lightgbm_backupdata/sorted(zip(lgbm.feature_importances_, train_data.columns)), columns=['Value','Feature'])
```


```python
def extract_df(/docs/projects/capstone/Modeling/lightgbm_backupdata/feature, df, colnames):
    '''
    extract each type of features and grass count to a single dataframe
    '''
    all_list = list(/docs/projects/capstone/Modeling/lightgbm_backupdata/filter(lambda x: feature in x, colnames))
/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/f/docs/projects/capstone/Modeling/lightgbm_backupdata/i/docs/projects/capstone/Modeling/lightgbm_backupdata/n/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/_/docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/f/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/=/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/p/docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/D/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/F/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/m/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata/    for lst in all_list:
        one_df = df[df['Feature'] == lst]
        final_df = pd.concat(/docs/projects/capstone/Modeling/lightgbm_backupdata/[final_df, one_df], axis=0)

    return final_df
```


```python
/docs/projects/capstone/Modeling/lightgbm_backupdata/c/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/n/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/m/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/=/docs/projects/capstone/Modeling/lightgbm_backupdata/ /docs/projects/capstone/Modeling/lightgbm_backupdata/f/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/u/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/_/docs/projects/capstone/Modeling/lightgbm_backupdata/i/docs/projects/capstone/Modeling/lightgbm_backupdata/m/docs/projects/capstone/Modeling/lightgbm_backupdata/p/docs/projects/capstone/Modeling/lightgbm_backupdata/_/docs/projects/capstone/Modeling/lightgbm_backupdata/d/docs/projects/capstone/Modeling/lightgbm_backupdata/f/docs/projects/capstone/Modeling/lightgbm_backupdata/[/docs/projects/capstone/Modeling/lightgbm_backupdata/'/docs/projects/capstone/Modeling/lightgbm_backupdata/F/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/a/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/u/docs/projects/capstone/Modeling/lightgbm_backupdata/r/docs/projects/capstone/Modeling/lightgbm_backupdata/e/docs/projects/capstone/Modeling/lightgbm_backupdata/'/docs/projects/capstone/Modeling/lightgbm_backupdata/]/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/_/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/i/docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata/avg_imp_dict = {}
for feature in feature_type:
    single_df = extract_df(/docs/projects/capstone/Modeling/lightgbm_backupdata/feature, feature_imp_df, colnames)
    avg_imp_dict[feature] = round(/docs/projects/capstone/Modeling/lightgbm_backupdata/single_df['Value'].mean(), 4)
 
```


```python
feature_avg_imp = pd.DataFrame.from_dict(/docs/projects/capstone/Modeling/lightgbm_backupdata/avg_imp_dict, orient="index")
feature_avg_imp = feature_avg_imp.rename(/docs/projects/capstone/Modeling/lightgbm_backupdata/columns = {0: 'Avg value'})
feature_avg_imp = feature_avg_imp.sort_values(/docs/projects/capstone/Modeling/lightgbm_backupdata/by='Avg value', ascending = False)
```


```python
plt.figure(/docs/projects/capstone/Modeling/lightgbm_backupdata/figsize=(12,4))
plt.bar(/docs/projects/capstone/Modeling/lightgbm_backupdata/feature_avg_imp.index.to_list(), feature_avg_imp['Avg value'])
plt.xticks(/docs/projects/capstone/Modeling/lightgbm_backupdata/rotation=90)
plt.title(/docs/projects/capstone/Modeling/lightgbm_backupdata/"average feature importance valuefor each type of features")
plt.ylabel(/docs/projects/capstone/Modeling/lightgbm_backupdata/"average feature importance value")
plt.xlabel(/docs/projects/capstone/Modeling/lightgbm_backupdata/"types of features")
for a, b in zip(/docs/projects/capstone/Modeling/lightgbm_backupdata/feature_avg_imp.index.to_list(), feature_avg_imp['Avg value']):
    plt.text(/docs/projects/capstone/Modeling/lightgbm_backupdata/a, b, b, ha='center', va='bottom', fontsize=8)
/docs/projects/capstone/Modeling/lightgbm_backupdata/p/docs/projects/capstone/Modeling/lightgbm_backupdata/l/docs/projects/capstone/Modeling/lightgbm_backupdata/t/docs/projects/capstone/Modeling/lightgbm_backupdata/./docs/projects/capstone/Modeling/lightgbm_backupdata/s/docs/projects/capstone/Modeling/lightgbm_backupdata/h/docs/projects/capstone/Modeling/lightgbm_backupdata/o/docs/projects/capstone/Modeling/lightgbm_backupdata/w/docs/projects/capstone/Modeling/lightgbm_backupdata/(/docs/projects/capstone/Modeling/lightgbm_backupdata/)/docs/projects/capstone/Modeling/lightgbm_backupdata/
/docs/projects/capstone/Modeling/lightgbm_backupdata/```


    
![png](/docs/projects/capstone/Modeling/lightgbm_backupdata/backup_modelling_files/backup_modelling_20_0.png)
    



```python

```
