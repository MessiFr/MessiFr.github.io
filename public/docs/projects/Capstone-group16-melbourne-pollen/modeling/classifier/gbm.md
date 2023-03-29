# Classification Models

In this notebook, we will perform data preprocessing and build classifiers for the pollen count data. We will also apply feature selection methods on the dataset. We will try several classification models to compare and discuss the most reasonable and appropriate model.


```python
from logging import warning
import pickle
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import Normalizer

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, recall_score, f1_score
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(/docs/projects/capstone/Modeling/classifier/'ignore')
```

## Preprocessing

At the beginning, we assign the label for each instance according to the pollun counts. And the label spliting standard is based on [Melbourne Pollen](/docs/projects/capstone/Modeling/classifier/https://www.melbournepollen.com.au/)

$$ 
\text{Label} =
    \left \{ 
    \begin{aligned} 
        1 \quad & \text{if}\quad 0  \leq \text{pollen count} < 20 \\ 
        2 \quad & \text{if}\quad 20 \leq \text{pollen count} < 50 \\ 
        3 \quad & \text{if}\quad 50 \leq \text{pollen count} < 100 \\ 
        4 \quad & \text{if}\quad         \text{pollen count} \geq 100 \\ 
        
    \end{aligned} 
    \right. 
$$


```python
# load weather data
weather_dict = pickle.load(/docs/projects/capstone/Modeling/classifier/open('../weather_v2.pkl', "rb"))


# add grass count data and assigned by labels
grass_df = pd.read_csv(/docs/projects/capstone/Modeling/classifier/'../preprocessing/clean_data.csv')

grass_df['Count Date'] = grass_df['Count Date'].apply(/docs/projects/capstone/Modeling/classifier/lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))
grass_df['non_grass'] = grass_df['Total'] - grass_df['grass_count']

label = [[100, 4], [50, 3], [20, 2], [-1, 1]]

def get_label(/docs/projects/capstone/Modeling/classifier/x):
    for i in label:
        if x >= i[0]:
            return i[1]

grass_df['label'] = grass_df['grass_count'].apply(/docs/projects/capstone/Modeling/classifier/lambda x:get_label(x))
```


```python
# data from 2008
data_2008 = pd.read_csv(/docs/projects/capstone/Modeling/classifier/"../preprocessing/data_2008.csv")
/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/2/docs/projects/capstone/Modeling/classifier/0/docs/projects/capstone/Modeling/classifier/0/docs/projects/capstone/Modeling/classifier/8/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/h/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/```




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
      <td>(/docs/projects/capstone/Modeling/classifier/145,-38)</td>
      <td>285.875</td>
      <td>280.1</td>
      <td>290.8</td>
      <td>4.78008</td>
      <td>285.75</td>
      <td>280.4</td>
      <td>290.3</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>0.499166</td>
      <td>3.300</td>
      <td>1.499999</td>
      <td>6.0</td>
      <td>1.961293</td>
      <td>13.875</td>
      <td>13.0</td>
      <td>14.9</td>
      <td>0.880814</td>
      <td>2000-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(/docs/projects/capstone/Modeling/classifier/145,-38)</td>
      <td>285.875</td>
      <td>280.1</td>
      <td>290.8</td>
      <td>4.78008</td>
      <td>285.75</td>
      <td>280.4</td>
      <td>290.3</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>1.175797</td>
      <td>-2.450</td>
      <td>-8.000000</td>
      <td>2.9</td>
      <td>5.402777</td>
      <td>16.925</td>
      <td>13.6</td>
      <td>20.0</td>
      <td>2.780138</td>
      <td>2000-01-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(/docs/projects/capstone/Modeling/classifier/145,-38)</td>
      <td>285.875</td>
      <td>280.1</td>
      <td>290.8</td>
      <td>4.78008</td>
      <td>285.75</td>
      <td>280.4</td>
      <td>290.3</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>2.538208</td>
      <td>3.100</td>
      <td>1.099999</td>
      <td>5.7</td>
      <td>1.971464</td>
      <td>22.625</td>
      <td>18.8</td>
      <td>24.7</td>
      <td>2.617091</td>
      <td>2000-01-03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(/docs/projects/capstone/Modeling/classifier/145,-38)</td>
      <td>285.875</td>
      <td>280.1</td>
      <td>290.8</td>
      <td>4.78008</td>
      <td>285.75</td>
      <td>280.4</td>
      <td>290.3</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>2.053453</td>
      <td>4.850</td>
      <td>3.500000</td>
      <td>6.2</td>
      <td>1.161895</td>
      <td>16.400</td>
      <td>12.3</td>
      <td>20.2</td>
      <td>3.799123</td>
      <td>2000-01-04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(/docs/projects/capstone/Modeling/classifier/145,-38)</td>
      <td>285.875</td>
      <td>280.1</td>
      <td>290.8</td>
      <td>4.78008</td>
      <td>285.75</td>
      <td>280.4</td>
      <td>290.3</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>...</td>
      <td>0.618467</td>
      <td>3.975</td>
      <td>2.800000</td>
      <td>5.6</td>
      <td>1.286792</td>
      <td>14.550</td>
      <td>13.4</td>
      <td>15.1</td>
      <td>0.776745</td>
      <td>2000-01-05</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>



For every city count the number of instances


```python
grass_df['Name'] = grass_df['Name'].apply(/docs/projects/capstone/Modeling/classifier/lambda x:x.split('_')[0])
/docs/projects/capstone/Modeling/classifier/g/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/[/docs/projects/capstone/Modeling/classifier/'/docs/projects/capstone/Modeling/classifier/N/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/'/docs/projects/capstone/Modeling/classifier/]/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/v/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/```




    Sydney                 4454
    Tasmania               4049
    Melbourne              2842
    NSW-Westmead           1342
    Canberra(/docs/projects/capstone/Modeling/classifier/2007-2017)    1301
    Canberra                995
    NSW-Gosford             854
    NT                      809
    Burwood                 406
    Creswick                404
    Geelong                 401
    Dookie                  374
    Churchill               365
    Hamilton                355
    Bendigo                 352
    NSW-Wagga               247
    Name: Name, dtype: int64



Filter the data in **Melbourne** and merge the pollen data and the weather data as the train dataset


```python
# weather_dict[1]
grass_columns = ['Count Date', 'Elevation', 'Total', 'Location', 'SchColTime', 'grass_count', 'label']
data = grass_df.loc[grass_df['Name']=='Melbourne'][grass_columns]
data

/docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/b/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/=/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/p/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/D/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/F/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/weather_dict[1]['date'] = weather_dict[1].index
for i in [1, 25, 28, 29]:
    tmp = pd.merge(/docs/projects/capstone/Modeling/classifier/data.loc[data['Location']==i], weather_dict[1], left_on='Count Date', right_on='date', how='left')
    melbourne_df = pd.concat(/docs/projects/capstone/Modeling/classifier/[melbourne_df, tmp])

full_data = melbourne_df.drop(/docs/projects/capstone/Modeling/classifier/['Count Date', 'SchColTime', 'date'], axis=1)
/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/h/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/```




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
      <th>Elevation</th>
      <th>Total</th>
      <th>Location</th>
      <th>grass_count</th>
      <th>label</th>
      <th>av_abl_ht</th>
      <th>accum_prcp</th>
      <th>av_lwsfcdown</th>
      <th>av_mslp</th>
      <th>av_qsair_scrn</th>
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
      <th>0</th>
      <td>28.0</td>
      <td>76.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>567.409722</td>
      <td>0.014479</td>
      <td>291.515625</td>
      <td>101588.000000</td>
      <td>0.004978</td>
      <td>...</td>
      <td>3.625</td>
      <td>35.250</td>
      <td>87.500</td>
      <td>87.500</td>
      <td>486.375</td>
      <td>1792.332031</td>
      <td>24080.046875</td>
      <td>71414.628906</td>
      <td>209709.917969</td>
      <td>415718.011719</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28.0</td>
      <td>335.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>672.985000</td>
      <td>0.044749</td>
      <td>316.624375</td>
      <td>101377.280000</td>
      <td>0.004932</td>
      <td>...</td>
      <td>4.375</td>
      <td>36.250</td>
      <td>89.250</td>
      <td>89.250</td>
      <td>487.500</td>
      <td>2489.031250</td>
      <td>24086.992188</td>
      <td>71454.339844</td>
      <td>209820.257812</td>
      <td>415804.148438</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28.0</td>
      <td>857.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1351.370000</td>
      <td>0.038105</td>
      <td>320.018125</td>
      <td>100350.040000</td>
      <td>0.005645</td>
      <td>...</td>
      <td>3.500</td>
      <td>36.500</td>
      <td>90.875</td>
      <td>90.875</td>
      <td>487.375</td>
      <td>2488.683594</td>
      <td>24090.761719</td>
      <td>71490.339844</td>
      <td>209930.417969</td>
      <td>415890.109375</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28.0</td>
      <td>235.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>896.015000</td>
      <td>0.011231</td>
      <td>318.508125</td>
      <td>101197.400000</td>
      <td>0.005693</td>
      <td>...</td>
      <td>2.750</td>
      <td>36.125</td>
      <td>91.000</td>
      <td>91.000</td>
      <td>487.000</td>
      <td>2488.316406</td>
      <td>24092.531250</td>
      <td>71522.316406</td>
      <td>210040.222656</td>
      <td>415975.867188</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28.0</td>
      <td>263.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1343.250000</td>
      <td>0.055731</td>
      <td>309.666466</td>
      <td>101231.692308</td>
      <td>0.005118</td>
      <td>...</td>
      <td>3.250</td>
      <td>36.125</td>
      <td>91.125</td>
      <td>91.125</td>
      <td>487.500</td>
      <td>2587.183594</td>
      <td>24192.238281</td>
      <td>71650.304688</td>
      <td>210248.914062</td>
      <td>416160.636719</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 176 columns</p>
</div>



Generate the full dataset with dropped nan values and time columns


```python
full_data = melbourne_df.drop(/docs/projects/capstone/Modeling/classifier/['Count Date', 'SchColTime', 'date'], axis=1).dropna()
/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/=/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/b/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/p/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/[/docs/projects/capstone/Modeling/classifier/'/docs/projects/capstone/Modeling/classifier/C/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/D/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/'/docs/projects/capstone/Modeling/classifier/,/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/'/docs/projects/capstone/Modeling/classifier/S/docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/h/docs/projects/capstone/Modeling/classifier/C/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/T/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/'/docs/projects/capstone/Modeling/classifier/,/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/'/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/'/docs/projects/capstone/Modeling/classifier/]/docs/projects/capstone/Modeling/classifier/,/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/x/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/=/docs/projects/capstone/Modeling/classifier/1/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/p/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier//docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/h/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/```




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
      <th>Elevation</th>
      <th>Total</th>
      <th>Location</th>
      <th>grass_count</th>
      <th>label</th>
      <th>av_abl_ht</th>
      <th>accum_prcp</th>
      <th>av_lwsfcdown</th>
      <th>av_mslp</th>
      <th>av_qsair_scrn</th>
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
      <th>0</th>
      <td>28.0</td>
      <td>76.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>567.409722</td>
      <td>0.014479</td>
      <td>291.515625</td>
      <td>101588.000000</td>
      <td>0.004978</td>
      <td>...</td>
      <td>3.625</td>
      <td>35.250</td>
      <td>87.500</td>
      <td>87.500</td>
      <td>486.375</td>
      <td>1792.332031</td>
      <td>24080.046875</td>
      <td>71414.628906</td>
      <td>209709.917969</td>
      <td>415718.011719</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28.0</td>
      <td>335.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>672.985000</td>
      <td>0.044749</td>
      <td>316.624375</td>
      <td>101377.280000</td>
      <td>0.004932</td>
      <td>...</td>
      <td>4.375</td>
      <td>36.250</td>
      <td>89.250</td>
      <td>89.250</td>
      <td>487.500</td>
      <td>2489.031250</td>
      <td>24086.992188</td>
      <td>71454.339844</td>
      <td>209820.257812</td>
      <td>415804.148438</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28.0</td>
      <td>857.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1351.370000</td>
      <td>0.038105</td>
      <td>320.018125</td>
      <td>100350.040000</td>
      <td>0.005645</td>
      <td>...</td>
      <td>3.500</td>
      <td>36.500</td>
      <td>90.875</td>
      <td>90.875</td>
      <td>487.375</td>
      <td>2488.683594</td>
      <td>24090.761719</td>
      <td>71490.339844</td>
      <td>209930.417969</td>
      <td>415890.109375</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28.0</td>
      <td>235.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>896.015000</td>
      <td>0.011231</td>
      <td>318.508125</td>
      <td>101197.400000</td>
      <td>0.005693</td>
      <td>...</td>
      <td>2.750</td>
      <td>36.125</td>
      <td>91.000</td>
      <td>91.000</td>
      <td>487.000</td>
      <td>2488.316406</td>
      <td>24092.531250</td>
      <td>71522.316406</td>
      <td>210040.222656</td>
      <td>415975.867188</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28.0</td>
      <td>263.0</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1343.250000</td>
      <td>0.055731</td>
      <td>309.666466</td>
      <td>101231.692308</td>
      <td>0.005118</td>
      <td>...</td>
      <td>3.250</td>
      <td>36.125</td>
      <td>91.125</td>
      <td>91.125</td>
      <td>487.500</td>
      <td>2587.183594</td>
      <td>24192.238281</td>
      <td>71650.304688</td>
      <td>210248.914062</td>
      <td>416160.636719</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 176 columns</p>
</div>



#### Randomly Re-Sample
Generate variables in dataframe as `X_origin`, choose the label as target as `Y_origin`. </br>
Since the labels in raw data doesn't distribute evenly, so we use the `RandomOverSample` method to resample the data. ***


```python
/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/b/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/v/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/```




    1.0    1159
    2.0     464
    3.0     250
    4.0     196
    Name: label, dtype: int64




```python
X_origin = full_data.drop(/docs/projects/capstone/Modeling/classifier/['label', 'Total', 'Location', 'grass_count'], axis=1).reset_index(drop=True)
X_origin = full_data.drop(['label', 'Total', 'Location', 'grass_count'], axis=1).reset_index(/docs/projects/capstone/Modeling/classifier/drop=True)
Y_origin = full_data['label']

X_origin, Y_origin = RandomOverSampler(/docs/projects/capstone/Modeling/classifier/random_state=0).fit_resample(X_origin, Y_origin)
/docs/projects/capstone/Modeling/classifier/X_origin, Y_origin = RandomOverSampler(random_state=0).fit_resample(/docs/projects/capstone/Modeling/classifier/X_origin, Y_origin)
print(/docs/projects/capstone/Modeling/classifier/Y_origin.value_counts())
```

    1.0    1159
    2.0    1159
    3.0    1159
    4.0    1159
    Name: label, dtype: int64



```python
X_origin.shape
```




    (/docs/projects/capstone/Modeling/classifier/4636, 172)



# Modelling

### Benchmark

The `sklearn.ensemble.GradientBoostingClassifier` is used as benchmark model.


```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score


X_train, X_test, Y_train, Y_test = train_test_split(/docs/projects/capstone/Modeling/classifier/X_origin, Y_origin, test_size=0.33, random_state=88)

/docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/=/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/G/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/B/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/g/docs/projects/capstone/Modeling/classifier/C/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/model.fit(/docs/projects/capstone/Modeling/classifier/X_train, Y_train)
model.score(/docs/projects/capstone/Modeling/classifier/X_test, Y_test)
```




    0.8522875816993464



### Model Evaluation


```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

Y_pred = model.predict(/docs/projects/capstone/Modeling/classifier/X_test)

# confusion_matrix(/docs/projects/capstone/Modeling/classifier/Y_test, Y_pred)
cm = confusion_matrix(/docs/projects/capstone/Modeling/classifier/Y_test, Y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/p/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/p/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/p/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/h/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/w/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/```


    
![png](/docs/projects/capstone/Modeling/classifier/gbm_files/gbm_18_0.png)
    



```python
from sklearn.metrics import classification_report

class_report = classification_report(/docs/projects/capstone/Modeling/classifier/Y_test, Y_pred)
print(/docs/projects/capstone/Modeling/classifier/class_report)
```

                  precision    recall  f1-score   support
    
             1.0       0.85      0.76      0.80       378
             2.0       0.80      0.76      0.78       407
             3.0       0.85      0.93      0.89       353
             4.0       0.90      0.97      0.94       392
    
        accuracy                           0.85      1530
       macro avg       0.85      0.85      0.85      1530
    weighted avg       0.85      0.85      0.85      1530
    



```python
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

model = OneVsRestClassifier(/docs/projects/capstone/Modeling/classifier/GradientBoostingClassifier())

y_score = model.fit(/docs/projects/capstone/Modeling/classifier/X_train, Y_train).decision_function(X_test)
y_score = model.fit(X_train, Y_train).decision_function(/docs/projects/capstone/Modeling/classifier/X_test)
```


```python
def transform(/docs/projects/capstone/Modeling/classifier/y):
    result = np.zeros(/docs/projects/capstone/Modeling/classifier/[len(y), 4])
    for i in range(/docs/projects/capstone/Modeling/classifier/len(y)):
        result[i, int(/docs/projects/capstone/Modeling/classifier/y[i]-1)] += 1
    return result

Y_test = transform(/docs/projects/capstone/Modeling/classifier/Y_test.to_numpy())
```


```python
n_classes = 4
lw = 2

/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/p/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/=/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier//docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/p/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/=/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier//docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/=/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/for i in range(/docs/projects/capstone/Modeling/classifier/n_classes):
    fpr[i], tpr[i], _ = roc_curve(/docs/projects/capstone/Modeling/classifier/Y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(/docs/projects/capstone/Modeling/classifier/fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(/docs/projects/capstone/Modeling/classifier/Y_test.ravel(), y_score.ravel())
/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/p/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/[/docs/projects/capstone/Modeling/classifier/"/docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/"/docs/projects/capstone/Modeling/classifier/]/docs/projects/capstone/Modeling/classifier/,/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/p/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/[/docs/projects/capstone/Modeling/classifier/"/docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/"/docs/projects/capstone/Modeling/classifier/]/docs/projects/capstone/Modeling/classifier/,/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/=/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/v/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/Y/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/v/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/,/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/y/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/v/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/roc_auc["micro"] = auc(/docs/projects/capstone/Modeling/classifier/fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(/docs/projects/capstone/Modeling/classifier/np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(/docs/projects/capstone/Modeling/classifier/all_fpr)
for i in range(/docs/projects/capstone/Modeling/classifier/n_classes):
    mean_tpr += np.interp(/docs/projects/capstone/Modeling/classifier/all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(/docs/projects/capstone/Modeling/classifier/fpr["macro"], tpr["macro"])

# Plot all ROC curves
/docs/projects/capstone/Modeling/classifier/p/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/g/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (/docs/projects/capstone/Modeling/classifier/area = {0:0.2f})".format(roc_auc["micro"]),
    label="micro-average ROC curve (area = {0:0.2f})".format(/docs/projects/capstone/Modeling/classifier/roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (/docs/projects/capstone/Modeling/classifier/area = {0:0.2f})".format(roc_auc["macro"]),
    label="macro-average ROC curve (area = {0:0.2f})".format(/docs/projects/capstone/Modeling/classifier/roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(/docs/projects/capstone/Modeling/classifier/["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(/docs/projects/capstone/Modeling/classifier/range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (/docs/projects/capstone/Modeling/classifier/area = {1:0.2f})".format(i, roc_auc[i]),
        label="ROC curve of class {0} (area = {1:0.2f})".format(/docs/projects/capstone/Modeling/classifier/i, roc_auc[i]),
    )

plt.plot(/docs/projects/capstone/Modeling/classifier/[0, 1], [0, 1], "k--", lw=lw)
plt.xlim(/docs/projects/capstone/Modeling/classifier/[0.0, 1.0])
plt.ylim(/docs/projects/capstone/Modeling/classifier/[0.0, 1.05])
plt.xlabel(/docs/projects/capstone/Modeling/classifier/"False Positive Rate")
plt.ylabel(/docs/projects/capstone/Modeling/classifier/"True Positive Rate")
plt.title(/docs/projects/capstone/Modeling/classifier/"ROC of Grandient Boosting Classifier")
plt.legend(/docs/projects/capstone/Modeling/classifier/loc="lower right")
/docs/projects/capstone/Modeling/classifier/p/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/h/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/w/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/```


    
![png](/docs/projects/capstone/Modeling/classifier/gbm_files/gbm_22_0.png)
    


Check the overfitting of model.


```python
from support_evaluation import check_overfitting
check_overfitting(/docs/projects/capstone/Modeling/classifier/GradientBoostingClassifier(), X_origin, Y_origin)
```

    [Parallel(/docs/projects/capstone/Modeling/classifier/n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(/docs/projects/capstone/Modeling/classifier/n_jobs=-1)]: Done   2 out of  25 | elapsed:   15.5s remaining:  3.0min
    [Parallel(/docs/projects/capstone/Modeling/classifier/n_jobs=-1)]: Done  25 out of  25 | elapsed:  5.5min finished



    
![png](/docs/projects/capstone/Modeling/classifier/gbm_files/gbm_24_1.png)
    


The plot of learning curve of `GradientBoostingClassifier` model presents that the classification accuracy of training score is always larger than cross validation score, which presents the overfitting of model.


```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import time

def grid_search(/docs/projects/capstone/Modeling/classifier/pipeline, parameters, X, y):
    gs = GridSearchCV(/docs/projects/capstone/Modeling/classifier/pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
    
    print(/docs/projects/capstone/Modeling/classifier/'Performing grid search...')
    print(/docs/projects/capstone/Modeling/classifier/'pipeline:', [name for name, _ in pipeline.steps])
    print(/docs/projects/capstone/Modeling/classifier/'parameters:')
    print(/docs/projects/capstone/Modeling/classifier/parameters)
/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/0/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/=/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/    gs.fit(/docs/projects/capstone/Modeling/classifier/X, y)
    print(/docs/projects/capstone/Modeling/classifier/'done in %0.3fs' % (time.time() - t0))
/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/p/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/    
    # print out best 5 results
    mean_score = gs.cv_results_['mean_test_score']
    param_set = gs.cv_results_['params']
/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/g/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/[/docs/projects/capstone/Modeling/classifier/-/docs/projects/capstone/Modeling/classifier/5/docs/projects/capstone/Modeling/classifier/:/docs/projects/capstone/Modeling/classifier/]/docs/projects/capstone/Modeling/classifier/:/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/        print(/docs/projects/capstone/Modeling/classifier/param_set[i])
        print(/docs/projects/capstone/Modeling/classifier/gs.cv_results_['mean_test_score'][i])
        print(/docs/projects/capstone/Modeling/classifier/'='*30)
    
    return gs
```


```python
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_regression, f_classif, mutual_info_regression


pipeline = Pipeline(
    [
        (/docs/projects/capstone/Modeling/classifier/'kbest', SelectKBest()),
        (/docs/projects/capstone/Modeling/classifier/'clf', GradientBoostingClassifier())
    ]
)

# parameters={
#     'clf__hidden_layer_sizes': [(/docs/projects/capstone/Modeling/classifier/50,50,50), (50,100,50), (100,)],
#     'clf__hidden_layer_sizes': [(50,50,50), (/docs/projects/capstone/Modeling/classifier/50,100,50), (100,)],
#     'clf__hidden_layer_sizes': [(50,50,50), (50,/docs/projects/capstone/Modeling/classifier/100,50), (/docs/projects/capstone/Modeling/classifier/100,)],
#     'clf__activation': ['tanh', 'relu'],
#     'clf__solver': ['sgd', 'adam'],
#     'clf__alpha': [0.0001, 0.05],
#     'clf__learning_rate': ['constant','adaptive'],
# }

parameters = {
    'kbest__score_func': (/docs/projects/capstone/Modeling/classifier/f_classif, mutual_info_classif),
    'kbest__k':(/docs/projects/capstone/Modeling/classifier/20, 50, 100, 150, 'all'),
    # "clf__loss":["deviance"],
    "clf__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
    # "clf__max_features":["log2","sqrt"],
    # "clf__criterion": ["friedman_mse",  "mae"],
    }

```


```python
gs = grid_search(/docs/projects/capstone/Modeling/classifier/pipeline, parameters, X_origin, Y_origin)
```

    Performing grid search...
    pipeline: ['kbest', 'clf']
    parameters:
    {'kbest__score_func': (/docs/projects/capstone/Modeling/classifier/<function f_regression at 0x13cc84c10>, <function mutual_info_regression at 0x13cc87370>), 'kbest__k': (20, 50, 100, 150, 'all'), 'clf__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]}
    {'kbest__score_func': (<function f_regression at 0x13cc84c10>, <function mutual_info_regression at 0x13cc87370>), 'kbest__k': (/docs/projects/capstone/Modeling/classifier/20, 50, 100, 150, 'all'), 'clf__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]}
    Fitting 5 folds for each of 50 candidates, totalling 250 fits
    done in 2256.210s
    
    {'clf__learning_rate': 0.2, 'kbest__k': 50, 'kbest__score_func': <function f_regression at 0x13cc84c10>}
    0.836739528698434
    ==============================
    {'clf__learning_rate': 0.2, 'kbest__k': 150, 'kbest__score_func': <function f_regression at 0x13cc84c10>}
    0.8391046386192016
    ==============================
    {'clf__learning_rate': 0.2, 'kbest__k': 150, 'kbest__score_func': <function mutual_info_regression at 0x13cc87370>}
    0.8442877469032473
    ==============================
    {'clf__learning_rate': 0.2, 'kbest__k': 20, 'kbest__score_func': <function f_regression at 0x13cc84c10>}
    0.8449126771565674
    ==============================
    {'clf__learning_rate': 0.2, 'kbest__k': 100, 'kbest__score_func': <function mutual_info_regression at 0x13cc87370>}
    0.8455815478183238
    ==============================



```python

```


```python
def print_scores(/docs/projects/capstone/Modeling/classifier/y_test, y_pred):
    print(/docs/projects/capstone/Modeling/classifier/'='*10 + 'Evaluation results' + '='*10)
    print(/docs/projects/capstone/Modeling/classifier/'The accuracy  : {}'.format(accuracy_score(y_test, y_pred)))
    print(/docs/projects/capstone/Modeling/classifier/'  The recall  : {}'.format(recall_score(y_test, y_pred, average='weighted')))
    print(/docs/projects/capstone/Modeling/classifier/'      The f1  : {}'.format(f1_score(y_test, y_pred, average='weighted')))
```


```python
classif_y_test = []
for i in Y_test:
    for j in range(/docs/projects/capstone/Modeling/classifier/4):
        if i[j] == 1:
            classif_y_test.append(/docs/projects/capstone/Modeling/classifier/j+1)
# classif_y_test
print_scores(/docs/projects/capstone/Modeling/classifier/classif_y_test, Y_pred)
```

    ==========Evaluation results==========
    The accuracy  : 0.8522875816993464
      The recall  : 0.8522875816993464
          The f1  : 0.8499602749809874



```python
regression_test_df = pd.read_csv(/docs/projects/capstone/Modeling/classifier/'../../preprocessing/result_baseline.csv')
regression_test_df['test_label'] = regression_test_df.test.apply(/docs/projects/capstone/Modeling/classifier/lambda x:get_label(x))
regression_test_df['pred_label'] = regression_test_df.tune_param.apply(/docs/projects/capstone/Modeling/classifier/lambda x:get_label(x))
/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/g/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/h/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/```




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
      <th>Unnamed: 0</th>
      <th>test</th>
      <th>baseline</th>
      <th>tune_param</th>
      <th>fea_selc</th>
      <th>test_label</th>
      <th>pred_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-10-02</td>
      <td>37.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>7.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-10-03</td>
      <td>75.0</td>
      <td>18.0</td>
      <td>15.0</td>
      <td>7.0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-10-04</td>
      <td>69.0</td>
      <td>17.0</td>
      <td>20.0</td>
      <td>9.0</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-10-05</td>
      <td>47.0</td>
      <td>14.0</td>
      <td>9.0</td>
      <td>25.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-10-06</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
regression_gfs_df = pd.read_csv(/docs/projects/capstone/Modeling/classifier/'../../preprocessing/resulr_gfs.csv')
regression_gfs_df['test_label'] = regression_gfs_df.test.apply(/docs/projects/capstone/Modeling/classifier/lambda x:get_label(x))
regression_gfs_df['pred_label'] = regression_gfs_df.tune_param.apply(/docs/projects/capstone/Modeling/classifier/lambda x:get_label(x))
/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/g/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/g/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/h/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/```




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
      <th>Unnamed: 0</th>
      <th>test</th>
      <th>baseline</th>
      <th>tune_param</th>
      <th>test_label</th>
      <th>pred_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2007-10-01</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>11.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007-10-02</td>
      <td>6.0</td>
      <td>29.0</td>
      <td>23.0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2007-10-03</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2007-10-04</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2007-10-05</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python

print(/docs/projects/capstone/Modeling/classifier/"regression for backup")
print_scores(/docs/projects/capstone/Modeling/classifier/regression_test_df.test_label, regression_test_df.pred_label)

print(/docs/projects/capstone/Modeling/classifier/"\nregression for gfs")
print_scores(/docs/projects/capstone/Modeling/classifier/regression_gfs_df.test_label, regression_gfs_df.pred_label)
```

    regression for backup
    ==========Evaluation results==========
    The accuracy  : 0.5163934426229508
      The recall  : 0.5163934426229508
          The f1  : 0.541521456685391
    
    regression for gfs
    ==========Evaluation results==========
    The accuracy  : 0.5054945054945055
      The recall  : 0.5054945054945055
          The f1  : 0.47511706852366187



```python

```


```python

```
