```python
import pandas as pd
import numpy as np
import prophet
import matplotlib.pyplot as plt
```


```python
data = pd.read_csv(/docs/projects/capstone/Modeling/naive_forecasting/'../preprocessing/melbourne_data.csv')
data = data.loc[(/docs/projects/capstone/Modeling/naive_forecasting/data['month'] != 1) & (data['month'] != 9)]
data = data.loc[(data['month'] != 1) & (/docs/projects/capstone/Modeling/naive_forecasting/data['month'] != 9)]
data = data.groupby(/docs/projects/capstone/Modeling/naive_forecasting/by=["date_1"]).mean()
/docs/projects/capstone/Modeling/naive_forecasting/d/docs/projects/capstone/Modeling/naive_forecasting/a/docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/a/docs/projects/capstone/Modeling/naive_forecasting/ /docs/projects/capstone/Modeling/naive_forecasting/=/docs/projects/capstone/Modeling/naive_forecasting/ /docs/projects/capstone/Modeling/naive_forecasting/d/docs/projects/capstone/Modeling/naive_forecasting/a/docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/a/docs/projects/capstone/Modeling/naive_forecasting/./docs/projects/capstone/Modeling/naive_forecasting/g/docs/projects/capstone/Modeling/naive_forecasting/r/docs/projects/capstone/Modeling/naive_forecasting/o/docs/projects/capstone/Modeling/naive_forecasting/u/docs/projects/capstone/Modeling/naive_forecasting/p/docs/projects/capstone/Modeling/naive_forecasting/b/docs/projects/capstone/Modeling/naive_forecasting/y/docs/projects/capstone/Modeling/naive_forecasting/(/docs/projects/capstone/Modeling/naive_forecasting/b/docs/projects/capstone/Modeling/naive_forecasting/y/docs/projects/capstone/Modeling/naive_forecasting/=/docs/projects/capstone/Modeling/naive_forecasting/[/docs/projects/capstone/Modeling/naive_forecasting/"/docs/projects/capstone/Modeling/naive_forecasting/d/docs/projects/capstone/Modeling/naive_forecasting/a/docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/e/docs/projects/capstone/Modeling/naive_forecasting/_/docs/projects/capstone/Modeling/naive_forecasting/1/docs/projects/capstone/Modeling/naive_forecasting/"/docs/projects/capstone/Modeling/naive_forecasting/]/docs/projects/capstone/Modeling/naive_forecasting/)/docs/projects/capstone/Modeling/naive_forecasting/./docs/projects/capstone/Modeling/naive_forecasting/m/docs/projects/capstone/Modeling/naive_forecasting/e/docs/projects/capstone/Modeling/naive_forecasting/a/docs/projects/capstone/Modeling/naive_forecasting/n/docs/projects/capstone/Modeling/naive_forecasting/(/docs/projects/capstone/Modeling/naive_forecasting/)/docs/projects/capstone/Modeling/naive_forecasting/
/docs/projects/capstone/Modeling/naive_forecasting/data = data[['Total']]
/docs/projects/capstone/Modeling/naive_forecasting/d/docs/projects/capstone/Modeling/naive_forecasting/a/docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/a/docs/projects/capstone/Modeling/naive_forecasting/./docs/projects/capstone/Modeling/naive_forecasting/h/docs/projects/capstone/Modeling/naive_forecasting/e/docs/projects/capstone/Modeling/naive_forecasting/a/docs/projects/capstone/Modeling/naive_forecasting/d/docs/projects/capstone/Modeling/naive_forecasting/(/docs/projects/capstone/Modeling/naive_forecasting/)/docs/projects/capstone/Modeling/naive_forecasting/
/docs/projects/capstone/Modeling/naive_forecasting/```




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
      <th>Total</th>
    </tr>
    <tr>
      <th>date_1</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10-01</th>
      <td>174.400000</td>
    </tr>
    <tr>
      <th>10-02</th>
      <td>219.333333</td>
    </tr>
    <tr>
      <th>10-03</th>
      <td>306.523810</td>
    </tr>
    <tr>
      <th>10-04</th>
      <td>249.500000</td>
    </tr>
    <tr>
      <th>10-05</th>
      <td>241.285714</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.plot(/docs/projects/capstone/Modeling/naive_forecasting/figsize=(20, 4))
/docs/projects/capstone/Modeling/naive_forecasting/p/docs/projects/capstone/Modeling/naive_forecasting/l/docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/./docs/projects/capstone/Modeling/naive_forecasting/g/docs/projects/capstone/Modeling/naive_forecasting/r/docs/projects/capstone/Modeling/naive_forecasting/i/docs/projects/capstone/Modeling/naive_forecasting/d/docs/projects/capstone/Modeling/naive_forecasting/(/docs/projects/capstone/Modeling/naive_forecasting/)/docs/projects/capstone/Modeling/naive_forecasting/
/docs/projects/capstone/Modeling/naive_forecasting/plt.legend(/docs/projects/capstone/Modeling/naive_forecasting/loc='best')
plt.title(/docs/projects/capstone/Modeling/naive_forecasting/'Melbourne Grass Count')
plt.show(/docs/projects/capstone/Modeling/naive_forecasting/block=False)
```


    
![png](/docs/projects/capstone/Modeling/naive_forecasting/naive_forecasting_files/naive_forecasting_2_0.png)
    



```python
import seaborn as sns
fig = plt.subplots(/docs/projects/capstone/Modeling/naive_forecasting/figsize=(20, 5))
ax = sns.boxplot(/docs/projects/capstone/Modeling/naive_forecasting/x=data['Total'],whis=1.5)
```


    
![png](/docs/projects/capstone/Modeling/naive_forecasting/naive_forecasting_files/naive_forecasting_3_0.png)
    



```python
fig = data.Total.hist(/docs/projects/capstone/Modeling/naive_forecasting/figsize = (20,5))
```


    
![png](/docs/projects/capstone/Modeling/naive_forecasting/naive_forecasting_files/naive_forecasting_4_0.png)
    



```python
from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 20,24
decomposition = sm.tsa.seasonal_decompose(/docs/projects/capstone/Modeling/naive_forecasting/data.Total, model='additive', period=30) # additive seasonal index
/docs/projects/capstone/Modeling/naive_forecasting/f/docs/projects/capstone/Modeling/naive_forecasting/i/docs/projects/capstone/Modeling/naive_forecasting/g/docs/projects/capstone/Modeling/naive_forecasting/ /docs/projects/capstone/Modeling/naive_forecasting/=/docs/projects/capstone/Modeling/naive_forecasting/ /docs/projects/capstone/Modeling/naive_forecasting/d/docs/projects/capstone/Modeling/naive_forecasting/e/docs/projects/capstone/Modeling/naive_forecasting/c/docs/projects/capstone/Modeling/naive_forecasting/o/docs/projects/capstone/Modeling/naive_forecasting/m/docs/projects/capstone/Modeling/naive_forecasting/p/docs/projects/capstone/Modeling/naive_forecasting/o/docs/projects/capstone/Modeling/naive_forecasting/s/docs/projects/capstone/Modeling/naive_forecasting/i/docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/i/docs/projects/capstone/Modeling/naive_forecasting/o/docs/projects/capstone/Modeling/naive_forecasting/n/docs/projects/capstone/Modeling/naive_forecasting/./docs/projects/capstone/Modeling/naive_forecasting/p/docs/projects/capstone/Modeling/naive_forecasting/l/docs/projects/capstone/Modeling/naive_forecasting/o/docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/(/docs/projects/capstone/Modeling/naive_forecasting/)/docs/projects/capstone/Modeling/naive_forecasting/
/docs/projects/capstone/Modeling/naive_forecasting//docs/projects/capstone/Modeling/naive_forecasting/p/docs/projects/capstone/Modeling/naive_forecasting/l/docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/./docs/projects/capstone/Modeling/naive_forecasting/s/docs/projects/capstone/Modeling/naive_forecasting/h/docs/projects/capstone/Modeling/naive_forecasting/o/docs/projects/capstone/Modeling/naive_forecasting/w/docs/projects/capstone/Modeling/naive_forecasting/(/docs/projects/capstone/Modeling/naive_forecasting/)/docs/projects/capstone/Modeling/naive_forecasting/
/docs/projects/capstone/Modeling/naive_forecasting/```


    
![png](/docs/projects/capstone/Modeling/naive_forecasting/naive_forecasting_files/naive_forecasting_5_0.png)
    



```python
decomposition = sm.tsa.seasonal_decompose(/docs/projects/capstone/Modeling/naive_forecasting/data.Total, model='multiplicative', period=30) # multiplicative seasonal index
/docs/projects/capstone/Modeling/naive_forecasting/f/docs/projects/capstone/Modeling/naive_forecasting/i/docs/projects/capstone/Modeling/naive_forecasting/g/docs/projects/capstone/Modeling/naive_forecasting/ /docs/projects/capstone/Modeling/naive_forecasting/=/docs/projects/capstone/Modeling/naive_forecasting/ /docs/projects/capstone/Modeling/naive_forecasting/d/docs/projects/capstone/Modeling/naive_forecasting/e/docs/projects/capstone/Modeling/naive_forecasting/c/docs/projects/capstone/Modeling/naive_forecasting/o/docs/projects/capstone/Modeling/naive_forecasting/m/docs/projects/capstone/Modeling/naive_forecasting/p/docs/projects/capstone/Modeling/naive_forecasting/o/docs/projects/capstone/Modeling/naive_forecasting/s/docs/projects/capstone/Modeling/naive_forecasting/i/docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/i/docs/projects/capstone/Modeling/naive_forecasting/o/docs/projects/capstone/Modeling/naive_forecasting/n/docs/projects/capstone/Modeling/naive_forecasting/./docs/projects/capstone/Modeling/naive_forecasting/p/docs/projects/capstone/Modeling/naive_forecasting/l/docs/projects/capstone/Modeling/naive_forecasting/o/docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/(/docs/projects/capstone/Modeling/naive_forecasting/)/docs/projects/capstone/Modeling/naive_forecasting/
/docs/projects/capstone/Modeling/naive_forecasting//docs/projects/capstone/Modeling/naive_forecasting/p/docs/projects/capstone/Modeling/naive_forecasting/l/docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/./docs/projects/capstone/Modeling/naive_forecasting/s/docs/projects/capstone/Modeling/naive_forecasting/h/docs/projects/capstone/Modeling/naive_forecasting/o/docs/projects/capstone/Modeling/naive_forecasting/w/docs/projects/capstone/Modeling/naive_forecasting/(/docs/projects/capstone/Modeling/naive_forecasting/)/docs/projects/capstone/Modeling/naive_forecasting/
/docs/projects/capstone/Modeling/naive_forecasting/```


    
![png](/docs/projects/capstone/Modeling/naive_forecasting/naive_forecasting_files/naive_forecasting_6_0.png)
    



```python
train_len = 60
train = data[0:train_len] # first 120 months as training set
test = data[train_len:] # last 24 months as out-of-time test set
```


```python
/docs/projects/capstone/Modeling/naive_forecasting/y/docs/projects/capstone/Modeling/naive_forecasting/_/docs/projects/capstone/Modeling/naive_forecasting/h/docs/projects/capstone/Modeling/naive_forecasting/a/docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/_/docs/projects/capstone/Modeling/naive_forecasting/n/docs/projects/capstone/Modeling/naive_forecasting/a/docs/projects/capstone/Modeling/naive_forecasting/i/docs/projects/capstone/Modeling/naive_forecasting/v/docs/projects/capstone/Modeling/naive_forecasting/e/docs/projects/capstone/Modeling/naive_forecasting/ /docs/projects/capstone/Modeling/naive_forecasting/=/docs/projects/capstone/Modeling/naive_forecasting/ /docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/e/docs/projects/capstone/Modeling/naive_forecasting/s/docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/./docs/projects/capstone/Modeling/naive_forecasting/c/docs/projects/capstone/Modeling/naive_forecasting/o/docs/projects/capstone/Modeling/naive_forecasting/p/docs/projects/capstone/Modeling/naive_forecasting/y/docs/projects/capstone/Modeling/naive_forecasting/(/docs/projects/capstone/Modeling/naive_forecasting/)/docs/projects/capstone/Modeling/naive_forecasting/
/docs/projects/capstone/Modeling/naive_forecasting/y_hat_naive['naive_forecast'] = train['Total'][train_len-1]
```


```python
plt.figure(/docs/projects/capstone/Modeling/naive_forecasting/figsize=(20,5))
/docs/projects/capstone/Modeling/naive_forecasting/p/docs/projects/capstone/Modeling/naive_forecasting/l/docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/./docs/projects/capstone/Modeling/naive_forecasting/g/docs/projects/capstone/Modeling/naive_forecasting/r/docs/projects/capstone/Modeling/naive_forecasting/i/docs/projects/capstone/Modeling/naive_forecasting/d/docs/projects/capstone/Modeling/naive_forecasting/(/docs/projects/capstone/Modeling/naive_forecasting/)/docs/projects/capstone/Modeling/naive_forecasting/
/docs/projects/capstone/Modeling/naive_forecasting/plt.plot(/docs/projects/capstone/Modeling/naive_forecasting/train['Total'], label='Train')
plt.plot(/docs/projects/capstone/Modeling/naive_forecasting/test['Total'], label='Test')
plt.plot(/docs/projects/capstone/Modeling/naive_forecasting/y_hat_naive['naive_forecast'], label='Naive forecast')
plt.legend(/docs/projects/capstone/Modeling/naive_forecasting/loc='best')
plt.title(/docs/projects/capstone/Modeling/naive_forecasting/'Naive Method')
/docs/projects/capstone/Modeling/naive_forecasting/p/docs/projects/capstone/Modeling/naive_forecasting/l/docs/projects/capstone/Modeling/naive_forecasting/t/docs/projects/capstone/Modeling/naive_forecasting/./docs/projects/capstone/Modeling/naive_forecasting/s/docs/projects/capstone/Modeling/naive_forecasting/h/docs/projects/capstone/Modeling/naive_forecasting/o/docs/projects/capstone/Modeling/naive_forecasting/w/docs/projects/capstone/Modeling/naive_forecasting/(/docs/projects/capstone/Modeling/naive_forecasting/)/docs/projects/capstone/Modeling/naive_forecasting/
/docs/projects/capstone/Modeling/naive_forecasting/```


    
![png](/docs/projects/capstone/Modeling/naive_forecasting/naive_forecasting_files/naive_forecasting_9_0.png)
    



```python
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(/docs/projects/capstone/Modeling/naive_forecasting/mean_squared_error(test['Total'], y_hat_naive['naive_forecast'])).round(2)
rmse = np.sqrt(mean_squared_error(test['Total'], y_hat_naive['naive_forecast'])).round(/docs/projects/capstone/Modeling/naive_forecasting/2)
mape = np.round(/docs/projects/capstone/Modeling/naive_forecasting/np.mean(np.abs(test['Total']-y_hat_naive['naive_forecast'])/test['Total'])*100,2)

results = pd.DataFrame(/docs/projects/capstone/Modeling/naive_forecasting/{'Method':['Naive method'], 'MAPE': [mape], 'RMSE': [rmse]})
results = results[['Method', 'RMSE', 'MAPE']]
results
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
      <th>Method</th>
      <th>RMSE</th>
      <th>MAPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Naive method</td>
      <td>63.79</td>
      <td>65.49</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
