```python
import pandas as pd
import numpy as np
import prophet
import matplotlib.pyplot as plt
```


```python
data = pd.read_csv('../preprocessing/melbourne_data.csv')
data = data.loc[(data['month'] != 1) & (data['month'] != 9)]
data = data.groupby(by=["date_1"]).mean()
data = data[['Total']]
data.head()
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
data.plot(figsize=(20, 4))
plt.grid()
plt.legend(loc='best')
plt.title('Melbourne Grass Count')
plt.show(block=False)
```


    
![png](/docs/projects/capstone/Modeling/naive_forecasting/naive_forecasting_files/naive_forecasting_2_0.png)
    



```python
import seaborn as sns
fig = plt.subplots(figsize=(20, 5))
ax = sns.boxplot(x=data['Total'],whis=1.5)
```


    
![png](/docs/projects/capstone/Modeling/naive_forecasting/naive_forecasting_files/naive_forecasting_3_0.png)
    



```python
fig = data.Total.hist(figsize = (20,5))
```


    
![png](/docs/projects/capstone/Modeling/naive_forecasting/naive_forecasting_files/naive_forecasting_4_0.png)
    



```python
from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 20,24
decomposition = sm.tsa.seasonal_decompose(data.Total, model='additive', period=30) # additive seasonal index
fig = decomposition.plot()
plt.show()
```


    
![png](/docs/projects/capstone/Modeling/naive_forecasting/naive_forecasting_files/naive_forecasting_5_0.png)
    



```python
decomposition = sm.tsa.seasonal_decompose(data.Total, model='multiplicative', period=30) # multiplicative seasonal index
fig = decomposition.plot()
plt.show()
```


    
![png](/docs/projects/capstone/Modeling/naive_forecasting/naive_forecasting_files/naive_forecasting_6_0.png)
    



```python
train_len = 60
train = data[0:train_len] # first 120 months as training set
test = data[train_len:] # last 24 months as out-of-time test set
```


```python
y_hat_naive = test.copy()
y_hat_naive['naive_forecast'] = train['Total'][train_len-1]
```


```python
plt.figure(figsize=(20,5))
plt.grid()
plt.plot(train['Total'], label='Train')
plt.plot(test['Total'], label='Test')
plt.plot(y_hat_naive['naive_forecast'], label='Naive forecast')
plt.legend(loc='best')
plt.title('Naive Method')
plt.show()
```


    
![png](/docs/projects/capstone/Modeling/naive_forecasting/naive_forecasting_files/naive_forecasting_9_0.png)
    



```python
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(test['Total'], y_hat_naive['naive_forecast'])).round(2)
mape = np.round(np.mean(np.abs(test['Total']-y_hat_naive['naive_forecast'])/test['Total'])*100,2)

results = pd.DataFrame({'Method':['Naive method'], 'MAPE': [mape], 'RMSE': [rmse]})
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
