```python
import pickle
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(/docs/projects/capstone/feature_analysis/"ignore")

```

## Loading data
- weather_v2.pkl: source on VM
- grass_clean_mel.csv: generate from "pollen_count.ipynb"


```python
# weather dataset provided by the client
weather_dict = pickle.load(/docs/projects/capstone/feature_analysis/open('../data/weather_v2.pkl', "rb"))
# Melbourne weather dataset
mel_df = weather_dict[1]

# preprocessed Melbourne grass pollen count dataset 
grass_df = pd.read_csv(/docs/projects/capstone/feature_analysis/'grass_clean_mel.csv')
```


```python
# change data type of Melbourne grass pollen count data
grass_df['Count Date'] = grass_df['Count Date'].apply(/docs/projects/capstone/feature_analysis/lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))

# Merge Melbourne grass pollen count data and Melboure weather data
grass_df = grass_df[[/docs/projects/capstone/feature_analysis/'Count Date', 'grass_count']].set_index(/docs/projects/capstone/feature_analysis/'Count Date').join(mel_df)
grass_df = grass_df[['Count Date', 'grass_count']].set_index('Count Date').join(/docs/projects/capstone/feature_analysis/mel_df)
/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis/```


```python
# convert count to pollen level 
label = [[50, 3], [20, 2], [0, 1]]

def get_label(/docs/projects/capstone/feature_analysis/x):
    for i in label:
        if x >= i[0]:
            return i[1]

grass_df.insert(/docs/projects/capstone/feature_analysis/0, 'grass_level', grass_df['grass_count'].apply(lambda x:get_label(x)))
```


```python
/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/[/docs/projects/capstone/feature_analysis/'/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/v/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/'/docs/projects/capstone/feature_analysis/]/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/v/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/x/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis//docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/[/docs/projects/capstone/feature_analysis/'/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/v/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/'/docs/projects/capstone/feature_analysis/]/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/v/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/x/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis/```




    1    253
    2     83
    3     90
    Name: grass_level, dtype: int64




```python
grass_pollen_level_dict = {1: "low", 2: 'medium', 3: 'high'}

#plt.figure(/docs/projects/capstone/feature_analysis/figsize=(10,6))
ax = sns.barplot(/docs/projects/capstone/feature_analysis/x=grass_df['grass_level'].value_counts().sort_index().rename(index=grass_pollen_level_dict).index, 
/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/x/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/b/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/x/docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/[/docs/projects/capstone/feature_analysis/'/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/v/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/'/docs/projects/capstone/feature_analysis/]/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/v/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/x/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/x/docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/v/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/x/docs/projects/capstone/feature_analysis/,/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis/ax = sns.barplot(x=grass_df['grass_level'].value_counts().sort_index().rename(/docs/projects/capstone/feature_analysis/index=grass_pollen_level_dict).index, 
/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/y/docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/[/docs/projects/capstone/feature_analysis/'/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/v/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/'/docs/projects/capstone/feature_analysis/]/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/v/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/x/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis//docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/y/docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/[/docs/projects/capstone/feature_analysis/'/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/v/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/'/docs/projects/capstone/feature_analysis/]/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/v/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/x/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis/ax.set(/docs/projects/capstone/feature_analysis/xlabel='grass pollen level', ylabel='count', title='grass pollen level with count')
ax.bar_label(/docs/projects/capstone/feature_analysis/ax.containers[0])
```




    [Text(/docs/projects/capstone/feature_analysis/0, 0, '253'), Text(0, 0, '83'), Text(0, 0, '90')]
    [Text(0, 0, '253'), Text(/docs/projects/capstone/feature_analysis/0, 0, '83'), Text(0, 0, '90')]
    [Text(0, 0, '253'), Text(0, 0, '83'), Text(/docs/projects/capstone/feature_analysis/0, 0, '90')]




    
![png](/docs/projects/capstone/feature_analysis/feature_analysis_report_files/feature_analysis_report_6_1.png)
    



```python
# df with numerical predicted variables and raw variables
/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/y/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis/count_df = count_df.drop(/docs/projects/capstone/feature_analysis/['grass_level'], axis=1).reset_index(drop=True)
count_df = count_df.drop(['grass_level'], axis=1).reset_index(/docs/projects/capstone/feature_analysis/drop=True)
count_df.head(/docs/projects/capstone/feature_analysis/5)
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
      <td>0.0</td>
      <td>567.409722</td>
      <td>0.014479</td>
      <td>291.515625</td>
      <td>101588.000000</td>
      <td>0.004978</td>
      <td>85.603299</td>
      <td>281.953993</td>
      <td>1.618056</td>
      <td>0.277778</td>
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
      <td>0.0</td>
      <td>672.985000</td>
      <td>0.044749</td>
      <td>316.624375</td>
      <td>101377.280000</td>
      <td>0.004932</td>
      <td>199.541875</td>
      <td>285.437500</td>
      <td>-1.055000</td>
      <td>0.590000</td>
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
      <td>3.0</td>
      <td>1351.370000</td>
      <td>0.038105</td>
      <td>320.018125</td>
      <td>100350.040000</td>
      <td>0.005645</td>
      <td>184.114375</td>
      <td>289.835625</td>
      <td>2.420000</td>
      <td>-1.575000</td>
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
      <td>0.0</td>
      <td>896.015000</td>
      <td>0.011231</td>
      <td>318.508125</td>
      <td>101197.400000</td>
      <td>0.005693</td>
      <td>178.778750</td>
      <td>286.084375</td>
      <td>3.255000</td>
      <td>-0.600000</td>
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
      <td>0.0</td>
      <td>1343.250000</td>
      <td>0.055731</td>
      <td>309.666466</td>
      <td>101231.692308</td>
      <td>0.005118</td>
      <td>225.740385</td>
      <td>287.077524</td>
      <td>4.971154</td>
      <td>-1.826923</td>
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
<p>5 rows × 172 columns</p>
</div>




```python
# df with categorical predicted variables and raw variables
/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/b/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/y/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis/label_df = label_df.drop(/docs/projects/capstone/feature_analysis/['grass_count'], axis=1).reset_index(drop=True)
label_df = label_df.drop(['grass_count'], axis=1).reset_index(/docs/projects/capstone/feature_analysis/drop=True)
label_df.head(/docs/projects/capstone/feature_analysis/5)
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
      <th>grass_level</th>
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
      <td>1</td>
      <td>567.409722</td>
      <td>0.014479</td>
      <td>291.515625</td>
      <td>101588.000000</td>
      <td>0.004978</td>
      <td>85.603299</td>
      <td>281.953993</td>
      <td>1.618056</td>
      <td>0.277778</td>
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
      <td>1</td>
      <td>672.985000</td>
      <td>0.044749</td>
      <td>316.624375</td>
      <td>101377.280000</td>
      <td>0.004932</td>
      <td>199.541875</td>
      <td>285.437500</td>
      <td>-1.055000</td>
      <td>0.590000</td>
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
      <td>1</td>
      <td>1351.370000</td>
      <td>0.038105</td>
      <td>320.018125</td>
      <td>100350.040000</td>
      <td>0.005645</td>
      <td>184.114375</td>
      <td>289.835625</td>
      <td>2.420000</td>
      <td>-1.575000</td>
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
      <td>1</td>
      <td>896.015000</td>
      <td>0.011231</td>
      <td>318.508125</td>
      <td>101197.400000</td>
      <td>0.005693</td>
      <td>178.778750</td>
      <td>286.084375</td>
      <td>3.255000</td>
      <td>-0.600000</td>
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
      <td>1</td>
      <td>1343.250000</td>
      <td>0.055731</td>
      <td>309.666466</td>
      <td>101231.692308</td>
      <td>0.005118</td>
      <td>225.740385</td>
      <td>287.077524</td>
      <td>4.971154</td>
      <td>-1.826923</td>
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
<p>5 rows × 172 columns</p>
</div>




```python
# independent variables
X = count_df.drop(/docs/projects/capstone/feature_analysis/['grass_count'], axis=1).reset_index(drop=True)
X = count_df.drop(['grass_count'], axis=1).reset_index(/docs/projects/capstone/feature_analysis/drop=True)

# target/predicted variable i.e grass pollen count (/docs/projects/capstone/feature_analysis/numerical)
Y_count = count_df['grass_count']
# target/predicted variable i.e grass pollen count (/docs/projects/capstone/feature_analysis/categorical)
Y_label = label_df['grass_level']


feature_name = list(/docs/projects/capstone/feature_analysis/X.columns)
num_feats = 15
```

# Regression: grass count

## SelectKBest

The SelectKBest technique to extract the best features in a dataset based on the highest k-score. We can use this method for classification and regression data by modifying the "score function" option. It is important to determine which score function and the value of k to use.</br>

- Regression: f_regression</br>


```python
# regression
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
```


```python
# f_regression: correlation
f_regression_selector = SelectKBest(/docs/projects/capstone/feature_analysis/f_regression, k=num_feats)
f_regression_selector.fit(/docs/projects/capstone/feature_analysis/X, Y_count)
/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis//docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/X/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/[/docs/projects/capstone/feature_analysis/:/docs/projects/capstone/feature_analysis/,/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/]/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis/#print(/docs/projects/capstone/feature_analysis/str(len(f_regression_feature)), 'SelectKBest f_regression selected features')

#f_regression_feature
```


```python
print(/docs/projects/capstone/feature_analysis/str(len(f_regression_feature)), 'selected features from SelectKBest f_regression')
f_regression_df = pd.DataFrame(/docs/projects/capstone/feature_analysis/f_regression_feature)
f_regression_df['Explanation'] = ['surface downwelling shortwave flux in air', 
                                  'air temperature',
                                  'mean of meridional speed of air moving towards the northward at 10m', 
                                  'sum of surface downwelling shortwave flux in air',
                                  'sum of meridional speed of air moving towards the northward at 10m', 
                                  'maximun air temperature', 
                                  'maximun air temperature increase in 1h', 
                                  'maximun air temperature in the afternoon', 
                                  'minimun air temperature in the afternoon', 
                                  'maximun air temperature increase for 1h in the afternoon',
                                  'maximun air temperature increase for 3h in a day',
                                  'maximun air temperature decrease for 3h in a day',
                                  'mean wind speed from the north',
                                  'sum of surface downwelling shortwave flux in air for the past 1 day',
                                  'heat sum for the phase for the past 1 day']

f_regression_df = f_regression_df.rename(/docs/projects/capstone/feature_analysis/columns={0: "Feature"})
with pd.option_context(/docs/projects/capstone/feature_analysis/'display.max_colwidth', None):
    display(/docs/projects/capstone/feature_analysis/f_regression_df)
```

    15 selected features from SelectKBest f_regression



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
      <th>Explanation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>av_swsfcdown</td>
      <td>surface downwelling shortwave flux in air</td>
    </tr>
    <tr>
      <th>1</th>
      <td>av_temp_scrn</td>
      <td>air temperature</td>
    </tr>
    <tr>
      <th>2</th>
      <td>av_vwnd10m</td>
      <td>mean of meridional speed of air moving towards the northward at 10m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>av_swsfcdown_sum</td>
      <td>sum of surface downwelling shortwave flux in air</td>
    </tr>
    <tr>
      <th>4</th>
      <td>av_vwnd10m_sum</td>
      <td>sum of meridional speed of air moving towards the northward at 10m</td>
    </tr>
    <tr>
      <th>5</th>
      <td>av_temp_scrn_max</td>
      <td>maximun air temperature</td>
    </tr>
    <tr>
      <th>6</th>
      <td>av_temp_scrn_max_1h_rise</td>
      <td>maximun air temperature increase in 1h</td>
    </tr>
    <tr>
      <th>7</th>
      <td>av_temp_scrn_max_afternoon</td>
      <td>maximun air temperature in the afternoon</td>
    </tr>
    <tr>
      <th>8</th>
      <td>av_temp_scrn_min_afternoon</td>
      <td>minimun air temperature in the afternoon</td>
    </tr>
    <tr>
      <th>9</th>
      <td>av_temp_scrn_max_afternoon_1hrise</td>
      <td>maximun air temperature increase for 1h in the afternoon</td>
    </tr>
    <tr>
      <th>10</th>
      <td>av_temp_scrn_max_day_3hrise</td>
      <td>maximun air temperature increase for 3h in a day</td>
    </tr>
    <tr>
      <th>11</th>
      <td>av_temp_scrn_max_day_3hfall</td>
      <td>maximun air temperature decrease for 3h in a day</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Wind-North_mean</td>
      <td>mean wind speed from the north</td>
    </tr>
    <tr>
      <th>13</th>
      <td>av_swsfcdown_sum_1D</td>
      <td>sum of surface downwelling shortwave flux in air for the past 1 day</td>
    </tr>
    <tr>
      <th>14</th>
      <td>thermal_time_1D</td>
      <td>heat sum for the phase for the past 1 day</td>
    </tr>
  </tbody>
</table>
</div>



```python
# mi_regression_selector = SelectKBest(/docs/projects/capstone/feature_analysis/mutual_info_regression, k=num_feats)
# mi_regression_selector.fit(/docs/projects/capstone/feature_analysis/X, Y_count)
/docs/projects/capstone/feature_analysis/#/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis//docs/projects/capstone/feature_analysis/#/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/X/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/[/docs/projects/capstone/feature_analysis/:/docs/projects/capstone/feature_analysis/,/docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/]/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis/# # print(/docs/projects/capstone/feature_analysis/str(len(mi_regression_feature)), 'SelectKBest mutual_info_regression selected features')

# pd.DataFrame(/docs/projects/capstone/feature_analysis/mi_regression_feature)
```


```python
# print(/docs/projects/capstone/feature_analysis/str(len(mi_regression_feature)), 'selected features from SelectKBest mutual_info_regression')
# mi_regression_df = pd.DataFrame(/docs/projects/capstone/feature_analysis/mi_regression_feature)
# mi_regression_df['Explanation'] = ['air temperature',
#                                    'maximun air temperature',
#                                    'maximum air pressure at sea level decrease for 3h',
#                                    'maximun air temperature in the afternoon', 
#                                    'minimun air temperature in the afternoon',
#                                    'sum of air temperature in the afternoon',
#                                    'sum of air temperature in a day',
#                                    'sum of days of surface downwelling shortwave flux in air for the past 10 days',
#                                    'hours of surface downwelling shortwave flux in air for the past 30 days',
#                                    'hours of surface downwelling shortwave flux in air for the past 90 days',
#                                    'sum of days of surface downwelling shortwave flux in air for the past 90 days',
#                                    'sum of days of chilling temperatures where in the range of 0°C to 8°C for the past 30 days',
#                                    'hours of chilling temperatures where in the range of 0°C to 8°C for the past 90 days',
#                                    'heat sum for the phase for the past 180 days',
#                                    'moisture content of soil layer for the past 90 days']

# mi_regression_df = mi_regression_df.rename(/docs/projects/capstone/feature_analysis/columns={0: "Feature"})


# with pd.option_context(/docs/projects/capstone/feature_analysis/'display.max_colwidth', None):
#     display(/docs/projects/capstone/feature_analysis/mi_regression_df)

```


```python
# summary

# # put all selection together
# feature_selection_df_count = pd.DataFrame({'Feature':feature_name, 'Pearson':f_regression_support, 
#                                            'Mutual_info':mi_regression_support})
# # count the selected times for each feature
# feature_selection_df_count['Total'] = np.sum(/docs/projects/capstone/feature_analysis/feature_selection_df_count, axis=1)


# # display the top 30
# feature_selection_df_count = feature_selection_df_count.sort_values(/docs/projects/capstone/feature_analysis/['Total','Feature'] , ascending=False)
# feature_selection_df_count.index = range(/docs/projects/capstone/feature_analysis/1, len(feature_selection_df_count)+1)
# feature_selection_df_count = feature_selection_df_count.loc[feature_selection_df_count['Total'] == 2]
# # feature_selection_df_count['Explanation'] = ['minimun air temperature in the afternoon',
# #                                              'maximun air temperature in the afternoon',
# #                                              'maximun air temperature',
# #                                              'air temperature']

# # with pd.option_context(/docs/projects/capstone/feature_analysis/'display.max_colwidth', None):
# #     display(/docs/projects/capstone/feature_analysis/feature_selection_df_count)

# feature_selection_df_count
```

# Classification: grass count level

## SelectKBest
- Classification: f_classif, mutual_info_classif, chi2</br>


```python
# classification
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler

# If using 'chi2' score function, need to use MinMaxScaler in X since X contains negative values.

# Pearson's chi square test (/docs/projects/capstone/feature_analysis/goodness of fit) does not apply to negative values. It's logical because 
# the chi square test assumes frequencies distribution and a frequency can't be a negative number. 
# Consequently, sklearn.feature_selection.chi2 asserts the input is non-negative.

/docs/projects/capstone/feature_analysis/X/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/M/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/M/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/x/docs/projects/capstone/feature_analysis/S/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/X/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis//docs/projects/capstone/feature_analysis/X_norm = MinMaxScaler().fit_transform(/docs/projects/capstone/feature_analysis/X)
chi_selector = SelectKBest(/docs/projects/capstone/feature_analysis/chi2, k=num_feats)
chi_selector.fit(/docs/projects/capstone/feature_analysis/X_norm, Y_label)
/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/h/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/h/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis//docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/h/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/X/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/[/docs/projects/capstone/feature_analysis/:/docs/projects/capstone/feature_analysis/,/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/h/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/]/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis/# print(/docs/projects/capstone/feature_analysis/str(len(chi_feature)), 'selected features')

# chi_feature
```


```python
print(/docs/projects/capstone/feature_analysis/str(len(chi_feature)), 'selected features from SelectKBest chi2')
/docs/projects/capstone/feature_analysis/chi_feature_df = pd.DataFrame(/docs/projects/capstone/feature_analysis/chi_feature)
chi_feature_df['Explanation'] = ['air temperature',
                                 'maximun air temperature',
                                 'maximun air temperature increase in 1h',
                                 'maximun air temperature in the afternoon', 
                                 'minimun air temperature in the afternoon',
                                 'maximun air temperature increase for 3h in a day',
                                 'hours of precipitation in the morning', 
                                 'mean speed of wind from north',
                                 'total speed of wind from north',
                                 'total speed of wind from south',
                                 'hours of forcing temperature for the past 1 day',
                                 'hours of chilling temperatures for the past 10 days',
                                 'hours of chilling temperatures for the past 30 days',
                                 'sum of days of chilling temperatures for the past 30 days',
                                 'heat sum for the phase for the past 1 day']

chi_feature_df = chi_feature_df.rename(/docs/projects/capstone/feature_analysis/columns={0: "Feature"})
with pd.option_context(/docs/projects/capstone/feature_analysis/'display.max_colwidth', None):
    display(/docs/projects/capstone/feature_analysis/chi_feature_df)
```

    15 selected features from SelectKBest chi2



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
      <th>Explanation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>av_temp_scrn</td>
      <td>air temperature</td>
    </tr>
    <tr>
      <th>1</th>
      <td>av_temp_scrn_max</td>
      <td>maximun air temperature</td>
    </tr>
    <tr>
      <th>2</th>
      <td>av_temp_scrn_max_1h_rise</td>
      <td>maximun air temperature increase in 1h</td>
    </tr>
    <tr>
      <th>3</th>
      <td>av_temp_scrn_max_afternoon</td>
      <td>maximun air temperature in the afternoon</td>
    </tr>
    <tr>
      <th>4</th>
      <td>av_temp_scrn_min_afternoon</td>
      <td>minimun air temperature in the afternoon</td>
    </tr>
    <tr>
      <th>5</th>
      <td>av_temp_scrn_max_day_3hrise</td>
      <td>maximun air temperature increase for 3h in a day</td>
    </tr>
    <tr>
      <th>6</th>
      <td>morning_hrs_of_precp</td>
      <td>hours of precipitation in the morning</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Wind-North_mean</td>
      <td>mean speed of wind from north</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Wind-North_total</td>
      <td>total speed of wind from north</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Wind-South_total</td>
      <td>total speed of wind from south</td>
    </tr>
    <tr>
      <th>10</th>
      <td>forcing_temp_numhours_1D</td>
      <td>hours of forcing temperature for the past 1 day</td>
    </tr>
    <tr>
      <th>11</th>
      <td>chilling_temp_numhours_10D</td>
      <td>hours of chilling temperatures for the past 10 days</td>
    </tr>
    <tr>
      <th>12</th>
      <td>chilling_temp_numhours_30D</td>
      <td>hours of chilling temperatures for the past 30 days</td>
    </tr>
    <tr>
      <th>13</th>
      <td>chilling_temp_sum_30D</td>
      <td>sum of days of chilling temperatures for the past 30 days</td>
    </tr>
    <tr>
      <th>14</th>
      <td>thermal_time_1D</td>
      <td>heat sum for the phase for the past 1 day</td>
    </tr>
  </tbody>
</table>
</div>



```python
# f_classif_selector = SelectKBest(/docs/projects/capstone/feature_analysis/f_classif, k=num_feats)
# f_classif_selector.fit(/docs/projects/capstone/feature_analysis/X_norm, Y_label)
/docs/projects/capstone/feature_analysis/#/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis//docs/projects/capstone/feature_analysis/#/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/X/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/[/docs/projects/capstone/feature_analysis/:/docs/projects/capstone/feature_analysis/,/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/]/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis/# print(/docs/projects/capstone/feature_analysis/str(len(f_classif_feature)), 'selected features')

# f_classif_feature
```


```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(/docs/projects/capstone/feature_analysis/estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
rfe_selector.fit(/docs/projects/capstone/feature_analysis/X_norm, Y_label)

/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis//docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/X/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/[/docs/projects/capstone/feature_analysis/:/docs/projects/capstone/feature_analysis/,/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/]/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis/# print(/docs/projects/capstone/feature_analysis/str(len(rfe_feature)), 'selected features')

# rfe_feature
```

    Fitting estimator with 171 features.
    Fitting estimator with 161 features.
    Fitting estimator with 151 features.
    Fitting estimator with 141 features.
    Fitting estimator with 131 features.
    Fitting estimator with 121 features.
    Fitting estimator with 111 features.
    Fitting estimator with 101 features.
    Fitting estimator with 91 features.
    Fitting estimator with 81 features.
    Fitting estimator with 71 features.
    Fitting estimator with 61 features.
    Fitting estimator with 51 features.
    Fitting estimator with 41 features.
    Fitting estimator with 31 features.
    Fitting estimator with 21 features.



```python
print(/docs/projects/capstone/feature_analysis/str(len(rfe_feature)), 'selected features from RFE')
/docs/projects/capstone/feature_analysis/rfe_feature_df = pd.DataFrame(/docs/projects/capstone/feature_analysis/rfe_feature)
rfe_feature_df['Explanation'] = ['air temperature',
                                 'maximun 10 meter wind vertival-component (/docs/projects/capstone/feature_analysis/Mean) decrease in 1h',
                                 'maximun air pressure at sea level decrease in 3h in a day',
                                 'maximun specific_humidity decrease in 1h in the afternoon', 
                                 'minimun air temperature in the afternoon',
                                 'total hours of precipitation',
                                 'mean speed of wind from north',
                                 'total speed of wind from south',
                                 'sum of surface downwelling shortwave flux in air for the past 180 days',
                                 'sum of forcing temperature for the past 1 day',
                                 'hours of chilling temperatures for the past 30 days',
                                 'hours of chilling temperatures for the past 90 days',
                                 'hours of chilling temperatures for the past 180 days',
                                 'sum of topt for the past 30 days',
                                 'moisture content of soil layer for the past 180 days']

rfe_feature_df = rfe_feature_df.rename(/docs/projects/capstone/feature_analysis/columns={0: "Feature"})
with pd.option_context(/docs/projects/capstone/feature_analysis/'display.max_colwidth', None):
    display(/docs/projects/capstone/feature_analysis/rfe_feature_df)
```

    15 selected features from RFE



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
      <th>Explanation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>av_temp_scrn</td>
      <td>air temperature</td>
    </tr>
    <tr>
      <th>1</th>
      <td>av_wndgust10m_max_1h_fall</td>
      <td>maximun 10 meter wind vertival-component (/docs/projects/capstone/feature_analysis/Mean) decrease in 1h</td>
    </tr>
    <tr>
      <th>2</th>
      <td>av_mslp_max_day_3hfall</td>
      <td>maximun air pressure at sea level decrease in 3h in a day</td>
    </tr>
    <tr>
      <th>3</th>
      <td>av_qsair_scrn_max_afternoon_1hfall</td>
      <td>maximun specific_humidity decrease in 1h in the afternoon</td>
    </tr>
    <tr>
      <th>4</th>
      <td>av_temp_scrn_min_afternoon</td>
      <td>minimun air temperature in the afternoon</td>
    </tr>
    <tr>
      <th>5</th>
      <td>total_hrs_of_precp</td>
      <td>total hours of precipitation</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Wind-North_mean</td>
      <td>mean speed of wind from north</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Wind-South_total</td>
      <td>total speed of wind from south</td>
    </tr>
    <tr>
      <th>8</th>
      <td>av_swsfcdown_sum_180D</td>
      <td>sum of surface downwelling shortwave flux in air for the past 180 days</td>
    </tr>
    <tr>
      <th>9</th>
      <td>forcing_temp_sum_1D</td>
      <td>sum of forcing temperature for the past 1 day</td>
    </tr>
    <tr>
      <th>10</th>
      <td>chilling_temp_numhours_30D</td>
      <td>hours of chilling temperatures for the past 30 days</td>
    </tr>
    <tr>
      <th>11</th>
      <td>chilling_temp_numhours_90D</td>
      <td>hours of chilling temperatures for the past 90 days</td>
    </tr>
    <tr>
      <th>12</th>
      <td>chilling_temp_numhours_180D</td>
      <td>hours of chilling temperatures for the past 180 days</td>
    </tr>
    <tr>
      <th>13</th>
      <td>topt_sum_30D</td>
      <td>sum of topt for the past 30 days</td>
    </tr>
    <tr>
      <th>14</th>
      <td>soil_mois_180D</td>
      <td>moisture content of soil layer for the past 180 days</td>
    </tr>
  </tbody>
</table>
</div>



```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

embeded_lr_selector = SelectFromModel(/docs/projects/capstone/feature_analysis/LogisticRegression(penalty="l1", solver='liblinear'), max_features=num_feats)
embeded_lr_selector.fit(/docs/projects/capstone/feature_analysis/X_norm, Y_label)

/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/b/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/b/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/g/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis//docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/b/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/f/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/a/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/=/docs/projects/capstone/feature_analysis/ /docs/projects/capstone/feature_analysis/X/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/[/docs/projects/capstone/feature_analysis/:/docs/projects/capstone/feature_analysis/,/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/b/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/e/docs/projects/capstone/feature_analysis/d/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/_/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/p/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/r/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/]/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/c/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/u/docs/projects/capstone/feature_analysis/m/docs/projects/capstone/feature_analysis/n/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/./docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/o/docs/projects/capstone/feature_analysis/l/docs/projects/capstone/feature_analysis/i/docs/projects/capstone/feature_analysis/s/docs/projects/capstone/feature_analysis/t/docs/projects/capstone/feature_analysis/(/docs/projects/capstone/feature_analysis/)/docs/projects/capstone/feature_analysis/
/docs/projects/capstone/feature_analysis/# print(/docs/projects/capstone/feature_analysis/str(len(embeded_lr_feature)), 'selected features')

# embeded_lr_feature
```


```python
print(/docs/projects/capstone/feature_analysis/str(len(embeded_lr_feature)), 'selected features from Lasso regularization')
/docs/projects/capstone/feature_analysis/embeded_lr_feature_df = pd.DataFrame(/docs/projects/capstone/feature_analysis/embeded_lr_feature)
embeded_lr_feature_df['Explanation'] = ['air temperature',
                                        'maximum air temperature',
                                        'maximun air pressure at sea level decrease in 1h in the afternoon',
                                        'maximum air temperature decrease in 3h in a day',
                                        'mean speed of wind from north',
                                        'total speed of wind from south',
                                        'sum of accumulated total precipitation amount at the surface for the past 90 days',
                                        'sum of surface downwelling shortwave flux in air for the past 180 days',
                                        'hours of forcing temperature for the past 1 day',
                                        'sum of forcing temperature for the past 1 day',
                                        'hours of chilling temperatures for the past 30 days',
                                        'hours of chilling temperatures for the past 90 days',
                                        'hours of chilling temperatures for the past 180 days',
                                        'heat sum for the phase for the past 30 days',
                                        'moisture content of soil layer for the past 180 days']

embeded_lr_feature_df = embeded_lr_feature_df.rename(/docs/projects/capstone/feature_analysis/columns={0: "Feature"})
with pd.option_context(/docs/projects/capstone/feature_analysis/'display.max_colwidth', None):
    display(/docs/projects/capstone/feature_analysis/embeded_lr_feature_df)
```

    15 selected features from Lasso regularization



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
      <th>Explanation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>av_temp_scrn</td>
      <td>air temperature</td>
    </tr>
    <tr>
      <th>1</th>
      <td>av_temp_scrn_max</td>
      <td>maximum air temperature</td>
    </tr>
    <tr>
      <th>2</th>
      <td>av_mslp_max_afternoon_1hfall</td>
      <td>maximun air pressure at sea level decrease in 1h in the afternoon</td>
    </tr>
    <tr>
      <th>3</th>
      <td>av_temp_scrn_max_day_3hfall</td>
      <td>maximum air temperature decrease in 3h in a day</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wind-North_mean</td>
      <td>mean speed of wind from north</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Wind-South_total</td>
      <td>total speed of wind from south</td>
    </tr>
    <tr>
      <th>6</th>
      <td>accum_prcp_numhours_90D</td>
      <td>sum of accumulated total precipitation amount at the surface for the past 90 days</td>
    </tr>
    <tr>
      <th>7</th>
      <td>av_swsfcdown_sum_180D</td>
      <td>sum of surface downwelling shortwave flux in air for the past 180 days</td>
    </tr>
    <tr>
      <th>8</th>
      <td>forcing_temp_numhours_1D</td>
      <td>hours of forcing temperature for the past 1 day</td>
    </tr>
    <tr>
      <th>9</th>
      <td>forcing_temp_sum_1D</td>
      <td>sum of forcing temperature for the past 1 day</td>
    </tr>
    <tr>
      <th>10</th>
      <td>chilling_temp_numhours_30D</td>
      <td>hours of chilling temperatures for the past 30 days</td>
    </tr>
    <tr>
      <th>11</th>
      <td>chilling_temp_numhours_90D</td>
      <td>hours of chilling temperatures for the past 90 days</td>
    </tr>
    <tr>
      <th>12</th>
      <td>chilling_temp_numhours_180D</td>
      <td>hours of chilling temperatures for the past 180 days</td>
    </tr>
    <tr>
      <th>13</th>
      <td>thermal_time_90D</td>
      <td>heat sum for the phase for the past 30 days</td>
    </tr>
    <tr>
      <th>14</th>
      <td>soil_mois_180D</td>
      <td>moisture content of soil layer for the past 180 days</td>
    </tr>
  </tbody>
</table>
</div>



```python
pd.set_option(/docs/projects/capstone/feature_analysis/'display.max_rows', None)
# put all selection together
feature_selection_df_label = pd.DataFrame({'Feature':feature_name, 'Chi-2':chi_support, 
                                     'RFE':rfe_support, 'L1(/docs/projects/capstone/feature_analysis/Logistics)':embeded_lr_support,})
# count the selected times for each feature
feature_selection_df_label['Total'] = np.sum(/docs/projects/capstone/feature_analysis/feature_selection_df_label, axis=1)

# Display most common features
feature_selection_df_label = feature_selection_df_label.sort_values(/docs/projects/capstone/feature_analysis/['Total','Feature'] , ascending=False)
feature_selection_df_label = feature_selection_df_label.loc[feature_selection_df_label['Total'] >= 2]
feature_selection_df_label.index = range(/docs/projects/capstone/feature_analysis/1, len(feature_selection_df_label)+1)

feature_selection_df_label['Explanation'] = ['hours of chilling temperatures for the past 30 days',
                                             'air temperature', 
                                             'total speed of wind from south',
                                             'mean speed of wind from north',
                                             'moisture content of soil layer for the past 180 days',
                                             'sum of forcing temperature for the past 1 day',
                                             'hours of forcing temperature for the past 1 day',
                                             'hours of chilling temperatures for the past 90 days',
                                             'hours of chilling temperatures for the past 180 days',
                                             'minimun air temperature in the afternoon',
                                             'maximum air temperature',
                                             'sum of surface downwelling shortwave flux in air for the past 180 days']

with pd.option_context(/docs/projects/capstone/feature_analysis/'display.max_colwidth', None):
    display(/docs/projects/capstone/feature_analysis/feature_selection_df_label)
    
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
      <th>Chi-2</th>
      <th>RFE</th>
      <th>L1(/docs/projects/capstone/feature_analysis/Logistics)</th>
      <th>Total</th>
      <th>Explanation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>chilling_temp_numhours_30D</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>3</td>
      <td>hours of chilling temperatures for the past 30 days</td>
    </tr>
    <tr>
      <th>2</th>
      <td>av_temp_scrn</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>3</td>
      <td>air temperature</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wind-South_total</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>3</td>
      <td>total speed of wind from south</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wind-North_mean</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>3</td>
      <td>mean speed of wind from north</td>
    </tr>
    <tr>
      <th>5</th>
      <td>soil_mois_180D</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>2</td>
      <td>moisture content of soil layer for the past 180 days</td>
    </tr>
    <tr>
      <th>6</th>
      <td>forcing_temp_sum_1D</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>2</td>
      <td>sum of forcing temperature for the past 1 day</td>
    </tr>
    <tr>
      <th>7</th>
      <td>forcing_temp_numhours_1D</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>2</td>
      <td>hours of forcing temperature for the past 1 day</td>
    </tr>
    <tr>
      <th>8</th>
      <td>chilling_temp_numhours_90D</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>2</td>
      <td>hours of chilling temperatures for the past 90 days</td>
    </tr>
    <tr>
      <th>9</th>
      <td>chilling_temp_numhours_180D</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>2</td>
      <td>hours of chilling temperatures for the past 180 days</td>
    </tr>
    <tr>
      <th>10</th>
      <td>av_temp_scrn_min_afternoon</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>2</td>
      <td>minimun air temperature in the afternoon</td>
    </tr>
    <tr>
      <th>11</th>
      <td>av_temp_scrn_max</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>2</td>
      <td>maximum air temperature</td>
    </tr>
    <tr>
      <th>12</th>
      <td>av_swsfcdown_sum_180D</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>2</td>
      <td>sum of surface downwelling shortwave flux in air for the past 180 days</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Summary of methods of feature selection

chi_feature_df = chi_feature_df.rename(/docs/projects/capstone/feature_analysis/columns={'Feature': 'Chi-2'})
rfe_feature_df = rfe_feature_df.rename(/docs/projects/capstone/feature_analysis/columns={'Feature': 'RFE'})
embeded_lr_feature_df = embeded_lr_feature_df.rename(/docs/projects/capstone/feature_analysis/columns={'Feature': 'L1(Logistics)'})

df1 = pd.concat(/docs/projects/capstone/feature_analysis/[chi_feature_df, rfe_feature_df], axis=1)
df2 = pd.concat(/docs/projects/capstone/feature_analysis/[df1, embeded_lr_feature_df], axis=1)

df2 = df2.drop(/docs/projects/capstone/feature_analysis/['Explanation'], axis=1).reset_index(drop=True)
df2 = df2.drop(['Explanation'], axis=1).reset_index(/docs/projects/capstone/feature_analysis/drop=True)

print(/docs/projects/capstone/feature_analysis/str(len(embeded_lr_feature)), 'selected features from SelectKBest chi2, RFE and Lasso regularization')
df2
```

    15 selected features from SelectKBest chi2, RFE and Lasso regularization





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
      <th>Chi-2</th>
      <th>RFE</th>
      <th>L1(/docs/projects/capstone/feature_analysis/Logistics)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>av_temp_scrn</td>
      <td>av_temp_scrn</td>
      <td>av_temp_scrn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>av_temp_scrn_max</td>
      <td>av_wndgust10m_max_1h_fall</td>
      <td>av_temp_scrn_max</td>
    </tr>
    <tr>
      <th>2</th>
      <td>av_temp_scrn_max_1h_rise</td>
      <td>av_mslp_max_day_3hfall</td>
      <td>av_mslp_max_afternoon_1hfall</td>
    </tr>
    <tr>
      <th>3</th>
      <td>av_temp_scrn_max_afternoon</td>
      <td>av_qsair_scrn_max_afternoon_1hfall</td>
      <td>av_temp_scrn_max_day_3hfall</td>
    </tr>
    <tr>
      <th>4</th>
      <td>av_temp_scrn_min_afternoon</td>
      <td>av_temp_scrn_min_afternoon</td>
      <td>Wind-North_mean</td>
    </tr>
    <tr>
      <th>5</th>
      <td>av_temp_scrn_max_day_3hrise</td>
      <td>total_hrs_of_precp</td>
      <td>Wind-South_total</td>
    </tr>
    <tr>
      <th>6</th>
      <td>morning_hrs_of_precp</td>
      <td>Wind-North_mean</td>
      <td>accum_prcp_numhours_90D</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Wind-North_mean</td>
      <td>Wind-South_total</td>
      <td>av_swsfcdown_sum_180D</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Wind-North_total</td>
      <td>av_swsfcdown_sum_180D</td>
      <td>forcing_temp_numhours_1D</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Wind-South_total</td>
      <td>forcing_temp_sum_1D</td>
      <td>forcing_temp_sum_1D</td>
    </tr>
    <tr>
      <th>10</th>
      <td>forcing_temp_numhours_1D</td>
      <td>chilling_temp_numhours_30D</td>
      <td>chilling_temp_numhours_30D</td>
    </tr>
    <tr>
      <th>11</th>
      <td>chilling_temp_numhours_10D</td>
      <td>chilling_temp_numhours_90D</td>
      <td>chilling_temp_numhours_90D</td>
    </tr>
    <tr>
      <th>12</th>
      <td>chilling_temp_numhours_30D</td>
      <td>chilling_temp_numhours_180D</td>
      <td>chilling_temp_numhours_180D</td>
    </tr>
    <tr>
      <th>13</th>
      <td>chilling_temp_sum_30D</td>
      <td>topt_sum_30D</td>
      <td>thermal_time_90D</td>
    </tr>
    <tr>
      <th>14</th>
      <td>thermal_time_1D</td>
      <td>soil_mois_180D</td>
      <td>soil_mois_180D</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
