```python
%matplotlib inline

import numpy as np
from scipy import signal
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


ds=pd.read_pickle(/docs/projects/capstone/Modeling/time_series/'/Users/demi/Desktop/pollen data/weather_v2.pkl')
df=ds[1]
df1=df.loc['2000-01-01':'2021-01-31']
# df1=df.loc[df.index<='2021-1-31']


df1['Count Date']=df1.index
df1['Count Date']=df1['Count Date'].astype(/docs/projects/capstone/Modeling/time_series/'datetime64[ns]')
df1
```

    /var/folders/q1/7fhq9ggs78g6wf7c38dhzh9h0000gn/T/ipykernel_78036/3767438390.py:17: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df1['Count Date']=df1.index
    /var/folders/q1/7fhq9ggs78g6wf7c38dhzh9h0000gn/T/ipykernel_78036/3767438390.py:18: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df1['Count Date']=df1['Count Date'].astype(/docs/projects/capstone/Modeling/time_series/'datetime64[ns]')





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
      <th>thermal_time_10D</th>
      <th>thermal_time_30D</th>
      <th>thermal_time_90D</th>
      <th>thermal_time_180D</th>
      <th>soil_mois_1D</th>
      <th>soil_mois_10D</th>
      <th>soil_mois_30D</th>
      <th>soil_mois_90D</th>
      <th>soil_mois_180D</th>
      <th>Count Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-02</th>
      <td>753.565000</td>
      <td>0.000156</td>
      <td>293.490000</td>
      <td>101769.360000</td>
      <td>0.006113</td>
      <td>411.645000</td>
      <td>289.752500</td>
      <td>-0.855000</td>
      <td>4.340000</td>
      <td>8.495000</td>
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
      <th>2000-01-03</th>
      <td>882.295000</td>
      <td>0.011341</td>
      <td>340.160000</td>
      <td>101239.120000</td>
      <td>0.007275</td>
      <td>333.087500</td>
      <td>294.993125</td>
      <td>0.780000</td>
      <td>-1.270000</td>
      <td>8.580000</td>
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
      <th>2000-01-04</th>
      <td>642.775000</td>
      <td>0.241001</td>
      <td>354.760625</td>
      <td>101026.800000</td>
      <td>0.008047</td>
      <td>274.170000</td>
      <td>289.227500</td>
      <td>3.050000</td>
      <td>4.835000</td>
      <td>12.920000</td>
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
      <th>2000-01-05</th>
      <td>850.585000</td>
      <td>0.021159</td>
      <td>332.816250</td>
      <td>101789.000000</td>
      <td>0.006387</td>
      <td>309.916250</td>
      <td>287.412500</td>
      <td>3.655000</td>
      <td>7.555000</td>
      <td>16.020000</td>
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
      <th>2000-01-06</th>
      <td>772.685000</td>
      <td>0.006436</td>
      <td>339.945000</td>
      <td>101948.000000</td>
      <td>0.007227</td>
      <td>344.033125</td>
      <td>289.284375</td>
      <td>0.785000</td>
      <td>5.965000</td>
      <td>10.745000</td>
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
      <th>2021-01-27</th>
      <td>834.791667</td>
      <td>0.075095</td>
      <td>352.086666</td>
      <td>100960.166667</td>
      <td>0.007849</td>
      <td>130.549167</td>
      <td>289.445415</td>
      <td>-0.145833</td>
      <td>4.887499</td>
      <td>10.920833</td>
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
      <th>2021-01-28</th>
      <td>963.125000</td>
      <td>0.000219</td>
      <td>335.512920</td>
      <td>101483.541667</td>
      <td>0.007803</td>
      <td>334.555417</td>
      <td>293.186250</td>
      <td>-2.487500</td>
      <td>2.991666</td>
      <td>9.762500</td>
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
      <th>2021-01-29</th>
      <td>896.600000</td>
      <td>0.484703</td>
      <td>395.593999</td>
      <td>101408.280000</td>
      <td>0.009936</td>
      <td>132.476798</td>
      <td>295.356401</td>
      <td>-2.292000</td>
      <td>-0.148000</td>
      <td>5.888000</td>
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
      <th>2021-01-30</th>
      <td>632.791667</td>
      <td>0.407535</td>
      <td>395.608746</td>
      <td>100911.333333</td>
      <td>0.011993</td>
      <td>86.142084</td>
      <td>292.612085</td>
      <td>0.550000</td>
      <td>-1.445833</td>
      <td>7.416667</td>
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
      <th>2021-01-31</th>
      <td>748.458333</td>
      <td>0.001671</td>
      <td>359.496667</td>
      <td>101510.916667</td>
      <td>0.008391</td>
      <td>275.843329</td>
      <td>291.180419</td>
      <td>0.341666</td>
      <td>3.920833</td>
      <td>9.358333</td>
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
<p>7701 rows × 172 columns</p>
</div>




```python
d2=pd.read_csv(/docs/projects/capstone/Modeling/time_series/'/Users/demi/Desktop/pollen data/melbourne_grass_preprocessed.csv')
# df2=d1.loc[d1['Location']==1]
d2['Count Date']=d2['Count Date'].astype(/docs/projects/capstone/Modeling/time_series/'datetime64[ns]')
df=pd.merge(/docs/projects/capstone/Modeling/time_series/df1,d2,on='Count Date')

# df=df.drop(/docs/projects/capstone/Modeling/time_series/['Location','QCL','Latitude','Longitude','Elevation','Continuation Location','Sample Time','SchColTime','Name','State','NameMLFile'],axis=1)
# df['Count']=df['Count'].fillna(/docs/projects/capstone/Modeling/time_series/method='bfill').astype(int)
# df['Count']=df['Count'].fillna(method='bfill').astype(/docs/projects/capstone/Modeling/time_series/int)
# print(/docs/projects/capstone/Modeling/time_series/df['Count'].isnull().values==True)
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
      <th>thermal_time_30D</th>
      <th>thermal_time_90D</th>
      <th>thermal_time_180D</th>
      <th>soil_mois_1D</th>
      <th>soil_mois_10D</th>
      <th>soil_mois_30D</th>
      <th>soil_mois_90D</th>
      <th>soil_mois_180D</th>
      <th>Count Date</th>
      <th>grass_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>753.565000</td>
      <td>0.000156</td>
      <td>293.490000</td>
      <td>101769.360000</td>
      <td>0.006113</td>
      <td>411.645000</td>
      <td>289.752500</td>
      <td>-0.855000</td>
      <td>4.340000</td>
      <td>8.495000</td>
      <td>...</td>
      <td>6.375000</td>
      <td>6.375000</td>
      <td>6.375000</td>
      <td>2343.648438</td>
      <td>2721.484375</td>
      <td>2721.484375</td>
      <td>2721.484375</td>
      <td>2721.484375</td>
      <td>2000-01-02</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>882.295000</td>
      <td>0.011341</td>
      <td>340.160000</td>
      <td>101239.120000</td>
      <td>0.007275</td>
      <td>333.087500</td>
      <td>294.993125</td>
      <td>0.780000</td>
      <td>-1.270000</td>
      <td>8.580000</td>
      <td>...</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>13.000000</td>
      <td>2341.230469</td>
      <td>4969.046875</td>
      <td>4969.046875</td>
      <td>4969.046875</td>
      <td>4969.046875</td>
      <td>2000-01-03</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>642.775000</td>
      <td>0.241001</td>
      <td>354.760625</td>
      <td>101026.800000</td>
      <td>0.008047</td>
      <td>274.170000</td>
      <td>289.227500</td>
      <td>3.050000</td>
      <td>4.835000</td>
      <td>12.920000</td>
      <td>...</td>
      <td>17.375000</td>
      <td>17.375000</td>
      <td>17.375000</td>
      <td>2340.359375</td>
      <td>7215.773438</td>
      <td>7215.773438</td>
      <td>7215.773438</td>
      <td>7215.773438</td>
      <td>2000-01-04</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>850.585000</td>
      <td>0.021159</td>
      <td>332.816250</td>
      <td>101789.000000</td>
      <td>0.006387</td>
      <td>309.916250</td>
      <td>287.412500</td>
      <td>3.655000</td>
      <td>7.555000</td>
      <td>16.020000</td>
      <td>...</td>
      <td>20.250000</td>
      <td>20.250000</td>
      <td>20.250000</td>
      <td>2339.445312</td>
      <td>9461.621094</td>
      <td>9461.621094</td>
      <td>9461.621094</td>
      <td>9461.621094</td>
      <td>2000-01-05</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>772.685000</td>
      <td>0.006436</td>
      <td>339.945000</td>
      <td>101948.000000</td>
      <td>0.007227</td>
      <td>344.033125</td>
      <td>289.284375</td>
      <td>0.785000</td>
      <td>5.965000</td>
      <td>10.745000</td>
      <td>...</td>
      <td>23.000000</td>
      <td>23.000000</td>
      <td>23.000000</td>
      <td>2338.539062</td>
      <td>11706.601562</td>
      <td>11706.601562</td>
      <td>11706.601562</td>
      <td>11706.601562</td>
      <td>2000-01-06</td>
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
      <th>7696</th>
      <td>834.791667</td>
      <td>0.075095</td>
      <td>352.086666</td>
      <td>100960.166667</td>
      <td>0.007849</td>
      <td>130.549167</td>
      <td>289.445415</td>
      <td>-0.145833</td>
      <td>4.887499</td>
      <td>10.920833</td>
      <td>...</td>
      <td>184.374908</td>
      <td>184.374908</td>
      <td>787.493103</td>
      <td>7547.602417</td>
      <td>76221.471649</td>
      <td>229415.330170</td>
      <td>692386.868195</td>
      <td>961095.985832</td>
      <td>2021-01-27</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7697</th>
      <td>963.125000</td>
      <td>0.000219</td>
      <td>335.512920</td>
      <td>101483.541667</td>
      <td>0.007803</td>
      <td>334.555417</td>
      <td>293.186250</td>
      <td>-2.487500</td>
      <td>2.991666</td>
      <td>9.762500</td>
      <td>...</td>
      <td>186.924896</td>
      <td>186.924896</td>
      <td>789.674408</td>
      <td>7535.679962</td>
      <td>76149.324097</td>
      <td>229251.895264</td>
      <td>692180.215698</td>
      <td>967718.756729</td>
      <td>2021-01-28</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>7698</th>
      <td>896.600000</td>
      <td>0.484703</td>
      <td>395.593999</td>
      <td>101408.280000</td>
      <td>0.009936</td>
      <td>132.476798</td>
      <td>295.356401</td>
      <td>-2.292000</td>
      <td>-0.148000</td>
      <td>5.888000</td>
      <td>...</td>
      <td>185.274902</td>
      <td>185.274902</td>
      <td>790.161865</td>
      <td>7836.774933</td>
      <td>76074.451599</td>
      <td>229086.352783</td>
      <td>691967.418121</td>
      <td>974334.436188</td>
      <td>2021-01-29</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>7699</th>
      <td>632.791667</td>
      <td>0.407535</td>
      <td>395.608746</td>
      <td>100911.333333</td>
      <td>0.011993</td>
      <td>86.142084</td>
      <td>292.612085</td>
      <td>0.550000</td>
      <td>-1.445833</td>
      <td>7.416667</td>
      <td>...</td>
      <td>182.549927</td>
      <td>182.549927</td>
      <td>789.127441</td>
      <td>7542.737488</td>
      <td>76027.771698</td>
      <td>228949.192780</td>
      <td>691782.330627</td>
      <td>980975.300159</td>
      <td>2021-01-30</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7700</th>
      <td>748.458333</td>
      <td>0.001671</td>
      <td>359.496667</td>
      <td>101510.916667</td>
      <td>0.008391</td>
      <td>275.843329</td>
      <td>291.180419</td>
      <td>0.341666</td>
      <td>3.920833</td>
      <td>9.358333</td>
      <td>...</td>
      <td>185.149872</td>
      <td>185.149872</td>
      <td>791.405579</td>
      <td>7537.892426</td>
      <td>76006.474243</td>
      <td>228821.920197</td>
      <td>691599.008057</td>
      <td>987616.377117</td>
      <td>2021-01-31</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>7701 rows × 173 columns</p>
</div>




```python
print(/docs/projects/capstone/Modeling/time_series/df['Count Date'])
/docs/projects/capstone/Modeling/time_series/df['Count Date'] = pd.to_datetime(/docs/projects/capstone/Modeling/time_series/df['Count Date'])
df = df.set_index(/docs/projects/capstone/Modeling/time_series/'Count Date')
df
```

    0      2000-01-02
    1      2000-01-03
    2      2000-01-04
    3      2000-01-05
    4      2000-01-06
              ...    
    7696   2021-01-27
    7697   2021-01-28
    7698   2021-01-29
    7699   2021-01-30
    7700   2021-01-31
    Name: Count Date, Length: 7701, dtype: datetime64[ns]





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
      <th>thermal_time_10D</th>
      <th>thermal_time_30D</th>
      <th>thermal_time_90D</th>
      <th>thermal_time_180D</th>
      <th>soil_mois_1D</th>
      <th>soil_mois_10D</th>
      <th>soil_mois_30D</th>
      <th>soil_mois_90D</th>
      <th>soil_mois_180D</th>
      <th>grass_count</th>
    </tr>
    <tr>
      <th>Count Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-01-02</th>
      <td>753.565000</td>
      <td>0.000156</td>
      <td>293.490000</td>
      <td>101769.360000</td>
      <td>0.006113</td>
      <td>411.645000</td>
      <td>289.752500</td>
      <td>-0.855000</td>
      <td>4.340000</td>
      <td>8.495000</td>
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
      <td>0.0</td>
    </tr>
    <tr>
      <th>2000-01-03</th>
      <td>882.295000</td>
      <td>0.011341</td>
      <td>340.160000</td>
      <td>101239.120000</td>
      <td>0.007275</td>
      <td>333.087500</td>
      <td>294.993125</td>
      <td>0.780000</td>
      <td>-1.270000</td>
      <td>8.580000</td>
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
      <td>0.0</td>
    </tr>
    <tr>
      <th>2000-01-04</th>
      <td>642.775000</td>
      <td>0.241001</td>
      <td>354.760625</td>
      <td>101026.800000</td>
      <td>0.008047</td>
      <td>274.170000</td>
      <td>289.227500</td>
      <td>3.050000</td>
      <td>4.835000</td>
      <td>12.920000</td>
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
      <td>0.0</td>
    </tr>
    <tr>
      <th>2000-01-05</th>
      <td>850.585000</td>
      <td>0.021159</td>
      <td>332.816250</td>
      <td>101789.000000</td>
      <td>0.006387</td>
      <td>309.916250</td>
      <td>287.412500</td>
      <td>3.655000</td>
      <td>7.555000</td>
      <td>16.020000</td>
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
      <td>0.0</td>
    </tr>
    <tr>
      <th>2000-01-06</th>
      <td>772.685000</td>
      <td>0.006436</td>
      <td>339.945000</td>
      <td>101948.000000</td>
      <td>0.007227</td>
      <td>344.033125</td>
      <td>289.284375</td>
      <td>0.785000</td>
      <td>5.965000</td>
      <td>10.745000</td>
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
      <th>2021-01-27</th>
      <td>834.791667</td>
      <td>0.075095</td>
      <td>352.086666</td>
      <td>100960.166667</td>
      <td>0.007849</td>
      <td>130.549167</td>
      <td>289.445415</td>
      <td>-0.145833</td>
      <td>4.887499</td>
      <td>10.920833</td>
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
      <td>1.0</td>
    </tr>
    <tr>
      <th>2021-01-28</th>
      <td>963.125000</td>
      <td>0.000219</td>
      <td>335.512920</td>
      <td>101483.541667</td>
      <td>0.007803</td>
      <td>334.555417</td>
      <td>293.186250</td>
      <td>-2.487500</td>
      <td>2.991666</td>
      <td>9.762500</td>
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
      <td>7.0</td>
    </tr>
    <tr>
      <th>2021-01-29</th>
      <td>896.600000</td>
      <td>0.484703</td>
      <td>395.593999</td>
      <td>101408.280000</td>
      <td>0.009936</td>
      <td>132.476798</td>
      <td>295.356401</td>
      <td>-2.292000</td>
      <td>-0.148000</td>
      <td>5.888000</td>
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
      <td>2.0</td>
    </tr>
    <tr>
      <th>2021-01-30</th>
      <td>632.791667</td>
      <td>0.407535</td>
      <td>395.608746</td>
      <td>100911.333333</td>
      <td>0.011993</td>
      <td>86.142084</td>
      <td>292.612085</td>
      <td>0.550000</td>
      <td>-1.445833</td>
      <td>7.416667</td>
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
      <td>0.0</td>
    </tr>
    <tr>
      <th>2021-01-31</th>
      <td>748.458333</td>
      <td>0.001671</td>
      <td>359.496667</td>
      <td>101510.916667</td>
      <td>0.008391</td>
      <td>275.843329</td>
      <td>291.180419</td>
      <td>0.341666</td>
      <td>3.920833</td>
      <td>9.358333</td>
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
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>7701 rows × 172 columns</p>
</div>




```python
# df=pd.read_csv(/docs/projects/capstone/Modeling/time_series/'/Users/demi/Desktop/pollen data/Grass.csv')
df['grass_count'].plot(/docs/projects/capstone/Modeling/time_series/figsize=(16,8))
# size of script
plt.tick_params(/docs/projects/capstone/Modeling/time_series/labelsize=10)
# generate grids
/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/g/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/```


    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_3_0.png)
    



```python
plt.style.use(/docs/projects/capstone/Modeling/time_series/{'figure.figsize':(5,5)})
df['grass_count'].plot(/docs/projects/capstone/Modeling/time_series/kind='hist',bins=20)
```




    <AxesSubplot:ylabel='Frequency'>




    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_4_1.png)
    



```python
# mean of the year 2018
df_M=pd.DataFrame(/docs/projects/capstone/Modeling/time_series/df['grass_count']['2018'])
df_M.resample(/docs/projects/capstone/Modeling/time_series/'M').mean().T
/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/M/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/M/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/T/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/```




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
      <th>Count Date</th>
      <th>2018-01-31</th>
      <th>2018-02-28</th>
      <th>2018-03-31</th>
      <th>2018-04-30</th>
      <th>2018-05-31</th>
      <th>2018-06-30</th>
      <th>2018-07-31</th>
      <th>2018-08-31</th>
      <th>2018-09-30</th>
      <th>2018-10-31</th>
      <th>2018-11-30</th>
      <th>2018-12-31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>grass_count</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.333333</td>
      <td>14.451613</td>
      <td>37.7</td>
      <td>25.709677</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.style.use(/docs/projects/capstone/Modeling/time_series/{'figure.figsize':(8,5)})

sns.heatmap(/docs/projects/capstone/Modeling/time_series/df_M.resample('M').mean().T)
/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/M/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/M/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/T/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/```




    <AxesSubplot:xlabel='Count Date'>




    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_6_1.png)
    



```python
# Apply the Moving Average function by a subset of size 7 days.

fig,axes=plt.subplots(/docs/projects/capstone/Modeling/time_series/2,2)
plt.subplot(/docs/projects/capstone/Modeling/time_series/2,2,1)
# Select the subset data from 2017-9 to 2017-12
df2017=df.loc['2017-9':'2017-12']['Count']      
df2017_mean = df2017.T.rolling(/docs/projects/capstone/Modeling/time_series/window=7).mean()
/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/2/docs/projects/capstone/Modeling/time_series/0/docs/projects/capstone/Modeling/time_series/1/docs/projects/capstone/Modeling/time_series/7/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/2/docs/projects/capstone/Modeling/time_series/0/docs/projects/capstone/Modeling/time_series/1/docs/projects/capstone/Modeling/time_series/7/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/T/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/g/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/7/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/df2017_mean.plot(/docs/projects/capstone/Modeling/time_series/figsize=(32,10))
plt.subplot(/docs/projects/capstone/Modeling/time_series/2,2,2)
# Select the subset data from 2018-9 to 2018-12
df2018=df.loc['2018-10':'2018-12']['Count']      
df2018_mean = df2017.T.rolling(/docs/projects/capstone/Modeling/time_series/window=7).mean()
/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/2/docs/projects/capstone/Modeling/time_series/0/docs/projects/capstone/Modeling/time_series/1/docs/projects/capstone/Modeling/time_series/8/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/2/docs/projects/capstone/Modeling/time_series/0/docs/projects/capstone/Modeling/time_series/1/docs/projects/capstone/Modeling/time_series/7/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/T/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/g/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/7/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/df2018_mean.plot(/docs/projects/capstone/Modeling/time_series/figsize=(32,10))
plt.subplot(/docs/projects/capstone/Modeling/time_series/2,2,3)
# Select the subset data from 2019-9 to 2019-12
df2019=df.loc['2019-10':'2019-12']['Count']      
df2019_mean = df2017.T.rolling(/docs/projects/capstone/Modeling/time_series/window=7).mean()
/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/2/docs/projects/capstone/Modeling/time_series/0/docs/projects/capstone/Modeling/time_series/1/docs/projects/capstone/Modeling/time_series/9/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/2/docs/projects/capstone/Modeling/time_series/0/docs/projects/capstone/Modeling/time_series/1/docs/projects/capstone/Modeling/time_series/7/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/T/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/g/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/7/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/df2019_mean.plot(/docs/projects/capstone/Modeling/time_series/figsize=(32,10))
plt.subplot(/docs/projects/capstone/Modeling/time_series/2,2,4)
# Select the subset data from 2020-9 to 2020-12
df2020=df.loc['2020-10':'2020-12']['Count']      
df2020_mean = df2017.T.rolling(/docs/projects/capstone/Modeling/time_series/window=7).mean()
/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/2/docs/projects/capstone/Modeling/time_series/0/docs/projects/capstone/Modeling/time_series/2/docs/projects/capstone/Modeling/time_series/0/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/2/docs/projects/capstone/Modeling/time_series/0/docs/projects/capstone/Modeling/time_series/1/docs/projects/capstone/Modeling/time_series/7/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/T/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/g/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/7/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/df2020_mean.plot(/docs/projects/capstone/Modeling/time_series/figsize=(32,10))
```




    <AxesSubplot:xlabel='Count Date'>




    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_7_1.png)
    



```python
df_mean = df['Count'].T.rolling(/docs/projects/capstone/Modeling/time_series/window=10).mean()
/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/[/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/C/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/u/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/]/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/T/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/g/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/1/docs/projects/capstone/Modeling/time_series/0/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/df_mean.plot(/docs/projects/capstone/Modeling/time_series/figsize=(32,10))
```




    <AxesSubplot:xlabel='Count Date'>




    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_8_1.png)
    



```python
df_mean = df['grass_count'].T.rolling(/docs/projects/capstone/Modeling/time_series/window=10).mean()
/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/[/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/g/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/c/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/u/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/]/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/T/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/g/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/1/docs/projects/capstone/Modeling/time_series/0/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/df_mean.plot(/docs/projects/capstone/Modeling/time_series/figsize=(32,10))
```




    <AxesSubplot:xlabel='Count Date'>




    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_9_1.png)
    


### Time-series decomposition


```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Additive Decomposition
result_add = seasonal_decompose(/docs/projects/capstone/Modeling/time_series/df2017, model='additive')

# Plot
plt.rcParams.update(/docs/projects/capstone/Modeling/time_series/{'figure.figsize': (7,10)})
/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/u/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/u/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/A/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/v/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/D/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/c/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/,/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/z/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/1/docs/projects/capstone/Modeling/time_series/2/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/result_add.plot().suptitle(/docs/projects/capstone/Modeling/time_series/'Additive Decomposition', fontsize=12)
/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/```


    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_11_0.png)
    


unstable with seasonality

### Autocorrelation


```python
from scipy.stats import pearsonr

a = df['grass_count']
a = a.fillna(/docs/projects/capstone/Modeling/time_series/method='bfill')
b = a.shift(/docs/projects/capstone/Modeling/time_series/1)

print(/docs/projects/capstone/Modeling/time_series/pearsonr(a[1:], b[1:]))
```

    PearsonRResult(/docs/projects/capstone/Modeling/time_series/statistic=0.5552438295581152, pvalue=0.0)



```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
fig, axes = plt.subplots(/docs/projects/capstone/Modeling/time_series/2,1)
plot_acf(/docs/projects/capstone/Modeling/time_series/a, ax=axes[0])
plot_pacf(/docs/projects/capstone/Modeling/time_series/a, ax=axes[1])

/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/g/docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/y/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/u/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series//docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/
## lags are on the horizontal
```

    /opt/homebrew/lib/python3.9/site-packages/statsmodels/graphics/tsaplots.py:348: FutureWarning: The default method 'yw' can produce PACF values outside of the [-1,1] interval. After 0.13, the default will change tounadjusted Yule-Walker (/docs/projects/capstone/Modeling/time_series/'ywm'). You can use this method now by setting method=/docs/projects/capstone/Modeling/time_series/'ywm'.
      warnings.warn(



    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_15_1.png)
    


You can observe that the PACF lag 1 is quite significant since is well above the significance line. Lag 2 turns out to be significant as well, managing to cross the significance limit (/docs/projects/capstone/Modeling/time_series/blue region). So take p as 2.

Couple of lags are well above the significance line. So, let’s tentatively fix q as 2.


```python

plt.style.use(/docs/projects/capstone/Modeling/time_series/{'figure.figsize':(13,7)})

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(/docs/projects/capstone/Modeling/time_series/df2018)
plt.title(/docs/projects/capstone/Modeling/time_series/'autorelation')
# 设置坐标文字大小
plt.tick_params(/docs/projects/capstone/Modeling/time_series/labelsize=10)

# plt.xticks(/docs/projects/capstone/Modeling/time_series/np.linspace(0, 3650, 7))
plt.yticks(/docs/projects/capstone/Modeling/time_series/np.linspace(-1, 1, 20))
```




    ([<matplotlib.axis.YTick at 0x286290220>,
      <matplotlib.axis.YTick at 0x287bf2e50>,
      <matplotlib.axis.YTick at 0x287bf2460>,
      <matplotlib.axis.YTick at 0x287c27d90>,
      <matplotlib.axis.YTick at 0x287c32880>,
      <matplotlib.axis.YTick at 0x287c380a0>,
      <matplotlib.axis.YTick at 0x287c38760>,
      <matplotlib.axis.YTick at 0x287c3e040>,
      <matplotlib.axis.YTick at 0x287c3e640>,
      <matplotlib.axis.YTick at 0x287c38f10>,
      <matplotlib.axis.YTick at 0x287c32d60>,
      <matplotlib.axis.YTick at 0x287c3e6a0>,
      <matplotlib.axis.YTick at 0x287c46490>,
      <matplotlib.axis.YTick at 0x287c46be0>,
      <matplotlib.axis.YTick at 0x287c4c370>,
      <matplotlib.axis.YTick at 0x287c4cac0>,
      <matplotlib.axis.YTick at 0x287c4c730>,
      <matplotlib.axis.YTick at 0x287c464c0>,
      <matplotlib.axis.YTick at 0x287c550a0>,
      <matplotlib.axis.YTick at 0x287c55760>],
     [Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, ''),
      Text(/docs/projects/capstone/Modeling/time_series/0, 0, '')])




    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_17_1.png)
    


### Stationary


```python
from statsmodels.tsa.stattools import adfuller
a = df['Count']
a = a.fillna(/docs/projects/capstone/Modeling/time_series/method='bfill')
print(adfuller(a,   
				maxlag=None, 
				regression='c', 
				autolag='AIC',   # automatically choose best lag in [0, 1,...,maxlag] method；
				store=False, 
				regresults=False)
				)


```

    (/docs/projects/capstone/Modeling/time_series/-4.03425679820627, 0.0012413820990973753, 6, 423, {'1%': -3.4459042013025836, '5%': -2.8683970525583358, '10%': -2.5704225783970176}, 4229.037182676861)


p-value is 0.0012413820990973753, so it is stationary

## simple exponential smoothing


```python
from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing, Holt

/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/1/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/D/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/F/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/df1['ds']=df['Count Date']
df1['y']=df['grass_count']
/docs/projects/capstone/Modeling/time_series/df1['ds'] = pd.to_datetime(/docs/projects/capstone/Modeling/time_series/df1['ds'])
df1 = df1.set_index(/docs/projects/capstone/Modeling/time_series/'ds')
# split data
train_data=df1[0:310]
test_data=df1[304:]
/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/
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
      <th>y</th>
    </tr>
    <tr>
      <th>ds</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2000-11-01</th>
      <td>90.0</td>
    </tr>
    <tr>
      <th>2000-11-02</th>
      <td>68.0</td>
    </tr>
    <tr>
      <th>2000-11-03</th>
      <td>27.0</td>
    </tr>
    <tr>
      <th>2000-11-04</th>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2000-11-05</th>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.plot(/docs/projects/capstone/Modeling/time_series/marker='o', color='blueviolet',legend=True)
fit1 = SimpleExpSmoothing(/docs/projects/capstone/Modeling/time_series/train_data).fit(smoothing_level=0.2,optimized=False)
fit1 = SimpleExpSmoothing(train_data).fit(/docs/projects/capstone/Modeling/time_series/smoothing_level=0.2,optimized=False)
fcast1 = fit1.forecast(/docs/projects/capstone/Modeling/time_series/300).rename(r'$\alpha=0.2$')
fcast1 = fit1.forecast(300).rename(/docs/projects/capstone/Modeling/time_series/r'$\alpha=0.2$')
# plot
fcast1.plot(/docs/projects/capstone/Modeling/time_series/marker='o', color='blue', legend=True)
fit1.fittedvalues.plot(/docs/projects/capstone/Modeling/time_series/marker='o',  color='blue')



fit2 = SimpleExpSmoothing(/docs/projects/capstone/Modeling/time_series/train_data).fit(smoothing_level=0.6,optimized=False)
fit2 = SimpleExpSmoothing(train_data).fit(/docs/projects/capstone/Modeling/time_series/smoothing_level=0.6,optimized=False)
fcast2 = fit2.forecast(/docs/projects/capstone/Modeling/time_series/len(test_data)).rename(r'$\alpha=0.6$')
fcast2 = fit2.forecast(len(test_data)).rename(/docs/projects/capstone/Modeling/time_series/r'$\alpha=0.6$')
# plot
fcast2.plot(/docs/projects/capstone/Modeling/time_series/marker='o', color='red',legend=True)
fit2.fittedvalues.plot(/docs/projects/capstone/Modeling/time_series/marker='o', color='red')


fit3 = SimpleExpSmoothing(/docs/projects/capstone/Modeling/time_series/train_data).fit()
/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/3/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/S/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/E/docs/projects/capstone/Modeling/time_series/x/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/S/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/g/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/fcast3 = fit3.forecast(/docs/projects/capstone/Modeling/time_series/len(test_data)).rename(r'$\alpha=%s$'%fit3.model.params['smoothing_level'])
fcast3 = fit3.forecast(len(test_data)).rename(/docs/projects/capstone/Modeling/time_series/r'$\alpha=%s$'%fit3.model.params['smoothing_level'])
# plot
fcast3.plot(/docs/projects/capstone/Modeling/time_series/marker='o', color='green', legend=True)
fit3.fittedvalues.plot(/docs/projects/capstone/Modeling/time_series/marker='o', color='green')

/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/```

    /opt/homebrew/lib/python3.9/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(/docs/projects/capstone/Modeling/time_series/dates, freq)
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(/docs/projects/capstone/Modeling/time_series/dates, freq)
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(/docs/projects/capstone/Modeling/time_series/dates, freq)



    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_23_1.png)
    


## Hold's method


```python
df1.plot(/docs/projects/capstone/Modeling/time_series/marker='o', color='blueviolet',legend=True)
fit1 = Holt(/docs/projects/capstone/Modeling/time_series/train_data).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fit1 = Holt(train_data).fit(/docs/projects/capstone/Modeling/time_series/smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast1 = fit1.forecast(/docs/projects/capstone/Modeling/time_series/120).rename("Holt's linear trend")
fcast1 = fit1.forecast(120).rename(/docs/projects/capstone/Modeling/time_series/"Holt's linear trend")
fit1.fittedvalues.plot(/docs/projects/capstone/Modeling/time_series/marker="o", color='blue')
fcast1.plot(/docs/projects/capstone/Modeling/time_series/color='blue', marker="o", legend=True)

fit3 = Holt(/docs/projects/capstone/Modeling/time_series/train_data, damped=True).fit(smoothing_level=0.8, smoothing_slope=0.2)
fit3 = Holt(train_data, damped=True).fit(/docs/projects/capstone/Modeling/time_series/smoothing_level=0.8, smoothing_slope=0.2)
fcast3 = fit3.forecast(/docs/projects/capstone/Modeling/time_series/12).rename("Additive damped trend")
fcast3 = fit3.forecast(12).rename(/docs/projects/capstone/Modeling/time_series/"Additive damped trend")
fit3.fittedvalues.plot(/docs/projects/capstone/Modeling/time_series/marker="o", color='green')
fcast3.plot(/docs/projects/capstone/Modeling/time_series/color='green', marker="o", legend=True)

/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/```

    /opt/homebrew/lib/python3.9/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(/docs/projects/capstone/Modeling/time_series/dates, freq)
    /var/folders/q1/7fhq9ggs78g6wf7c38dhzh9h0000gn/T/ipykernel_78036/1622727271.py:2: FutureWarning: the 'smoothing_slope'' keyword is deprecated, use 'smoothing_trend' instead.
      fit1 = Holt(/docs/projects/capstone/Modeling/time_series/train_data).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
      fit1 = Holt(train_data).fit(/docs/projects/capstone/Modeling/time_series/smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
    /var/folders/q1/7fhq9ggs78g6wf7c38dhzh9h0000gn/T/ipykernel_78036/1622727271.py:7: FutureWarning: the 'damped'' keyword is deprecated, use 'damped_trend' instead.
      fit3 = Holt(/docs/projects/capstone/Modeling/time_series/train_data, damped=True).fit(smoothing_level=0.8, smoothing_slope=0.2)
      fit3 = Holt(train_data, damped=True).fit(/docs/projects/capstone/Modeling/time_series/smoothing_level=0.8, smoothing_slope=0.2)
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(/docs/projects/capstone/Modeling/time_series/dates, freq)
    /var/folders/q1/7fhq9ggs78g6wf7c38dhzh9h0000gn/T/ipykernel_78036/1622727271.py:7: FutureWarning: the 'smoothing_slope'' keyword is deprecated, use 'smoothing_trend' instead.
      fit3 = Holt(/docs/projects/capstone/Modeling/time_series/train_data, damped=True).fit(smoothing_level=0.8, smoothing_slope=0.2)
      fit3 = Holt(train_data, damped=True).fit(/docs/projects/capstone/Modeling/time_series/smoothing_level=0.8, smoothing_slope=0.2)



    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_25_1.png)
    


## Holt's winters


```python
df1.plot(/docs/projects/capstone/Modeling/time_series/marker='o', color='blueviolet',legend=True)
fit1 = ExponentialSmoothing(/docs/projects/capstone/Modeling/time_series/train_data, seasonal_periods=4, trend='add', seasonal='add').fit()
/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/1/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/E/docs/projects/capstone/Modeling/time_series/x/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/S/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/g/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/,/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/4/docs/projects/capstone/Modeling/time_series/,/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/,/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/# fit3 = ExponentialSmoothing(/docs/projects/capstone/Modeling/time_series/train_data, seasonal_periods=4, trend='add', seasonal='add', damped=True).fit()
/docs/projects/capstone/Modeling/time_series/#/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/3/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/E/docs/projects/capstone/Modeling/time_series/x/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/S/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/g/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/,/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/4/docs/projects/capstone/Modeling/time_series/,/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/,/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/,/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/T/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/u/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/fit1.fittedvalues.plot(/docs/projects/capstone/Modeling/time_series/style='--', color='red')
fit1.forecast(/docs/projects/capstone/Modeling/time_series/120).plot(style='--', marker='o', color='red', legend=True)
fit1.forecast(120).plot(/docs/projects/capstone/Modeling/time_series/style='--', marker='o', color='red', legend=True)
/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/```

    /opt/homebrew/lib/python3.9/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(/docs/projects/capstone/Modeling/time_series/dates, freq)



    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_27_1.png)
    


## ARIMA



```python
from statsmodels.tsa.arima.model import ARIMA

# 2,0,2 ARIMA Model
model = ARIMA(/docs/projects/capstone/Modeling/time_series/df1, order=(3,0,1))#p,d,q
/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/print(/docs/projects/capstone/Modeling/time_series/model_fit.summary())
```

    /opt/homebrew/lib/python3.9/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(/docs/projects/capstone/Modeling/time_series/dates, freq)
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(/docs/projects/capstone/Modeling/time_series/dates, freq)
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(/docs/projects/capstone/Modeling/time_series/dates, freq)
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/tsa/statespace/sarimax.py:966: UserWarning: Non-stationary starting autoregressive parameters found. Using zeros as starting parameters.
      warn('Non-stationary starting autoregressive parameters'
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/tsa/statespace/sarimax.py:978: UserWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.
      warn('Non-invertible starting MA parameters found.'


                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 7701
    Model:                 ARIMA(/docs/projects/capstone/Modeling/time_series/3, 0, 1)   Log Likelihood              -35188.301
    Date:                Wed, 12 Oct 2022   AIC                          70388.601
    Time:                        12:24:40   BIC                          70430.296
    Sample:                    01-02-2000   HQIC                         70402.900
                             - 01-31-2021                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          9.3870      5.518      1.701      0.089      -1.429      20.203
    ar.L1          1.1927      0.006    191.803      0.000       1.181       1.205
    ar.L2         -0.3479      0.006    -53.540      0.000      -0.361      -0.335
    ar.L3          0.1374      0.005     27.229      0.000       0.128       0.147
    ma.L1         -0.8400      0.005   -163.338      0.000      -0.850      -0.830
    sigma2       544.9080      2.394    227.659      0.000     540.217     549.599
    ===================================================================================
    Ljung-Box (/docs/projects/capstone/Modeling/time_series/L1) (Q):                   0.61   Jarque-Bera (JB):            750889.23
    Ljung-Box (L1) (/docs/projects/capstone/Modeling/time_series/Q):                   0.61   Jarque-Bera (JB):            750889.23
    Ljung-Box (L1) (Q):                   0.61   Jarque-Bera (/docs/projects/capstone/Modeling/time_series/JB):            750889.23
    Prob(/docs/projects/capstone/Modeling/time_series/Q):                              0.43   Prob(JB):                         0.00
    Prob(Q):                              0.43   Prob(/docs/projects/capstone/Modeling/time_series/JB):                         0.00
    /docs/projects/capstone/Modeling/time_series/Heteroskedasticity (/docs/projects/capstone/Modeling/time_series/H):               1.04   Skew:                             4.57
    Prob(/docs/projects/capstone/Modeling/time_series/H) (two-sided):                  0.29   Kurtosis:                        50.50
    Prob(H) (/docs/projects/capstone/Modeling/time_series/two-sided):                  0.29   Kurtosis:                        50.50
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (/docs/projects/capstone/Modeling/time_series/complex-step).



```python
import pmdarima as pm
/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/model = pm.auto_arima(df1, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=8, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(/docs/projects/capstone/Modeling/time_series/model.summary())
```

    Performing stepwise search to minimize aic
     ARIMA(/docs/projects/capstone/Modeling/time_series/1,0,1)(0,0,0)[0]             : AIC=70806.920, Time=0.84 sec
     ARIMA(1,0,1)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0]             : AIC=70806.920, Time=0.84 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/0,0,0)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0]             : AIC=75108.108, Time=0.07 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/0,0,0)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0]             : AIC=75108.108, Time=0.07 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/1,0,0)(0,0,0)[0]             : AIC=71756.761, Time=0.08 sec
     ARIMA(1,0,0)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0]             : AIC=71756.761, Time=0.08 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/0,0,1)(0,0,0)[0]             : AIC=72808.756, Time=0.40 sec
     ARIMA(0,0,1)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0]             : AIC=72808.756, Time=0.40 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/2,0,1)(0,0,0)[0]             : AIC=70520.237, Time=1.06 sec
     ARIMA(2,0,1)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0]             : AIC=70520.237, Time=1.06 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/2,0,0)(0,0,0)[0]             : AIC=71604.928, Time=0.26 sec
     ARIMA(2,0,0)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0]             : AIC=71604.928, Time=0.26 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/3,0,1)(0,0,0)[0]             : AIC=70399.024, Time=1.09 sec
     ARIMA(3,0,1)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0]             : AIC=70399.024, Time=1.09 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/3,0,0)(0,0,0)[0]             : AIC=71316.743, Time=0.32 sec
     ARIMA(3,0,0)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0]             : AIC=71316.743, Time=0.32 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/3,0,2)(0,0,0)[0]             : AIC=70479.297, Time=1.58 sec
     ARIMA(3,0,2)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0]             : AIC=70479.297, Time=1.58 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/2,0,2)(0,0,0)[0]             : AIC=70451.165, Time=1.06 sec
     ARIMA(2,0,2)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0]             : AIC=70451.165, Time=1.06 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/3,0,1)(0,0,0)[0] intercept   : AIC=70388.600, Time=2.60 sec
     ARIMA(3,0,1)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0] intercept   : AIC=70388.600, Time=2.60 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/2,0,1)(0,0,0)[0] intercept   : AIC=70510.479, Time=1.87 sec
     ARIMA(2,0,1)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0] intercept   : AIC=70510.479, Time=1.87 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/3,0,0)(0,0,0)[0] intercept   : AIC=71217.808, Time=0.48 sec
     ARIMA(3,0,0)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0] intercept   : AIC=71217.808, Time=0.48 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/3,0,2)(0,0,0)[0] intercept   : AIC=70469.505, Time=2.75 sec
     ARIMA(3,0,2)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0] intercept   : AIC=70469.505, Time=2.75 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/2,0,0)(0,0,0)[0] intercept   : AIC=71460.298, Time=0.36 sec
     ARIMA(2,0,0)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0] intercept   : AIC=71460.298, Time=0.36 sec
     ARIMA(/docs/projects/capstone/Modeling/time_series/2,0,2)(0,0,0)[0] intercept   : AIC=70441.285, Time=2.16 sec
     ARIMA(2,0,2)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0] intercept   : AIC=70441.285, Time=2.16 sec
    
    Best model:  ARIMA(/docs/projects/capstone/Modeling/time_series/3,0,1)(0,0,0)[0] intercept
    Best model:  ARIMA(3,0,1)(/docs/projects/capstone/Modeling/time_series/0,0,0)[0] intercept
    Total fit time: 17.305 seconds
                                   SARIMAX Results                                
    ==============================================================================
    Dep. Variable:                      y   No. Observations:                 7701
    Model:               SARIMAX(/docs/projects/capstone/Modeling/time_series/3, 0, 1)   Log Likelihood              -35188.300
    Date:                Wed, 12 Oct 2022   AIC                          70388.600
    Time:                        12:21:07   BIC                          70430.294
    Sample:                    01-02-2000   HQIC                         70402.898
                             - 01-31-2021                                         
    Covariance Type:                  opg                                         
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    intercept      0.1650      0.112      1.477      0.140      -0.054       0.384
    ar.L1          1.1927      0.006    191.776      0.000       1.181       1.205
    ar.L2         -0.3478      0.006    -53.534      0.000      -0.361      -0.335
    ar.L3          0.1374      0.005     27.227      0.000       0.128       0.147
    ma.L1         -0.8400      0.005   -163.305      0.000      -0.850      -0.830
    sigma2       544.9156      2.394    227.657      0.000     540.224     549.607
    ===================================================================================
    Ljung-Box (/docs/projects/capstone/Modeling/time_series/L1) (Q):                   0.61   Jarque-Bera (JB):            750912.58
    Ljung-Box (L1) (/docs/projects/capstone/Modeling/time_series/Q):                   0.61   Jarque-Bera (JB):            750912.58
    Ljung-Box (L1) (Q):                   0.61   Jarque-Bera (/docs/projects/capstone/Modeling/time_series/JB):            750912.58
    Prob(/docs/projects/capstone/Modeling/time_series/Q):                              0.43   Prob(JB):                         0.00
    Prob(Q):                              0.43   Prob(/docs/projects/capstone/Modeling/time_series/JB):                         0.00
    /docs/projects/capstone/Modeling/time_series/Heteroskedasticity (/docs/projects/capstone/Modeling/time_series/H):               1.04   Skew:                             4.57
    Prob(/docs/projects/capstone/Modeling/time_series/H) (two-sided):                  0.29   Kurtosis:                        50.51
    Prob(H) (/docs/projects/capstone/Modeling/time_series/two-sided):                  0.29   Kurtosis:                        50.51
    ===================================================================================
    
    Warnings:
    [1] Covariance matrix calculated using the outer product of gradients (/docs/projects/capstone/Modeling/time_series/complex-step).



```python
model.plot_diagnostics(/docs/projects/capstone/Modeling/time_series/figsize=(7,5))
/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/```


    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_31_0.png)
    


Top left: The residual errors seem to fluctuate around a mean of zero and have a uniform variance.

Top Right: The density plot suggest normal distribution with mean near zero.

Bottom left: Some dots don't fall in line with the red line. Any significant deviations would imply the distribution is skewed.

Bottom Right: The Correlogram, aka, ACF plot shows the residual errors are not autocorrelated. Any autocorrelation would imply that there is some pattern in the residual errors which are not explained in the model. So you will need to look for more X’s (/docs/projects/capstone/Modeling/time_series/predictors) to the model.


```python
# Build Model
model = ARIMA(/docs/projects/capstone/Modeling/time_series/train_data, order=(3, 0, 1))  
/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/print(/docs/projects/capstone/Modeling/time_series/fitted.summary)
```

    /opt/homebrew/lib/python3.9/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(/docs/projects/capstone/Modeling/time_series/dates, freq)
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(/docs/projects/capstone/Modeling/time_series/dates, freq)
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/tsa/base/tsa_model.py:471: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      self._init_dates(/docs/projects/capstone/Modeling/time_series/dates, freq)


    <bound method SARIMAXResults.summary of <statsmodels.tsa.arima.model.ARIMAResults object at 0x29d2f5490>>



```python
from pmdarima.model_selection import train_test_split

# Load/split your data
train, test = train_test_split(/docs/projects/capstone/Modeling/time_series/df1, train_size=7575)
# Fit your model
model = pm.auto_arima(/docs/projects/capstone/Modeling/time_series/train, seasonal=True, m=12)
# make your forecasts
forecasts = model.predict(/docs/projects/capstone/Modeling/time_series/len(test))  # predict N steps into the future

```


    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_34_0.png)
    



```python
# Visualize the forecasts (/docs/projects/capstone/Modeling/time_series/blue=train, green=forecasts)
f, ax = plt.subplots(/docs/projects/capstone/Modeling/time_series/1)
f.set_figheight(/docs/projects/capstone/Modeling/time_series/5)
f.set_figwidth(/docs/projects/capstone/Modeling/time_series/18)

# fig = model.plot(/docs/projects/capstone/Modeling/time_series/forecasts, ax=ax)
# x = np.arange(/docs/projects/capstone/Modeling/time_series/len(df1))
x=df1.index
plt.plot(/docs/projects/capstone/Modeling/time_series/ x[-126:],test['y'], color='r')
plt.plot(/docs/projects/capstone/Modeling/time_series/x[:-126], train, c='blue')
plt.plot(/docs/projects/capstone/Modeling/time_series/x[-126:], forecasts, c='green')
/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/```


    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_35_0.png)
    


# Prophet model


```python
from fbprophet import Prophet

/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/1/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/D/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/F/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/df1['ds']=df['Count Date']
df1['y']=df['Count']
/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/1/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/```




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
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-09-26</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-09-27</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-09-28</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-09-29</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-09-30</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# split data
train_data=df1[0:310]
test_data=df1[311:]
# fit model
/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/P/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/m.fit(/docs/projects/capstone/Modeling/time_series/train_data)
```

    INFO:fbprophet:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
    /opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)


    Initial log joint probability = -5.91056
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
          79       520.778   1.49117e-08       73.8555      0.4369      0.4369      103   
    Optimization terminated normally: 
      Convergence detected: relative gradient magnitude is below tolerance





    <fbprophet.forecaster.Prophet at 0x10efa80d0>




```python
# forecast the following 365 days
future = m.make_future_dataframe(/docs/projects/capstone/Modeling/time_series/periods=365)
forecast = m.predict(/docs/projects/capstone/Modeling/time_series/future)
```

    /opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)
    /opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)



```python
fig1 = m.plot(/docs/projects/capstone/Modeling/time_series/forecast)
fig2 = m.plot_components(/docs/projects/capstone/Modeling/time_series/forecast)
```

    /opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)
    /opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)



    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_40_1.png)
    



    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_40_2.png)
    



```python
from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(/docs/projects/capstone/Modeling/time_series/m, initial='300 days', period='80 days', horizon = '200 days')
```

    INFO:fbprophet:Making 5 forecasts with cutoffs between 2019-04-30 00:00:00 and 2020-03-15 00:00:00
    WARNING:fbprophet:Seasonality has period of 365.25 days which is larger than initial window. Consider increasing initial.
      0%|          | 0/5 [00:00<?, ?it/s]/opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)
    /opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)
    /opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)


    Initial log joint probability = -8.92491
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
          99       222.436    2.6595e-07       91.3855      0.5156      0.5156      122   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         114       222.436     5.982e-09       95.6543      0.3475      0.3475      144   
    Optimization terminated normally: 
      Convergence detected: absolute parameter change was below tolerance


     20%|██        | 1/5 [00:00<00:02,  1.95it/s]/opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)
    /opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)
    /opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)


    Initial log joint probability = -8.92491
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
          99       222.436    2.6595e-07       91.3855      0.5156      0.5156      122   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         114       222.436     5.982e-09       95.6543      0.3475      0.3475      144   
    Optimization terminated normally: 
      Convergence detected: absolute parameter change was below tolerance


     40%|████      | 2/5 [00:10<00:17,  5.86s/it]/opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)
    /opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)
    /opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)


    Initial log joint probability = -8.50317
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
          99       280.555   1.54573e-06       97.6026           1           1      139   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         114       280.604    0.00063059       100.179   5.684e-06       0.001      199  LS failed, Hessian reset 
         192       280.645   6.61465e-09       91.1806      0.3919           1      304   
    Optimization terminated normally: 
      Convergence detected: absolute parameter change was below tolerance


     60%|██████    | 3/5 [00:10<00:06,  3.44s/it]/opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)
    /opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)
    /opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)


    Initial log joint probability = -5.29624
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
          99       518.732   5.74391e-05       87.0682           1           1      130   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         160       518.747   3.38075e-08       78.8067           1           1      213   
    Optimization terminated normally: 
      Convergence detected: relative gradient magnitude is below tolerance


     80%|████████  | 4/5 [00:11<00:02,  2.26s/it]/opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)
    /opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)
    /opt/homebrew/lib/python3.9/site-packages/fbprophet/forecaster.py:891: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      components = components.append(/docs/projects/capstone/Modeling/time_series/new_comp)


    Initial log joint probability = -4.90379
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
          93       517.096   5.78554e-09       77.0077     0.06107           1      129   
    Optimization terminated normally: 
      Convergence detected: absolute parameter change was below tolerance


    100%|██████████| 5/5 [00:11<00:00,  2.33s/it]



```python
from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(/docs/projects/capstone/Modeling/time_series/df_cv)
/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/```

    INFO:fbprophet:Skipping MAPE because y close to 0





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
      <th>horizon</th>
      <th>mse</th>
      <th>rmse</th>
      <th>mae</th>
      <th>mdape</th>
      <th>coverage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25 days</td>
      <td>2167.082374</td>
      <td>46.551932</td>
      <td>30.467985</td>
      <td>1.174400</td>
      <td>0.714286</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26 days</td>
      <td>2194.676976</td>
      <td>46.847380</td>
      <td>31.102259</td>
      <td>0.884694</td>
      <td>0.714286</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27 days</td>
      <td>2214.642132</td>
      <td>47.059984</td>
      <td>31.605025</td>
      <td>0.884694</td>
      <td>0.714286</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28 days</td>
      <td>1718.767043</td>
      <td>41.458015</td>
      <td>29.236320</td>
      <td>1.282265</td>
      <td>0.732143</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29 days</td>
      <td>1242.364808</td>
      <td>35.247196</td>
      <td>27.151282</td>
      <td>1.282265</td>
      <td>0.714286</td>
    </tr>
  </tbody>
</table>
</div>



# VAR model


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
```


```python
/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/=/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/target = df['grass_count']
print(/docs/projects/capstone/Modeling/time_series/'X.shape:', df.shape)
 
 
# select feature by person coefficient
X = np.array(/docs/projects/capstone/Modeling/time_series/df)
Y = np.array(/docs/projects/capstone/Modeling/time_series/target)
skb = SelectKBest(/docs/projects/capstone/Modeling/time_series/score_func=f_regression, k=7)
skb.fit(/docs/projects/capstone/Modeling/time_series/X, Y.ravel())
print(/docs/projects/capstone/Modeling/time_series/'selected features:', [df.columns[i] for i in skb.get_support(indices = True)])
/docs/projects/capstone/Modeling/time_series/X_selected = skb.transform(/docs/projects/capstone/Modeling/time_series/X)
print(/docs/projects/capstone/Modeling/time_series/'X_selected.shape:', X_selected.shape)


```

    X.shape: (/docs/projects/capstone/Modeling/time_series/7695, 172)
    selected features: ['av_swsfcdown', 'av_swsfcdown_numhours_1D', 'av_swsfcdown_numhours_10D', 'av_swsfcdown_sum_10D', 'forcing_temp_numhours_180D', 'forcing_temp_sum_180D', 'grass_count']
    X_selected.shape: (/docs/projects/capstone/Modeling/time_series/7695, 7)



```python
X_selected=pd.DataFrame(/docs/projects/capstone/Modeling/time_series/X_selected,columns=[df.columns[i] for i in skb.get_support(indices = True)])
X_selected
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
      <th>av_swsfcdown</th>
      <th>av_swsfcdown_numhours_1D</th>
      <th>av_swsfcdown_numhours_10D</th>
      <th>av_swsfcdown_sum_10D</th>
      <th>forcing_temp_numhours_180D</th>
      <th>forcing_temp_sum_180D</th>
      <th>grass_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>411.645000</td>
      <td>16.0</td>
      <td>21.0</td>
      <td>14612.296875</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>333.087500</td>
      <td>16.0</td>
      <td>36.0</td>
      <td>22118.546875</td>
      <td>1.0</td>
      <td>0.553125</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>274.170000</td>
      <td>16.0</td>
      <td>51.0</td>
      <td>28420.921875</td>
      <td>1.0</td>
      <td>0.553125</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>309.916250</td>
      <td>16.0</td>
      <td>66.0</td>
      <td>35528.796875</td>
      <td>1.0</td>
      <td>0.553125</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>344.033125</td>
      <td>16.0</td>
      <td>81.0</td>
      <td>43629.343750</td>
      <td>1.0</td>
      <td>0.553125</td>
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
    </tr>
    <tr>
      <th>7690</th>
      <td>130.549167</td>
      <td>15.0</td>
      <td>152.0</td>
      <td>71764.820016</td>
      <td>113.0</td>
      <td>341.590070</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7691</th>
      <td>334.555417</td>
      <td>15.0</td>
      <td>152.0</td>
      <td>72318.180003</td>
      <td>113.0</td>
      <td>341.590070</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>7692</th>
      <td>132.476798</td>
      <td>16.0</td>
      <td>152.0</td>
      <td>67785.839900</td>
      <td>113.0</td>
      <td>341.590070</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>7693</th>
      <td>86.142084</td>
      <td>15.0</td>
      <td>152.0</td>
      <td>63012.549925</td>
      <td>113.0</td>
      <td>341.590070</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7694</th>
      <td>275.843329</td>
      <td>15.0</td>
      <td>152.0</td>
      <td>60367.329937</td>
      <td>113.0</td>
      <td>341.590070</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>7695 rows × 7 columns</p>
</div>



## Grangers Causality test


```python
from statsmodels.tsa.stattools import grangercausalitytests

maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(/docs/projects/capstone/Modeling/time_series/data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (/docs/projects/capstone/Modeling/time_series/0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(/docs/projects/capstone/Modeling/time_series/np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    df = pd.DataFrame(np.zeros((len(/docs/projects/capstone/Modeling/time_series/variables), len(/docs/projects/capstone/Modeling/time_series/variables))), columns=/docs/projects/capstone/Modeling/time_series/variables, index=/docs/projects/capstone/Modeling/time_series/variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(/docs/projects/capstone/Modeling/time_series/data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(/docs/projects/capstone/Modeling/time_series/test_result[i+1][0][test][1],4) for i in range(maxlag)]
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(/docs/projects/capstone/Modeling/time_series/maxlag)]
            if verbose: print(/docs/projects/capstone/Modeling/time_series/f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(/docs/projects/capstone/Modeling/time_series/p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index=[var for var in variables]
    # df.drop(/docs/projects/capstone/Modeling/time_series/[i for i in variables if i!='grass_count'])
    # df=df.drop(/docs/projects/capstone/Modeling/time_series/[1,2,3,4,5,6])
    return df

gm=grangers_causation_matrix(/docs/projects/capstone/Modeling/time_series/X_selected, variables = X_selected.columns)
```

    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 2, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 3, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 4, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 5, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 6, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 2, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 3, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 4, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 5, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 6, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 7, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 8, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 9, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 10, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 11, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 12, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 2, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 3, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 4, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 5, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 6, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 7, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 8, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 9, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 10, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 11, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 12, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 2, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 3, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 4, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 6, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 2, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 3, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 4, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 3, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 5, but rank is 1
      warnings.warn('covariance of constraints does not have full '
    /opt/homebrew/lib/python3.9/site-packages/statsmodels/base/model.py:1871: ValueWarning: covariance of constraints does not have full rank. The number of constraints is 7, but rank is 1
      warnings.warn('covariance of constraints does not have full '



```python
gm=gm.drop(/docs/projects/capstone/Modeling/time_series/[i for i in X_selected.columns if i!='grass_count'])

```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (/docs/projects/capstone/Modeling/time_series/most recent call last)

/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series///docs/projects/capstone/Modeling/time_series/U/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series///docs/projects/capstone/Modeling/time_series/D/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series///docs/projects/capstone/Modeling/time_series/D/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/k/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series///docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series///docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/y/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/b/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/C/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/5/docs/projects/capstone/Modeling/time_series/0/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/</docs/projects/capstone/Modeling/time_series/c/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/:/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/1/docs/projects/capstone/Modeling/time_series/>/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/    ----> <a href='vscode-notebook-cell:/Users/Demi/Desktop/pollen%20data/time%20series.ipynb#Y146sZmlsZQ%3D%3D?line=0'>1</a> gm=gm.drop(/docs/projects/capstone/Modeling/time_series/[i for i in X_selected.columns if i!='grass_count'])
          <a href='vscode-notebook-cell:/Users/Demi/Desktop/pollen%20data/time%20series.ipynb#Y146sZmlsZQ%3D%3D?line=1'>2</a> print(/docs/projects/capstone/Modeling/time_series/gm)


    File /opt/homebrew/lib/python3.9/site-packages/pandas/util/_decorators.py:311, in deprecate_nonkeyword_arguments.<locals>.decorate.<locals>.wrapper(/docs/projects/capstone/Modeling/time_series/*args, **kwargs)
        305 if len(/docs/projects/capstone/Modeling/time_series/args) > num_allow_/docs/projects/capstone/Modeling/time_series/args:
        306     warnings.warn(
        307         msg.format(/docs/projects/capstone/Modeling/time_series/arguments=arguments),
        308         FutureWarning,
        309         stacklevel=stacklevel,
        310     )
    --> 311 return func(/docs/projects/capstone/Modeling/time_series/*args, **kwargs)


    File /opt/homebrew/lib/python3.9/site-packages/pandas/core/frame.py:4954, in DataFrame.drop(/docs/projects/capstone/Modeling/time_series/self, labels, axis, index, columns, level, inplace, errors)
       4806 @deprecate_nonkeyword_arguments(/docs/projects/capstone/Modeling/time_series/version=None, allowed_args=["self", "labels"])
       4807 def drop(
       4808     self,
       (/docs/projects/capstone/Modeling/time_series/...)
       4815     errors: str = "raise",
       4816 ):
       4817     """
       4818     Drop specified labels from rows or columns.
       4819 
       (/docs/projects/capstone/Modeling/time_series/...)
       4952             weight  1.0     0.8
       4953     """
/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/-/docs/projects/capstone/Modeling/time_series/>/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/4/docs/projects/capstone/Modeling/time_series/9/docs/projects/capstone/Modeling/time_series/5/docs/projects/capstone/Modeling/time_series/4/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/u/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/u/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/       4955         labels=labels,
       4956         axis=axis,
       4957         index=index,
       4958         columns=columns,
       4959         level=level,
       4960         inplace=inplace,
       4961         errors=errors,
       4962     )


    File /opt/homebrew/lib/python3.9/site-packages/pandas/core/generic.py:4267, in NDFrame.drop(/docs/projects/capstone/Modeling/time_series/self, labels, axis, index, columns, level, inplace, errors)
/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/4/docs/projects/capstone/Modeling/time_series/2/docs/projects/capstone/Modeling/time_series/6/docs/projects/capstone/Modeling/time_series/5/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/x/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/,/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/b/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/x/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/:/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/       4266     if labels is not None:
    -> 4267         obj = obj._drop_axis(/docs/projects/capstone/Modeling/time_series/labels, axis, level=level, errors=errors)
       4269 if inplace:
       4270     self._update_inplace(/docs/projects/capstone/Modeling/time_series/obj)


    File /opt/homebrew/lib/python3.9/site-packages/pandas/core/generic.py:4311, in NDFrame._drop_axis(/docs/projects/capstone/Modeling/time_series/self, labels, axis, level, errors, consolidate, only_slice)
       4309         new_axis = axis.drop(/docs/projects/capstone/Modeling/time_series/labels, level=level, errors=errors)
       4310     else:
    -> 4311         new_axis = axis.drop(/docs/projects/capstone/Modeling/time_series/labels, errors=errors)
       4312     indexer = axis.get_indexer(/docs/projects/capstone/Modeling/time_series/new_axis)
       4314 # Case for non-unique axis
       4315 else:


    File /opt/homebrew/lib/python3.9/site-packages/pandas/core/indexes/base.py:6644, in Index.drop(/docs/projects/capstone/Modeling/time_series/self, labels, errors)
/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/6/docs/projects/capstone/Modeling/time_series/6/docs/projects/capstone/Modeling/time_series/4/docs/projects/capstone/Modeling/time_series/2/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/k/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/y/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/:/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/       6643     if errors != "ignore":
    -> 6644         raise KeyError(/docs/projects/capstone/Modeling/time_series/f"{list(labels[mask])} not found in axis")
       6645     indexer = indexer[~mask]
       6646 return self.delete(/docs/projects/capstone/Modeling/time_series/indexer)


    KeyError: "['av_swsfcdown', 'av_swsfcdown_numhours_1D', 'av_swsfcdown_numhours_10D', 'av_swsfcdown_sum_10D', 'forcing_temp_numhours_180D', 'forcing_temp_sum_180D'] not found in axis"



```python
pd.set_option(/docs/projects/capstone/Modeling/time_series/'display.max_colwidth',50)
print(/docs/projects/capstone/Modeling/time_series/gm)
```

                 av_swsfcdown_x  av_swsfcdown_numhours_1D_x  \
    grass_count             0.0                         0.0   
    
                 av_swsfcdown_numhours_10D_x  av_swsfcdown_sum_10D_x  \
    grass_count                          0.0                     0.0   
    
                 forcing_temp_numhours_180D_x  forcing_temp_sum_180D_x  \
    grass_count                           0.0                      0.0   
    
                 grass_count_x  
    grass_count            1.0  


If a given p-value is < 0.05, then, the corresponding X series (/docs/projects/capstone/Modeling/time_series/column) causes the Y (row).
If a given p-value is < 0.05, then, the corresponding X series (column) causes the Y (/docs/projects/capstone/Modeling/time_series/row).
Almost all the variables in the system are interchangeably causing each other except 9 values > 0.05

## Cointegration test

the presence of a statistically significant connection between two or more time series


```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(/docs/projects/capstone/Modeling/time_series/df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(/docs/projects/capstone/Modeling/time_series/df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(/docs/projects/capstone/Modeling/time_series/1-alpha)]]
    def adjust(/docs/projects/capstone/Modeling/time_series/val, length= 6): return str(val).ljust(length)
    def adjust(/docs/projects/capstone/Modeling/time_series/val, length= 6): return str(/docs/projects/capstone/Modeling/time_series/val).ljust(length)
    def adjust(val, /docs/projects/capstone/Modeling/time_series/length= 6): return str(val).ljust(/docs/projects/capstone/Modeling/time_series/length)

    # Summary
    print(/docs/projects/capstone/Modeling/time_series/'Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(/docs/projects/capstone/Modeling/time_series/df.columns, traces, cvts):
        print(/docs/projects/capstone/Modeling/time_series/adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
        print(adjust(col), ':: ', adjust(/docs/projects/capstone/Modeling/time_series/round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(/docs/projects/capstone/Modeling/time_series/cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(/docs/projects/capstone/Modeling/time_series/X_selected)

```

    Name   ::  Test Stat > C(/docs/projects/capstone/Modeling/time_series/95%)    =>   Signif  
     ----------------------------------------
    av_swsfcdown ::  4708.36   > 111.7797  =>   True
    av_swsfcdown_numhours_1D ::  2074.2    > 83.9383   =>   True
    av_swsfcdown_numhours_10D ::  1280.23   > 60.0627   =>   True
    av_swsfcdown_sum_10D ::  648.77    > 40.1749   =>   True
    forcing_temp_numhours_180D ::  300.54    > 24.2761   =>   True
    forcing_temp_sum_180D ::  25.78     > 12.3212   =>   True
    grass_count ::  0.56      > 4.1296    =>   False



```python
nobs = 200
df_train, df_test = X_selected[0:-nobs], X_selected[-nobs:]

# Check size
print(/docs/projects/capstone/Modeling/time_series/df_train.shape)  # (119, 8)
print(df_train.shape)  # (/docs/projects/capstone/Modeling/time_series/119, 8)
print(/docs/projects/capstone/Modeling/time_series/df_test.shape)  # (4, 8)
print(df_test.shape)  # (/docs/projects/capstone/Modeling/time_series/4, 8)
```

    (/docs/projects/capstone/Modeling/time_series/7495, 7)
    (/docs/projects/capstone/Modeling/time_series/200, 7)


### ADF test for stationarity


```python
def adfuller_test(/docs/projects/capstone/Modeling/time_series/series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(/docs/projects/capstone/Modeling/time_series/series, autolag='AIC')
    output = {'test_statistic':round(/docs/projects/capstone/Modeling/time_series/r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(/docs/projects/capstone/Modeling/time_series/r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(/docs/projects/capstone/Modeling/time_series/r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(/docs/projects/capstone/Modeling/time_series/val, length= 6): return str(val).ljust(length)
    def adjust(/docs/projects/capstone/Modeling/time_series/val, length= 6): return str(/docs/projects/capstone/Modeling/time_series/val).ljust(length)
    def adjust(val, /docs/projects/capstone/Modeling/time_series/length= 6): return str(val).ljust(/docs/projects/capstone/Modeling/time_series/length)

    # Print Summary
    print(/docs/projects/capstone/Modeling/time_series/f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(/docs/projects/capstone/Modeling/time_series/f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(/docs/projects/capstone/Modeling/time_series/f' Significance Level    = {signif}')
    print(/docs/projects/capstone/Modeling/time_series/f' Test Statistic        = {output["test_statistic"]}')
    print(/docs/projects/capstone/Modeling/time_series/f' No. Lags Chosen       = {output["n_lags"]}')

/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/k/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/y/docs/projects/capstone/Modeling/time_series/,/docs/projects/capstone/Modeling/time_series/v/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/[/docs/projects/capstone/Modeling/time_series/4/docs/projects/capstone/Modeling/time_series/]/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/:/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/        print(/docs/projects/capstone/Modeling/time_series/f' Critical value {adjust(key)} = {round(val, 3)}')
        print(f' Critical value {adjust(key)} = {round(/docs/projects/capstone/Modeling/time_series/val, 3)}')

    if p_value <= signif:
        print(/docs/projects/capstone/Modeling/time_series/f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(/docs/projects/capstone/Modeling/time_series/f" => Series is Stationary.")
    else:
        print(/docs/projects/capstone/Modeling/time_series/f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(/docs/projects/capstone/Modeling/time_series/f" => Series is Non-Stationary.")    
```


```python
# ADF Test on each column
/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/,/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/c/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/u/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/:/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/    adfuller_test(/docs/projects/capstone/Modeling/time_series/column, name=column.name)
    print(/docs/projects/capstone/Modeling/time_series/'\n')
```

        Augmented Dickey-Fuller Test on "av_swsfcdown" 
        -----------------------------------------------
     Null Hypothesis: Data has unit root. Non-Stationary.
     Significance Level    = 0.05
     Test Statistic        = -5.9437
     No. Lags Chosen       = 36
     Critical value 1%     = -3.431
     Critical value 5%     = -2.862
     Critical value 10%    = -2.567
     => P-Value = 0.0. Rejecting Null Hypothesis.
     => Series is Stationary.
    
    
        Augmented Dickey-Fuller Test on "av_swsfcdown_numhours_1D" 
        -----------------------------------------------
     Null Hypothesis: Data has unit root. Non-Stationary.
     Significance Level    = 0.05
     Test Statistic        = -5.8523
     No. Lags Chosen       = 33
     Critical value 1%     = -3.431
     Critical value 5%     = -2.862
     Critical value 10%    = -2.567
     => P-Value = 0.0. Rejecting Null Hypothesis.
     => Series is Stationary.
    
    
        Augmented Dickey-Fuller Test on "av_swsfcdown_numhours_10D" 
        -----------------------------------------------
     Null Hypothesis: Data has unit root. Non-Stationary.
     Significance Level    = 0.05
     Test Statistic        = -6.808
     No. Lags Chosen       = 35
     Critical value 1%     = -3.431
     Critical value 5%     = -2.862
     Critical value 10%    = -2.567
     => P-Value = 0.0. Rejecting Null Hypothesis.
     => Series is Stationary.
    
    
        Augmented Dickey-Fuller Test on "av_swsfcdown_sum_10D" 
        -----------------------------------------------
     Null Hypothesis: Data has unit root. Non-Stationary.
     Significance Level    = 0.05
     Test Statistic        = -8.06
     No. Lags Chosen       = 36
     Critical value 1%     = -3.431
     Critical value 5%     = -2.862
     Critical value 10%    = -2.567
     => P-Value = 0.0. Rejecting Null Hypothesis.
     => Series is Stationary.
    
    
        Augmented Dickey-Fuller Test on "forcing_temp_numhours_180D" 
        -----------------------------------------------
     Null Hypothesis: Data has unit root. Non-Stationary.
     Significance Level    = 0.05
     Test Statistic        = -10.7326
     No. Lags Chosen       = 35
     Critical value 1%     = -3.431
     Critical value 5%     = -2.862
     Critical value 10%    = -2.567
     => P-Value = 0.0. Rejecting Null Hypothesis.
     => Series is Stationary.
    
    
        Augmented Dickey-Fuller Test on "forcing_temp_sum_180D" 
        -----------------------------------------------
     Null Hypothesis: Data has unit root. Non-Stationary.
     Significance Level    = 0.05
     Test Statistic        = -9.2404
     No. Lags Chosen       = 36
     Critical value 1%     = -3.431
     Critical value 5%     = -2.862
     Critical value 10%    = -2.567
     => P-Value = 0.0. Rejecting Null Hypothesis.
     => Series is Stationary.
    
    
        Augmented Dickey-Fuller Test on "grass_count" 
        -----------------------------------------------
     Null Hypothesis: Data has unit root. Non-Stationary.
     Significance Level    = 0.05
     Test Statistic        = -9.7245
     No. Lags Chosen       = 34
     Critical value 1%     = -3.431
     Critical value 5%     = -2.862
     Critical value 10%    = -2.567
     => P-Value = 0.0. Rejecting Null Hypothesis.
     => Series is Stationary.
    
    


### Select the order of VAR model


```python
model = VAR(/docs/projects/capstone/Modeling/time_series/X_selected)
forecasting_model = VAR(/docs/projects/capstone/Modeling/time_series/df_train)
results_aic = []
for p in range(/docs/projects/capstone/Modeling/time_series/1,3):
  results = forecasting_model.fit(/docs/projects/capstone/Modeling/time_series/p)
  results_aic.append(/docs/projects/capstone/Modeling/time_series/results.aic)
```


```python
x = model.select_order(/docs/projects/capstone/Modeling/time_series/maxlags=20)
/docs/projects/capstone/Modeling/time_series/x/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/u/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/y/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/```




<table class="simpletable">
<caption>VAR Order Selection (/docs/projects/capstone/Modeling/time_series/* highlights the minimums)</caption>
<tr>
   <td></td>      <th>AIC</th>         <th>BIC</th>         <th>FPE</th>        <th>HQIC</th>    
</tr>
<tr>
  <th>0</th>  <td>     54.62</td>  <td>     54.63</td>  <td> 5.276e+23</td>  <td>     54.62</td> 
</tr>
<tr>
  <th>1</th>  <td>     32.49</td>  <td>     32.54</td>  <td> 1.293e+14</td>  <td>     32.51</td> 
</tr>
<tr>
  <th>2</th>  <td>     32.02</td>  <td>     32.12</td>  <td> 8.063e+13</td>  <td>     32.05</td> 
</tr>
<tr>
  <th>3</th>  <td>     31.92</td>  <td>     32.06</td>  <td> 7.297e+13</td>  <td>     31.97</td> 
</tr>
<tr>
  <th>4</th>  <td>     31.85</td>  <td>     32.03</td>  <td> 6.780e+13</td>  <td>     31.91</td> 
</tr>
<tr>
  <th>5</th>  <td>     31.78</td>  <td>     32.01</td>  <td> 6.357e+13</td>  <td>     31.86</td> 
</tr>
<tr>
  <th>6</th>  <td>     31.72</td>  <td>     31.99</td>  <td> 5.958e+13</td>  <td>     31.81</td> 
</tr>
<tr>
  <th>7</th>  <td>     31.61</td>  <td>     31.92</td>  <td> 5.330e+13</td>  <td>     31.72</td> 
</tr>
<tr>
  <th>8</th>  <td>     31.41</td>  <td>     31.77</td>  <td> 4.391e+13</td>  <td>     31.54</td> 
</tr>
<tr>
  <th>9</th>  <td>     31.13</td>  <td>     31.53</td>  <td> 3.298e+13</td>  <td>     31.27</td> 
</tr>
<tr>
  <th>10</th> <td>     31.03</td>  <td>     31.48</td>  <td> 2.999e+13</td>  <td>     31.19</td> 
</tr>
<tr>
  <th>11</th> <td>     30.24</td>  <td>     30.74</td>  <td> 1.363e+13</td>  <td>     30.41</td> 
</tr>
<tr>
  <th>12</th> <td>     30.19</td>  <td>     30.73*</td> <td> 1.298e+13</td>  <td>     30.38</td> 
</tr>
<tr>
  <th>13</th> <td>     30.16</td>  <td>     30.74</td>  <td> 1.252e+13</td>  <td>     30.36</td> 
</tr>
<tr>
  <th>14</th> <td>     30.14</td>  <td>     30.76</td>  <td> 1.225e+13</td>  <td>     30.35*</td>
</tr>
<tr>
  <th>15</th> <td>     30.13</td>  <td>     30.80</td>  <td> 1.212e+13</td>  <td>     30.36</td> 
</tr>
<tr>
  <th>16</th> <td>     30.12</td>  <td>     30.84</td>  <td> 1.208e+13</td>  <td>     30.37</td> 
</tr>
<tr>
  <th>17</th> <td>     30.11</td>  <td>     30.88</td>  <td> 1.199e+13</td>  <td>     30.38</td> 
</tr>
<tr>
  <th>18</th> <td>     30.10</td>  <td>     30.91</td>  <td> 1.187e+13</td>  <td>     30.38</td> 
</tr>
<tr>
  <th>19</th> <td>     30.10*</td> <td>     30.95</td>  <td> 1.186e+13*</td> <td>     30.40</td> 
</tr>
<tr>
  <th>20</th> <td>     30.11</td>  <td>     31.00</td>  <td> 1.192e+13</td>  <td>     30.42</td> 
</tr>
</table>




```python
model_fitted = model.fit(/docs/projects/capstone/Modeling/time_series/19)
/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/u/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/m/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/y/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/```




      Summary of Regression Results   
    ==================================
    Model:                         VAR
    Method:                        OLS
    Date:           Mon, 10, Oct, 2022
    Time:                     14:50:46
    --------------------------------------------------------------------
    No. of Equations:         7.00000    BIC:                    30.9524
    Nobs:                     7676.00    HQIC:                   30.3947
    Log likelihood:          -190842.    FPE:                1.18533e+13
    AIC:                      30.1036    Det(/docs/projects/capstone/Modeling/time_series/Omega_mle):     1.05009e+13
    --------------------------------------------------------------------
    Results for equation av_swsfcdown
    =================================================================================================
                                        coefficient       std. error           t-stat            prob
    -------------------------------------------------------------------------------------------------
    const                               -114.972058        17.439144           -6.593           0.000
    L1.av_swsfcdown                        0.420417         0.029302           14.348           0.000
    L1.av_swsfcdown_numhours_1D            2.706229         1.913396            1.414           0.157
    L1.av_swsfcdown_numhours_10D          -0.178510         0.666107           -0.268           0.789
    L1.av_swsfcdown_sum_10D               -0.002196         0.001112           -1.974           0.048
    L1.forcing_temp_numhours_180D         -1.936141         0.381888           -5.070           0.000
    L1.forcing_temp_sum_180D               0.054130         0.074370            0.728           0.467
    L1.grass_count                        -0.152839         0.024336           -6.280           0.000
    L2.av_swsfcdown                        0.012359         0.032794            0.377           0.706
    L2.av_swsfcdown_numhours_1D            2.153318         2.093798            1.028           0.304
    L2.av_swsfcdown_numhours_10D           0.155914         0.924647            0.169           0.866
    L2.av_swsfcdown_sum_10D                0.001237         0.001207            1.024           0.306
    L2.forcing_temp_numhours_180D          1.615679         0.615371            2.626           0.009
    L2.forcing_temp_sum_180D               0.005733         0.117834            0.049           0.961
    L2.grass_count                         0.023419         0.025594            0.915           0.360
    L3.av_swsfcdown                        0.031254         0.034275            0.912           0.362
    L3.av_swsfcdown_numhours_1D           -1.694779         2.116961           -0.801           0.423
    L3.av_swsfcdown_numhours_10D           1.773153         0.924740            1.917           0.055
    L3.av_swsfcdown_sum_10D               -0.000806         0.001219           -0.661           0.509
    L3.forcing_temp_numhours_180D         -0.099817         0.628183           -0.159           0.874
    L3.forcing_temp_sum_180D              -0.008334         0.118860           -0.070           0.944
    L3.grass_count                         0.000285         0.025573            0.011           0.991
    L4.av_swsfcdown                        0.117545         0.035218            3.338           0.001
    L4.av_swsfcdown_numhours_1D            2.316397         2.130031            1.087           0.277
    L4.av_swsfcdown_numhours_10D          -0.569222         0.925430           -0.615           0.538
    L4.av_swsfcdown_sum_10D               -0.002035         0.001225           -1.661           0.097
    L4.forcing_temp_numhours_180D         -0.022037         0.628829           -0.035           0.972
    L4.forcing_temp_sum_180D              -0.002973         0.118923           -0.025           0.980
    L4.grass_count                        -0.038383         0.025577           -1.501           0.133
    L5.av_swsfcdown                        0.115090         0.035542            3.238           0.001
    L5.av_swsfcdown_numhours_1D           -0.625979         2.133489           -0.293           0.769
    L5.av_swsfcdown_numhours_10D           0.556767         0.929676            0.599           0.549
    L5.av_swsfcdown_sum_10D                0.000334         0.001222            0.273           0.785
    L5.forcing_temp_numhours_180D          0.539791         0.629007            0.858           0.391
    L5.forcing_temp_sum_180D              -0.079994         0.118939           -0.673           0.501
    L5.grass_count                        -0.008797         0.025627           -0.343           0.731
    L6.av_swsfcdown                        0.094119         0.035309            2.666           0.008
    L6.av_swsfcdown_numhours_1D           -3.241039         2.140356           -1.514           0.130
    L6.av_swsfcdown_numhours_10D          -2.180109         0.931017           -2.342           0.019
    L6.av_swsfcdown_sum_10D                0.000519         0.001225            0.424           0.672
    L6.forcing_temp_numhours_180D         -0.440877         0.628840           -0.701           0.483
    L6.forcing_temp_sum_180D               0.023300         0.118903            0.196           0.845
    L6.grass_count                         0.030287         0.025764            1.176           0.240
    L7.av_swsfcdown                        0.041872         0.034621            1.209           0.226
    L7.av_swsfcdown_numhours_1D            0.167961         2.140441            0.078           0.937
    L7.av_swsfcdown_numhours_10D           2.288601         0.928436            2.465           0.014
    L7.av_swsfcdown_sum_10D                0.001220         0.001222            0.998           0.318
    L7.forcing_temp_numhours_180D          0.048285         0.628991            0.077           0.939
    L7.forcing_temp_sum_180D               0.063047         0.118799            0.531           0.596
    L7.grass_count                         0.019644         0.025804            0.761           0.446
    L8.av_swsfcdown                        0.089246         0.033372            2.674           0.007
    L8.av_swsfcdown_numhours_1D           -0.599447         2.137039           -0.281           0.779
    L8.av_swsfcdown_numhours_10D          -0.621096         0.930276           -0.668           0.504
    L8.av_swsfcdown_sum_10D               -0.001568         0.001219           -1.286           0.198
    L8.forcing_temp_numhours_180D          0.333458         0.629303            0.530           0.596
    L8.forcing_temp_sum_180D              -0.088538         0.118700           -0.746           0.456
    L8.grass_count                         0.003563         0.025825            0.138           0.890
    L9.av_swsfcdown                        0.070954         0.023982            2.959           0.003
    L9.av_swsfcdown_numhours_1D           -5.539002         2.128472           -2.602           0.009
    L9.av_swsfcdown_numhours_10D          -0.978630         0.919495           -1.064           0.287
    L9.av_swsfcdown_sum_10D                0.000522         0.001085            0.481           0.630
    L9.forcing_temp_numhours_180D         -0.339200         0.628994           -0.539           0.590
    L9.forcing_temp_sum_180D               0.062049         0.118563            0.523           0.601
    L9.grass_count                         0.004705         0.025841            0.182           0.856
    L10.av_swsfcdown                       0.052915         0.023419            2.259           0.024
    L10.av_swsfcdown_numhours_1D           0.786955         2.129336            0.370           0.712
    L10.av_swsfcdown_numhours_10D         -0.040530         0.887398           -0.046           0.964
    L10.av_swsfcdown_sum_10D               0.001845         0.000817            2.259           0.024
    L10.forcing_temp_numhours_180D        -0.026030         0.629282           -0.041           0.967
    L10.forcing_temp_sum_180D             -0.033534         0.118580           -0.283           0.777
    L10.grass_count                        0.040915         0.025901            1.580           0.114
    L11.av_swsfcdown                       0.045869         0.023848            1.923           0.054
    L11.av_swsfcdown_numhours_1D           3.702207         2.156878            1.716           0.086
    L11.av_swsfcdown_numhours_10D          0.003756         0.884935            0.004           0.997
    L11.av_swsfcdown_sum_10D              -0.000543         0.000740           -0.733           0.463
    L11.forcing_temp_numhours_180D         0.127954         0.629886            0.203           0.839
    L11.forcing_temp_sum_180D              0.025718         0.118641            0.217           0.828
    L11.grass_count                       -0.035925         0.026002           -1.382           0.167
    L12.av_swsfcdown                       0.020245         0.033598            0.603           0.547
    L12.av_swsfcdown_numhours_1D          -1.377390         2.165688           -0.636           0.525
    L12.av_swsfcdown_numhours_10D          1.310618         0.886011            1.479           0.139
    L12.av_swsfcdown_sum_10D              -0.000921         0.000740           -1.246           0.213
    L12.forcing_temp_numhours_180D         0.080511         0.629937            0.128           0.898
    L12.forcing_temp_sum_180D             -0.023639         0.118711           -0.199           0.842
    L12.grass_count                       -0.009252         0.025993           -0.356           0.722
    L13.av_swsfcdown                       0.036656         0.034783            1.054           0.292
    L13.av_swsfcdown_numhours_1D           4.271824         2.163274            1.975           0.048
    L13.av_swsfcdown_numhours_10D         -1.905254         0.886248           -2.150           0.032
    L13.av_swsfcdown_sum_10D               0.000560         0.000740            0.758           0.449
    L13.forcing_temp_numhours_180D        -0.418477         0.630315           -0.664           0.507
    L13.forcing_temp_sum_180D              0.056834         0.118821            0.478           0.632
    L13.grass_count                        0.012063         0.025957            0.465           0.642
    L14.av_swsfcdown                       0.001099         0.035482            0.031           0.975
    L14.av_swsfcdown_numhours_1D          -2.668318         2.168649           -1.230           0.219
    L14.av_swsfcdown_numhours_10D          2.742876         0.887161            3.092           0.002
    L14.av_swsfcdown_sum_10D              -0.000434         0.000740           -0.586           0.558
    L14.forcing_temp_numhours_180D         1.044288         0.630514            1.656           0.098
    L14.forcing_temp_sum_180D             -0.120128         0.118901           -1.010           0.312
    L14.grass_count                        0.042360         0.025927            1.634           0.102
    L15.av_swsfcdown                      -0.042329         0.035751           -1.184           0.236
    L15.av_swsfcdown_numhours_1D           1.156413         2.158927            0.536           0.592
    L15.av_swsfcdown_numhours_10D         -2.685252         0.887952           -3.024           0.002
    L15.av_swsfcdown_sum_10D               0.000353         0.000740            0.477           0.633
    L15.forcing_temp_numhours_180D        -0.432197         0.630445           -0.686           0.493
    L15.forcing_temp_sum_180D              0.044321         0.118847            0.373           0.709
    L15.grass_count                       -0.034204         0.025747           -1.328           0.184
    L16.av_swsfcdown                      -0.053667         0.035371           -1.517           0.129
    L16.av_swsfcdown_numhours_1D           0.683709         2.154904            0.317           0.751
    L16.av_swsfcdown_numhours_10D          0.991480         0.883105            1.123           0.262
    L16.av_swsfcdown_sum_10D               0.000800         0.000740            1.081           0.280
    L16.forcing_temp_numhours_180D        -0.101097         0.630436           -0.160           0.873
    L16.forcing_temp_sum_180D             -0.001091         0.118799           -0.009           0.993
    L16.grass_count                        0.012613         0.025665            0.491           0.623
    L17.av_swsfcdown                       0.012672         0.034454            0.368           0.713
    L17.av_swsfcdown_numhours_1D           0.340998         2.148172            0.159           0.874
    L17.av_swsfcdown_numhours_10D         -0.361656         0.885122           -0.409           0.683
    L17.av_swsfcdown_sum_10D              -0.000514         0.000740           -0.695           0.487
    L17.forcing_temp_numhours_180D        -0.303614         0.629976           -0.482           0.630
    L17.forcing_temp_sum_180D              0.092229         0.118813            0.776           0.438
    L17.grass_count                        0.046185         0.025692            1.798           0.072
    L18.av_swsfcdown                       0.006662         0.032717            0.204           0.839
    L18.av_swsfcdown_numhours_1D           5.012144         2.124908            2.359           0.018
    L18.av_swsfcdown_numhours_10D          0.093895         0.886472            0.106           0.916
    L18.av_swsfcdown_sum_10D              -0.000588         0.000731           -0.805           0.421
    L18.forcing_temp_numhours_180D         0.133237         0.619969            0.215           0.830
    L18.forcing_temp_sum_180D             -0.103861         0.117873           -0.881           0.378
    L18.grass_count                        0.036667         0.025678            1.428           0.153
    L19.av_swsfcdown                      -0.038972         0.027997           -1.392           0.164
    L19.av_swsfcdown_numhours_1D          -2.614040         1.926241           -1.357           0.175
    L19.av_swsfcdown_numhours_10D          0.403331         0.550650            0.732           0.464
    L19.av_swsfcdown_sum_10D               0.001138         0.000454            2.507           0.012
    L19.forcing_temp_numhours_180D         0.057918         0.384370            0.151           0.880
    L19.forcing_temp_sum_180D              0.039426         0.074245            0.531           0.595
    L19.grass_count                       -0.041353         0.024400           -1.695           0.090
    =================================================================================================
    
    Results for equation av_swsfcdown_numhours_1D
    =================================================================================================
                                        coefficient       std. error           t-stat            prob
    -------------------------------------------------------------------------------------------------
    const                                  1.898292         0.110301           17.210           0.000
    L1.av_swsfcdown                        0.000815         0.000185            4.399           0.000
    L1.av_swsfcdown_numhours_1D            0.461140         0.012102           38.104           0.000
    L1.av_swsfcdown_numhours_10D           0.010851         0.004213            2.576           0.010
    L1.av_swsfcdown_sum_10D               -0.000020         0.000007           -2.806           0.005
    L1.forcing_temp_numhours_180D         -0.004298         0.002415           -1.780           0.075
    L1.forcing_temp_sum_180D               0.000512         0.000470            1.089           0.276
    L1.grass_count                         0.000039         0.000154            0.252           0.801
    L2.av_swsfcdown                        0.000774         0.000207            3.730           0.000
    L2.av_swsfcdown_numhours_1D            0.187834         0.013243           14.184           0.000
    L2.av_swsfcdown_numhours_10D           0.000412         0.005848            0.070           0.944
    L2.av_swsfcdown_sum_10D               -0.000012         0.000008           -1.528           0.127
    L2.forcing_temp_numhours_180D          0.004064         0.003892            1.044           0.296
    L2.forcing_temp_sum_180D              -0.000596         0.000745           -0.799           0.424
    L2.grass_count                         0.000036         0.000162            0.220           0.826
    L3.av_swsfcdown                        0.000758         0.000217            3.495           0.000
    L3.av_swsfcdown_numhours_1D            0.116967         0.013390            8.736           0.000
    L3.av_swsfcdown_numhours_10D          -0.008758         0.005849           -1.497           0.134
    L3.av_swsfcdown_sum_10D               -0.000001         0.000008           -0.107           0.915
    L3.forcing_temp_numhours_180D         -0.002637         0.003973           -0.664           0.507
    L3.forcing_temp_sum_180D               0.000430         0.000752            0.572           0.567
    L3.grass_count                        -0.000112         0.000162           -0.694           0.488
    L4.av_swsfcdown                        0.000508         0.000223            2.280           0.023
    L4.av_swsfcdown_numhours_1D            0.030998         0.013472            2.301           0.021
    L4.av_swsfcdown_numhours_10D          -0.006830         0.005853           -1.167           0.243
    L4.av_swsfcdown_sum_10D                0.000015         0.000008            1.872           0.061
    L4.forcing_temp_numhours_180D          0.003653         0.003977            0.918           0.358
    L4.forcing_temp_sum_180D              -0.000513         0.000752           -0.682           0.495
    L4.grass_count                         0.000259         0.000162            1.600           0.110
    L5.av_swsfcdown                        0.000690         0.000225            3.071           0.002
    L5.av_swsfcdown_numhours_1D            0.093445         0.013494            6.925           0.000
    L5.av_swsfcdown_numhours_10D           0.007607         0.005880            1.294           0.196
    L5.av_swsfcdown_sum_10D               -0.000010         0.000008           -1.255           0.209
    L5.forcing_temp_numhours_180D          0.001354         0.003978            0.340           0.734
    L5.forcing_temp_sum_180D              -0.000072         0.000752           -0.096           0.923
    L5.grass_count                        -0.000228         0.000162           -1.407           0.159
    L6.av_swsfcdown                        0.000894         0.000223            4.005           0.000
    L6.av_swsfcdown_numhours_1D           -0.011070         0.013538           -0.818           0.413
    L6.av_swsfcdown_numhours_10D           0.015564         0.005889            2.643           0.008
    L6.av_swsfcdown_sum_10D               -0.000004         0.000008           -0.549           0.583
    L6.forcing_temp_numhours_180D         -0.005909         0.003977           -1.486           0.137
    L6.forcing_temp_sum_180D               0.000741         0.000752            0.986           0.324
    L6.grass_count                         0.000278         0.000163            1.707           0.088
    L7.av_swsfcdown                        0.000314         0.000219            1.436           0.151
    L7.av_swsfcdown_numhours_1D           -0.017131         0.013538           -1.265           0.206
    L7.av_swsfcdown_numhours_10D          -0.023998         0.005872           -4.087           0.000
    L7.av_swsfcdown_sum_10D                0.000018         0.000008            2.363           0.018
    L7.forcing_temp_numhours_180D          0.005300         0.003978            1.332           0.183
    L7.forcing_temp_sum_180D              -0.000399         0.000751           -0.531           0.596
    L7.grass_count                         0.000164         0.000163            1.007           0.314
    L8.av_swsfcdown                        0.000577         0.000211            2.735           0.006
    L8.av_swsfcdown_numhours_1D           -0.008637         0.013517           -0.639           0.523
    L8.av_swsfcdown_numhours_10D           0.014689         0.005884            2.497           0.013
    L8.av_swsfcdown_sum_10D               -0.000006         0.000008           -0.832           0.405
    L8.forcing_temp_numhours_180D         -0.005458         0.003980           -1.371           0.170
    L8.forcing_temp_sum_180D               0.000435         0.000751            0.580           0.562
    L8.grass_count                         0.000205         0.000163            1.253           0.210
    L9.av_swsfcdown                        0.000308         0.000152            2.030           0.042
    L9.av_swsfcdown_numhours_1D            0.000148         0.013462            0.011           0.991
    L9.av_swsfcdown_numhours_10D          -0.016909         0.005816           -2.907           0.004
    L9.av_swsfcdown_sum_10D                0.000009         0.000007            1.310           0.190
    L9.forcing_temp_numhours_180D          0.007665         0.003978            1.927           0.054
    L9.forcing_temp_sum_180D              -0.001123         0.000750           -1.498           0.134
    L9.grass_count                         0.000151         0.000163            0.924           0.355
    L10.av_swsfcdown                      -0.000004         0.000148           -0.024           0.981
    L10.av_swsfcdown_numhours_1D          -0.043968         0.013468           -3.265           0.001
    L10.av_swsfcdown_numhours_10D          0.029930         0.005613            5.333           0.000
    L10.av_swsfcdown_sum_10D               0.000010         0.000005            1.918           0.055
    L10.forcing_temp_numhours_180D        -0.006368         0.003980           -1.600           0.110
    L10.forcing_temp_sum_180D              0.001179         0.000750            1.573           0.116
    L10.grass_count                        0.000113         0.000164            0.691           0.490
    L11.av_swsfcdown                       0.000221         0.000151            1.467           0.142
    L11.av_swsfcdown_numhours_1D           0.025963         0.013642            1.903           0.057
    L11.av_swsfcdown_numhours_10D         -0.033997         0.005597           -6.074           0.000
    L11.av_swsfcdown_sum_10D              -0.000009         0.000005           -1.972           0.049
    L11.forcing_temp_numhours_180D         0.001733         0.003984            0.435           0.664
    L11.forcing_temp_sum_180D             -0.000731         0.000750           -0.974           0.330
    L11.grass_count                       -0.000148         0.000164           -0.899           0.369
    L12.av_swsfcdown                      -0.000173         0.000213           -0.815           0.415
    L12.av_swsfcdown_numhours_1D           0.000448         0.013698            0.033           0.974
    L12.av_swsfcdown_numhours_10D          0.017176         0.005604            3.065           0.002
    L12.av_swsfcdown_sum_10D               0.000001         0.000005            0.170           0.865
    L12.forcing_temp_numhours_180D        -0.000453         0.003984           -0.114           0.910
    L12.forcing_temp_sum_180D              0.000754         0.000751            1.004           0.315
    L12.grass_count                        0.000167         0.000164            1.017           0.309
    L13.av_swsfcdown                      -0.000460         0.000220           -2.092           0.036
    L13.av_swsfcdown_numhours_1D           0.045264         0.013682            3.308           0.001
    L13.av_swsfcdown_numhours_10D         -0.007445         0.005605           -1.328           0.184
    L13.av_swsfcdown_sum_10D               0.000000         0.000005            0.066           0.948
    L13.forcing_temp_numhours_180D        -0.001960         0.003987           -0.492           0.623
    L13.forcing_temp_sum_180D             -0.000325         0.000752           -0.433           0.665
    L13.grass_count                        0.000124         0.000164            0.754           0.451
    L14.av_swsfcdown                      -0.000555         0.000224           -2.474           0.013
    L14.av_swsfcdown_numhours_1D           0.007375         0.013716            0.538           0.591
    L14.av_swsfcdown_numhours_10D         -0.004048         0.005611           -0.721           0.471
    L14.av_swsfcdown_sum_10D               0.000004         0.000005            0.913           0.361
    L14.forcing_temp_numhours_180D         0.007372         0.003988            1.849           0.065
    L14.forcing_temp_sum_180D             -0.000654         0.000752           -0.870           0.385
    L14.grass_count                        0.000011         0.000164            0.066           0.947
    L15.av_swsfcdown                       0.000009         0.000226            0.041           0.967
    L15.av_swsfcdown_numhours_1D           0.020172         0.013655            1.477           0.140
    L15.av_swsfcdown_numhours_10D          0.011924         0.005616            2.123           0.034
    L15.av_swsfcdown_sum_10D              -0.000004         0.000005           -0.870           0.384
    L15.forcing_temp_numhours_180D        -0.008245         0.003987           -2.068           0.039
    L15.forcing_temp_sum_180D              0.001250         0.000752            1.663           0.096
    L15.grass_count                        0.000004         0.000163            0.022           0.982
    L16.av_swsfcdown                      -0.000605         0.000224           -2.703           0.007
    L16.av_swsfcdown_numhours_1D           0.017148         0.013630            1.258           0.208
    L16.av_swsfcdown_numhours_10D          0.010257         0.005586            1.836           0.066
    L16.av_swsfcdown_sum_10D               0.000007         0.000005            1.504           0.133
    L16.forcing_temp_numhours_180D         0.007378         0.003987            1.850           0.064
    L16.forcing_temp_sum_180D             -0.001829         0.000751           -2.434           0.015
    L16.grass_count                        0.000036         0.000162            0.224           0.823
    L17.av_swsfcdown                      -0.000353         0.000218           -1.621           0.105
    L17.av_swsfcdown_numhours_1D           0.023316         0.013587            1.716           0.086
    L17.av_swsfcdown_numhours_10D         -0.020052         0.005598           -3.582           0.000
    L17.av_swsfcdown_sum_10D              -0.000009         0.000005           -1.978           0.048
    L17.forcing_temp_numhours_180D        -0.002198         0.003985           -0.552           0.581
    L17.forcing_temp_sum_180D              0.000697         0.000751            0.928           0.353
    L17.grass_count                        0.000223         0.000163            1.372           0.170
    L18.av_swsfcdown                      -0.000100         0.000207           -0.483           0.629
    L18.av_swsfcdown_numhours_1D          -0.025225         0.013440           -1.877           0.061
    L18.av_swsfcdown_numhours_10D         -0.009493         0.005607           -1.693           0.090
    L18.av_swsfcdown_sum_10D               0.000006         0.000005            1.214           0.225
    L18.forcing_temp_numhours_180D        -0.007388         0.003921           -1.884           0.060
    L18.forcing_temp_sum_180D              0.001743         0.000746            2.338           0.019
    L18.grass_count                       -0.000029         0.000162           -0.177           0.860
    L19.av_swsfcdown                      -0.000337         0.000177           -1.905           0.057
    L19.av_swsfcdown_numhours_1D          -0.063169         0.012183           -5.185           0.000
    L19.av_swsfcdown_numhours_10D          0.010667         0.003483            3.063           0.002
    L19.av_swsfcdown_sum_10D               0.000002         0.000003            0.597           0.550
    L19.forcing_temp_numhours_180D         0.005257         0.002431            2.162           0.031
    L19.forcing_temp_sum_180D             -0.001383         0.000470           -2.945           0.003
    L19.grass_count                        0.000156         0.000154            1.010           0.313
    =================================================================================================
    
    Results for equation av_swsfcdown_numhours_10D
    =================================================================================================
                                        coefficient       std. error           t-stat            prob
    -------------------------------------------------------------------------------------------------
    const                                  1.822570         0.349000            5.222           0.000
    L1.av_swsfcdown                        0.000458         0.000586            0.781           0.435
    L1.av_swsfcdown_numhours_1D            0.367084         0.038292            9.587           0.000
    L1.av_swsfcdown_numhours_10D           1.053612         0.013330           79.038           0.000
    L1.av_swsfcdown_sum_10D                0.000005         0.000022            0.224           0.823
    L1.forcing_temp_numhours_180D         -0.008449         0.007643           -1.105           0.269
    L1.forcing_temp_sum_180D               0.001022         0.001488            0.687           0.492
    L1.grass_count                         0.000328         0.000487            0.674           0.501
    L2.av_swsfcdown                       -0.001119         0.000656           -1.706           0.088
    L2.av_swsfcdown_numhours_1D            0.200343         0.041902            4.781           0.000
    L2.av_swsfcdown_numhours_10D          -0.042356         0.018504           -2.289           0.022
    L2.av_swsfcdown_sum_10D                0.000059         0.000024            2.441           0.015
    L2.forcing_temp_numhours_180D          0.007913         0.012315            0.643           0.521
    L2.forcing_temp_sum_180D              -0.001156         0.002358           -0.490           0.624
    L2.grass_count                         0.000080         0.000512            0.156           0.876
    L3.av_swsfcdown                        0.001599         0.000686            2.331           0.020
    L3.av_swsfcdown_numhours_1D            0.087872         0.042366            2.074           0.038
    L3.av_swsfcdown_numhours_10D           0.042930         0.018506            2.320           0.020
    L3.av_swsfcdown_sum_10D               -0.000127         0.000024           -5.190           0.000
    L3.forcing_temp_numhours_180D         -0.004718         0.012571           -0.375           0.707
    L3.forcing_temp_sum_180D               0.000653         0.002379            0.275           0.784
    L3.grass_count                         0.000163         0.000512            0.319           0.750
    L4.av_swsfcdown                        0.001049         0.000705            1.488           0.137
    L4.av_swsfcdown_numhours_1D            0.134240         0.042627            3.149           0.002
    L4.av_swsfcdown_numhours_10D          -0.147642         0.018520           -7.972           0.000
    L4.av_swsfcdown_sum_10D                0.000029         0.000025            1.191           0.234
    L4.forcing_temp_numhours_180D          0.007452         0.012584            0.592           0.554
    L4.forcing_temp_sum_180D              -0.001374         0.002380           -0.577           0.564
    L4.grass_count                         0.000286         0.000512            0.558           0.577
    L5.av_swsfcdown                       -0.000510         0.000711           -0.717           0.473
    L5.av_swsfcdown_numhours_1D            0.208025         0.042696            4.872           0.000
    L5.av_swsfcdown_numhours_10D          -0.034407         0.018605           -1.849           0.064
    L5.av_swsfcdown_sum_10D                0.000052         0.000024            2.107           0.035
    L5.forcing_temp_numhours_180D          0.001059         0.012588            0.084           0.933
    L5.forcing_temp_sum_180D               0.000630         0.002380            0.265           0.791
    L5.grass_count                         0.000275         0.000513            0.536           0.592
    L6.av_swsfcdown                       -0.001503         0.000707           -2.127           0.033
    L6.av_swsfcdown_numhours_1D            0.102920         0.042834            2.403           0.016
    L6.av_swsfcdown_numhours_10D          -0.003883         0.018632           -0.208           0.835
    L6.av_swsfcdown_sum_10D                0.000045         0.000025            1.825           0.068
    L6.forcing_temp_numhours_180D         -0.010616         0.012585           -0.844           0.399
    L6.forcing_temp_sum_180D               0.001246         0.002380            0.524           0.601
    L6.grass_count                         0.000187         0.000516            0.362           0.717
    L7.av_swsfcdown                        0.001251         0.000693            1.806           0.071
    L7.av_swsfcdown_numhours_1D            0.038912         0.042835            0.908           0.364
    L7.av_swsfcdown_numhours_10D           0.030136         0.018580            1.622           0.105
    L7.av_swsfcdown_sum_10D               -0.000105         0.000024           -4.310           0.000
    L7.forcing_temp_numhours_180D          0.007011         0.012588            0.557           0.578
    L7.forcing_temp_sum_180D              -0.001001         0.002377           -0.421           0.674
    L7.grass_count                         0.000044         0.000516            0.085           0.932
    L8.av_swsfcdown                        0.000307         0.000668            0.460           0.645
    L8.av_swsfcdown_numhours_1D            0.054729         0.042767            1.280           0.201
    L8.av_swsfcdown_numhours_10D          -0.016278         0.018617           -0.874           0.382
    L8.av_swsfcdown_sum_10D                0.000053         0.000024            2.187           0.029
    L8.forcing_temp_numhours_180D         -0.001945         0.012594           -0.154           0.877
    L8.forcing_temp_sum_180D               0.000361         0.002375            0.152           0.879
    L8.grass_count                        -0.000011         0.000517           -0.022           0.982
    L9.av_swsfcdown                       -0.001964         0.000480           -4.093           0.000
    L9.av_swsfcdown_numhours_1D            0.117626         0.042596            2.761           0.006
    L9.av_swsfcdown_numhours_10D          -0.075092         0.018401           -4.081           0.000
    L9.av_swsfcdown_sum_10D                0.000072         0.000022            3.295           0.001
    L9.forcing_temp_numhours_180D         -0.000240         0.012588           -0.019           0.985
    L9.forcing_temp_sum_180D              -0.000052         0.002373           -0.022           0.982
    L9.grass_count                         0.000424         0.000517            0.820           0.412
    L10.av_swsfcdown                      -0.002285         0.000469           -4.876           0.000
    L10.av_swsfcdown_numhours_1D          -0.581541         0.042613          -13.647           0.000
    L10.av_swsfcdown_numhours_10D          0.144962         0.017759            8.163           0.000
    L10.av_swsfcdown_sum_10D               0.000009         0.000016            0.538           0.590
    L10.forcing_temp_numhours_180D         0.022481         0.012593            1.785           0.074
    L10.forcing_temp_sum_180D             -0.003595         0.002373           -1.515           0.130
    L10.grass_count                       -0.000253         0.000518           -0.487           0.626
    L11.av_swsfcdown                      -0.001893         0.000477           -3.966           0.000
    L11.av_swsfcdown_numhours_1D          -0.024257         0.043164           -0.562           0.574
    L11.av_swsfcdown_numhours_10D         -0.018933         0.017710           -1.069           0.285
    L11.av_swsfcdown_sum_10D              -0.000008         0.000015           -0.525           0.599
    L11.forcing_temp_numhours_180D        -0.025604         0.012606           -2.031           0.042
    L11.forcing_temp_sum_180D              0.003699         0.002374            1.558           0.119
    L11.grass_count                       -0.000121         0.000520           -0.233           0.816
    L12.av_swsfcdown                      -0.001921         0.000672           -2.858           0.004
    L12.av_swsfcdown_numhours_1D           0.067081         0.043341            1.548           0.122
    L12.av_swsfcdown_numhours_10D         -0.008780         0.017731           -0.495           0.620
    L12.av_swsfcdown_sum_10D               0.000013         0.000015            0.894           0.372
    L12.forcing_temp_numhours_180D         0.002308         0.012607            0.183           0.855
    L12.forcing_temp_sum_180D              0.000249         0.002376            0.105           0.917
    L12.grass_count                        0.000154         0.000520            0.296           0.768
    L13.av_swsfcdown                      -0.000129         0.000696           -0.186           0.853
    L13.av_swsfcdown_numhours_1D           0.146898         0.043292            3.393           0.001
    L13.av_swsfcdown_numhours_10D         -0.034634         0.017736           -1.953           0.051
    L13.av_swsfcdown_sum_10D              -0.000014         0.000015           -0.939           0.348
    L13.forcing_temp_numhours_180D         0.001200         0.012614            0.095           0.924
    L13.forcing_temp_sum_180D             -0.000824         0.002378           -0.347           0.729
    L13.grass_count                       -0.000164         0.000519           -0.316           0.752
    L14.av_swsfcdown                      -0.004132         0.000710           -5.819           0.000
    L14.av_swsfcdown_numhours_1D          -0.014762         0.043400           -0.340           0.734
    L14.av_swsfcdown_numhours_10D          0.002084         0.017754            0.117           0.907
    L14.av_swsfcdown_sum_10D               0.000019         0.000015            1.294           0.196
    L14.forcing_temp_numhours_180D         0.006655         0.012618            0.527           0.598
    L14.forcing_temp_sum_180D             -0.000126         0.002380           -0.053           0.958
    L14.grass_count                        0.000364         0.000519            0.701           0.483
    L15.av_swsfcdown                      -0.002387         0.000715           -3.336           0.001
    L15.av_swsfcdown_numhours_1D           0.033039         0.043205            0.765           0.444
    L15.av_swsfcdown_numhours_10D          0.010229         0.017770            0.576           0.565
    L15.av_swsfcdown_sum_10D              -0.000012         0.000015           -0.832           0.406
    L15.forcing_temp_numhours_180D        -0.009342         0.012617           -0.740           0.459
    L15.forcing_temp_sum_180D              0.001270         0.002378            0.534           0.593
    L15.grass_count                       -0.000120         0.000515           -0.233           0.816
    L16.av_swsfcdown                      -0.001277         0.000708           -1.804           0.071
    L16.av_swsfcdown_numhours_1D          -0.070230         0.043125           -1.629           0.103
    L16.av_swsfcdown_numhours_10D          0.032385         0.017673            1.832           0.067
    L16.av_swsfcdown_sum_10D               0.000001         0.000015            0.085           0.932
    L16.forcing_temp_numhours_180D         0.000743         0.012617            0.059           0.953
    L16.forcing_temp_sum_180D             -0.001122         0.002377           -0.472           0.637
    L16.grass_count                        0.000044         0.000514            0.085           0.932
    L17.av_swsfcdown                      -0.000513         0.000690           -0.744           0.457
    L17.av_swsfcdown_numhours_1D          -0.033570         0.042990           -0.781           0.435
    L17.av_swsfcdown_numhours_10D         -0.026648         0.017713           -1.504           0.132
    L17.av_swsfcdown_sum_10D               0.000005         0.000015            0.352           0.725
    L17.forcing_temp_numhours_180D         0.022123         0.012607            1.755           0.079
    L17.forcing_temp_sum_180D             -0.003346         0.002378           -1.407           0.159
    L17.grass_count                       -0.000077         0.000514           -0.150           0.881
    L18.av_swsfcdown                      -0.003469         0.000655           -5.298           0.000
    L18.av_swsfcdown_numhours_1D          -0.150050         0.042525           -3.529           0.000
    L18.av_swsfcdown_numhours_10D         -0.001074         0.017740           -0.061           0.952
    L18.av_swsfcdown_sum_10D              -0.000001         0.000015           -0.052           0.959
    L18.forcing_temp_numhours_180D        -0.012509         0.012407           -1.008           0.313
    L18.forcing_temp_sum_180D              0.003259         0.002359            1.382           0.167
    L18.grass_count                        0.000322         0.000514            0.626           0.531
    L19.av_swsfcdown                      -0.001563         0.000560           -2.789           0.005
    L19.av_swsfcdown_numhours_1D          -0.037569         0.038549           -0.975           0.330
    L19.av_swsfcdown_numhours_10D          0.009489         0.011020            0.861           0.389
    L19.av_swsfcdown_sum_10D              -0.000000         0.000009           -0.024           0.981
    L19.forcing_temp_numhours_180D        -0.005921         0.007692           -0.770           0.441
    L19.forcing_temp_sum_180D              0.000120         0.001486            0.080           0.936
    L19.grass_count                       -0.000054         0.000488           -0.111           0.912
    =================================================================================================
    
    Results for equation av_swsfcdown_sum_10D
    =================================================================================================
                                        coefficient       std. error           t-stat            prob
    -------------------------------------------------------------------------------------------------
    const                              -2521.033975       465.941755           -5.411           0.000
    L1.av_swsfcdown                       22.528849         0.782884           28.777           0.000
    L1.av_swsfcdown_numhours_1D          -73.962247        51.122415           -1.447           0.148
    L1.av_swsfcdown_numhours_10D         146.266313        17.797161            8.219           0.000
    L1.av_swsfcdown_sum_10D                0.380435         0.029722           12.800           0.000
    L1.forcing_temp_numhours_180D        -49.331379        10.203354           -4.835           0.000
    L1.forcing_temp_sum_180D               2.070613         1.987025            1.042           0.297
    L1.grass_count                        -3.854099         0.650220           -5.927           0.000
    L2.av_swsfcdown                        8.962155         0.876187           10.229           0.000
    L2.av_swsfcdown_numhours_1D           -2.366657        55.942410           -0.042           0.966
    L2.av_swsfcdown_numhours_10D         -60.461636        24.704857           -2.447           0.014
    L2.av_swsfcdown_sum_10D                0.219523         0.032250            6.807           0.000
    L2.forcing_temp_numhours_180D         33.053161        16.441589            2.010           0.044
    L2.forcing_temp_sum_180D               1.616590         3.148320            0.513           0.608
    L2.grass_count                         0.818929         0.683817            1.198           0.231
    L3.av_swsfcdown                        8.296827         0.915765            9.060           0.000
    L3.av_swsfcdown_numhours_1D         -150.084976        56.561302           -2.653           0.008
    L3.av_swsfcdown_numhours_10D          60.876985        24.707335            2.464           0.014
    L3.av_swsfcdown_sum_10D                0.039911         0.032580            1.225           0.221
    L3.forcing_temp_numhours_180D          7.309868        16.783888            0.436           0.663
    L3.forcing_temp_sum_180D              -3.297523         3.175719           -1.038           0.299
    L3.grass_count                        -0.683920         0.683261           -1.001           0.317
    L4.av_swsfcdown                        8.730636         0.940962            9.278           0.000
    L4.av_swsfcdown_numhours_1D          -19.441518        56.910501           -0.342           0.733
    L4.av_swsfcdown_numhours_10D         -81.794094        24.725772           -3.308           0.001
    L4.av_swsfcdown_sum_10D                0.005743         0.032741            0.175           0.861
    L4.forcing_temp_numhours_180D         -0.661043        16.801145           -0.039           0.969
    L4.forcing_temp_sum_180D               0.914692         3.177392            0.288           0.773
    L4.grass_count                        -0.760560         0.683370           -1.113           0.266
    L5.av_swsfcdown                        6.137217         0.949606            6.463           0.000
    L5.av_swsfcdown_numhours_1D           28.724559        57.002891            0.504           0.614
    L5.av_swsfcdown_numhours_10D         -34.847426        24.839229           -1.403           0.161
    L5.av_swsfcdown_sum_10D                0.122662         0.032661            3.756           0.000
    L5.forcing_temp_numhours_180D          6.691308        16.805912            0.398           0.691
    L5.forcing_temp_sum_180D              -1.595719         3.177840           -0.502           0.616
    L5.grass_count                        -0.125874         0.684713           -0.184           0.854
    L6.av_swsfcdown                        4.653052         0.943379            4.932           0.000
    L6.av_swsfcdown_numhours_1D          -33.365863        57.186377           -0.583           0.560
    L6.av_swsfcdown_numhours_10D         -71.692668        24.875066           -2.882           0.004
    L6.av_swsfcdown_sum_10D                0.062476         0.032730            1.909           0.056
    L6.forcing_temp_numhours_180D         -3.138443        16.801432           -0.187           0.852
    L6.forcing_temp_sum_180D              -0.699582         3.176856           -0.220           0.826
    L6.grass_count                         1.671763         0.688375            2.429           0.015
    L7.av_swsfcdown                        4.184321         0.925010            4.524           0.000
    L7.av_swsfcdown_numhours_1D           15.392848        57.188638            0.269           0.788
    L7.av_swsfcdown_numhours_10D          85.320182        24.806088            3.439           0.001
    L7.av_swsfcdown_sum_10D               -0.017155         0.032654           -0.525           0.599
    L7.forcing_temp_numhours_180D         -7.808108        16.805467           -0.465           0.642
    L7.forcing_temp_sum_180D               2.949440         3.174094            0.929           0.353
    L7.grass_count                         0.382529         0.689426            0.555           0.579
    L8.av_swsfcdown                        4.269989         0.891638            4.789           0.000
    L8.av_swsfcdown_numhours_1D          -42.903178        57.097732           -0.751           0.452
    L8.av_swsfcdown_numhours_10D         -20.818636        24.855254           -0.838           0.402
    L8.av_swsfcdown_sum_10D                0.019121         0.032581            0.587           0.557
    L8.forcing_temp_numhours_180D         20.414024        16.813827            1.214           0.225
    L8.forcing_temp_sum_180D              -3.829549         3.171443           -1.208           0.227
    L8.grass_count                         0.343453         0.689984            0.498           0.619
    L9.av_swsfcdown                        3.291603         0.640753            5.137           0.000
    L9.av_swsfcdown_numhours_1D          -83.837845        56.868837           -1.474           0.140
    L9.av_swsfcdown_numhours_10D         -23.311566        24.567222           -0.949           0.343
    L9.av_swsfcdown_sum_10D                0.034545         0.028994            1.191           0.233
    L9.forcing_temp_numhours_180D        -10.606604        16.805554           -0.631           0.528
    L9.forcing_temp_sum_180D               1.903715         3.167780            0.601           0.548
    L9.grass_count                         0.875888         0.690428            1.269           0.205
    L10.av_swsfcdown                      -3.792556         0.625724           -6.061           0.000
    L10.av_swsfcdown_numhours_1D         -65.492312        56.891919           -1.151           0.250
    L10.av_swsfcdown_numhours_10D         40.408189        23.709634            1.704           0.088
    L10.av_swsfcdown_sum_10D               0.071403         0.021827            3.271           0.001
    L10.forcing_temp_numhours_180D         2.939145        16.813248            0.175           0.861
    L10.forcing_temp_sum_180D             -1.924230         3.168242           -0.607           0.544
    L10.grass_count                       -1.529410         0.692026           -2.210           0.027
    L11.av_swsfcdown                     -20.892719         0.637165          -32.790           0.000
    L11.av_swsfcdown_numhours_1D         260.216482        57.627807            4.515           0.000
    L11.av_swsfcdown_numhours_10D        -15.013992        23.643842           -0.635           0.525
    L11.av_swsfcdown_sum_10D              -0.024466         0.019773           -1.237           0.216
    L11.forcing_temp_numhours_180D        -0.581398        16.829401           -0.035           0.972
    L11.forcing_temp_sum_180D              3.397603         3.169860            1.072           0.284
    L11.grass_count                        0.192892         0.694727            0.278           0.781
    L12.av_swsfcdown                      -8.757626         0.897689           -9.756           0.000
    L12.av_swsfcdown_numhours_1D          -4.752197        57.863194           -0.082           0.935
    L12.av_swsfcdown_numhours_10D         21.358573        23.672589            0.902           0.367
    L12.av_swsfcdown_sum_10D              -0.029702         0.019760           -1.503           0.133
    L12.forcing_temp_numhours_180D        -5.729581        16.830765           -0.340           0.734
    L12.forcing_temp_sum_180D             -1.217410         3.171742           -0.384           0.701
    L12.grass_count                       -0.749017         0.694478           -1.079           0.281
    L13.av_swsfcdown                      -6.171847         0.929324           -6.641           0.000
    L13.av_swsfcdown_numhours_1D         247.607337        57.798696            4.284           0.000
    L13.av_swsfcdown_numhours_10D        -85.360103        23.678914           -3.605           0.000
    L13.av_swsfcdown_sum_10D               0.024418         0.019758            1.236           0.217
    L13.forcing_temp_numhours_180D        -3.936625        16.840866           -0.234           0.815
    L13.forcing_temp_sum_180D              0.404544         3.174671            0.127           0.899
    L13.grass_count                        0.446701         0.693527            0.644           0.520
    L14.av_swsfcdown                      -5.840247         0.948020           -6.160           0.000
    L14.av_swsfcdown_numhours_1D         -99.001545        57.942289           -1.709           0.088
    L14.av_swsfcdown_numhours_10D         90.060011        23.703305            3.799           0.000
    L14.av_swsfcdown_sum_10D              -0.003754         0.019762           -0.190           0.849
    L14.forcing_temp_numhours_180D        21.434840        16.846184            1.272           0.203
    L14.forcing_temp_sum_180D             -2.960093         3.176813           -0.932           0.351
    L14.grass_count                        1.541954         0.692734            2.226           0.026
    L15.av_swsfcdown                      -5.280961         0.955214           -5.529           0.000
    L15.av_swsfcdown_numhours_1D          40.827094        57.682551            0.708           0.479
    L15.av_swsfcdown_numhours_10D        -63.838248        23.724439           -2.691           0.007
    L15.av_swsfcdown_sum_10D              -0.003408         0.019766           -0.172           0.863
    L15.forcing_temp_numhours_180D        -9.683931        16.844321           -0.575           0.565
    L15.forcing_temp_sum_180D              2.330302         3.175375            0.734           0.463
    L15.grass_count                       -0.454871         0.687917           -0.661           0.508
    L16.av_swsfcdown                      -2.867469         0.945036           -3.034           0.002
    L16.av_swsfcdown_numhours_1D         -33.232447        57.575061           -0.577           0.564
    L16.av_swsfcdown_numhours_10D         81.115426        23.594923            3.438           0.001
    L16.av_swsfcdown_sum_10D               0.011722         0.019777            0.593           0.553
    L16.forcing_temp_numhours_180D         3.319968        16.844087            0.197           0.844
    L16.forcing_temp_sum_180D             -1.280177         3.174088           -0.403           0.687
    L16.grass_count                       -0.661814         0.685724           -0.965           0.334
    L17.av_swsfcdown                      -0.768234         0.920559           -0.835           0.404
    L17.av_swsfcdown_numhours_1D           6.745920        57.395197            0.118           0.906
    L17.av_swsfcdown_numhours_10D        -71.211503        23.648831           -3.011           0.003
    L17.av_swsfcdown_sum_10D              -0.000019         0.019773           -0.001           0.999
    L17.forcing_temp_numhours_180D         9.208544        16.831795            0.547           0.584
    L17.forcing_temp_sum_180D             -1.531837         3.174469           -0.483           0.629
    L17.grass_count                        1.100970         0.686455            1.604           0.109
    L18.av_swsfcdown                      -2.275861         0.874125           -2.604           0.009
    L18.av_swsfcdown_numhours_1D          25.601051        56.773630            0.451           0.652
    L18.av_swsfcdown_numhours_10D         29.345161        23.684904            1.239           0.215
    L18.av_swsfcdown_sum_10D              -0.012206         0.019539           -0.625           0.532
    L18.forcing_temp_numhours_180D       -20.352116        16.564418           -1.229           0.219
    L18.forcing_temp_sum_180D              2.854832         3.149344            0.906           0.365
    L18.grass_count                        1.291392         0.686078            1.882           0.060
    L19.av_swsfcdown                      -1.745890         0.748034           -2.334           0.020
    L19.av_swsfcdown_numhours_1D          -7.531974        51.465609           -0.146           0.884
    L19.av_swsfcdown_numhours_10D          2.188818        14.712358            0.149           0.882
    L19.av_swsfcdown_sum_10D               0.025460         0.012128            2.099           0.036
    L19.forcing_temp_numhours_180D         4.398240        10.269645            0.428           0.668
    L19.forcing_temp_sum_180D             -0.021755         1.983697           -0.011           0.991
    L19.grass_count                       -1.258161         0.651920           -1.930           0.054
    =================================================================================================
    
    Results for equation forcing_temp_numhours_180D
    =================================================================================================
                                        coefficient       std. error           t-stat            prob
    -------------------------------------------------------------------------------------------------
    const                                 -1.300135         1.097147           -1.185           0.236
    L1.av_swsfcdown                        0.009829         0.001843            5.332           0.000
    L1.av_swsfcdown_numhours_1D           -0.033890         0.120377           -0.282           0.778
    L1.av_swsfcdown_numhours_10D           0.011088         0.041907            0.265           0.791
    L1.av_swsfcdown_sum_10D               -0.000085         0.000070           -1.213           0.225
    L1.forcing_temp_numhours_180D          1.478144         0.024026           61.523           0.000
    L1.forcing_temp_sum_180D              -0.021939         0.004679           -4.689           0.000
    L1.grass_count                         0.004019         0.001531            2.625           0.009
    L2.av_swsfcdown                       -0.000010         0.002063           -0.005           0.996
    L2.av_swsfcdown_numhours_1D           -0.130140         0.131727           -0.988           0.323
    L2.av_swsfcdown_numhours_10D          -0.014477         0.058172           -0.249           0.803
    L2.av_swsfcdown_sum_10D                0.000127         0.000076            1.678           0.093
    L2.forcing_temp_numhours_180D         -0.593398         0.038715          -15.327           0.000
    L2.forcing_temp_sum_180D               0.025234         0.007413            3.404           0.001
    L2.grass_count                        -0.002602         0.001610           -1.616           0.106
    L3.av_swsfcdown                        0.003069         0.002156            1.423           0.155
    L3.av_swsfcdown_numhours_1D           -0.093223         0.133184           -0.700           0.484
    L3.av_swsfcdown_numhours_10D           0.016843         0.058178            0.290           0.772
    L3.av_swsfcdown_sum_10D               -0.000117         0.000077           -1.527           0.127
    L3.forcing_temp_numhours_180D          0.140966         0.039521            3.567           0.000
    L3.forcing_temp_sum_180D              -0.008946         0.007478           -1.196           0.232
    L3.grass_count                         0.002364         0.001609            1.470           0.142
    L4.av_swsfcdown                        0.002460         0.002216            1.110           0.267
    L4.av_swsfcdown_numhours_1D            0.064551         0.134006            0.482           0.630
    L4.av_swsfcdown_numhours_10D          -0.002197         0.058221           -0.038           0.970
    L4.av_swsfcdown_sum_10D               -0.000041         0.000077           -0.527           0.598
    L4.forcing_temp_numhours_180D         -0.044061         0.039561           -1.114           0.265
    L4.forcing_temp_sum_180D               0.010365         0.007482            1.385           0.166
    L4.grass_count                        -0.001885         0.001609           -1.171           0.241
    L5.av_swsfcdown                       -0.000100         0.002236           -0.045           0.964
    L5.av_swsfcdown_numhours_1D           -0.003332         0.134224           -0.025           0.980
    L5.av_swsfcdown_numhours_10D          -0.039832         0.058489           -0.681           0.496
    L5.av_swsfcdown_sum_10D                0.000129         0.000077            1.677           0.093
    L5.forcing_temp_numhours_180D          0.026147         0.039573            0.661           0.509
    L5.forcing_temp_sum_180D              -0.002829         0.007483           -0.378           0.705
    L5.grass_count                         0.001466         0.001612            0.909           0.363
    L6.av_swsfcdown                       -0.000470         0.002221           -0.212           0.832
    L6.av_swsfcdown_numhours_1D            0.086354         0.134656            0.641           0.521
    L6.av_swsfcdown_numhours_10D           0.005579         0.058573            0.095           0.924
    L6.av_swsfcdown_sum_10D               -0.000018         0.000077           -0.238           0.812
    L6.forcing_temp_numhours_180D          0.066117         0.039562            1.671           0.095
    L6.forcing_temp_sum_180D              -0.014544         0.007481           -1.944           0.052
    L6.grass_count                        -0.003158         0.001621           -1.948           0.051
    L7.av_swsfcdown                        0.001591         0.002178            0.731           0.465
    L7.av_swsfcdown_numhours_1D           -0.097960         0.134661           -0.727           0.467
    L7.av_swsfcdown_numhours_10D           0.025809         0.058411            0.442           0.659
    L7.av_swsfcdown_sum_10D               -0.000035         0.000077           -0.452           0.651
    L7.forcing_temp_numhours_180D         -0.056411         0.039572           -1.426           0.154
    L7.forcing_temp_sum_180D               0.011403         0.007474            1.526           0.127
    L7.grass_count                        -0.001938         0.001623           -1.194           0.232
    L8.av_swsfcdown                        0.001160         0.002100            0.552           0.581
    L8.av_swsfcdown_numhours_1D            0.129101         0.134447            0.960           0.337
    L8.av_swsfcdown_numhours_10D          -0.014906         0.058526           -0.255           0.799
    L8.av_swsfcdown_sum_10D               -0.000022         0.000077           -0.287           0.774
    L8.forcing_temp_numhours_180D          0.006728         0.039591            0.170           0.865
    L8.forcing_temp_sum_180D              -0.007627         0.007468           -1.021           0.307
    L8.grass_count                        -0.001485         0.001625           -0.914           0.361
    L9.av_swsfcdown                        0.001974         0.001509            1.309           0.191
    L9.av_swsfcdown_numhours_1D           -0.112806         0.133908           -0.842           0.400
    L9.av_swsfcdown_numhours_10D          -0.011110         0.057848           -0.192           0.848
    L9.av_swsfcdown_sum_10D                0.000014         0.000068            0.212           0.832
    L9.forcing_temp_numhours_180D         -0.017809         0.039572           -0.450           0.653
    L9.forcing_temp_sum_180D               0.012051         0.007459            1.616           0.106
    L9.grass_count                         0.001570         0.001626            0.966           0.334
    L10.av_swsfcdown                       0.000440         0.001473            0.299           0.765
    L10.av_swsfcdown_numhours_1D           0.005955         0.133963            0.044           0.965
    L10.av_swsfcdown_numhours_10D          0.019635         0.055829            0.352           0.725
    L10.av_swsfcdown_sum_10D               0.000014         0.000051            0.263           0.792
    L10.forcing_temp_numhours_180D        -0.034957         0.039590           -0.883           0.377
    L10.forcing_temp_sum_180D              0.003286         0.007460            0.440           0.660
    L10.grass_count                        0.000015         0.001630            0.009           0.993
    L11.av_swsfcdown                       0.000309         0.001500            0.206           0.837
    L11.av_swsfcdown_numhours_1D          -0.028243         0.135695           -0.208           0.835
    L11.av_swsfcdown_numhours_10D         -0.002306         0.055674           -0.041           0.967
    L11.av_swsfcdown_sum_10D              -0.000008         0.000047           -0.174           0.862
    L11.forcing_temp_numhours_180D         0.042020         0.039628            1.060           0.289
    L11.forcing_temp_sum_180D             -0.009772         0.007464           -1.309           0.190
    L11.grass_count                        0.000656         0.001636            0.401           0.688
    L12.av_swsfcdown                      -0.000902         0.002114           -0.427           0.669
    L12.av_swsfcdown_numhours_1D           0.012181         0.136250            0.089           0.929
    L12.av_swsfcdown_numhours_10D         -0.010001         0.055742           -0.179           0.858
    L12.av_swsfcdown_sum_10D               0.000014         0.000047            0.299           0.765
    L12.forcing_temp_numhours_180D        -0.038242         0.039631           -0.965           0.335
    L12.forcing_temp_sum_180D              0.008791         0.007468            1.177           0.239
    L12.grass_count                        0.000236         0.001635            0.144           0.885
    L13.av_swsfcdown                      -0.000109         0.002188           -0.050           0.960
    L13.av_swsfcdown_numhours_1D          -0.010536         0.136098           -0.077           0.938
    L13.av_swsfcdown_numhours_10D         -0.026196         0.055756           -0.470           0.638
    L13.av_swsfcdown_sum_10D               0.000057         0.000047            1.223           0.221
    L13.forcing_temp_numhours_180D         0.036148         0.039655            0.912           0.362
    L13.forcing_temp_sum_180D             -0.000910         0.007475           -0.122           0.903
    L13.grass_count                        0.001414         0.001633            0.866           0.387
    L14.av_swsfcdown                      -0.001458         0.002232           -0.653           0.514
    L14.av_swsfcdown_numhours_1D           0.059062         0.136436            0.433           0.665
    L14.av_swsfcdown_numhours_10D          0.012797         0.055814            0.229           0.819
    L14.av_swsfcdown_sum_10D              -0.000030         0.000047           -0.653           0.514
    L14.forcing_temp_numhours_180D        -0.052159         0.039667           -1.315           0.189
    L14.forcing_temp_sum_180D              0.000199         0.007480            0.027           0.979
    L14.grass_count                       -0.000629         0.001631           -0.385           0.700
    L15.av_swsfcdown                      -0.001025         0.002249           -0.456           0.649
    L15.av_swsfcdown_numhours_1D           0.020359         0.135824            0.150           0.881
    L15.av_swsfcdown_numhours_10D          0.019045         0.055864            0.341           0.733
    L15.av_swsfcdown_sum_10D              -0.000050         0.000047           -1.083           0.279
    L15.forcing_temp_numhours_180D         0.076992         0.039663            1.941           0.052
    L15.forcing_temp_sum_180D             -0.011502         0.007477           -1.538           0.124
    L15.grass_count                       -0.001089         0.001620           -0.672           0.501
    L16.av_swsfcdown                       0.000043         0.002225            0.019           0.984
    L16.av_swsfcdown_numhours_1D           0.040958         0.135571            0.302           0.763
    L16.av_swsfcdown_numhours_10D         -0.023053         0.055559           -0.415           0.678
    L16.av_swsfcdown_sum_10D               0.000047         0.000047            1.011           0.312
    L16.forcing_temp_numhours_180D        -0.031298         0.039663           -0.789           0.430
    L16.forcing_temp_sum_180D              0.004791         0.007474            0.641           0.522
    L16.grass_count                       -0.000515         0.001615           -0.319           0.750
    L17.av_swsfcdown                       0.002421         0.002168            1.117           0.264
    L17.av_swsfcdown_numhours_1D          -0.001070         0.135148           -0.008           0.994
    L17.av_swsfcdown_numhours_10D          0.013617         0.055686            0.245           0.807
    L17.av_swsfcdown_sum_10D              -0.000031         0.000047           -0.655           0.512
    L17.forcing_temp_numhours_180D         0.014983         0.039634            0.378           0.705
    L17.forcing_temp_sum_180D             -0.003285         0.007475           -0.439           0.660
    L17.grass_count                        0.000894         0.001616            0.553           0.580
    L18.av_swsfcdown                      -0.000697         0.002058           -0.339           0.735
    L18.av_swsfcdown_numhours_1D           0.083285         0.133684            0.623           0.533
    L18.av_swsfcdown_numhours_10D         -0.018087         0.055771           -0.324           0.746
    L18.av_swsfcdown_sum_10D               0.000027         0.000046            0.578           0.563
    L18.forcing_temp_numhours_180D        -0.024771         0.039004           -0.635           0.525
    L18.forcing_temp_sum_180D              0.007430         0.007416            1.002           0.316
    L18.grass_count                       -0.001498         0.001615           -0.927           0.354
    L19.av_swsfcdown                      -0.002776         0.001761           -1.576           0.115
    L19.av_swsfcdown_numhours_1D           0.067649         0.121185            0.558           0.577
    L19.av_swsfcdown_numhours_10D          0.020643         0.034643            0.596           0.551
    L19.av_swsfcdown_sum_10D               0.000003         0.000029            0.099           0.921
    L19.forcing_temp_numhours_180D         0.004116         0.024182            0.170           0.865
    L19.forcing_temp_sum_180D             -0.001951         0.004671           -0.418           0.676
    L19.grass_count                       -0.000522         0.001535           -0.340           0.734
    =================================================================================================
    
    Results for equation forcing_temp_sum_180D
    =================================================================================================
                                        coefficient       std. error           t-stat            prob
    -------------------------------------------------------------------------------------------------
    const                                 -6.348823         5.620021           -1.130           0.259
    L1.av_swsfcdown                        0.031376         0.009443            3.323           0.001
    L1.av_swsfcdown_numhours_1D           -0.036156         0.616620           -0.059           0.953
    L1.av_swsfcdown_numhours_10D          -0.040477         0.214663           -0.189           0.850
    L1.av_swsfcdown_sum_10D               -0.000035         0.000358           -0.098           0.922
    L1.forcing_temp_numhours_180D          1.310197         0.123069           10.646           0.000
    L1.forcing_temp_sum_180D               1.129541         0.023967           47.129           0.000
    L1.grass_count                         0.006061         0.007843            0.773           0.440
    L2.av_swsfcdown                       -0.000087         0.010568           -0.008           0.993
    L2.av_swsfcdown_numhours_1D           -0.422697         0.674757           -0.626           0.531
    L2.av_swsfcdown_numhours_10D          -0.012736         0.297981           -0.043           0.966
    L2.av_swsfcdown_sum_10D                0.000344         0.000389            0.885           0.376
    L2.forcing_temp_numhours_180D         -1.884327         0.198312           -9.502           0.000
    L2.forcing_temp_sum_180D              -0.087158         0.037974           -2.295           0.022
    L2.grass_count                        -0.013191         0.008248           -1.599           0.110
    L3.av_swsfcdown                        0.019783         0.011046            1.791           0.073
    L3.av_swsfcdown_numhours_1D           -0.366595         0.682222           -0.537           0.591
    L3.av_swsfcdown_numhours_10D           0.198194         0.298011            0.665           0.506
    L3.av_swsfcdown_sum_10D               -0.000982         0.000393           -2.498           0.012
    L3.forcing_temp_numhours_180D          0.797445         0.202441            3.939           0.000
    L3.forcing_temp_sum_180D              -0.080387         0.038304           -2.099           0.036
    L3.grass_count                         0.012464         0.008241            1.512           0.130
    L4.av_swsfcdown                        0.009903         0.011350            0.873           0.383
    L4.av_swsfcdown_numhours_1D            0.334679         0.686434            0.488           0.626
    L4.av_swsfcdown_numhours_10D          -0.128869         0.298233           -0.432           0.666
    L4.av_swsfcdown_sum_10D                0.000204         0.000395            0.515           0.606
    L4.forcing_temp_numhours_180D         -0.446051         0.202649           -2.201           0.028
    L4.forcing_temp_sum_180D               0.078477         0.038325            2.048           0.041
    L4.grass_count                        -0.009354         0.008243           -1.135           0.256
    L5.av_swsfcdown                       -0.001973         0.011454           -0.172           0.863
    L5.av_swsfcdown_numhours_1D           -0.140357         0.687548           -0.204           0.838
    L5.av_swsfcdown_numhours_10D          -0.154606         0.299602           -0.516           0.606
    L5.av_swsfcdown_sum_10D                0.000549         0.000394            1.394           0.163
    L5.forcing_temp_numhours_180D          0.264969         0.202707            1.307           0.191
    L5.forcing_temp_sum_180D              -0.037804         0.038330           -0.986           0.324
    L5.grass_count                         0.009379         0.008259            1.136           0.256
    L6.av_swsfcdown                       -0.003248         0.011379           -0.285           0.775
    L6.av_swsfcdown_numhours_1D            0.401087         0.689761            0.581           0.561
    L6.av_swsfcdown_numhours_10D           0.025446         0.300034            0.085           0.932
    L6.av_swsfcdown_sum_10D                0.000019         0.000395            0.047           0.962
    L6.forcing_temp_numhours_180D          0.361325         0.202653            1.783           0.075
    L6.forcing_temp_sum_180D              -0.069149         0.038318           -1.805           0.071
    L6.grass_count                        -0.015658         0.008303           -1.886           0.059
    L7.av_swsfcdown                        0.000994         0.011157            0.089           0.929
    L7.av_swsfcdown_numhours_1D           -0.482272         0.689789           -0.699           0.484
    L7.av_swsfcdown_numhours_10D           0.088804         0.299202            0.297           0.767
    L7.av_swsfcdown_sum_10D               -0.000090         0.000394           -0.228           0.820
    L7.forcing_temp_numhours_180D         -0.317408         0.202701           -1.566           0.117
    L7.forcing_temp_sum_180D               0.056488         0.038285            1.475           0.140
    L7.grass_count                        -0.010343         0.008316           -1.244           0.214
    L8.av_swsfcdown                        0.002708         0.010755            0.252           0.801
    L8.av_swsfcdown_numhours_1D            0.664992         0.688692            0.966           0.334
    L8.av_swsfcdown_numhours_10D          -0.051643         0.299795           -0.172           0.863
    L8.av_swsfcdown_sum_10D               -0.000245         0.000393           -0.624           0.533
    L8.forcing_temp_numhours_180D         -0.043819         0.202802           -0.216           0.829
    L8.forcing_temp_sum_180D              -0.012153         0.038253           -0.318           0.751
    L8.grass_count                        -0.008498         0.008322           -1.021           0.307
    L9.av_swsfcdown                        0.006081         0.007729            0.787           0.431
    L9.av_swsfcdown_numhours_1D           -0.130567         0.685931           -0.190           0.849
    L9.av_swsfcdown_numhours_10D          -0.084142         0.296321           -0.284           0.776
    L9.av_swsfcdown_sum_10D                0.000176         0.000350            0.505           0.614
    L9.forcing_temp_numhours_180D         -0.192722         0.202703           -0.951           0.342
    L9.forcing_temp_sum_180D               0.068653         0.038209            1.797           0.072
    L9.grass_count                         0.014254         0.008328            1.712           0.087
    L10.av_swsfcdown                      -0.004810         0.007547           -0.637           0.524
    L10.av_swsfcdown_numhours_1D          -0.248440         0.686210           -0.362           0.717
    L10.av_swsfcdown_numhours_10D          0.052305         0.285977            0.183           0.855
    L10.av_swsfcdown_sum_10D               0.000136         0.000263            0.516           0.606
    L10.forcing_temp_numhours_180D        -0.117513         0.202795           -0.579           0.562
    L10.forcing_temp_sum_180D              0.029642         0.038214            0.776           0.438
    L10.grass_count                       -0.003929         0.008347           -0.471           0.638
    L11.av_swsfcdown                       0.000251         0.007685            0.033           0.974
    L11.av_swsfcdown_numhours_1D           0.206069         0.695086            0.296           0.767
    L11.av_swsfcdown_numhours_10D          0.059321         0.285183            0.208           0.835
    L11.av_swsfcdown_sum_10D              -0.000145         0.000238           -0.607           0.544
    L11.forcing_temp_numhours_180D         0.358025         0.202990            1.764           0.078
    L11.forcing_temp_sum_180D             -0.106871         0.038234           -2.795           0.005
    L11.grass_count                        0.000779         0.008380            0.093           0.926
    L12.av_swsfcdown                       0.000974         0.010828            0.090           0.928
    L12.av_swsfcdown_numhours_1D          -0.322222         0.697925           -0.462           0.644
    L12.av_swsfcdown_numhours_10D         -0.029830         0.285530           -0.104           0.917
    L12.av_swsfcdown_sum_10D               0.000061         0.000238            0.254           0.799
    L12.forcing_temp_numhours_180D        -0.414796         0.203007           -2.043           0.041
    L12.forcing_temp_sum_180D              0.108018         0.038256            2.824           0.005
    L12.grass_count                        0.000799         0.008377            0.095           0.924
    L13.av_swsfcdown                      -0.001436         0.011209           -0.128           0.898
    L13.av_swsfcdown_numhours_1D           0.117761         0.697147            0.169           0.866
    L13.av_swsfcdown_numhours_10D         -0.122465         0.285606           -0.429           0.668
    L13.av_swsfcdown_sum_10D               0.000254         0.000238            1.067           0.286
    L13.forcing_temp_numhours_180D         0.402374         0.203128            1.981           0.048
    L13.forcing_temp_sum_180D             -0.060710         0.038292           -1.585           0.113
    L13.grass_count                       -0.000823         0.008365           -0.098           0.922
    L14.av_swsfcdown                      -0.024616         0.011435           -2.153           0.031
    L14.av_swsfcdown_numhours_1D           0.454914         0.698879            0.651           0.515
    L14.av_swsfcdown_numhours_10D         -0.046498         0.285901           -0.163           0.871
    L14.av_swsfcdown_sum_10D               0.000025         0.000238            0.107           0.915
    L14.forcing_temp_numhours_180D        -0.262603         0.203193           -1.292           0.196
    L14.forcing_temp_sum_180D              0.000367         0.038318            0.010           0.992
    L14.grass_count                       -0.001910         0.008356           -0.229           0.819
    L15.av_swsfcdown                      -0.005479         0.011521           -0.476           0.634
    L15.av_swsfcdown_numhours_1D          -0.275053         0.695746           -0.395           0.693
    L15.av_swsfcdown_numhours_10D          0.070200         0.286156            0.245           0.806
    L15.av_swsfcdown_sum_10D              -0.000203         0.000238           -0.850           0.395
    L15.forcing_temp_numhours_180D         0.349163         0.203170            1.719           0.086
    L15.forcing_temp_sum_180D             -0.054442         0.038300           -1.421           0.155
    L15.grass_count                       -0.009177         0.008297           -1.106           0.269
    L16.av_swsfcdown                      -0.006011         0.011399           -0.527           0.598
    L16.av_swsfcdown_numhours_1D           0.608283         0.694450            0.876           0.381
    L16.av_swsfcdown_numhours_10D         -0.053404         0.284593           -0.188           0.851
    L16.av_swsfcdown_sum_10D               0.000149         0.000239            0.625           0.532
    L16.forcing_temp_numhours_180D        -0.242803         0.203167           -1.195           0.232
    L16.forcing_temp_sum_180D              0.051666         0.038285            1.350           0.177
    L16.grass_count                        0.000469         0.008271            0.057           0.955
    L17.av_swsfcdown                       0.011156         0.011103            1.005           0.315
    L17.av_swsfcdown_numhours_1D          -0.249522         0.692280           -0.360           0.719
    L17.av_swsfcdown_numhours_10D          0.084528         0.285244            0.296           0.767
    L17.av_swsfcdown_sum_10D              -0.000199         0.000238           -0.835           0.404
    L17.forcing_temp_numhours_180D         0.177582         0.203019            0.875           0.382
    L17.forcing_temp_sum_180D             -0.038431         0.038289           -1.004           0.316
    L17.grass_count                        0.005784         0.008280            0.699           0.485
    L18.av_swsfcdown                      -0.002944         0.010543           -0.279           0.780
    L18.av_swsfcdown_numhours_1D           0.442473         0.684783            0.646           0.518
    L18.av_swsfcdown_numhours_10D         -0.065489         0.285679           -0.229           0.819
    L18.av_swsfcdown_sum_10D               0.000043         0.000236            0.183           0.855
    L18.forcing_temp_numhours_180D        -0.108758         0.199794           -0.544           0.586
    L18.forcing_temp_sum_180D              0.025757         0.037986            0.678           0.498
    L18.grass_count                       -0.001220         0.008275           -0.147           0.883
    L19.av_swsfcdown                      -0.016388         0.009023           -1.816           0.069
    L19.av_swsfcdown_numhours_1D           0.606286         0.620760            0.977           0.329
    L19.av_swsfcdown_numhours_10D          0.061408         0.177455            0.346           0.729
    L19.av_swsfcdown_sum_10D               0.000088         0.000146            0.603           0.546
    L19.forcing_temp_numhours_180D         0.030641         0.123869            0.247           0.805
    L19.forcing_temp_sum_180D             -0.006519         0.023927           -0.272           0.785
    L19.grass_count                        0.001053         0.007863            0.134           0.893
    =================================================================================================
    
    Results for equation grass_count
    =================================================================================================
                                        coefficient       std. error           t-stat            prob
    -------------------------------------------------------------------------------------------------
    const                                  3.133940         8.294601            0.378           0.706
    L1.av_swsfcdown                        0.087029         0.013937            6.245           0.000
    L1.av_swsfcdown_numhours_1D           -0.287320         0.910071           -0.316           0.752
    L1.av_swsfcdown_numhours_10D          -0.513254         0.316821           -1.620           0.105
    L1.av_swsfcdown_sum_10D               -0.000057         0.000529           -0.107           0.915
    L1.forcing_temp_numhours_180D         -0.089495         0.181638           -0.493           0.622
    L1.forcing_temp_sum_180D              -0.031564         0.035373           -0.892           0.372
    L1.grass_count                         0.318036         0.011575           27.476           0.000
    L2.av_swsfcdown                       -0.018113         0.015598           -1.161           0.246
    L2.av_swsfcdown_numhours_1D            0.293688         0.995875            0.295           0.768
    L2.av_swsfcdown_numhours_10D          -0.016239         0.439791           -0.037           0.971
    L2.av_swsfcdown_sum_10D                0.000744         0.000574            1.295           0.195
    L2.forcing_temp_numhours_180D         -0.029730         0.292690           -0.102           0.919
    L2.forcing_temp_sum_180D               0.054873         0.056046            0.979           0.328
    L2.grass_count                        -0.021393         0.012173           -1.757           0.079
    L3.av_swsfcdown                        0.016778         0.016302            1.029           0.303
    L3.av_swsfcdown_numhours_1D            0.870762         1.006893            0.865           0.387
    L3.av_swsfcdown_numhours_10D           0.512151         0.439835            1.164           0.244
    L3.av_swsfcdown_sum_10D               -0.001622         0.000580           -2.796           0.005
    L3.forcing_temp_numhours_180D          0.108664         0.298783            0.364           0.716
    L3.forcing_temp_sum_180D              -0.034970         0.056534           -0.619           0.536
    L3.grass_count                         0.038223         0.012163            3.143           0.002
    L4.av_swsfcdown                       -0.023957         0.016751           -1.430           0.153
    L4.av_swsfcdown_numhours_1D            0.254694         1.013109            0.251           0.802
    L4.av_swsfcdown_numhours_10D           0.256959         0.440163            0.584           0.559
    L4.av_swsfcdown_sum_10D                0.001114         0.000583            1.911           0.056
    L4.forcing_temp_numhours_180D         -0.215366         0.299091           -0.720           0.471
    L4.forcing_temp_sum_180D               0.053857         0.056563            0.952           0.341
    L4.grass_count                         0.070733         0.012165            5.814           0.000
    L5.av_swsfcdown                       -0.007023         0.016905           -0.415           0.678
    L5.av_swsfcdown_numhours_1D            1.163821         1.014754            1.147           0.251
    L5.av_swsfcdown_numhours_10D          -0.368839         0.442183           -0.834           0.404
    L5.av_swsfcdown_sum_10D                0.000742         0.000581            1.277           0.202
    L5.forcing_temp_numhours_180D          0.440153         0.299175            1.471           0.141
    L5.forcing_temp_sum_180D              -0.109606         0.056571           -1.937           0.053
    L5.grass_count                         0.112694         0.012189            9.245           0.000
    L6.av_swsfcdown                       -0.010438         0.016794           -0.622           0.534
    L6.av_swsfcdown_numhours_1D           -0.525627         1.018020           -0.516           0.606
    L6.av_swsfcdown_numhours_10D           0.220140         0.442821            0.497           0.619
    L6.av_swsfcdown_sum_10D               -0.000873         0.000583           -1.498           0.134
    L6.forcing_temp_numhours_180D         -0.060480         0.299096           -0.202           0.840
    L6.forcing_temp_sum_180D               0.051114         0.056554            0.904           0.366
    L6.grass_count                         0.049599         0.012254            4.047           0.000
    L7.av_swsfcdown                        0.045318         0.016467            2.752           0.006
    L7.av_swsfcdown_numhours_1D           -0.891191         1.018061           -0.875           0.381
    L7.av_swsfcdown_numhours_10D           0.723761         0.441593            1.639           0.101
    L7.av_swsfcdown_sum_10D               -0.001818         0.000581           -3.127           0.002
    L7.forcing_temp_numhours_180D         -0.241709         0.299168           -0.808           0.419
    L7.forcing_temp_sum_180D               0.035121         0.056505            0.622           0.534
    L7.grass_count                         0.062060         0.012273            5.057           0.000
    L8.av_swsfcdown                        0.004262         0.015873            0.268           0.788
    L8.av_swsfcdown_numhours_1D           -0.542662         1.016442           -0.534           0.593
    L8.av_swsfcdown_numhours_10D          -0.825320         0.442468           -1.865           0.062
    L8.av_swsfcdown_sum_10D                0.001300         0.000580            2.241           0.025
    L8.forcing_temp_numhours_180D          0.195460         0.299316            0.653           0.514
    L8.forcing_temp_sum_180D              -0.051094         0.056457           -0.905           0.365
    L8.grass_count                         0.045508         0.012283            3.705           0.000
    L9.av_swsfcdown                        0.019464         0.011407            1.706           0.088
    L9.av_swsfcdown_numhours_1D           -1.392374         1.012368           -1.375           0.169
    L9.av_swsfcdown_numhours_10D           0.002119         0.437341            0.005           0.996
    L9.av_swsfcdown_sum_10D               -0.000190         0.000516           -0.369           0.712
    L9.forcing_temp_numhours_180D         -0.102313         0.299169           -0.342           0.732
    L9.forcing_temp_sum_180D               0.035448         0.056392            0.629           0.530
    L9.grass_count                         0.061243         0.012291            4.983           0.000
    L10.av_swsfcdown                      -0.003477         0.011139           -0.312           0.755
    L10.av_swsfcdown_numhours_1D          -0.213954         1.012778           -0.211           0.833
    L10.av_swsfcdown_numhours_10D          0.385191         0.422074            0.913           0.361
    L10.av_swsfcdown_sum_10D               0.000457         0.000389            1.177           0.239
    L10.forcing_temp_numhours_180D         0.027559         0.299306            0.092           0.927
    L10.forcing_temp_sum_180D             -0.012692         0.056400           -0.225           0.822
    L10.grass_count                       -0.001546         0.012319           -0.125           0.900
    L11.av_swsfcdown                       0.005838         0.011343            0.515           0.607
    L11.av_swsfcdown_numhours_1D           0.238297         1.025879            0.232           0.816
    L11.av_swsfcdown_numhours_10D         -0.520010         0.420903           -1.235           0.217
    L11.av_swsfcdown_sum_10D               0.000068         0.000352            0.193           0.847
    L11.forcing_temp_numhours_180D        -0.147449         0.299594           -0.492           0.623
    L11.forcing_temp_sum_180D              0.021702         0.056429            0.385           0.701
    L11.grass_count                        0.035805         0.012367            2.895           0.004
    L12.av_swsfcdown                       0.009867         0.015980            0.617           0.537
    L12.av_swsfcdown_numhours_1D          -0.487874         1.030069           -0.474           0.636
    L12.av_swsfcdown_numhours_10D          0.568590         0.421415            1.349           0.177
    L12.av_swsfcdown_sum_10D              -0.000413         0.000352           -1.175           0.240
    L12.forcing_temp_numhours_180D         0.398765         0.299618            1.331           0.183
    L12.forcing_temp_sum_180D             -0.053620         0.056463           -0.950           0.342
    L12.grass_count                        0.046983         0.012363            3.800           0.000
    L13.av_swsfcdown                       0.004887         0.016544            0.295           0.768
    L13.av_swsfcdown_numhours_1D           0.544509         1.028921            0.529           0.597
    L13.av_swsfcdown_numhours_10D         -0.722377         0.421527           -1.714           0.087
    L13.av_swsfcdown_sum_10D               0.000745         0.000352            2.119           0.034
    L13.forcing_temp_numhours_180D        -0.384333         0.299798           -1.282           0.200
    L13.forcing_temp_sum_180D              0.066430         0.056515            1.175           0.240
    L13.grass_count                        0.018054         0.012346            1.462           0.144
    L14.av_swsfcdown                      -0.051392         0.016876           -3.045           0.002
    L14.av_swsfcdown_numhours_1D          -0.369559         1.031477           -0.358           0.720
    L14.av_swsfcdown_numhours_10D          0.283274         0.421961            0.671           0.502
    L14.av_swsfcdown_sum_10D               0.000354         0.000352            1.007           0.314
    L14.forcing_temp_numhours_180D         0.133124         0.299892            0.444           0.657
    L14.forcing_temp_sum_180D             -0.021144         0.056553           -0.374           0.708
    L14.grass_count                       -0.002183         0.012332           -0.177           0.860
    L15.av_swsfcdown                       0.009572         0.017005            0.563           0.574
    L15.av_swsfcdown_numhours_1D           0.140070         1.026853            0.136           0.891
    L15.av_swsfcdown_numhours_10D         -0.088034         0.422338           -0.208           0.835
    L15.av_swsfcdown_sum_10D              -0.001009         0.000352           -2.867           0.004
    L15.forcing_temp_numhours_180D         0.020992         0.299859            0.070           0.944
    L15.forcing_temp_sum_180D             -0.018575         0.056527           -0.329           0.742
    L15.grass_count                        0.009202         0.012246            0.751           0.452
    L16.av_swsfcdown                       0.009325         0.016823            0.554           0.579
    L16.av_swsfcdown_numhours_1D           2.019447         1.024940            1.970           0.049
    L16.av_swsfcdown_numhours_10D          0.325968         0.420032            0.776           0.438
    L16.av_swsfcdown_sum_10D               0.000572         0.000352            1.626           0.104
    L16.forcing_temp_numhours_180D         0.062415         0.299855            0.208           0.835
    L16.forcing_temp_sum_180D             -0.009673         0.056504           -0.171           0.864
    L16.grass_count                       -0.041451         0.012207           -3.396           0.001
    L17.av_swsfcdown                      -0.011397         0.016388           -0.695           0.487
    L17.av_swsfcdown_numhours_1D           1.095895         1.021738            1.073           0.283
    L17.av_swsfcdown_numhours_10D          0.261662         0.420992            0.622           0.534
    L17.av_swsfcdown_sum_10D              -0.000428         0.000352           -1.216           0.224
    L17.forcing_temp_numhours_180D        -0.219019         0.299636           -0.731           0.465
    L17.forcing_temp_sum_180D              0.031665         0.056511            0.560           0.575
    L17.grass_count                        0.049153         0.012220            4.022           0.000
    L18.av_swsfcdown                      -0.022292         0.015561           -1.433           0.152
    L18.av_swsfcdown_numhours_1D          -1.975794         1.010673           -1.955           0.051
    L18.av_swsfcdown_numhours_10D         -0.189603         0.421634           -0.450           0.653
    L18.av_swsfcdown_sum_10D              -0.000138         0.000348           -0.398           0.691
    L18.forcing_temp_numhours_180D         0.074710         0.294876            0.253           0.800
    L18.forcing_temp_sum_180D              0.009534         0.056064            0.170           0.865
    L18.grass_count                       -0.012690         0.012213           -1.039           0.299
    L19.av_swsfcdown                      -0.011968         0.013316           -0.899           0.369
    L19.av_swsfcdown_numhours_1D          -1.325478         0.916180           -1.447           0.148
    L19.av_swsfcdown_numhours_10D         -0.179803         0.261906           -0.687           0.492
    L19.av_swsfcdown_sum_10D               0.000306         0.000216            1.417           0.157
    L19.forcing_temp_numhours_180D        -0.001855         0.182818           -0.010           0.992
    L19.forcing_temp_sum_180D             -0.010990         0.035313           -0.311           0.756
    L19.grass_count                       -0.018067         0.011605           -1.557           0.120
    =================================================================================================
    
    Correlation matrix of residuals
                                  av_swsfcdown  av_swsfcdown_numhours_1D  av_swsfcdown_numhours_10D  av_swsfcdown_sum_10D  forcing_temp_numhours_180D  forcing_temp_sum_180D  grass_count
    av_swsfcdown                      1.000000                  0.070566                   0.004515              0.902983                    0.061840               0.060232     0.037938
    av_swsfcdown_numhours_1D          0.070566                  1.000000                   0.288727              0.072125                    0.000760               0.003977     0.003863
    av_swsfcdown_numhours_10D         0.004515                  0.288727                   1.000000              0.183832                   -0.007246              -0.007123    -0.016251
    av_swsfcdown_sum_10D              0.902983                  0.072125                   0.183832              1.000000                    0.050891               0.049131     0.026709
    forcing_temp_numhours_180D        0.061840                  0.000760                  -0.007246              0.050891                    1.000000               0.876686     0.083722
    forcing_temp_sum_180D             0.060232                  0.003977                  -0.007123              0.049131                    0.876686               1.000000     0.047931
    grass_count                       0.037938                  0.003863                  -0.016251              0.026709                    0.083722               0.047931     1.000000
    




### Check for Serial Correlation of Residuals


```python
from statsmodels.stats.stattools import durbin_watson

out = durbin_watson(/docs/projects/capstone/Modeling/time_series/model_fitted.resid)
def adjust(/docs/projects/capstone/Modeling/time_series/val, length= 6): return str(val).ljust(length)
def adjust(/docs/projects/capstone/Modeling/time_series/val, length= 6): return str(/docs/projects/capstone/Modeling/time_series/val).ljust(length)
def adjust(val, /docs/projects/capstone/Modeling/time_series/length= 6): return str(val).ljust(/docs/projects/capstone/Modeling/time_series/length)
for col, val in zip(/docs/projects/capstone/Modeling/time_series/X_selected.columns, out):
    print(/docs/projects/capstone/Modeling/time_series/adjust(col), ':', round(val, 2))
    print(adjust(col), ':', round(/docs/projects/capstone/Modeling/time_series/val, 2))
```

    Count  : 2.11
    av_swsfcdown : 2.11
    av_temp_scrn : 2.08
    av_temp_scrn_max : 2.03
    av_temp_scrn_max_1h_rise : 2.02
    av_temp_scrn_max_afternoon : 2.03
    av_temp_scrn_min_afternoon : 2.05
    av_temp_scrn_max_day_3hrise : 2.04
    av_temp_scrn_max_day_3hfall : 1.98
    thermal_time_1D : 2.02


there is no significant serial correlation.

## Forecast


```python
nobs = 120
df_train, df_test =  X_selected[0:-nobs],X_selected[-nobs:]
print(/docs/projects/capstone/Modeling/time_series/df_train.shape)
print(/docs/projects/capstone/Modeling/time_series/df_test.shape)
```

    (/docs/projects/capstone/Modeling/time_series/7575, 7)
    (/docs/projects/capstone/Modeling/time_series/120, 7)



```python
# Get the lag order (/docs/projects/capstone/Modeling/time_series/we already know this)
lag_order = model_fitted.k_ar
print(/docs/projects/capstone/Modeling/time_series/lag_order) 

# Input data for forecasting
forecast_input = df_train.values[-lag_order:]
forecast_input

# Forecast
fc = model_fitted.forecast(/docs/projects/capstone/Modeling/time_series/y=forecast_input, steps=nobs) # nobs defined at top of program
df_forecast = pd.DataFrame(/docs/projects/capstone/Modeling/time_series/fc, index=X_selected.index[-nobs:], columns=X_selected.columns )
df_forecast
```

    19





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
      <th>av_swsfcdown</th>
      <th>av_swsfcdown_numhours_1D</th>
      <th>av_swsfcdown_numhours_10D</th>
      <th>av_swsfcdown_sum_10D</th>
      <th>forcing_temp_numhours_180D</th>
      <th>forcing_temp_sum_180D</th>
      <th>grass_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7575</th>
      <td>226.956079</td>
      <td>13.767396</td>
      <td>53.548465</td>
      <td>19656.306734</td>
      <td>0.889143</td>
      <td>4.515080</td>
      <td>64.269938</td>
    </tr>
    <tr>
      <th>7576</th>
      <td>93.226662</td>
      <td>14.378339</td>
      <td>50.899647</td>
      <td>18151.454863</td>
      <td>2.091688</td>
      <td>11.235209</td>
      <td>45.879837</td>
    </tr>
    <tr>
      <th>7577</th>
      <td>131.024275</td>
      <td>14.766558</td>
      <td>58.402320</td>
      <td>20684.124371</td>
      <td>3.099481</td>
      <td>17.223798</td>
      <td>-12.511248</td>
    </tr>
    <tr>
      <th>7578</th>
      <td>97.315053</td>
      <td>14.722950</td>
      <td>68.601553</td>
      <td>21900.496678</td>
      <td>3.670539</td>
      <td>20.518390</td>
      <td>-8.121121</td>
    </tr>
    <tr>
      <th>7579</th>
      <td>241.539024</td>
      <td>13.820285</td>
      <td>79.141845</td>
      <td>26955.751451</td>
      <td>4.603791</td>
      <td>27.173799</td>
      <td>1.153640</td>
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
    </tr>
    <tr>
      <th>7690</th>
      <td>282.642221</td>
      <td>15.614363</td>
      <td>150.383046</td>
      <td>68026.795613</td>
      <td>135.667491</td>
      <td>539.899884</td>
      <td>16.577869</td>
    </tr>
    <tr>
      <th>7691</th>
      <td>282.175396</td>
      <td>15.601817</td>
      <td>150.334638</td>
      <td>67977.748398</td>
      <td>137.469952</td>
      <td>546.575929</td>
      <td>16.417327</td>
    </tr>
    <tr>
      <th>7692</th>
      <td>281.683477</td>
      <td>15.588924</td>
      <td>150.281474</td>
      <td>67921.935709</td>
      <td>139.268810</td>
      <td>553.246290</td>
      <td>16.254382</td>
    </tr>
    <tr>
      <th>7693</th>
      <td>281.168129</td>
      <td>15.575693</td>
      <td>150.223643</td>
      <td>67859.344820</td>
      <td>141.063577</td>
      <td>559.909082</td>
      <td>16.089183</td>
    </tr>
    <tr>
      <th>7694</th>
      <td>280.630273</td>
      <td>15.562132</td>
      <td>150.161235</td>
      <td>67790.111033</td>
      <td>142.853743</td>
      <td>566.562254</td>
      <td>15.922166</td>
    </tr>
  </tbody>
</table>
<p>120 rows × 7 columns</p>
</div>




```python
ig, axes = plt.subplots(/docs/projects/capstone/Modeling/time_series/nrows=int(len(X_selected.columns)/2), ncols=2, dpi=150, figsize=(10,10))
ig, axes = plt.subplots(nrows=int(len(X_selected.columns)/2), ncols=2, dpi=150, figsize=(/docs/projects/capstone/Modeling/time_series/10,10))
for i, (/docs/projects/capstone/Modeling/time_series/col,ax) in enumerate(zip(X_selected.columns, axes.flatten())):
for i, (col,ax) in enumerate(/docs/projects/capstone/Modeling/time_series/zip(X_selected.columns, axes.flatten())):
    df_forecast[col].plot(/docs/projects/capstone/Modeling/time_series/legend=True, ax=ax).autoscale(axis='x',tight=True)
    df_forecast[col].plot(legend=True, ax=ax).autoscale(/docs/projects/capstone/Modeling/time_series/axis='x',tight=True)
    df_test[col][-nobs:].plot(/docs/projects/capstone/Modeling/time_series/legend=True, ax=ax);
    ax.set_title(/docs/projects/capstone/Modeling/time_series/col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position(/docs/projects/capstone/Modeling/time_series/'none')
    ax.yaxis.set_ticks_position(/docs/projects/capstone/Modeling/time_series/'none')
    ax.spines["top"].set_alpha(/docs/projects/capstone/Modeling/time_series/0)
    ax.tick_params(/docs/projects/capstone/Modeling/time_series/labelsize=6)

/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/i/docs/projects/capstone/Modeling/time_series/g/docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/y/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/u/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/;/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/```


    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_68_0.png)
    



```python
test_original = X_selected[-nobs:]
# /docs/projects/capstone/Modeling/time_series/test_original.index = pd.to_datetime(/docs/projects/capstone/Modeling/time_series/test_original.index)
test_original
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
      <th>av_swsfcdown</th>
      <th>av_swsfcdown_numhours_1D</th>
      <th>av_swsfcdown_numhours_10D</th>
      <th>av_swsfcdown_sum_10D</th>
      <th>forcing_temp_numhours_180D</th>
      <th>forcing_temp_sum_180D</th>
      <th>grass_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>7575</th>
      <td>237.758330</td>
      <td>14.0</td>
      <td>57.0</td>
      <td>21019.652594</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>7576</th>
      <td>114.223600</td>
      <td>15.0</td>
      <td>58.0</td>
      <td>20827.112662</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>47.0</td>
    </tr>
    <tr>
      <th>7577</th>
      <td>151.055200</td>
      <td>15.0</td>
      <td>69.0</td>
      <td>24477.129897</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7578</th>
      <td>111.615201</td>
      <td>15.0</td>
      <td>83.0</td>
      <td>27114.429910</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7579</th>
      <td>257.357501</td>
      <td>14.0</td>
      <td>97.0</td>
      <td>33291.009923</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>2.0</td>
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
    </tr>
    <tr>
      <th>7690</th>
      <td>130.549167</td>
      <td>15.0</td>
      <td>152.0</td>
      <td>71764.820016</td>
      <td>113.0</td>
      <td>341.59007</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7691</th>
      <td>334.555417</td>
      <td>15.0</td>
      <td>152.0</td>
      <td>72318.180003</td>
      <td>113.0</td>
      <td>341.59007</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>7692</th>
      <td>132.476798</td>
      <td>16.0</td>
      <td>152.0</td>
      <td>67785.839900</td>
      <td>113.0</td>
      <td>341.59007</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>7693</th>
      <td>86.142084</td>
      <td>15.0</td>
      <td>152.0</td>
      <td>63012.549925</td>
      <td>113.0</td>
      <td>341.59007</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7694</th>
      <td>275.843329</td>
      <td>15.0</td>
      <td>152.0</td>
      <td>60367.329937</td>
      <td>113.0</td>
      <td>341.59007</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>120 rows × 7 columns</p>
</div>




```python
test_original['grass_count'].plot(/docs/projects/capstone/Modeling/time_series/figsize=(12,5),legend=True)
/docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/c/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/[/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/g/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/c/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/u/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/]/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/plt.legend(/docs/projects/capstone/Modeling/time_series/labels=['grass_count','forecast_grass_count'])
```




    <matplotlib.legend.Legend at 0x29e7d16d0>




    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_70_1.png)
    



```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
y_true = test_original['grass_count']
y_pred = df_forecast['grass_count']
rmse = np.sqrt(/docs/projects/capstone/Modeling/time_series/mean_squared_error(y_true, y_pred))
print(/docs/projects/capstone/Modeling/time_series/'RMSE Error in forecasts = {}'.format(round(rmse, 2)))
```

    RMSE Error in forecasts = 61.8



```python
# Visualize the forecasts (/docs/projects/capstone/Modeling/time_series/blue=train, green=forecasts)
f, ax = plt.subplots(/docs/projects/capstone/Modeling/time_series/1)
f.set_figheight(/docs/projects/capstone/Modeling/time_series/5)
f.set_figwidth(/docs/projects/capstone/Modeling/time_series/18)

# fig = model.plot(/docs/projects/capstone/Modeling/time_series/forecasts, ax=ax)
# x = np.arange(/docs/projects/capstone/Modeling/time_series/len(df1))
x=df1.index
plt.plot(/docs/projects/capstone/Modeling/time_series/ x[-126:],test['y'], color='r')
plt.plot(/docs/projects/capstone/Modeling/time_series/x[:-126], train, c='blue')
plt.plot(/docs/projects/capstone/Modeling/time_series/x[-120:], df_forecast['grass_count'], c='green')
# test_original['grass_count'].plot(/docs/projects/capstone/Modeling/time_series/figsize=(12,5),legend=True)
/docs/projects/capstone/Modeling/time_series/#/docs/projects/capstone/Modeling/time_series/ /docs/projects/capstone/Modeling/time_series/d/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/f/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/e/docs/projects/capstone/Modeling/time_series/c/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/[/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/g/docs/projects/capstone/Modeling/time_series/r/docs/projects/capstone/Modeling/time_series/a/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/_/docs/projects/capstone/Modeling/time_series/c/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/u/docs/projects/capstone/Modeling/time_series/n/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/'/docs/projects/capstone/Modeling/time_series/]/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/plt.legend(/docs/projects/capstone/Modeling/time_series/labels=['grass_count','forecast_grass_count'])
/docs/projects/capstone/Modeling/time_series/p/docs/projects/capstone/Modeling/time_series/l/docs/projects/capstone/Modeling/time_series/t/docs/projects/capstone/Modeling/time_series/./docs/projects/capstone/Modeling/time_series/s/docs/projects/capstone/Modeling/time_series/h/docs/projects/capstone/Modeling/time_series/o/docs/projects/capstone/Modeling/time_series/w/docs/projects/capstone/Modeling/time_series/(/docs/projects/capstone/Modeling/time_series/)/docs/projects/capstone/Modeling/time_series/
/docs/projects/capstone/Modeling/time_series/```


    
![png](/docs/projects/capstone/Modeling/time_series/time%20series_files/time%20series_72_0.png)
    

