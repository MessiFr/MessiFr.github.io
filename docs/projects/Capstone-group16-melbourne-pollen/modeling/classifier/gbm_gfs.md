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

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, recall_score
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
# grass_df.head(/docs/projects/capstone/Modeling/classifier/2)
```


```python
# load weather data
weather_dict = pickle.load(/docs/projects/capstone/Modeling/classifier/open('../weather_v2.pkl', "rb"))


# add grass count data and assigned by labels
grass_df = pd.read_csv(/docs/projects/capstone/Modeling/classifier/'../../preprocessing/gfs_preprocessed_new.csv')
# grass_df

grass_df['date'] = grass_df['date'].apply(/docs/projects/capstone/Modeling/classifier/lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))
# grass_df['non_grass'] = grass_df['Total'] - grass_df['grass_count']

label = [[100, 4], [50, 3], [20, 2], [-1, 1]]

def get_label(/docs/projects/capstone/Modeling/classifier/x):
    for i in label:
        if x >= i[0]:
            return i[1]

grass_df['label'] = grass_df['grass_count'].apply(/docs/projects/capstone/Modeling/classifier/lambda x:get_label(x))
```


```python
/docs/projects/capstone/Modeling/classifier/g/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/b/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/v/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/```




    1    2551
    2     177
    3      96
    4      58
    Name: label, dtype: int64



Filter the data in **Melbourne** and merge the pollen data and the weather data as the train dataset


```python
# weather_dict[1]
full_data = grass_df.drop(/docs/projects/capstone/Modeling/classifier/['train', 'date', 'grass_count'], axis=1)
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
      <th>v_10m_max_4pm</th>
      <th>v_10m_sd_4pm</th>
      <th>pwat_mean_4pm</th>
      <th>pwat_min_4pm</th>
      <th>pwat_max_4pm</th>
      <th>pwat_sd_4pm</th>
      <th>year</th>
      <th>month</th>
      <th>season</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>285.875</td>
      <td>280.1</td>
      <td>290.8</td>
      <td>4.78008</td>
      <td>285.75</td>
      <td>280.4</td>
      <td>290.3</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>101654.0</td>
      <td>...</td>
      <td>6.0</td>
      <td>1.961293</td>
      <td>13.875</td>
      <td>13.0</td>
      <td>14.9</td>
      <td>0.880814</td>
      <td>2000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>285.875</td>
      <td>280.1</td>
      <td>290.8</td>
      <td>4.78008</td>
      <td>285.75</td>
      <td>280.4</td>
      <td>290.3</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>101654.0</td>
      <td>...</td>
      <td>2.9</td>
      <td>5.402777</td>
      <td>16.925</td>
      <td>13.6</td>
      <td>20.0</td>
      <td>2.780138</td>
      <td>2000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>285.875</td>
      <td>280.1</td>
      <td>290.8</td>
      <td>4.78008</td>
      <td>285.75</td>
      <td>280.4</td>
      <td>290.3</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>101654.0</td>
      <td>...</td>
      <td>5.7</td>
      <td>1.971464</td>
      <td>22.625</td>
      <td>18.8</td>
      <td>24.7</td>
      <td>2.617091</td>
      <td>2000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>285.875</td>
      <td>280.1</td>
      <td>290.8</td>
      <td>4.78008</td>
      <td>285.75</td>
      <td>280.4</td>
      <td>290.3</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>101654.0</td>
      <td>...</td>
      <td>6.2</td>
      <td>1.161895</td>
      <td>16.400</td>
      <td>12.3</td>
      <td>20.2</td>
      <td>3.799123</td>
      <td>2000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>285.875</td>
      <td>280.1</td>
      <td>290.8</td>
      <td>4.78008</td>
      <td>285.75</td>
      <td>280.4</td>
      <td>290.3</td>
      <td>4.430575</td>
      <td>101775.75</td>
      <td>101654.0</td>
      <td>...</td>
      <td>5.6</td>
      <td>1.286792</td>
      <td>14.550</td>
      <td>13.4</td>
      <td>15.1</td>
      <td>0.776745</td>
      <td>2000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 60 columns</p>
</div>




```python
level1 = full_data.loc[full_data.label == 1]
level2 = full_data.loc[full_data.label == 2]
level3 = full_data.loc[full_data.label == 3]
level4 = full_data.loc[full_data.label == 4]

# full_data = pd.concat(/docs/projects/capstone/Modeling/classifier/(level1[:177], level2, level3, level4), axis=1)
```


```python
full_data = pd.concat(/docs/projects/capstone/Modeling/classifier/(level1[:177], level2, level3, level4), axis=0)
```

#### Randomly Re-Sample
Generate variables in dataframe as `X_origin`, choose the label as target as `Y_origin`. </br>
Since the labels in raw data doesn't distribute evenly, so we use the `RandomOverSample` method to resample the data. ***


```python
X_origin = full_data.drop(/docs/projects/capstone/Modeling/classifier/['label'], axis=1).reset_index(drop=True)
X_origin = full_data.drop(['label'], axis=1).reset_index(/docs/projects/capstone/Modeling/classifier/drop=True)
Y_origin = full_data['label']

X_origin, Y_origin = RandomOverSampler(/docs/projects/capstone/Modeling/classifier/random_state=0).fit_resample(X_origin, Y_origin)
/docs/projects/capstone/Modeling/classifier/X_origin, Y_origin = RandomOverSampler(random_state=0).fit_resample(/docs/projects/capstone/Modeling/classifier/X_origin, Y_origin)
print(/docs/projects/capstone/Modeling/classifier/Y_origin.value_counts())
```

    1    177
    2    177
    3    177
    4    177
    Name: label, dtype: int64



```python
X_origin.shape
```




    (/docs/projects/capstone/Modeling/classifier/708, 59)



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




    0.8290598290598291



### Model Evaluation


```python

```


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


    
![png](/docs/projects/capstone/Modeling/classifier/gbm_gfs_files/gbm_gfs_17_0.png)
    



```python
from sklearn.metrics import classification_report

class_report = classification_report(/docs/projects/capstone/Modeling/classifier/Y_test, Y_pred)
print(/docs/projects/capstone/Modeling/classifier/class_report)
```

                  precision    recall  f1-score   support
    
               1       1.00      1.00      1.00        56
               2       0.91      0.62      0.74        66
               3       0.62      0.87      0.72        53
               4       0.86      0.86      0.86        59
    
        accuracy                           0.83       234
       macro avg       0.85      0.84      0.83       234
    weighted avg       0.86      0.83      0.83       234
    



```python
from sklearn.metrics import f1_score
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
    The accuracy  : 0.8290598290598291
      The recall  : 0.8290598290598291
          The f1  : 0.8297028100177706



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


    
![png](/docs/projects/capstone/Modeling/classifier/gbm_gfs_files/gbm_gfs_23_0.png)
    


Check the overfitting of model.


```python
from support_evaluation import check_overfitting
check_overfitting(/docs/projects/capstone/Modeling/classifier/GradientBoostingClassifier(), X_origin, Y_origin)
```

    [Parallel(/docs/projects/capstone/Modeling/classifier/n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(/docs/projects/capstone/Modeling/classifier/n_jobs=-1)]: Done   2 out of  25 | elapsed:   10.6s remaining:  2.0min
    [Parallel(/docs/projects/capstone/Modeling/classifier/n_jobs=-1)]: Done  25 out of  25 | elapsed:  3.4min finished



    
![png](/docs/projects/capstone/Modeling/classifier/gbm_gfs_files/gbm_gfs_25_1.png)
    


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
    {'kbest__score_func': (/docs/projects/capstone/Modeling/classifier/<function f_regression at 0x13d9bcd30>, <function mutual_info_regression at 0x111ef6f80>), 'kbest__k': (20, 50, 100, 150, 'all'), 'clf__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]}
    {'kbest__score_func': (<function f_regression at 0x13d9bcd30>, <function mutual_info_regression at 0x111ef6f80>), 'kbest__k': (/docs/projects/capstone/Modeling/classifier/20, 50, 100, 150, 'all'), 'clf__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2]}
    Fitting 5 folds for each of 50 candidates, totalling 250 fits
    done in 1915.043s
    
    {'clf__learning_rate': 0.05, 'kbest__k': 150, 'kbest__score_func': <function f_regression at 0x13d9bcd30>}
    nan
    ==============================
    {'clf__learning_rate': 0.05, 'kbest__k': 150, 'kbest__score_func': <function mutual_info_regression at 0x111ef6f80>}
    nan
    ==============================
    {'clf__learning_rate': 0.1, 'kbest__k': 150, 'kbest__score_func': <function mutual_info_regression at 0x111ef6f80>}
    nan
    ==============================
    {'clf__learning_rate': 0.1, 'kbest__k': 100, 'kbest__score_func': <function mutual_info_regression at 0x111ef6f80>}
    nan
    ==============================
    {'clf__learning_rate': 0.1, 'kbest__k': 100, 'kbest__score_func': <function f_regression at 0x13d9bcd30>}
    nan
    ==============================



```python
drop_columns = ['Unnamed: 0', 'grass_count', 'Total', 'date', 'year' ,'date_1', 'month', 'label']

data = pd.read_csv(/docs/projects/capstone/Modeling/classifier/'../preprocessing/melbourne_data.csv')

/docs/projects/capstone/Modeling/classifier/data['date'] = pd.to_datetime(/docs/projects/capstone/Modeling/classifier/data['date'])
/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/h/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
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
      <th>Elevation</th>
      <th>Total</th>
      <th>Location</th>
      <th>grass_count</th>
      <th>label</th>
      <th>av_abl_ht</th>
      <th>accum_prcp</th>
      <th>av_lwsfcdown</th>
      <th>av_mslp</th>
      <th>...</th>
      <th>thermal_time_180D</th>
      <th>soil_mois_1D</th>
      <th>soil_mois_10D</th>
      <th>soil_mois_30D</th>
      <th>soil_mois_90D</th>
      <th>soil_mois_180D</th>
      <th>date</th>
      <th>year</th>
      <th>date_1</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>769</td>
      <td>13.0</td>
      <td>76.0</td>
      <td>25</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>1490.955</td>
      <td>0.130126</td>
      <td>291.743750</td>
      <td>100769.72</td>
      <td>...</td>
      <td>476.875</td>
      <td>2537.160156</td>
      <td>25259.441406</td>
      <td>73932.160156</td>
      <td>214786.558594</td>
      <td>413919.671875</td>
      <td>2000-10-01</td>
      <td>2000</td>
      <td>10-01</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>770</td>
      <td>13.0</td>
      <td>63.0</td>
      <td>25</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1083.750</td>
      <td>0.011325</td>
      <td>312.027500</td>
      <td>101798.64</td>
      <td>...</td>
      <td>477.125</td>
      <td>2537.273438</td>
      <td>25255.527344</td>
      <td>73959.710938</td>
      <td>214964.300781</td>
      <td>414139.425781</td>
      <td>2000-10-02</td>
      <td>2000</td>
      <td>10-02</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2</th>
      <td>771</td>
      <td>13.0</td>
      <td>65.0</td>
      <td>25</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>990.995</td>
      <td>0.002729</td>
      <td>326.742500</td>
      <td>102008.00</td>
      <td>...</td>
      <td>476.250</td>
      <td>2536.863281</td>
      <td>25253.625000</td>
      <td>73984.609375</td>
      <td>215138.472656</td>
      <td>414358.984375</td>
      <td>2000-10-03</td>
      <td>2000</td>
      <td>10-03</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>772</td>
      <td>13.0</td>
      <td>498.0</td>
      <td>25</td>
      <td>12.0</td>
      <td>4.0</td>
      <td>1080.790</td>
      <td>0.000000</td>
      <td>306.050625</td>
      <td>101879.16</td>
      <td>...</td>
      <td>478.250</td>
      <td>2534.246094</td>
      <td>25254.082031</td>
      <td>74023.683594</td>
      <td>215475.507812</td>
      <td>414795.257812</td>
      <td>2000-10-05</td>
      <td>2000</td>
      <td>10-05</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>773</td>
      <td>13.0</td>
      <td>423.0</td>
      <td>25</td>
      <td>18.0</td>
      <td>4.0</td>
      <td>1495.965</td>
      <td>0.000030</td>
      <td>332.835625</td>
      <td>101365.80</td>
      <td>...</td>
      <td>476.375</td>
      <td>2532.039062</td>
      <td>25254.269531</td>
      <td>74037.093750</td>
      <td>215637.449219</td>
      <td>415010.820312</td>
      <td>2000-10-06</td>
      <td>2000</td>
      <td>10-06</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 181 columns</p>
</div>




```python
train_df = data.loc[data['date'] < pd.Timestamp(/docs/projects/capstone/Modeling/classifier/2020, 10, 1)]
eval_df = data.loc[data['date'] >= pd.Timestamp(/docs/projects/capstone/Modeling/classifier/2020, 10, 1)]
```


```python

```


```python
X_train, X_test, Y_train, Y_test = train_test_split(/docs/projects/capstone/Modeling/classifier/train_df.drop(drop_columns, axis=1), train_df.label, test_size=0.33, random_state=88)

print(/docs/projects/capstone/Modeling/classifier/X_train.shape)
print(/docs/projects/capstone/Modeling/classifier/X_test.shape)
```

    (/docs/projects/capstone/Modeling/classifier/1305, 173)
    (/docs/projects/capstone/Modeling/classifier/644, 173)



```python
/docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/=/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/G/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/B/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/g/docs/projects/capstone/Modeling/classifier/C/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/model.fit(/docs/projects/capstone/Modeling/classifier/X_train, Y_train)
model.score(/docs/projects/capstone/Modeling/classifier/X_test, Y_test)
```




    0.6754658385093167




```python

```


```python

```


```python
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_regression, f_classif, mutual_info_regression
from sklearn.model_selection import train_test_split, cross_val_score
```


```python
selector = SelectKBest(/docs/projects/capstone/Modeling/classifier/f_classif, k=20)
selector.fit(/docs/projects/capstone/Modeling/classifier/X_origin, Y_origin)

cols = selector.get_support(/docs/projects/capstone/Modeling/classifier/indices=True)
features_df_new = melbourne_df.iloc[:,cols]

melbourne_df[features_df_new.columns].head(/docs/projects/capstone/Modeling/classifier/5)


# features_df_new = data.iloc[:,cols]

# data[features_df_new.columns].head(/docs/projects/capstone/Modeling/classifier/5)
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
      <th>av_lwsfcdown</th>
      <th>av_abl_ht_sum</th>
      <th>av_lwsfcdown_sum</th>
      <th>av_mslp_max_1h_rise</th>
      <th>av_mslp_max_1h_fall</th>
      <th>av_qsair_scrn_max</th>
      <th>av_qsair_scrn_sum_afternoon</th>
      <th>av_qsair_scrn_max_afternoon_1hrise</th>
      <th>av_qsair_scrn_max_day_3hfall</th>
      <th>av_temp_scrn_max_afternoon</th>
      <th>av_temp_scrn_min_afternoon</th>
      <th>av_wndgust10m_sum_afternoon</th>
      <th>av_wndgust10m_max_afternoon_1hfall</th>
      <th>av_wndgust10m_max_day_3hfall</th>
      <th>morning_precp</th>
      <th>morning_hrs_of_precp</th>
      <th>next_morning_hrs_of_precp</th>
      <th>av_swsfcdown_numhours_30D</th>
      <th>topt_numhours_30D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>567.409722</td>
      <td>291.515625</td>
      <td>10213.375</td>
      <td>5247.281250</td>
      <td>126.0</td>
      <td>-81.0</td>
      <td>0.005371</td>
      <td>0.024658</td>
      <td>0.000000</td>
      <td>-0.000732</td>
      <td>286.078125</td>
      <td>284.500000</td>
      <td>43.750</td>
      <td>-2.125</td>
      <td>-6.000</td>
      <td>0.000035</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>379.0</td>
      <td>722.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>672.985000</td>
      <td>316.624375</td>
      <td>16824.625</td>
      <td>7915.609375</td>
      <td>63.0</td>
      <td>-122.0</td>
      <td>0.005615</td>
      <td>0.037598</td>
      <td>0.000488</td>
      <td>-0.000977</td>
      <td>289.093750</td>
      <td>285.453125</td>
      <td>53.500</td>
      <td>-0.500</td>
      <td>-4.500</td>
      <td>0.141462</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>380.0</td>
      <td>722.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1351.370000</td>
      <td>320.018125</td>
      <td>33784.250</td>
      <td>8000.453125</td>
      <td>155.0</td>
      <td>-183.0</td>
      <td>0.006592</td>
      <td>0.041748</td>
      <td>0.000732</td>
      <td>-0.000732</td>
      <td>297.750000</td>
      <td>291.109375</td>
      <td>85.250</td>
      <td>-2.375</td>
      <td>-5.500</td>
      <td>0.623656</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>381.0</td>
      <td>722.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>896.015000</td>
      <td>318.508125</td>
      <td>22400.375</td>
      <td>7962.703125</td>
      <td>104.0</td>
      <td>-26.0</td>
      <td>0.006592</td>
      <td>0.044922</td>
      <td>0.000732</td>
      <td>-0.001465</td>
      <td>289.765625</td>
      <td>285.640625</td>
      <td>83.000</td>
      <td>-1.750</td>
      <td>-3.875</td>
      <td>0.245373</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>382.0</td>
      <td>722.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1343.250000</td>
      <td>309.666466</td>
      <td>34924.500</td>
      <td>8051.328125</td>
      <td>200.0</td>
      <td>-81.0</td>
      <td>0.005615</td>
      <td>0.041016</td>
      <td>0.000244</td>
      <td>-0.000977</td>
      <td>290.890625</td>
      <td>287.906250</td>
      <td>98.375</td>
      <td>-1.500</td>
      <td>-2.875</td>
      <td>0.003570</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>384.0</td>
      <td>723.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
features_df_new.columns
```




    Index(['av_abl_ht', 'av_lwsfcdown', 'av_abl_ht_sum', 'av_lwsfcdown_sum',
           'av_mslp_max_1h_rise', 'av_mslp_max_1h_fall', 'av_qsair_scrn_max',
           'av_qsair_scrn_sum_afternoon', 'av_qsair_scrn_max_afternoon_1hrise',
           'av_qsair_scrn_max_day_3hfall', 'av_temp_scrn_max_afternoon',
           'av_temp_scrn_min_afternoon', 'av_wndgust10m_sum_afternoon',
           'av_wndgust10m_max_afternoon_1hfall', 'av_wndgust10m_max_day_3hfall',
           'morning_precp', 'morning_hrs_of_precp', 'next_morning_hrs_of_precp',
           'av_swsfcdown_numhours_30D', 'topt_numhours_30D'],
          dtype='object')




```python
# melbourne_df[['Count Date', 'Total', *features_df_new.columns]]
melbourne_df.head(/docs/projects/capstone/Modeling/classifier/5)
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
      <th>Elevation</th>
      <th>Total</th>
      <th>Location</th>
      <th>SchColTime</th>
      <th>grass_count</th>
      <th>label</th>
      <th>av_abl_ht</th>
      <th>accum_prcp</th>
      <th>av_lwsfcdown</th>
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
      <td>2017-09-26</td>
      <td>28.0</td>
      <td>76.0</td>
      <td>1</td>
      <td>09:00:00</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>567.409722</td>
      <td>0.014479</td>
      <td>291.515625</td>
      <td>...</td>
      <td>35.250</td>
      <td>87.500</td>
      <td>87.500</td>
      <td>486.375</td>
      <td>1792.332031</td>
      <td>24080.046875</td>
      <td>71414.628906</td>
      <td>209709.917969</td>
      <td>415718.011719</td>
      <td>2017-09-26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-09-27</td>
      <td>28.0</td>
      <td>335.0</td>
      <td>1</td>
      <td>09:00:00</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>672.985000</td>
      <td>0.044749</td>
      <td>316.624375</td>
      <td>...</td>
      <td>36.250</td>
      <td>89.250</td>
      <td>89.250</td>
      <td>487.500</td>
      <td>2489.031250</td>
      <td>24086.992188</td>
      <td>71454.339844</td>
      <td>209820.257812</td>
      <td>415804.148438</td>
      <td>2017-09-27</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-09-28</td>
      <td>28.0</td>
      <td>857.0</td>
      <td>1</td>
      <td>09:00:00</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1351.370000</td>
      <td>0.038105</td>
      <td>320.018125</td>
      <td>...</td>
      <td>36.500</td>
      <td>90.875</td>
      <td>90.875</td>
      <td>487.375</td>
      <td>2488.683594</td>
      <td>24090.761719</td>
      <td>71490.339844</td>
      <td>209930.417969</td>
      <td>415890.109375</td>
      <td>2017-09-28</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-09-29</td>
      <td>28.0</td>
      <td>235.0</td>
      <td>1</td>
      <td>09:00:00</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>896.015000</td>
      <td>0.011231</td>
      <td>318.508125</td>
      <td>...</td>
      <td>36.125</td>
      <td>91.000</td>
      <td>91.000</td>
      <td>487.000</td>
      <td>2488.316406</td>
      <td>24092.531250</td>
      <td>71522.316406</td>
      <td>210040.222656</td>
      <td>415975.867188</td>
      <td>2017-09-29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-09-30</td>
      <td>28.0</td>
      <td>263.0</td>
      <td>1</td>
      <td>09:00:00</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1343.250000</td>
      <td>0.055731</td>
      <td>309.666466</td>
      <td>...</td>
      <td>36.125</td>
      <td>91.125</td>
      <td>91.125</td>
      <td>487.500</td>
      <td>2587.183594</td>
      <td>24192.238281</td>
      <td>71650.304688</td>
      <td>210248.914062</td>
      <td>416160.636719</td>
      <td>2017-09-30</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 179 columns</p>
</div>




```python
plotData = melbourne_df.loc[melbourne_df['Count Date'] > pd.Timestamp(/docs/projects/capstone/Modeling/classifier/2019, 10, 1)]
plotData = plotData.loc[plotData['Count Date'] < pd.Timestamp(/docs/projects/capstone/Modeling/classifier/2020, 1, 1)]
```


```python
def normalize(/docs/projects/capstone/Modeling/classifier/df):
/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/=/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/c/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/p/docs/projects/capstone/Modeling/classifier/y/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/    for feature_name in df.columns:
        try:
/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/x/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/v/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/=/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/[/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/]/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/x/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier//docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/v/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/=/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/d/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/[/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/_/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/a/docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/]/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/m/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/n/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/            result[feature_name] = (/docs/projects/capstone/Modeling/classifier/df[feature_name] - min_value) / (max_value - min_value)
            result[feature_name] = (df[feature_name] - min_value) / (/docs/projects/capstone/Modeling/classifier/max_value - min_value)
        except:
            continue
    return result
```


```python
/docs/projects/capstone/Modeling/classifier/plotData = normalize(/docs/projects/capstone/Modeling/classifier/plotData)
```


```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import statsmodels

# feature = 'topt_numhours_30D'
# feature = 'av_wndgust10m_sum_afternoon'
feature = 'av_qsair_scrn_max_afternoon_1hrise'

/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/g/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/=/docs/projects/capstone/Modeling/classifier/ /docs/projects/capstone/Modeling/classifier/g/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/F/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/g/docs/projects/capstone/Modeling/classifier/u/docs/projects/capstone/Modeling/classifier/r/docs/projects/capstone/Modeling/classifier/e/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/
# fig = make_subplots(/docs/projects/capstone/Modeling/classifier/rows=1, cols=2)

fig.add_trace(go.Scatter(x=plotData['Count Date'], y=plotData['label'], 
                mode='lines', name='grass_count'))

fig.add_trace(go.Scatter(x=plotData['Count Date'], y=plotData[feature], 
                opacity=0.65, mode='markers', name=feature))


# fig = px.line(/docs/projects/capstone/Modeling/classifier/plotData, x="Count Date", y="grass_count")
# fig = px.scatter(/docs/projects/capstone/Modeling/classifier/plotData, x="Count Date", y="av_lwsfcdown", opacity=0.65)
# fig.update_layout(/docs/projects/capstone/Modeling/classifier/height=300, width=500, title_text="Side By Side Subplots")
/docs/projects/capstone/Modeling/classifier/f/docs/projects/capstone/Modeling/classifier/i/docs/projects/capstone/Modeling/classifier/g/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/h/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/w/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/```




```python
import seaborn as sns

sns.set(/docs/projects/capstone/Modeling/classifier/rc={'figure.figsize':(16,10)})
sns.heatmap(/docs/projects/capstone/Modeling/classifier/features_df_new.corr(),
            annot=True,
            linewidths=.5,
            center=0,
            cbar=False,
            # cmap="PiYG"
            )
/docs/projects/capstone/Modeling/classifier/p/docs/projects/capstone/Modeling/classifier/l/docs/projects/capstone/Modeling/classifier/t/docs/projects/capstone/Modeling/classifier/./docs/projects/capstone/Modeling/classifier/s/docs/projects/capstone/Modeling/classifier/h/docs/projects/capstone/Modeling/classifier/o/docs/projects/capstone/Modeling/classifier/w/docs/projects/capstone/Modeling/classifier/(/docs/projects/capstone/Modeling/classifier/)/docs/projects/capstone/Modeling/classifier/
/docs/projects/capstone/Modeling/classifier/```


    
![png](/docs/projects/capstone/Modeling/classifier/gbm_gfs_files/gbm_gfs_45_0.png)
    



```python
features_df_new.columns
```




    Index(['av_abl_ht', 'av_lwsfcdown', 'av_abl_ht_sum', 'av_lwsfcdown_sum',
           'av_mslp_max_1h_rise', 'av_mslp_max_1h_fall', 'av_qsair_scrn_max',
           'av_qsair_scrn_sum_afternoon', 'av_qsair_scrn_max_afternoon_1hrise',
           'av_qsair_scrn_max_day_3hfall', 'av_temp_scrn_max_afternoon',
           'av_temp_scrn_min_afternoon', 'av_wndgust10m_sum_afternoon',
           'av_wndgust10m_max_afternoon_1hfall', 'av_wndgust10m_max_day_3hfall',
           'morning_precp', 'morning_hrs_of_precp', 'next_morning_hrs_of_precp',
           'av_swsfcdown_numhours_30D', 'topt_numhours_30D'],
          dtype='object')




```python

```
