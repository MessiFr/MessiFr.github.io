# COMP90051 Workshop 3
## Linear regression

***

In this worksheet, we'll explore ordinary least squares (/docs/notes/sml/OLS) regression: both simple linear regression and basis function regression. Our key objectives are:

* to implement OLS regression using numerical linear algebra functions from the `numpy` library
* to practice using the scikit-learn interface for linear regression
* to implement basis function regression as an extension of OLS linear regression

Firstly, we'll import the relevant libraries (/docs/notes/sml/`numpy`, `matplotlib`, etc.).


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(/docs/notes/sml/'darkgrid')
plt.rcParams['figure.dpi'] = 108
```

### 1. The Boston Housing dataset

Throughout this worksheet, we'll use the _Boston Housing dataset_ as an example. 
It contains data about towns in the Boston area, which can be used to predict median house values. 
There are 506 observations in total, each of which is described by 13 features, such as _per capita crime rate_, _percentage of population with a lower socio-economic status_, etc. 
You can read more about the features [here](/docs/notes/sml/http://lib.stat.cmu.edu/datasets/boston).

Let's begin by loading the data (/docs/notes/sml/from scikit-learn) and converting to a Pandas dataframe.


```python
from sklearn.datasets import load_boston
/docs/notes/sml/b/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/_/docs/notes/sml/b/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/ds = pd.DataFrame(/docs/notes/sml/boston.data, columns=boston.feature_names)
y = pd.Series(/docs/notes/sml/boston.target, name='MEDV')
/docs/notes/sml/d/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```

    /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function load_boston is deprecated; `load_boston` is deprecated in 1.0 and will be removed in 1.2.
    
        The Boston housing prices dataset has an ethical problem. You can refer to
        the documentation of this function for further details.
    
        The scikit-learn maintainers therefore strongly discourage the use of this
        dataset unless the purpose of the code is to study and educate about
        ethical issues in data science and machine learning.
    
        In this special case, you can fetch the dataset from the original
        source::
    
            import pandas as pd
            import numpy as np
    
            data_url = "http://lib.stat.cmu.edu/datasets/boston"
            raw_df = pd.read_csv(/docs/notes/sml/data_url, sep="\s+", skiprows=22, header=None)
            data = np.hstack(/docs/notes/sml/[raw_df.values[::2, :], raw_df.values[1::2, :2]])
            target = raw_df.values[1::2, 2]
    
        Alternative datasets include the California housing dataset (i.e.
        :func:`~sklearn.datasets.fetch_california_housing`) and the Ames housing
        dataset. You can load the datasets as follows::
    
            from sklearn.datasets import fetch_california_housing
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/u/docs/notes/sml/s/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/g/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/f/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/_/docs/notes/sml/c/docs/notes/sml/a/docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/f/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/n/docs/notes/sml/i/docs/notes/sml/a/docs/notes/sml/_/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/u/docs/notes/sml/s/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/g/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/    
        for the California housing dataset and::
    
            from sklearn.datasets import fetch_openml
            housing = fetch_openml(/docs/notes/sml/name="house_prices", as_frame=True)
    
        for the Ames housing dataset.
      warnings.warn(/docs/notes/sml/msg, category=FutureWarning)





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
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>



To keep things simple, we'll work with a single feature called `LSTAT` for now. 
It corresponds to the percentage of the population in the town classified as 'lower status' by the US Census service in 1978. 
Note that the response variable (/docs/notes/sml/the median house value in the town) is denoted `MEDV`.
Plotting the  `MEDV` vs. `LSTAT` we see that a linear model appears plausible:


```python
features = ['LSTAT']

#ds['LSTAT'] = ds['LSTAT'].apply(/docs/notes/sml/lambda x: x/100.)
for f in features:
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/f/docs/notes/sml/i/docs/notes/sml/g/docs/notes/sml/u/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/    plt.scatter(/docs/notes/sml/ds[f], y, marker='.')
    plt.xlabel(/docs/notes/sml/f)
    plt.ylabel(/docs/notes/sml/r'Median House Value ($\times 10^3$ USD)')
```


    
![png](/docs/notes/sml/worksheet03_solutions_files/worksheet03_solutions_5_0.png)
    


***
**Question:** Have we made any mistakes in our analysis so far?

_Answer: Yes, a minor one. It's important that the model development process is not informed at all by the test data.
Technically, we shouldn't even visualise the test data, in case any observed trends/patterns inform our choice of model._
***

Let's now randomly split the data into training and test sets.
This is necessary, so we can assess the generalisation error of our model later on.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(/docs/notes/sml/ds, y, test_size=0.2, random_state=90051)
print(/docs/notes/sml/"Training set has {} instances. Test set has {} instances.".format(X_train.shape[0], X_test.shape[0]))
# select subset of the features
X_train_s = X_train[features].values
X_test_s = X_test[features].values
```

    Training set has 404 instances. Test set has 102 instances.


### 2. Linear algebra solution

Let's now fit a linear regression model to the single-featured Boston Housing data. 
In order to appreciate the lower-level details, we'll start by fitting the model using numerical linear algebra functions. 
The same approach is used under the hood in machine learning libraries, such as scikit-learn.

***
**Exercise:** In lectures, we derived an analytic expression for the optimal weights $\mathbf{w}^* = \left[\mathbf{X}^\top \mathbf{X}\right]^{-1} \mathbf{X}^\top \mathbf{y}$. Attempt the derivation yourself using the following matrix calculus identities: 

$$
\begin{gather}
    \nabla_{\mathbf{w}} \mathbf{w}^\top \mathbf{x} = \nabla_{\mathbf{w}} \mathbf{x}^\top \mathbf{w} = \mathbf{x}^\top \\
    \nabla_{\mathbf{w}} \mathbf{A}\mathbf{w} = \mathbf{A} \\
    \nabla_{\mathbf{w}} \mathbf{w}^\top \mathbf{A}\mathbf{w} = \mathbf{w}^\top \left(/docs/notes/sml/\mathbf{A}^\top + \mathbf{A}\right)
\end{gather}
$$

where vector $\mathbf{x}$ and matrix $\mathbf{A}$ are constants (/docs/notes/sml/independent of $\mathbf{w}$).

**Solution:**
Under a decision theoretic framework, the empirical risk can be expressed as 
$$
\hat{R}(/docs/notes/sml/\mathbf{w}) = \frac{1}{n} \sum_{i = 1}^{n} (\mathbf{x}_i^\top /docs/notes/sml/\mathbf{w} - y_i)^2  = \|\mathbf{X} /docs/notes/sml/\mathbf{w} - \mathbf{y}\|_2^2 = /docs/notes/sml/\mathbf{w}^\top \mathbf{X}^\top \mathbf{X} /docs/notes/sml/\mathbf{w} - /docs/notes/sml/\mathbf{w}^\top \mathbf{X}^\top \mathbf{y}  - \mathbf{y}^\top \mathbf{X} /docs/notes/sml/\mathbf{w} + \mathbf{y}^\top \mathbf{y}
\hat{R}(/docs/notes/sml/\mathbf{w}) = \frac{1}{n} \sum_{i = 1}^{n} (/docs/notes/sml/\mathbf{x}_i^\top /docs/notes/sml/\mathbf{w} - y_i)^2  = \|\mathbf{X} /docs/notes/sml/\mathbf{w} - \mathbf{y}\|_2^2 = /docs/notes/sml/\mathbf{w}^\top \mathbf{X}^\top \mathbf{X} /docs/notes/sml/\mathbf{w} - /docs/notes/sml/\mathbf{w}^\top \mathbf{X}^\top \mathbf{y}  - \mathbf{y}^\top \mathbf{X} /docs/notes/sml/\mathbf{w} + \mathbf{y}^\top \mathbf{y}
$$
where 
$\mathbf{X} = \begin{pmatrix} \mathbf{x}_1^\top \\ \mathbf{x}_2^\top \\ \vdots \\ \mathbf{x}_n^\top \end{pmatrix}$ is the _design matrix_, 
$\mathbf{y} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}$ and 
$\mathbf{w} = \begin{pmatrix} w_0 \\ w_1 \\ \vdots \\ w_n \end{pmatrix}$.

The optimal weight vector is a minimiser of the empirical risk, i.e. $/docs/notes/sml/\mathbf{w}^\star \in \arg \min_{/docs/notes/sml/\mathbf{w}} \hat{R}(/docs/notes/sml/\mathbf{w})$. 
To get the _normal equations_ we solve for the stationary points of $\hat{R}(/docs/notes/sml/\mathbf{w})$, i.e. $\nabla_{/docs/notes/sml/\mathbf{w}} \hat{R}(/docs/notes/sml/\mathbf{w}) = 0$.
To get the _normal equations_ we solve for the stationary points of $\hat{R}(/docs/notes/sml/\mathbf{w})$, i.e. $\nabla_{/docs/notes/sml/\mathbf{w}} \hat{R}(/docs/notes/sml/\mathbf{w}) = 0$.
We have

$$
\nabla_{/docs/notes/sml/\mathbf{w}} \hat{R}(/docs/notes/sml/\mathbf{w}) = \nabla_{/docs/notes/sml/\mathbf{w}} (/docs/notes/sml/\mathbf{w}^T \mathbf{X}^\top \mathbf{X} /docs/notes/sml/\mathbf{w}) - \nabla_{/docs/notes/sml/\mathbf{w}} ( /docs/notes/sml/\mathbf{w}^\top \mathbf{X}^\top \mathbf{y}) - \nabla_{/docs/notes/sml/\mathbf{w}} (\mathbf{y}^\top \mathbf{X} /docs/notes/sml/\mathbf{w}) = 2 /docs/notes/sml/\mathbf{w}^\top \mathbf{X}^\top \mathbf{X} - 2 \mathbf{y}^\top \mathbf{X}
\nabla_{/docs/notes/sml/\mathbf{w}} \hat{R}(/docs/notes/sml/\mathbf{w}) = \nabla_{/docs/notes/sml/\mathbf{w}} (/docs/notes/sml//docs/notes/sml/\mathbf{w}^T \mathbf{X}^\top \mathbf{X} /docs/notes/sml/\mathbf{w}) - \nabla_{/docs/notes/sml/\mathbf{w}} ( /docs/notes/sml/\mathbf{w}^\top \mathbf{X}^\top \mathbf{y}) - \nabla_{/docs/notes/sml/\mathbf{w}} (\mathbf{y}^\top \mathbf{X} /docs/notes/sml/\mathbf{w}) = 2 /docs/notes/sml/\mathbf{w}^\top \mathbf{X}^\top \mathbf{X} - 2 \mathbf{y}^\top \mathbf{X}
\nabla_{\mathbf{w}} \hat{R}(\mathbf{w}) = \nabla_{\mathbf{w}} (/docs/notes/sml/\mathbf{w}^T \mathbf{X}^\top \mathbf{X} \mathbf{w}) - \nabla_{\mathbf{w}} (/docs/notes/sml/ \mathbf{w}^\top \mathbf{X}^\top \mathbf{y}) - \nabla_{\mathbf{w}} (\mathbf{y}^\top \mathbf{X} \mathbf{w}) = 2 \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} - 2 \mathbf{y}^\top \mathbf{X}
\nabla_{\mathbf{w}} \hat{R}(\mathbf{w}) = \nabla_{\mathbf{w}} (/docs/notes/sml/\mathbf{w}^T \mathbf{X}^\top \mathbf{X} \mathbf{w}) - \nabla_{\mathbf{w}} ( \mathbf{w}^\top \mathbf{X}^\top \mathbf{y}) - \nabla_{\mathbf{w}} (/docs/notes/sml/\mathbf{y}^\top \mathbf{X} \mathbf{w}) = 2 \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} - 2 \mathbf{y}^\top \mathbf{X}
\nabla_{/docs/notes/sml/\mathbf{w}} \hat{R}(/docs/notes/sml/\mathbf{w}) = \nabla_{/docs/notes/sml/\mathbf{w}} (/docs/notes/sml/\mathbf{w}^T \mathbf{X}^\top \mathbf{X} /docs/notes/sml/\mathbf{w}) - \nabla_{/docs/notes/sml/\mathbf{w}} (/docs/notes/sml/ /docs/notes/sml/\mathbf{w}^\top \mathbf{X}^\top \mathbf{y}) - \nabla_{/docs/notes/sml/\mathbf{w}} (\mathbf{y}^\top \mathbf{X} /docs/notes/sml/\mathbf{w}) = 2 /docs/notes/sml/\mathbf{w}^\top \mathbf{X}^\top \mathbf{X} - 2 \mathbf{y}^\top \mathbf{X}
\nabla_{\mathbf{w}} \hat{R}(\mathbf{w}) = \nabla_{\mathbf{w}} (/docs/notes/sml/\mathbf{w}^T \mathbf{X}^\top \mathbf{X} \mathbf{w}) - \nabla_{\mathbf{w}} (/docs/notes/sml/ \mathbf{w}^\top \mathbf{X}^\top \mathbf{y}) - \nabla_{\mathbf{w}} (\mathbf{y}^\top \mathbf{X} \mathbf{w}) = 2 \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} - 2 \mathbf{y}^\top \mathbf{X}
\nabla_{\mathbf{w}} \hat{R}(\mathbf{w}) = \nabla_{\mathbf{w}} (\mathbf{w}^T \mathbf{X}^\top \mathbf{X} \mathbf{w}) - \nabla_{\mathbf{w}} (/docs/notes/sml/ \mathbf{w}^\top \mathbf{X}^\top \mathbf{y}) - \nabla_{\mathbf{w}} (/docs/notes/sml/\mathbf{y}^\top \mathbf{X} \mathbf{w}) = 2 \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} - 2 \mathbf{y}^\top \mathbf{X}
\nabla_{/docs/notes/sml/\mathbf{w}} \hat{R}(/docs/notes/sml/\mathbf{w}) = \nabla_{/docs/notes/sml/\mathbf{w}} (/docs/notes/sml/\mathbf{w}^T \mathbf{X}^\top \mathbf{X} /docs/notes/sml/\mathbf{w}) - \nabla_{/docs/notes/sml/\mathbf{w}} ( /docs/notes/sml/\mathbf{w}^\top \mathbf{X}^\top \mathbf{y}) - \nabla_{/docs/notes/sml/\mathbf{w}} (/docs/notes/sml/\mathbf{y}^\top \mathbf{X} /docs/notes/sml/\mathbf{w}) = 2 /docs/notes/sml/\mathbf{w}^\top \mathbf{X}^\top \mathbf{X} - 2 \mathbf{y}^\top \mathbf{X}
\nabla_{\mathbf{w}} \hat{R}(\mathbf{w}) = \nabla_{\mathbf{w}} (/docs/notes/sml/\mathbf{w}^T \mathbf{X}^\top \mathbf{X} \mathbf{w}) - \nabla_{\mathbf{w}} ( \mathbf{w}^\top \mathbf{X}^\top \mathbf{y}) - \nabla_{\mathbf{w}} (/docs/notes/sml/\mathbf{y}^\top \mathbf{X} \mathbf{w}) = 2 \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} - 2 \mathbf{y}^\top \mathbf{X}
\nabla_{\mathbf{w}} \hat{R}(\mathbf{w}) = \nabla_{\mathbf{w}} (\mathbf{w}^T \mathbf{X}^\top \mathbf{X} \mathbf{w}) - \nabla_{\mathbf{w}} (/docs/notes/sml/ \mathbf{w}^\top \mathbf{X}^\top \mathbf{y}) - \nabla_{\mathbf{w}} (/docs/notes/sml/\mathbf{y}^\top \mathbf{X} \mathbf{w}) = 2 \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} - 2 \mathbf{y}^\top \mathbf{X}
$$

which implies $\mathbf{w}^* = \left[\mathbf{X}^\top \mathbf{X}\right]^{-1} \mathbf{X}^\top \mathbf{y}$
***

Although we can express $\mathbf{w}^\star$ explicitly in terms of the matrix inverse $(/docs/notes/sml/\mathbf{X}^\top \mathbf{X})^{-1}$, this isn't an efficient way to compute $\mathbf{w}$ numerically (we typically never compute the inverse of a matrix exactly when solving the system $A\mathbf{x} = b$ for numerical stability). It is better instead to solve the following system of linear equations:
Although we can express $\mathbf{w}^\star$ explicitly in terms of the matrix inverse $(/docs/notes/sml/\mathbf{X}^\top \mathbf{X})^{-1}$, this isn't an efficient way to compute $\mathbf{w}$ numerically (/docs/notes/sml/we typically never compute the inverse of a matrix exactly when solving the system $A\mathbf{x} = b$ for numerical stability). It is better instead to solve the following system of linear equations:
$$\mathbf{X}^\top\mathbf{X} \mathbf{w}^\star = \mathbf{X}^\top\mathbf{y}$$

***
**Exercise:** Use `np.linalg.solve` to solve for $\mathbf{w}^\star$ using the single-featured training data.
_Hint: You can enter `/docs/notes/sml/np.linalg.solve?` or `help(/docs/notes/sml/np.linalg.solve)` to see the docstring (help file)._
_Hint: You can enter `/docs/notes/sml/np.linalg.solve?` or `help(/docs/notes/sml/np.linalg.solve)` to see the docstring (/docs/notes/sml/help file)._
***


```python
# Prepend a column of 1's to the design matrices (/docs/notes/sml/since we absorbed the bias term in the weights vector)
X_train_b = np.column_stack(/docs/notes/sml/(np.ones_like(X_train_s), X_train_s))
X_test_b = np.column_stack(/docs/notes/sml/(np.ones_like(X_test_s), X_test_s))
print(/docs/notes/sml/'Design matrix shape:', X_train_s.shape)

w = np.linalg.solve(/docs/notes/sml/np.dot(X_train_b.T, X_train_b), np.dot(X_train_b.T, y_train)) # fill in
w = np.linalg.solve(/docs/notes/sml/np.dot(X_train_b.T, X_train_b), np.dot(/docs/notes/sml/X_train_b.T, y_train)) # fill in
print(/docs/notes/sml/'Weights:', w)
```

    Design matrix shape: (/docs/notes/sml/404, 1)
    Weights: [34.51530004 -0.95801769]


Let's check our implementation by plotting the predictions on the test data.


```python
def predict(/docs/notes/sml/X, w):
    """Return the predicted response for a given design matrix and weights vector
    """
    return np.dot(/docs/notes/sml/X, w)

X_grid = np.linspace(/docs/notes/sml/X_train_s.min(), X_train_s.max(), num=1001)
/docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/i/docs/notes/sml/d/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/n/docs/notes/sml/p/docs/notes/sml/./docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/s/docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/e/docs/notes/sml/(/docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/x/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/n/docs/notes/sml/u/docs/notes/sml/m/docs/notes/sml/=/docs/notes/sml/1/docs/notes/sml/0/docs/notes/sml/0/docs/notes/sml/1/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/x = np.column_stack(/docs/notes/sml/(np.ones_like(X_grid), X_grid))
y = predict(/docs/notes/sml/x, w)
plt.plot(/docs/notes/sml/X_grid, y, 'k-', label='Prediction')
plt.scatter(/docs/notes/sml/X_train_s, y_train, color='b', marker='.', label='Train')
#plt.scatter(/docs/notes/sml/X_test_s, y_test, color='r', marker='.', label='Test')
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/plt.ylabel(/docs/notes/sml/"$y$ (Median House Value)")
plt.xlabel(/docs/notes/sml/"$x$ (LSTAT)")
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/w/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```


    
![png](/docs/notes/sml/worksheet03_solutions_files/worksheet03_solutions_11_0.png)
    


We'll compute the mean error term over the training and test sets to assess model performance.


```python
def mean_squared_error(/docs/notes/sml/y_true, y_pred):
    return np.mean(/docs/notes/sml/(y_pred - y_true)**2) 

y_pred_train = predict(/docs/notes/sml/X_train_b, w)
y_pred_test = predict(/docs/notes/sml/X_test_b, w)
print(/docs/notes/sml/'Train MSE:', mean_squared_error(y_pred_train, y_train))
print(/docs/notes/sml/'Test MSE:', mean_squared_error(y_pred_test, y_test))
```

    Train MSE: 38.632216441608094
    Test MSE: 38.00420488101305


### 4. Linear regression using scikit-learn

Now that you have a good understanding of what's going on under the hood, you can use the functionality in scikit-learn to solve linear regression problems you encounter in the future. Using the `LinearRegression` module, fitting a linear regression model becomes a one-liner as shown below.


```python
from sklearn.linear_model import LinearRegression
/docs/notes/sml/l/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/L/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/R/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/f/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/s/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/y/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml///docs/notes/sml/d/docs/notes/sml/o/docs/notes/sml/c/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/n/docs/notes/sml/o/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/s/docs/notes/sml/m/docs/notes/sml/l/docs/notes/sml///docs/notes/sml/l/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/L/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/R/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/f/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml///docs/notes/sml/d/docs/notes/sml/o/docs/notes/sml/c/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/n/docs/notes/sml/o/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/s/docs/notes/sml/m/docs/notes/sml/l/docs/notes/sml///docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/s/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/y/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```

The `LinearRegression` module provides access to the bias weight $w_0$ under the `intercept_` property


```python
lr.intercept_
```




    34.5153000408642



and the non-bias weights under the `coef_` property


```python
lr.coef_
```




    array(/docs/notes/sml/[-0.95801769])



You should check that these results match the solution you obtained previously. Note that sklearn also uses a numerical linear algebra solver under the hood.

Finally, what happens if we use the other 12 features available in the dataset?


```python
/docs/notes/sml/l/docs/notes/sml/r/docs/notes/sml/_/docs/notes/sml/f/docs/notes/sml/u/docs/notes/sml/l/docs/notes/sml/l/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/L/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/R/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/f/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/y/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml///docs/notes/sml/d/docs/notes/sml/o/docs/notes/sml/c/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/n/docs/notes/sml/o/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/s/docs/notes/sml/m/docs/notes/sml/l/docs/notes/sml///docs/notes/sml/l/docs/notes/sml/r/docs/notes/sml/_/docs/notes/sml/f/docs/notes/sml/u/docs/notes/sml/l/docs/notes/sml/l/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/L/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/R/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/f/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml///docs/notes/sml/d/docs/notes/sml/o/docs/notes/sml/c/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/n/docs/notes/sml/o/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/s/docs/notes/sml/m/docs/notes/sml/l/docs/notes/sml///docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/y/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/y_pred_train = lr_full.predict(/docs/notes/sml/X_train)
y_pred_test = lr_full.predict(/docs/notes/sml/X_test)

print(/docs/notes/sml/'Train MSE:', mean_squared_error(y_pred_train, y_train))
print(/docs/notes/sml/'Test MSE:', mean_squared_error(y_pred_test, y_test))
```

    Train MSE: 20.05928429120229
    Test MSE: 30.726949873388705


As measured by the MSE, our predictions are looking better. Although we obtained a performance boost here, in real problems you should be cautious of blindly including features in your analysis just because you can.

## 5. Basis expansion

Linear regression is simple and easy to interpret, however it cannot capture nonlinear relationships between the response and features. 
To deal with this problem, we can extend linear regression by mapping the features into another space where the relationship is (/docs/notes/sml/ideally) linear. 
This is known as _basis expansion_.
Specifically, we map the original feature vector $/docs/notes/sml/\mathbf{x} \in \mathbb{R}^m$ to a new feature vector $\varphi(/docs/notes/sml/\mathbf{x}) \in \mathbb{R}^k$. 
We then perform linear regression as before, replacing the original feature vectors with the new feature vectors: $y = w_0 + \sum_{i = 1}^{m} w_i \cdot \varphi_i(/docs/notes/sml/\mathbf{x})$. 
Note that this function is nonlinear in $\mathbf{x}$, but linear in $\mathbf{w}$. 

All of the previous results for simple linear regression carry over as you would expect by making the replacement $/docs/notes/sml/\mathbf{x} \to \varphi(/docs/notes/sml/\mathbf{x})$. 
For instance, the normal equation becomes: 

$$
\mathbf{w}^\star = \left[\mathbf{\Phi}^\top \mathbf{\Phi}\right]^{-1} \mathbf{\Phi}^\top \mathbf{y}
$$

where $\mathbf{\Phi} = \begin{pmatrix} \varphi(/docs/notes/sml/\mathbf{x}_1)^\top \\ \vdots \\ \varphi(\mathbf{x}_n)^\top \end{pmatrix}$ denotes the transformed design matrix.
where $\mathbf{\Phi} = \begin{pmatrix} \varphi(/docs/notes/sml/\mathbf{x}_1)^\top \\ \vdots \\ \varphi(/docs/notes/sml/\mathbf{x}_n)^\top \end{pmatrix}$ denotes the transformed design matrix.


There are many possible choices for the mapping $\varphi(/docs/notes/sml/\mathbf{x})$, but we'll focus on using polynomial basis functions in the single-feature case, e.g. $\varphi(x) = [1, x, x^2, \ldots, x^{k - 1}]^\top$ (note the first element corresponds to a bias term). 
There are many possible choices for the mapping $\varphi(/docs/notes/sml/\mathbf{/docs/notes/sml/x})$, but we'll focus on using polynomial basis functions in the single-feature case, e.g. $\varphi(/docs/notes/sml/x) = [1, /docs/notes/sml/x, /docs/notes/sml/x^2, \ldots, /docs/notes/sml/x^{k - 1}]^\top$ (note the first element corresponds to a bias term). 
There are many possible choices for the mapping $\varphi(\mathbf{/docs/notes/sml/x})$, but we'll focus on using polynomial basis functions in the single-feature case, e.g. $\varphi(/docs/notes/sml/x) = [1, /docs/notes/sml/x, /docs/notes/sml/x^2, \ldots, /docs/notes/sml/x^{k - 1}]^\top$ (/docs/notes/sml/note the first element corresponds to a bias term). 
There are many possible choices for the mapping $\varphi(/docs/notes/sml/\mathbf{x})$, but we'll focus on using polynomial basis functions in the single-feature case, e.g. $\varphi(x) = [1, x, x^2, \ldots, x^{k - 1}]^\top$ (/docs/notes/sml/note the first element corresponds to a bias term). 
There are many possible choices for the mapping $\varphi(\mathbf{/docs/notes/sml/x})$, but we'll focus on using polynomial basis functions in the single-feature case, e.g. $\varphi(/docs/notes/sml/x) = [1, /docs/notes/sml/x, /docs/notes/sml/x^2, \ldots, /docs/notes/sml/x^{k - 1}]^\top$ (/docs/notes/sml/note the first element corresponds to a bias term). 

We can compute the transformed design matrix using a built-in class from scikit-learn called `PolynomialFeatures`.
We'll start by considering polynomial features of degree 2.


```python
from sklearn.preprocessing import PolynomialFeatures
degree = 2
poly = PolynomialFeatures(/docs/notes/sml/degree=degree)
Phi_train = poly.fit_transform(/docs/notes/sml/X_train_s)
Phi_test = poly.fit_transform(/docs/notes/sml/X_test_s)
print(/docs/notes/sml/"Original design matrix (first 5 rows):\n", X_train_s[0:5], "\n")
print(/docs/notes/sml/"Transformed design matrix (first 5 rows):\n", Phi_train[0:5])
```

    Original design matrix (/docs/notes/sml/first 5 rows):
     [[12.8 ]
     [13.09]
     [27.38]
     [17.44]
     [ 1.92]] 
    
    Transformed design matrix (/docs/notes/sml/first 5 rows):
     [[  1.      12.8    163.84  ]
     [  1.      13.09   171.3481]
     [  1.      27.38   749.6644]
     [  1.      17.44   304.1536]
     [  1.       1.92     3.6864]]


Now let's perform linear regression on the transformed training data and plot the resulting model. 


```python
lr_poly = LinearRegression(/docs/notes/sml/fit_intercept=False).fit(Phi_train, y_train)
lr_poly = LinearRegression(/docs/notes/sml/fit_intercept=False).fit(/docs/notes/sml/Phi_train, y_train)

X_grid = np.linspace(/docs/notes/sml/X_train_s.min(), X_train_s.max(), num=1001)
/docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/i/docs/notes/sml/d/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/n/docs/notes/sml/p/docs/notes/sml/./docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/s/docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/e/docs/notes/sml/(/docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/x/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/n/docs/notes/sml/u/docs/notes/sml/m/docs/notes/sml/=/docs/notes/sml/1/docs/notes/sml/0/docs/notes/sml/0/docs/notes/sml/1/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/Phi_grid = poly.fit_transform(/docs/notes/sml/X_grid[:,np.newaxis])
y = lr_poly.predict(/docs/notes/sml/Phi_grid)
plt.plot(/docs/notes/sml/X_grid, y, 'k-', label='Prediction')
plt.scatter(/docs/notes/sml/X_train_s, y_train, color='b', marker='.', label='Train')
plt.scatter(/docs/notes/sml/X_test_s, y_test, color='r', marker='.', label='Test')
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/plt.ylabel(/docs/notes/sml/"$y$ (Median House Value)")
plt.xlabel(/docs/notes/sml/"$x$ (LSTAT)")
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/w/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```


    
![png](/docs/notes/sml/worksheet03_solutions_files/worksheet03_solutions_26_0.png)
    


Seems like a better fit than the linear model! Let's take a look at the error terms on the train/test set:


```python
y_pred_train_poly = lr_poly.predict(/docs/notes/sml/Phi_train)
y_pred_test_poly = lr_poly.predict(/docs/notes/sml/Phi_test)
print(/docs/notes/sml/'Train MSE for polynomial features of degree {}: {:.3f}'.format(degree, mean_squared_error(y_pred_train_poly, y_train)))
print(/docs/notes/sml/'Test MSE for polynomial features of degree {}: {:.3f}'.format(degree, mean_squared_error(y_pred_test_poly, y_test)))

print(/docs/notes/sml/'Train MSE using linear features only: {:.3f}'.format(mean_squared_error(lr.predict(X_train_s), y_train)))
print(/docs/notes/sml/'Test MSE using linear features only: {:.3f}'.format(mean_squared_error(lr.predict(X_test_s), y_test)))
```

    Train MSE for polynomial features of degree 2: 29.535
    Test MSE for polynomial features of degree 2: 33.760
    Train MSE using linear features only: 38.632
    Test MSE using linear features only: 38.004


Strange, a large reduction on the train MSE but not so much on the test MSE. 
Lets scan across a range of powers. 
What do you expect to happen as we increase the maximum polynomial order on the training set? 
Take a minute to discuss with your fellow students before executing the next cell.


```python
degrees = list(/docs/notes/sml/range(12))
/docs/notes/sml/m/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/m/docs/notes/sml/s/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/m/docs/notes/sml/s/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/
X_grid = np.linspace(/docs/notes/sml/min(X_train_s.min(), X_test_s.min()), 
/docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/i/docs/notes/sml/d/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/n/docs/notes/sml/p/docs/notes/sml/./docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/s/docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/e/docs/notes/sml/(/docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/
/docs/notes/sml/                     max(/docs/notes/sml/X_train_s.max(), X_test_s.max()), num=1001)
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/x/docs/notes/sml/(/docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/x/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/x/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/n/docs/notes/sml/u/docs/notes/sml/m/docs/notes/sml/=/docs/notes/sml/1/docs/notes/sml/0/docs/notes/sml/0/docs/notes/sml/1/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/
plt.figure(/docs/notes/sml/figsize=(20,16))
for i, degree in enumerate(/docs/notes/sml/degrees):
    plt.subplot(/docs/notes/sml/len(degrees)//2, 2, i+1) 
    
    # Transform features
    poly = PolynomialFeatures(/docs/notes/sml/degree=degree)
    Phi_train, Phi_test = poly.fit_transform(/docs/notes/sml/X_train_s), poly.fit_transform(X_test_s)
    Phi_train, Phi_test = poly.fit_transform(/docs/notes/sml/X_train_s), poly.fit_transform(/docs/notes/sml/X_test_s)
    Phi_grid = poly.fit_transform(/docs/notes/sml/X_grid[:,np.newaxis])
    
    # Fit model
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/#/docs/notes/sml/l/docs/notes/sml/r/docs/notes/sml/_/docs/notes/sml/p/docs/notes/sml/o/docs/notes/sml/l/docs/notes/sml/y/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/L/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/R/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/f/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/P/docs/notes/sml/h/docs/notes/sml/i/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/y/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml///docs/notes/sml/d/docs/notes/sml/o/docs/notes/sml/c/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/n/docs/notes/sml/o/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/s/docs/notes/sml/m/docs/notes/sml/l/docs/notes/sml///docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/#/docs/notes/sml/l/docs/notes/sml/r/docs/notes/sml/_/docs/notes/sml/p/docs/notes/sml/o/docs/notes/sml/l/docs/notes/sml/y/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/L/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/R/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/f/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml///docs/notes/sml/d/docs/notes/sml/o/docs/notes/sml/c/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/n/docs/notes/sml/o/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/s/docs/notes/sml/m/docs/notes/sml/l/docs/notes/sml///docs/notes/sml/P/docs/notes/sml/h/docs/notes/sml/i/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/y/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/    #models.append(/docs/notes/sml/lr_poly)
    w = np.linalg.solve(/docs/notes/sml/np.dot(Phi_train.T, Phi_train), np.dot(Phi_train.T, y_train))
    w = np.linalg.solve(/docs/notes/sml/np.dot(Phi_train.T, Phi_train), np.dot(/docs/notes/sml/Phi_train.T, y_train))
    models.append(/docs/notes/sml/w)
    
    # Evaluate
    #train_mse = mean_squared_error(/docs/notes/sml/lr_poly.predict(Phi_train), y_train)
    #/docs/notes/sml/train_mses.append(/docs/notes/sml/train_mse)
    #test_mse = mean_squared_error(/docs/notes/sml/lr_poly.predict(Phi_test), y_test)
    #/docs/notes/sml/test_mses.append(/docs/notes/sml/test_mse)
    train_mse = mean_squared_error(/docs/notes/sml/predict(Phi_train, w), y_train)
    /docs/notes/sml/train_mses.append(/docs/notes/sml/train_mse)
    test_mse = mean_squared_error(/docs/notes/sml/predict(Phi_test, w), y_test)
    /docs/notes/sml/test_mses.append(/docs/notes/sml/test_mse)
    
    plt.plot(/docs/notes/sml/X_grid, predict(Phi_grid,w), 'k', label='Prediction')
    plt.scatter(/docs/notes/sml/X_train_s, y_train, color='b', marker='.', label='Train')
    plt.scatter(/docs/notes/sml/X_test_s, y_test, color='r', marker='.', label='Test')
    plt.title(/docs/notes/sml/'Degree {} | Train MSE {:.3f}, Test MSE {:.3f}'.format(degree, train_mse, test_mse))
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/    
plt.suptitle(/docs/notes/sml/'Polynomial regression for different polynomial degrees', y=1.05, fontsize=32)
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/g/docs/notes/sml/h/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/y/docs/notes/sml/o/docs/notes/sml/u/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```


    
![png](/docs/notes/sml/worksheet03_solutions_files/worksheet03_solutions_30_0.png)
    


Let's plot mean-squared error vs. polynomial degree for the train and test sets.


```python
plt.plot(/docs/notes/sml/degrees, train_mses, color='b', label='Train')
plt.plot(/docs/notes/sml/degrees, test_mses, color='r', label='Test')
plt.title(/docs/notes/sml/'MSE vs. polynomial degree')
plt.ylabel(/docs/notes/sml/'MSE')
plt.xlabel(/docs/notes/sml/'Polynomial degree')
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/w/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```


    
![png](/docs/notes/sml/worksheet03_solutions_files/worksheet03_solutions_32_0.png)
    


***
**Question**: 🤨 What's going on here? Does this match your earlier findings, or your intuition about which model was most appropriate? Why isn't test error behaving the same as training error?

_Answer:_ This can be explained by the _bias-variance trade-off_ associated with predictive models. 
An overly simplistic model (/docs/notes/sml/e.g. simple linear regression) may suffer from _high bias_ or _underfitting_, while an overly complex model (e.g. high-degree polynomial regression) may suffer from _high variance_ or _overfitting_.
An overly simplistic model (/docs/notes/sml/e.g. simple linear regression) may suffer from _high bias_ or _underfitting_, while an overly complex model (/docs/notes/sml/e.g. high-degree polynomial regression) may suffer from _high variance_ or _overfitting_.
We'll discuss this more in lecture 5, when we cover regularisation as a strategy to avoid overfitting.

***
## Bonus: Ridge regression (/docs/notes/sml/this section is optional)

One solution for managing the bias-variance trade-off is regularisation. 
In the context of regression, one can simply add a penalty term to the least-squares cost function in order to encourage weight vectors that are sparse and/or small in magnitude.
In this section, we'll experiment with ridge regression, where a $L_2$ (/docs/notes/sml/Tikhonov) penalty term is added to the cost function as follows:

$$
C(/docs/notes/sml/\mathbf{w}) = \| \mathbf{y} - \mathbf{X} /docs/notes/sml/\mathbf{w} \|_2^2 + \alpha \| /docs/notes/sml/\mathbf{w} \|_2^2
$$

***
**Exercise:** Repeat the previous section on polynomial regression with an $L_2$ penalty term and $\alpha = 0.002$. You may find the `sklearn.linear_model.Ridge` class useful.

_Note: You'll need to rescale the `LSTAT` feature (/docs/notes/sml/e.g. divide by 100) in order to avoid numerical issues._
***

We'll start by importing `Ridge` and rescaling `LSTAT`.


```python
from sklearn.linear_model import Ridge
X_train_s = X_train_s / 100.0
X_test_s = X_test_s / 100.0
```

/docs/notes/sml/W/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/ /docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/p/docs/notes/sml/y/docs/notes/sml/ /docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/f/docs/notes/sml/r/docs/notes/sml/o/docs/notes/sml/m/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/v/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/u/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/e/docs/notes/sml/c/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/k/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/g/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/e/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/t/docs/notes/sml/ /docs/notes/sml/`/docs/notes/sml/L/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/R/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/`/docs/notes/sml/ /docs/notes/sml/-/docs/notes/sml/>/docs/notes/sml/ /docs/notes/sml/`/docs/notes/sml/R/docs/notes/sml/i/docs/notes/sml/d/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/(/docs/notes/sml/a/docs/notes/sml/l/docs/notes/sml/p/docs/notes/sml/h/docs/notes/sml/a/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/0/docs/notes/sml/./docs/notes/sml/0/docs/notes/sml/0/docs/notes/sml/2/docs/notes/sml/)/docs/notes/sml/`/docs/notes/sml/./docs/notes/sml/
/docs/notes/sml///docs/notes/sml/d/docs/notes/sml/o/docs/notes/sml/c/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/n/docs/notes/sml/o/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/s/docs/notes/sml/m/docs/notes/sml/l/docs/notes/sml///docs/notes/sml/W/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/ /docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/p/docs/notes/sml/y/docs/notes/sml/ /docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/f/docs/notes/sml/r/docs/notes/sml/o/docs/notes/sml/m/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/v/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/u/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/e/docs/notes/sml/c/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/k/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/g/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/e/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/t/docs/notes/sml/ /docs/notes/sml/`/docs/notes/sml/L/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/R/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/`/docs/notes/sml/ /docs/notes/sml/-/docs/notes/sml/>/docs/notes/sml/ /docs/notes/sml/`/docs/notes/sml/R/docs/notes/sml/i/docs/notes/sml/d/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/(/docs/notes/sml///docs/notes/sml/d/docs/notes/sml/o/docs/notes/sml/c/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/n/docs/notes/sml/o/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml///docs/notes/sml/s/docs/notes/sml/m/docs/notes/sml/l/docs/notes/sml///docs/notes/sml/a/docs/notes/sml/l/docs/notes/sml/p/docs/notes/sml/h/docs/notes/sml/a/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/0/docs/notes/sml/./docs/notes/sml/0/docs/notes/sml/0/docs/notes/sml/2/docs/notes/sml/)/docs/notes/sml/`/docs/notes/sml/./docs/notes/sml/
/docs/notes/sml/

```python
degrees = list(/docs/notes/sml/range(12))
/docs/notes/sml/m/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/m/docs/notes/sml/s/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/m/docs/notes/sml/s/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/
X_grid = np.linspace(/docs/notes/sml/min(X_train_s.min(), X_test_s.min()), 
/docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/i/docs/notes/sml/d/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/n/docs/notes/sml/p/docs/notes/sml/./docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/s/docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/e/docs/notes/sml/(/docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/
/docs/notes/sml/                     max(/docs/notes/sml/X_train_s.max(), X_test_s.max()), num=1001)
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/x/docs/notes/sml/(/docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/x/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/X/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/x/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/n/docs/notes/sml/u/docs/notes/sml/m/docs/notes/sml/=/docs/notes/sml/1/docs/notes/sml/0/docs/notes/sml/0/docs/notes/sml/1/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/
plt.figure(/docs/notes/sml/figsize=(20,16))
for i, degree in enumerate(/docs/notes/sml/degrees):
    plt.subplot(/docs/notes/sml/len(degrees)//2, 2, i+1) 
    
    # Transform features
    poly = PolynomialFeatures(/docs/notes/sml/degree=degree)
    Phi_train, Phi_test = poly.fit_transform(/docs/notes/sml/X_train_s), poly.fit_transform(X_test_s)
    Phi_train, Phi_test = poly.fit_transform(/docs/notes/sml/X_train_s), poly.fit_transform(/docs/notes/sml/X_test_s)
    Phi_grid = poly.fit_transform(/docs/notes/sml/X_grid[:,np.newaxis])
    
    # Fit model
    lr_poly = Ridge(/docs/notes/sml/alpha = 0.002).fit(Phi_train, y_train)
    lr_poly = Ridge(/docs/notes/sml/alpha = 0.002).fit(/docs/notes/sml/Phi_train, y_train)
    models.append(/docs/notes/sml/lr_poly)
    
    # Evaluate
    train_mse = mean_squared_error(/docs/notes/sml/lr_poly.predict(Phi_train), y_train)
    /docs/notes/sml/train_mses.append(/docs/notes/sml/train_mse)
    test_mse = mean_squared_error(/docs/notes/sml/lr_poly.predict(Phi_test), y_test)
    /docs/notes/sml/test_mses.append(/docs/notes/sml/test_mse)
    
    plt.plot(/docs/notes/sml/X_grid, lr_poly.predict(Phi_grid), 'k', label='Prediction')
    plt.scatter(/docs/notes/sml/X_train_s, y_train, color='b', marker='.', label='Train')
    #plt.scatter(/docs/notes/sml/X_test_s, y_test, color='r', marker='.', label='Test')
    plt.title(/docs/notes/sml/'Degree {} | Train MSE {:.3f}'.format(degree, train_mse))
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/    
plt.suptitle(/docs/notes/sml/'Polynomial ridge regression for different polynomial degrees', y=1.05, fontsize=32)
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/g/docs/notes/sml/h/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/y/docs/notes/sml/o/docs/notes/sml/u/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```


    
![png](/docs/notes/sml/worksheet03_solutions_files/worksheet03_solutions_38_0.png)
    


Notice that the model is no longer overfitting for larger polynomial degrees.
We can confirm this by plotting the mean-squared error vs. the degree.


```python
plt.plot(/docs/notes/sml/degrees, train_mses, color='b', label='Train')
plt.plot(/docs/notes/sml/degrees, test_mses, color='r', label='Test')
plt.title(/docs/notes/sml/'MSE vs. polynomial degree')
plt.ylabel(/docs/notes/sml/'MSE')
plt.xlabel(/docs/notes/sml/'Polynomial degree')
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/w/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```


    
![png](/docs/notes/sml/worksheet03_solutions_files/worksheet03_solutions_40_0.png)
    


Finally, we'll plot the L2 norm of the weights versus the polynomial degree. You should compare this with the non-regularized values!


```python
w_L2 = [np.sum(/docs/notes/sml/m.coef_**2) for m in models]
plt.plot(/docs/notes/sml/degrees, w_L2)
plt.xlabel(/docs/notes/sml/'Polynomial degree')
plt.ylabel(/docs/notes/sml/r'$\| \mathbf{w} \|_2^2$')
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/w/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```


    
![png](/docs/notes/sml/worksheet03_solutions_files/worksheet03_solutions_42_0.png)
    


You may want to experiment with different settings for the regularisation parameter $\alpha$. 
How could you select the 'best' value of $\alpha$ to achieve a good balance between training error and generalisation error?
