# COMP90051 Workshop 5b
## Support Vector Machines
***

In this section, we'll explore how the SVM hyperparameters (/docs/notes/sml/i.e. the penalty parameter, the kernel, and any kernel parameters) affect the decision surface.


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from timeit import default_timer as timer
sns.set_style(/docs/notes/sml/'darkgrid')
plt.rcParams['figure.dpi'] = 108

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
```

### 1. Data set
To make visualisation and training easy, we'll consider a small binary classification data set called `cats.csv` (/docs/notes/sml/available from the LMS). 
It contains observations for 150 cats.
There are two features: heart and body weight measured in kilograms.
The target variable is the sex of the cat (/docs/notes/sml/we encode 'male' as`-1` and 'female' as `+1`).

\[Note: the data set originates from the following paper: R. A. Fisher (/docs/notes/sml/1947) _The analysis of covariance method for the relation between a part and the whole_, Biometrics **3**, 65–68\]

Ensure that `cats.csv` is located in the same directory as this notebook, then run the following code block to read the CSV file using `pandas`.


```python
full_df = pd.read_csv(/docs/notes/sml/'cats.csv')
full_df.SEX = full_df.SEX.map(/docs/notes/sml/{'M': -1, 'F': 1})
/docs/notes/sml/f/docs/notes/sml/u/docs/notes/sml/l/docs/notes/sml/l/docs/notes/sml/_/docs/notes/sml/d/docs/notes/sml/f/docs/notes/sml/./docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```




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
      <th>HWT</th>
      <th>BWT</th>
      <th>SEX</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>7.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>7.4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>9.5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.1</td>
      <td>7.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.1</td>
      <td>7.3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Let's split the data into train/test sets so that we can evaluate our trained SVM.
(/docs/notes/sml/Note that this is likely to be unreliable for such a small data set.)


```python
train_df, test_df = train_test_split(/docs/notes/sml/full_df, test_size=0.2, random_state=1)
```

Since SVMs incorporate a penalty term for the weights (/docs/notes/sml/proportional to $\|\mathbf{w}\|_2^2$), it's usually beneficial to _standardise_ the features so that they vary on roughly the same scale.

***
**Exercise:** Complete the code block below to standardise the features, so that each feature has zero mean/unit variance.

_Hint: use `StandardScaler` imported above._
***


```python
/docs/notes/sml/s/docs/notes/sml/c/docs/notes/sml/a/docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/S/docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/n/docs/notes/sml/d/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/d/docs/notes/sml/S/docs/notes/sml/c/docs/notes/sml/a/docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/X_train = scaler.fit_transform(/docs/notes/sml/train_df.drop('SEX', axis=1)) # fill in
y_train = train_df.SEX # fill in

X_test = scaler.transform(/docs/notes/sml/test_df.drop('SEX', axis=1)) # fill in
y_test = test_df.SEX # fill in
```


```python
X_train
```




    array([[-0.47767243,  0.09678416],
           [ 0.52474234, -0.10126172],
           [ 0.72522529,  0.73053096],
           [-0.07670652, -0.18048007],
           [-1.07912129,  0.13639333],
           [-0.47767243, -0.73500852],
           [-0.27718947, -0.9330544 ],
           [ 0.52474234,  0.69092179],
           [ 0.12377643,  1.12662271],
           [-0.87863834, -1.32914615],
           [-1.27960424, -1.32914615],
           [ 1.52715711,  0.88896766],
           [-0.07670652, -0.85383605],
           [-0.27718947, -0.7746177 ],
           [ 0.72522529,  0.33443921],
           [ 1.92812301,  0.13639333],
           [ 0.52474234, -0.25969842],
           [-1.07912129, -0.616181  ],
           [-1.07912129, -1.21031862],
           [ 0.32425939,  0.45326674],
           [ 0.92570825,  0.92857684],
           [ 0.92570825,  0.37404839],
           [-1.27960424, -0.33891677],
           [-0.07670652, -1.05188192],
           [ 0.32425939, -0.22008924],
           [-1.07912129, -1.4083645 ],
           [-1.4800872 , -0.45774429],
           [-0.67815538, -1.32914615],
           [ 1.32667415,  0.61170344],
           [ 2.32908892,  1.48310529],
           [-1.27960424, -1.01227275],
           [ 0.72522529,  1.44349612],
           [ 1.1261912 ,  1.36427777],
           [-1.07912129, -0.41813512],
           [ 0.92570825,  1.12662271],
           [ 0.52474234,  0.37404839],
           [-1.07912129, -0.41813512],
           [ 0.72522529,  0.57209426],
           [-0.27718947, -0.49735347],
           [ 0.52474234,  0.92857684],
           [-0.67815538, -0.616181  ],
           [-0.07670652,  0.53248509],
           [ 0.12377643,  1.04740436],
           [ 0.92570825,  0.65131261],
           [-1.27960424, -0.7746177 ],
           [-0.47767243, -0.65579017],
           [-0.67815538, -0.73500852],
           [ 0.92570825,  0.49287591],
           [-0.27718947, -0.22008924],
           [ 0.52474234,  0.61170344],
           [ 2.32908892,  3.89926498],
           [-0.87863834, -0.22008924],
           [-0.87863834, -0.45774429],
           [-0.47767243, -0.73500852],
           [ 1.32667415,  0.21561168],
           [ 1.1261912 ,  0.53248509],
           [ 0.12377643, -0.18048007],
           [ 0.32425939,  0.25522086],
           [ 1.1261912 ,  1.87919704],
           [ 0.12377643, -0.616181  ],
           [ 1.72764006,  0.45326674],
           [ 1.52715711,  1.95841539],
           [-0.47767243, -1.0914911 ],
           [-1.07912129, -0.37852594],
           [-0.07670652, -0.33891677],
           [-0.07670652,  0.05717498],
           [-0.07670652, -0.41813512],
           [-1.07912129, -1.0914911 ],
           [-0.87863834, -0.22008924],
           [-0.27718947, -0.06165254],
           [ 0.32425939, -0.22008924],
           [-0.87863834, -0.41813512],
           [ 1.52715711,  0.41365756],
           [ 0.52474234,  1.24545024],
           [ 1.1261912 ,  0.33443921],
           [-1.27960424, -0.22008924],
           [-0.87863834, -0.65579017],
           [-0.47767243,  0.80974931],
           [-1.27960424, -1.36875533],
           [ 1.72764006,  1.04740436],
           [-0.67815538, -1.7252379 ],
           [ 2.12860597,  2.4337255 ],
           [ 0.52474234,  1.04740436],
           [-1.4800872 , -1.64601955],
           [ 1.32667415,  0.69092179],
           [ 0.12377643, -0.25969842],
           [-1.27960424, -0.9330544 ],
           [-0.07670652, -0.41813512],
           [-0.67815538, -1.0914911 ],
           [-1.4800872 , -1.44797368],
           [-1.4800872 , -1.28953697],
           [-1.07912129,  0.01756581],
           [-0.87863834, -0.65579017],
           [-0.67815538, -1.0914911 ],
           [-0.67815538, -0.53696265],
           [-1.27960424, -0.97266357],
           [-1.27960424, -0.85383605],
           [ 0.52474234,  0.80974931],
           [-1.07912129, -0.7746177 ],
           [-0.07670652,  0.73053096],
           [ 0.32425939, -0.02204337],
           [-0.47767243, -0.53696265],
           [-1.07912129, -1.36875533],
           [ 0.52474234, -0.02204337],
           [-0.87863834, -0.89344522],
           [-0.87863834, -0.37852594],
           [ 1.52715711,  2.5921622 ],
           [-0.47767243,  0.80974931],
           [ 1.32667415,  1.48310529],
           [-0.07670652, -0.65579017],
           [ 1.52715711,  1.99802457],
           [ 1.72764006,  1.64154199],
           [-0.27718947, -1.17070945],
           [ 2.12860597,  1.64154199],
           [-0.27718947, -0.22008924]])



Let's plot the training data. Notice that it's not linearly separable.


```python
plt.scatter(/docs/notes/sml/X_train[y_train==1,0], X_train[y_train==1,1], label="Female ($y=1$)", c='r')
plt.scatter(/docs/notes/sml/X_train[y_train==-1,0], X_train[y_train==-1,1], label="Male ($y=-1$)", c='b')
plt.xlabel(/docs/notes/sml/"Heart weight")
plt.ylabel(/docs/notes/sml/"Body weight")
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/w/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```


    
![png](/docs/notes/sml/worksheet05b_solutions_files/worksheet05b_solutions_10_0.png)
    


### 2. Parameter grid search
Since the data is clearly not linearly separable, we're going to fit a kernelised SVM.
To do this, we'll use the `sklearn.svm.SVC` class, which is a wrapper for the popular [LIBSVM](/docs/notes/sml/https://www.csie.ntu.edu.tw/~cjlin/libsvm/) library.
\[Aside: LIBSVM solves the dual problem using a variant of the [sequential minimal optimisation (/docs/notes/sml/SMO) algorithm](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf).\]
\[Aside: LIBSVM solves the dual problem using a variant of the [sequential minimal optimisation (/docs/notes/sml/SMO) algorithm](/docs/notes/sml/https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf).\]
The corresponding primal problem is as follows:

$$
\begin{align}
\min_{\mathbf{w}, b, \xi} \phantom{=} & \frac{1}{2} \mathbf{w}^T \mathbf{w} + C \sum_{i = 1}^{n} \xi_i \\
      \mathrm{subject~to} \phantom{=} & y_{i}(/docs/notes/sml/\mathbf{w}^T \cdot \phi(\mathbf{x_i}) + b) \geq 1 - \xi_i \\
                          \phantom{=} & \xi_i \geq 0 \ \forall i
\end{align}
$$

Here $C$ is the penalty parameter, $\mathbf{w}$ are the weights, $b$ is the bias and $\phi$ is a mapping to a higher dimensional space---related to the kernel through $K(/docs/notes/sml/\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$.
Here $C$ is the penalty parameter, $\mathbf{w}$ are the weights, $b$ is the bias and $\phi$ is a mapping to a higher dimensional space---related to the kernel through $K(/docs/notes/sml/\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(/docs/notes/sml/\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$.
Here $C$ is the penalty parameter, $\mathbf{w}$ are the weights, $b$ is the bias and $\phi$ is a mapping to a higher dimensional space---related to the kernel through $K(/docs/notes/sml/\mathbf{x}_i, /docs/notes/sml/\mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(/docs/notes/sml/\mathbf{x}_j) \rangle$.
Here $C$ is the penalty parameter, $\mathbf{w}$ are the weights, $b$ is the bias and $\phi$ is a mapping to a higher dimensional space---related to the kernel through $K(/docs/notes/sml/\mathbf{x}_i, /docs/notes/sml/\mathbf{x}_j) = \langle \phi(/docs/notes/sml/\mathbf{x}_i), \phi(/docs/notes/sml/\mathbf{x}_j) \rangle$.
For now, we'll use the radial basis function (/docs/notes/sml/RBF) kernel, which is parameterised in terms of $\gamma$ as follows:

$$
K(/docs/notes/sml/\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)
K(/docs/notes/sml/\mathbf{x}_i, \mathbf{x}_j) = \exp(/docs/notes/sml/-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)
$$

Returning to our classification problem: it's unclear how to set appropriate values for $C$ and $\gamma$ (/docs/notes/sml/named `C` and `gamma` in `sklearn`).
A simple way around this is to do an exhaustive cross validation grid search.
Below we define an evenly-spaced grid in log-space.


```python
C_range = np.logspace(/docs/notes/sml/-2, 5, 8)
gamma_range = np.logspace(/docs/notes/sml/-6, 1, 16)

# Visualise the grid
xx, yy = np.meshgrid(/docs/notes/sml/C_range, gamma_range)
plt.plot(/docs/notes/sml/xx, yy, 'ko')
plt.xscale(/docs/notes/sml/'log')
plt.yscale(/docs/notes/sml/'log')
plt.xlabel(/docs/notes/sml/'$C$')
plt.ylabel(/docs/notes/sml/r'$\gamma$')
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/w/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```


    
![png](/docs/notes/sml/worksheet05b_solutions_files/worksheet05b_solutions_12_0.png)
    


To do the grid search, we'll use the built-in `sklearn.model_selection.GridSearchCV` class.
It evaluates the model for each combination of parameter values using cross validation, and selects the combination with the best score.

We'll use `StratifiedShuffleSplit` for cross validation (/docs/notes/sml/it effectively generates bootstrap samples from the training data, while preserving the class ratio).


```python
cv = StratifiedShuffleSplit(/docs/notes/sml/n_splits=30, test_size=0.1, random_state=1)
grid = GridSearchCV(/docs/notes/sml/SVC(kernel='rbf'), param_grid={'gamma': gamma_range, 'C': C_range}, cv=cv)
grid.fit(/docs/notes/sml/X_train, y_train)
print(/docs/notes/sml/"The best parameters are {0.best_params_} with an accuracy of {0.best_score_:.3g}".format(grid))
```

    The best parameters are {'C': 10.0, 'gamma': 0.04641588833612782} with an accuracy of 0.828


***
**Question:** Why aren't we using k-fold cross validation?

_Answer:_ Because our training data is too small: it only contains 115 instances. We hope to get a more precise estimate of the accuracy by using a high number of bootstrap samples.
***

Below we visualise the cross validation accuracy over the grid of parameters.


```python
scores = grid.cv_results_['mean_test_score'].reshape(/docs/notes/sml/C_range.size, gamma_range.size)

plt.figure(/docs/notes/sml/figsize=(8, 6))
plt.imshow(/docs/notes/sml/scores, cmap='viridis')
plt.colorbar(/docs/notes/sml/shrink=0.7)
plt.xticks(/docs/notes/sml/np.arange(len(gamma_range)), ["%.2e" % gamma for gamma in gamma_range], rotation=90)
plt.yticks(/docs/notes/sml/np.arange(len(C_range)), ["%1.e" % C for C in C_range])
plt.title(/docs/notes/sml/'Cross validation accuracy')
plt.xlabel(/docs/notes/sml/r'$\gamma$')
plt.ylabel(/docs/notes/sml/'$C$')
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/w/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```


    
![png](/docs/notes/sml/worksheet05b_solutions_files/worksheet05b_solutions_16_0.png)
    


***
**Question:** Interpret this plot. Is there a clear winning combination of parameters?

_Answer:_ There are a few parameter combinations that achieve high cross validation accuracy (/docs/notes/sml/the yellow squares). 
With such a small training set, there's likely to be significant variance (/docs/notes/sml/noise) in these estimates.
Hence, we can't be confident that our chosen parameters are truly "the best". If faced with a tie or near-tie, it makes sense to break the tie in a conservative direction, i.e., by using a smaller gamma (/docs/notes/sml/less non-linearity, as the RBF decays more slowly with distance) and a lower value of C (less of a penalty of margin violations relative to weight shrinkage.)
Hence, we can't be confident that our chosen parameters are truly "the best". If faced with a tie or near-tie, it makes sense to break the tie in a conservative direction, i.e., by using a smaller gamma (/docs/notes/sml/less non-linearity, as the RBF decays more slowly with distance) and a lower value of C (/docs/notes/sml/less of a penalty of margin violations relative to weight shrinkage.)
***

Now that we've found the "best" parameters, let's fit the SVM on the entire training set (/docs/notes/sml/without cross-validation).

(/docs/notes/sml/Note: we actually fit all parameter combinations, as they're needed for a plot generated below.)


```python
classifiers = {(/docs/notes/sml/C, gamma) : SVC(C=/docs/notes/sml/C, gamma=gamma, kernel='rbf').fit(X_train, y_train) 
classifiers = {(/docs/notes/sml/C, gamma) : SVC(/docs/notes/sml/C=/docs/notes/sml/C, gamma=gamma, kernel='rbf').fit(X_train, y_train) 
classifiers = {(/docs/notes/sml/C, gamma) : SVC(/docs/notes/sml/C=/docs/notes/sml/C, gamma=gamma, kernel='rbf').fit(/docs/notes/sml/X_train, y_train) 
classifiers = {(/docs/notes/sml/C, gamma) : SVC(C=/docs/notes/sml/C, gamma=gamma, kernel='rbf').fit(/docs/notes/sml/X_train, y_train) 
classifiers = {(/docs/notes/sml/C, gamma) : SVC(/docs/notes/sml/C=/docs/notes/sml/C, gamma=gamma, kernel='rbf').fit(/docs/notes/sml/X_train, y_train) 
               for C in C_range
               for gamma in gamma_range}
```

Below we evaluate the "best" classifier on the test set.


```python
best_params = (/docs/notes/sml/grid.best_params_["C"], grid.best_params_["gamma"])
best_svm = classifiers[best_params]
best_train_acc = best_svm.score(/docs/notes/sml/X_train, y_train)
best_test_acc = best_svm.score(/docs/notes/sml/X_test, y_test) 
print(/docs/notes/sml/"The SVM with parameters C={0[0]:.3g}, gamma={0[1]:.3g} has training accuracy {1:.3g} and test accuracy {2:.3g}.".format(best_params, best_train_acc, best_test_acc))
```

    The SVM with parameters C=10, gamma=0.0464 has training accuracy 0.817 and test accuracy 0.724.


***
**Question:** How does this compare to the training accuracy?

_Answer:_ The test accuracy is significantly lower than the training accuracy. This could be explained by:

* Overfitting
* Lack of data (/docs/notes/sml/high variance in estimated metrics)
* Lack of informative features
***

Below we visualise the decision functions for all parameter combinations (/docs/notes/sml/double-click output to expand to 100%)


```python
fig, ax = plt.subplots(/docs/notes/sml/C_range.size, gamma_range.size, figsize=(50,20))
border = 0.2

# Build meshgrid over the feature space
X_min = np.amin(/docs/notes/sml/X_train, axis=0)
X_max = np.amax(/docs/notes/sml/X_train, axis=0)
xx, yy = np.meshgrid(/docs/notes/sml/np.linspace(X_min[0] - border, X_max[0] + border, 100), 
                     np.linspace(/docs/notes/sml/X_min[1] - border, X_max[1] + border, 100))

# Plot training data + decision function for all feature combinations
for (/docs/notes/sml/i, C) in enumerate(C_range):
for (/docs/notes/sml/i, C) in enumerate(/docs/notes/sml/C_range):
    for (/docs/notes/sml/j, gamma) in enumerate(gamma_range):
    for (/docs/notes/sml/j, gamma) in enumerate(/docs/notes/sml/gamma_range):
        clf = classifiers[(/docs/notes/sml/C, gamma)]
        Z = clf.decision_function(/docs/notes/sml/np.c_[xx.ravel(), yy.ravel()])
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/Z/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/c/docs/notes/sml/l/docs/notes/sml/f/docs/notes/sml/./docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/c/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/f/docs/notes/sml/u/docs/notes/sml/n/docs/notes/sml/c/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/n/docs/notes/sml/p/docs/notes/sml/./docs/notes/sml/c/docs/notes/sml/_/docs/notes/sml/[/docs/notes/sml/x/docs/notes/sml/x/docs/notes/sml/./docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/v/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/y/docs/notes/sml/y/docs/notes/sml/./docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/v/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/]/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/        Z = Z.reshape(/docs/notes/sml/xx.shape)

        ax[i,j].set_title(/docs/notes/sml/"gamma={0.gamma:.3g}; C={0.C:.3g}".format(clf), 
                           size='medium')

        ax[i,j].pcolormesh(/docs/notes/sml/xx, yy, -Z, cmap=plt.cm.RdBu)
        ax[i,j].scatter(/docs/notes/sml/X_train[y_train==1,0], X_train[y_train==1,1], c='r', edgecolors='k')
        ax[i,j].scatter(/docs/notes/sml/X_train[y_train==-1,0], X_train[y_train==-1,1], c='b', edgecolors='k')
        ax[i,j].set_xticks(/docs/notes/sml/[])
        ax[i,j].set_yticks(/docs/notes/sml/[])
        ax[i,j].axis(/docs/notes/sml/'tight')
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/w/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```


    
![png](/docs/notes/sml/worksheet05b_solutions_files/worksheet05b_solutions_22_0.png)
    


***
**Question:** Explain how `gamma` and `C` affect the decision surface qualitatively.

_Answer:_

* Larger values of `gamma` implies less smoothing (/docs/notes/sml/a "spiky" decision boundary). This can result in overfitting
* Larger values of `C` implies less regularisation (/docs/notes/sml/larger weights are not penalised). This can also result in overfitting.

**Extension activity:** Re-run this section using a different kernel (/docs/notes/sml/e.g. the built-in polynomial kernel or a custom kernel).
***
