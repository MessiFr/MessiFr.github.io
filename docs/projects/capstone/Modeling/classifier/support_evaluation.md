```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import numpy as np
from matplotlib import pyplot as plt

# cross_val_score(model,X_origin, Y_origin)
def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def check_overfitting(model, X, Y):
    '''
    Use `learning_curve` to determine the performance of model 
    working in training dataset and the full dataset to check 
    if the model is overfitting.
    '''
    train_size, train_scores, evaluation_scores =  learning_curve(model,X, Y, \
                                                                train_sizes=np.linspace(0.1,1,5), cv=5, n_jobs= -1, verbose = -1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    evaluation_scores_mean = np.mean(evaluation_scores, axis=1)
    evaluation_scores_std = np.std(evaluation_scores, axis=1)

    plt.plot(train_size,train_scores_mean, label = "training score", color = 'g')
    plt.fill_between(train_size, train_scores_mean - train_scores_std,
                    train_scores_mean+train_scores_std, alpha = 0.2, color = 'g' )
    plt.plot(train_size,evaluation_scores_mean, label = "cross validation score", color = 'b')
    plt.fill_between(train_size, evaluation_scores_mean - evaluation_scores_std,
                    evaluation_scores_mean+evaluation_scores_std, alpha = 0.2, color = 'b' )
    plt.legend(loc = 'best')

    plt.title(f"learning curve for Gradient Boosting Classifier")
    plt.xlabel("amount of training instance")
    plt.ylabel("classification accuracy (%)")
```