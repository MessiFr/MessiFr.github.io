```python
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import numpy as np
from matplotlib import pyplot as plt

# cross_val_score(/docs/projects/capstone/Modeling/classifier/model,X_origin, Y_origin)
def namestr(/docs/projects/capstone/Modeling/classifier/obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def check_overfitting(/docs/projects/capstone/Modeling/classifier/model, X, Y):
    '''
    Use `learning_curve` to determine the performance of model 
    working in training dataset and the full dataset to check 
    if the model is overfitting.
    '''
    train_size, train_scores, evaluation_scores =  learning_curve(model,X, Y, \
                                                                train_sizes=np.linspace(/docs/projects/capstone/Modeling/classifier/0.1,1,5), cv=5, n_jobs= -1, verbose = -1)

    train_scores_mean = np.mean(/docs/projects/capstone/Modeling/classifier/train_scores, axis=1)
    train_scores_std = np.std(/docs/projects/capstone/Modeling/classifier/train_scores, axis=1)
    evaluation_scores_mean = np.mean(/docs/projects/capstone/Modeling/classifier/evaluation_scores, axis=1)
    evaluation_scores_std = np.std(/docs/projects/capstone/Modeling/classifier/evaluation_scores, axis=1)

    plt.plot(/docs/projects/capstone/Modeling/classifier/train_size,train_scores_mean, label = "training score", color = 'g')
    plt.fill_between(train_size, train_scores_mean - train_scores_std,
                    train_scores_mean+train_scores_std, alpha = 0.2, color = 'g' )
    plt.plot(/docs/projects/capstone/Modeling/classifier/train_size,evaluation_scores_mean, label = "cross validation score", color = 'b')
    plt.fill_between(train_size, evaluation_scores_mean - evaluation_scores_std,
                    evaluation_scores_mean+evaluation_scores_std, alpha = 0.2, color = 'b' )
    plt.legend(/docs/projects/capstone/Modeling/classifier/loc = 'best')

    plt.title(/docs/projects/capstone/Modeling/classifier/f"learning curve for Gradient Boosting Classifier")
    plt.xlabel(/docs/projects/capstone/Modeling/classifier/"amount of training instance")
    plt.ylabel(/docs/projects/capstone/Modeling/classifier/"classification accuracy (%)")
    ```