## The Perceptron and PyTorch  
***
In this worksheet, we'll implement the perceptron (a building block of neural networks) from scratch. 
Our key objectives are:

* to review the steps involved in the perceptron training algorithm
* to assess how the perceptron behaves in two distinct scenarios (separable vs. non-separable data)
* learn how to use PyTorch, a high performance deep learning API, to implement the perceptron with automatic differentiation


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
%matplotlib inline

from sklearn.datasets import make_classification
```

### 1. Synthetic data set
We'll use the built-in `make_classification` function from `sklearn` to generate a synthetic binary classification data set.
The main advantage of using synthetic data is that we have complete control over the distribution. 
This is useful for studying machine learning algorithms under specific conditions.
In particular, we'll be varying the *degree of separability* between the two classes by adjusting the `class_sep` parameter below.
To begin, we'll work with a data set that is almost linearly separable (with `class_sep = 1.75`).

**Note:** In this worksheet we deviate from the standard `0`/`1` encoding for binary class labels used in `sklearn`. 
We use `-1` in place of `0` for the "negative" class to make the mathematical presentation of the perceptron algorithm easier to digest.


```python
FIGURE_RESOLUTION = 128
plt.rcParams['figure.dpi'] = FIGURE_RESOLUTION

def create_toy_data(n_samples=150, class_sep=1.75):
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, 
                               n_redundant=0, n_clusters_per_class=1, flip_y=0,
                               class_sep=class_sep, random_state=12)
    y[y==0] = -1 # encode "negative" class using -1 rather than 0
    plt.plot(X[y==-1,0], X[y==-1,1], ".", label="$y = -1$")
    plt.plot(X[y==1,0], X[y==1,1], ".", label="$y = 1$")
    plt.legend()
    plt.xlabel("$x_0$")
    plt.ylabel("$x_1$")
    plt.show()

    return X,y

X, y = create_toy_data(class_sep = 1.75)
```


    
![png](/docs/notes/sml/worksheet06_solutions_files/worksheet06_solutions_3_0.png)
    


**Question:** Is the perceptron a suitable classifier for this data set?

**Answer:** *Strictly speaking, the perceptron is only guaranteed to converge to a "reasonable" solution if the data is linearly seperable.
This data is not quite linearly separable, so one could argue that the perceptron is unsuitable.*
  
In preparation for training and evaluating a perceptron on this data, we randomly partition the data into train/test sets.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=90051)
print("Training set has {} instances. Test set has {} instances.".format(X_train.shape[0], X_test.shape[0]))
```

    Training set has 105 instances. Test set has 45 instances.


### 2. Definition of the perceptron

Recall from lectures that a perceptron is a linear binary classifier which maps an input vector $\mathbf{x} \in \mathbb{R}^D$ to a binary output $y \in \{-1,1\}$ given by
$$
\begin{align*}
f(\mathbf{x}; \mathbf{w}, b) &= \begin{cases}
    1 & \mathrm{if} \ s(\mathbf{x}; \mathbf{w}, b) \geq 0, \\
    -1 & \mathrm{otherwise},
\end{cases}
\end{align*}
$$
where $s(\mathbf{x}; \mathbf{w}, b) = \mathbf{w} \cdot \mathbf{x} + b$. 
Here $\mathbf{w} \in \mathbb{R}^D$ is a vector of weights (one for each feature) and $b$ is the bias term. Note that for this worksheet we are including an explicit bias, rather than incorporating it as a constant column in $\mathbf{X}$.

Let start by implementing the weighted sum function $s(\mathbf{x}; \mathbf{w}, b)$ below. This is the 'inner product' $\langle\mathbf{w}, \mathbf{x}\rangle = \sum_i w_i x_i$, which captures the geometrical alignment of the weight and instance vectors. 


```python
def weighted_sum(X, w, b):
    """
    Returns an array containing the weighted sum s(x) for each instance x in the feature matrix
    
    Arguments:
    X : numpy array, shape: (n_instances, n_features)
        feature matrix
    w : numpy array, shape: (n_features,)
        weights vector
    b : float
        bias term
    """
    return np.dot(X,w) + b # fill in

def predict(X, w, b):
    """
    Returns an array containing the predicted binary labels (-1/1) for each instance in the feature matrix
    
    Arguments:
    X : numpy array, shape: (n_instances, n_features)
        feature matrix
    w : numpy array, shape: (n_features,)
        weights vector
    b : float
        bias term
    """
    return np.where(weighted_sum(X, w, b) >= 0, 1, -1)
```

### 3. Perceptron training algorithm

We will now implement the Perceptron training algorithm first proposed by Rosenblatt in 1957. 

This is an online algorithm in the sense that it examples one point $\mathbf{x}_t$ at a time. At each stage, the algorithm maintains a weight vector $\mathbf{w}_t$ which defines the learned (hyper)plane separating the two classes. For convenience one typically starts with $\mathbf{w}_0 = 0$. The label of the point $\mathbf{x}_t$ is predicted using the sign of the dot product: $\hat{y}_t = \mathrm{sign}(\mathbf{w}_t \cdot \mathbf{x}_t)$. If $\mathbf{x}_t$ is misclassified, $y_t \mathbf{x} \cdot \mathbf{w}_t$ is negative and we make a correction to the weight vector $\mathbf{w}_{t+1} = \mathbf{w}_t + \eta y_t \mathbf{x}_t$ for some learning rate $\eta >0$. To motivate this, consider the dot product of the weight vector after the update with the misclassified instance: 

\begin{align}
y_t (\mathbf{w}_t + \eta y_t \mathbf{x}_t) \cdot \mathbf{x}_t &= y_t \mathbf{w}_t \cdot \mathbf{x}_t + \eta ||\mathbf{x}_t ||^2\\
&\geq y_t \mathbf{w}_t \cdot \mathbf{x}_t
\end{align}

So we reduce the severity of the error by an additive correction $\eta || \mathbf{x}_t ||^2$. We repeat this procedure until convergence or termination after a set limit of iterations. Pseudocode of the algorithm is shown below (taken from _Foundations of Machine Learning, Mohri, 2019_).

<img src="https://i.imgur.com/N6NZrAC.png" alt="Perceptron alg." width="750"/>

**Note:** After iterating through all of the training instances $t=[1, \ldots, T]$, we have completed one *epoch*. The above pseudocode runs for one epoch. Typically, multiple epochs are required to get close to the optimal solution.

The above training procedure is equivalent to performing sequential gradient descent on the following objective function:

\begin{align}
    F(\mathbf{w}) &= \frac{1}{T} \sum_{t=1}^T \max\left(0, -y_t (\mathbf{w} \cdot \mathbf{x}_t)\right) \\
    \mathbf{w}_{t+1} &\leftarrow \mathbf{w}_t - \eta \nabla_{\mathbf{w}} F(\mathbf{w}) 
\end{align}

To motivate the form of $F(\mathbf{w})$, note that if $\mathbf{x}_t$ is misclassified, $-y_t \mathbf{w}_t \cdot \mathbf{x}_t > 0$. In this case the gradient is $\nabla_{\mathbf{w}} F(\mathbf{w}) = -y_t \mathbf{x}_t$, and zero if the instance is correctly classified. Omitting technical details about the case where $\mathbf{w}_t \cdot \mathbf{x}_t = 0$ (we can resolve the non-differentiability by working with _subgradients_), the stochastic online update becomes:

$$ \mathbf{w}_{t+1} \leftarrow
  \begin{cases}
    \mathbf{w}_t + \eta y_t \mathbf{x}_t, & \text{if } y_t\left(\mathbf{x}_t \cdot \mathbf{w}_t\right) \leq 0; \\
    \mathbf{w}_t, & \text{if } y_t\left(\mathbf{x}_t \cdot \mathbf{w}_t\right) > 0 \\
  \end{cases}
$$

***

**Exercise:** Find weights of a perceptron of the following dataset, with learning rate $\eta =0.1$


$$\mathbf{X} = \begin{pmatrix} 1&1&1 \\ 1&1&2 \\ 1&0&0 \\ 1&-1 & 0\end{pmatrix},  \mathbf{y} = \begin{pmatrix} 1 \\ 1 \\ -1 \\ -1 \end{pmatrix}$$



_**Answer:** Follow the pseudocode of the algorithm described above, we initialize the $\mathbf{w}$ with zeros,  $\mathbf{w} = [0,0,0]$. The first instance $\mathbf{x}_1$ is [1,1,1], we compute the inner product $\mathbf{x}_1^t\mathbf{w} = 0$ and make the prediction $\hat{y}_1 = 1$, the prediction matched with groud truth ($\hat{y}_1={y}_1$) so we do not update the weights. We will not update for the second instance as well. The prediction for the third instance is still 1 but the label is -1, thus we need to update the weights with $\mathbf{w} = \mathbf{w} + \eta y \mathbf{x}$:_ 

$$
w_0 \leftarrow 0  + 0.1 \times -1 \times 1\\
w_1 \leftarrow 0  + 0.1 \times -1 \times 0\\
w_2 \leftarrow 0  + 0.1 \times -1 \times 0\\
$$

_The new weight vector is $\mathbf{w} = [-0.1, 0, 0] $, we use it to predict the fourth instance, we get $\hat{y}=-1$, which matches the truth, so we do nothing._

_Once we have done all instances, we have completed one epoch. We can start from scratch and conduct the other epoch training until all the weight is not updated in one completed epoch, which means our model achieves the capability of perfect classification._
***

Let's now write a function called `train` which implements sequential gradient descent. The function should implement the pseudocode shown above, repeated for `n_epochs`. 


```python
def train(X, y, w, b, n_epochs=10, eta=0.1):
    """
    Returns updated weight vector w and bias term b for
    the Perceptron algorithm
    
    Arguments:
    X : numpy array, shape: (n_instances, n_features)
        feature matrix
    y : numpy array, shape: (n_instances,)
        target class labels relative to X
    n_epochs : int
        number of epochs (full sweeps through X)
    w : numpy array, shape: (n_features,)
        initial guess for weights vector w_0
    b : float
        initial guess for bias term b_0
    eta : positive float
        step size (default: 0.1)
    """
    errors = 0
    for t in range(n_epochs):
        # loop over individual elements in X
        for i in range(X.shape[0]):
            yhat = predict(X[i,:], w, b) # fill in
            if yhat != y[i]:
                errors += 1
                # Update if misclassified, else do nothing
                w += eta * y[i] * X[i,:] # fill in
                b += eta * y[i] # fill in
                
    print('Number of errors {} / {} iterations'.format(errors, n_epochs * X.shape[0]))
    return w, b
```

Test your implementation by running it for 5 epochs.
You should get the following result for the weights and bias term:
`w = [ 0.78555827 -0.11419669]; b = 0.1`


```python
# Initialise weights and bias to zero
w = np.zeros(X.shape[1]); b = 0.0

w, b = train(X_train, y_train, w, b, n_epochs=5)
print("w = {}; b = {}".format(w, b))
```

    Number of errors 25 / 525 iterations
    w = [ 0.78555827 -0.11419669]; b = 0.1


### 4. Evaluation

Now that we've trained the perceptron, let's see how it performs.

Below we plot the data (training and test sets) along with the decision boundary (which is defined by $\{\mathbf{x}: \hat{\mathbf{w}} \cdot \mathbf{x}= 0$\})
**Note:** It's not necessary to understand how the `plot_results` function works.


```python
def plot_results(X_train, y_train, X_test, y_test, score_fn, threshold = 0):
    # Plot training set
    plt.plot(X_train[y_train==-1,0], X_train[y_train==-1,1], ".", label=r"$y=-1$, train")
    plt.plot(X_train[y_train==1,0], X_train[y_train==1,1], ".", label=r"$y=1$, train")
    plt.gca().set_prop_cycle(None) # reset colour cycle

    # Plot test set
    plt.plot(X_test[y_test==-1,0], X_test[y_test==-1,1], "x", label=r"$y=-1$, test")
    plt.plot(X_test[y_test==1,0], X_test[y_test==1,1], "x", label=r"$y=1$, test")

    # Compute axes limits
    border = 1
    x0_lower = X[:,0].min() - border
    x0_upper = X[:,0].max() + border
    x1_lower = X[:,1].min() - border
    x1_upper = X[:,1].max() + border

    # Generate grid over feature space
    resolution = 0.01
    x0, x1 = np.mgrid[x0_lower:x0_upper:resolution, x1_lower:x1_upper:resolution]
    grid = np.c_[x0.ravel(), x1.ravel()]
    s = score_fn(grid).reshape(x0.shape)

    # Plot decision boundary (where s(x) == 0)
    plt.contour(x0, x1, s, levels=[0], cmap="Greys", vmin=-0.2, vmax=0.2)

    plt.legend()
    plt.xlabel("$x_0$")
    plt.ylabel("$x_1$")
    plt.show()
    
plot_results(X_train, y_train, X_test, y_test, lambda X: weighted_sum(X, w, b))
```


    
![png](/docs/notes/sml/worksheet06_solutions_files/worksheet06_solutions_18_0.png)
    


**Question:** How well does the decision boundary separate the points in the two classes? How does it perform if you reduce the `class_sep` parameter in the `create_toy_dataset` function above?

**Answer:** *The decision boundary is acceptable for `class_sep=1.75`, but very poor for `class_sep=0.5`.*

To evaluate the perceptron quantitatively, we'll use the error rate (proportion of misclassified instances).
The error rate is a reasonable evaluation measure to use for this data since the classes are well balanced.

Complete the `evaluate` function below.


```python
def evaluate(X, y, w, b):
    """
    Returns the proportion of misclassified instances (error rate)
    
    Arguments:
    X : numpy array, shape: (n_instances, n_features)
        feature matrix
    Y : numpy array, shape: (n_instances,)
        target class labels relative to X
    w : numpy array, shape: (n_features,)
        weights vector
    b : float
        bias term
    """
    return np.mean(predict(X, w, b) != y) # fill in

print(evaluate(X_train, y_train, w, b))
```

    0.01904761904761905


The code block above computes the error rate on the training data, which is not a good idea in general. (Why?)

Compute the error rate on the test set instead.


```python
print(evaluate(X_test, y_test, w, b)) # fill in
```

    0.022222222222222223


**Question:** How does this compare to the error on the training set? Is it what you expected?

**Answer:** *The error on the test set is a little higher than the error on the training set (for `class_sep=1.75`). This suggests we may have a problem with overfitting (but it's difficult to say for such a small sample size).*

### 5. Repeat with class overlap
By now you've probably concluded that the perceptron performs well on this data (`class_sep=1.75`), which is to be expected as it's roughly linearly separable. 
However, we know (from lectures) that the perceptron can fail on non-linearly separable data.
To test this scenario, re-generate the synthetic data set with `class_sep=0.5` and repeat Sections 2–4.

**Question:** What do you find? Pay particular attention to the error vs. training epochs (you may need to add a `print` command in the inner training loop). Do you notice anything unusual?

_**Answer:** As mentioned in the previous answer, the model becomes highly unstable—the optimisation algorithm doesn't really converge to a good "approximate" solution. As a result, the train/test errors fluctuate wildly._

### 6. Move over numpy, hello PyTorch

[Pytorch](https://pytorch.org/) is a open-source Python library designed for fast matrix computations on CPU/GPU. This includes both standard linear algebra and deep learning-specific operations. It is based on the neural network backend of the [Torch library](http://torch.ch/). A central feature of Pytorch is its use of Automatic on-the-fly differentiation ([Autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)) to compute derivatives of (almost) all computations involving tensors, so we can make use of gradient-based updates to optimize some objective function. In this last section we will introduce some fundamental operations in Pytorch and reimplement the Perceptron algorithm.

To illustrate the interface of pytorch, consider the following code which does some simple matrix operations:


```python
!pip3 install torch
```

    Collecting torch
      Downloading torch-1.12.1-cp310-none-macosx_10_9_x86_64.whl (133.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m133.8/133.8 MB[0m [31m231.1 kB/s[0m eta [36m0:00:00[0m00:01[0m00:18[0m
    [?25hCollecting typing-extensions
      Downloading typing_extensions-4.3.0-py3-none-any.whl (25 kB)
    Installing collected packages: typing-extensions, torch
    Successfully installed torch-1.12.1 typing-extensions-4.3.0
    [33mWARNING: You are using pip version 22.0.4; however, version 22.2.2 is available.
    You should consider upgrading via the '/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 -m pip install --upgrade pip' command.[0m[33m
    [0m


```python
# !pip3 install torch
import torch
# Create matrices with entries drawn from the standard normal, and multiply them
# torch.dot()
x_pt = torch.randn(size=[4,4])
y_pt = torch.randn(size=[4,8])
z_pt = torch.matmul(x_pt, y_pt)
z_pt
```

    /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm





    tensor([[ 0.4490,  0.1314, -2.9952,  4.0268,  0.5722, -1.2777,  2.4954, -2.2401],
            [-0.4978,  0.2263,  4.2176, -2.7984, -0.5106,  1.7246, -2.8656,  0.5869],
            [-0.4366,  0.6214,  0.4276,  1.0189,  0.1151,  0.8781, -0.1773, -1.1379],
            [-0.3372,  0.5151, -0.6911,  1.3622,  0.2096,  0.4561,  0.4885, -0.9397]])



On the surface, this looks similar to numpy, albeit with a slightly different API. Behind the scenes, each time we evaluate a Pytorch operation, the library silently defines a new computation graph which is executed, and this is highly optimised on either CPU or GPU processors.

We will not give a full tutorial of the basic operations here, other than to say that the basic API is extremely similar to numpy. Please take a few minutes to run through the [excellent introduction at the official Pytorch repository](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html). You may want to do this in your own time, either before or after the workshop.

#### Autograd
To compute the gradients of tensors with respect to other tensors, a full record of operations performed on the Tensor must be retained. This can be achieved by setting the attribute `.requires_grad = True`. After the computation is finished, calling `.backward()` on the output tensor walks through the computation history in reverse order to automatically compute the gradients using a method based on the chain rule, which can be used for optimization of arbitrary loss functions.


```python
mu = 0
x = torch.ones([2,3], requires_grad=True)
loss = torch.exp(-(x-mu)**2/(2)).mean()
loss.backward()
x.grad
```




    tensor([[-0.1011, -0.1011, -0.1011],
            [-0.1011, -0.1011, -0.1011]])



The gradient for the tensor `x` is accumulated into the `.grad` attribute. We would use this in some gradient based update rule, such as vanilla SGD. You might want to verify that the gradient above is correct, by calculating the gradient by hand.

Autograd can be very useful when evaluating the derivatives of loss functions constructed as a sequence of nested operations (an example is the cross-entropy loss when using a neural network for classification) where the analytic derivative is difficult to compute symbolically.

Note that autograd is **not** symbolic or numerical (finite-differences) differentiation, both of which have trouble handling gradients of relatively long nested functions. It is essentially a dynamic-programming approach to symbolic differentiation.

#### Back to the Perceptron

Armed with this shiny deep learning API, let's rewrite the perceptron training algorithm. We have two options: just using the standard perceptron update formula, or providing torch with the loss function and letting it compute the gradient automatically. We'll go with the latter, a more general solution, which is similar to how you might train a logistic regerssion classification model, regression, counting likelihood etc. In these cases you would change the `loss =` line to reflect the different loss function, e.g., logistic cross entropy, squared error (Gaussian), Poisson loss for the three cases, respectively.

Review the code below, and use the [PyTorch API](https://pytorch.org/docs/stable/torch.html#) to understand the operations. Some step are a little cryptic, and relate to enumerating over the training data, setting the shape of the tensors (e.g., `squeeze` which removes redundant dimensions of size 1, `requires_grad_` which flags a parameter for which `backward` will compute the gradient.) Take a look at the [SGD class, and other optimisers](https://pytorch.org/docs/stable/optim.html) and their various parameters.


```python
def train_perceptron_SGD_pytorch(X, y, n_epochs=30, eta=0.1):
    
    # Create iterable dataset in Torch format - technical details, 
    # don't worry too much about this!
    X = torch.cat([torch.ones(X.shape[0],1), X], dim=1)
    train_ds = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1)
    
    # Create trainable weight vector, tell Torch to track operations on w
    w = torch.zeros(X.shape[1]).double().requires_grad_()
    
    # Setup the optimizer. This implements the basic gradient descent update
    optimizer = torch.optim.SGD([w], lr=eta)
    
    for _ in range(n_epochs):
        for i, (xi,yi) in enumerate(train_loader):
            xi = torch.squeeze(xi)

            #Compute loss F(w)
            loss = torch.max(torch.Tensor([0]), -yi * (torch.dot(w,xi)))
            
            loss.backward()               # Backward pass (compute parameter gradients)
            optimizer.step()              # Update weight parameter using SGD
            optimizer.zero_grad()         # Reset gradients to zero for next ieration
        
    w = w.detach()  # Tell Torch to stop tracking operations for autograd
    print('Trained weights:', w)
    return w.numpy()
```

Now run the function above to verify it reaches a similar solution to the `numpy` code above. 


```python
w = train_perceptron_SGD_pytorch(torch.from_numpy(X_train), torch.from_numpy(y_train),n_epochs = 5)

plot_results(X_train, y_train, X_test, y_test, score_fn=lambda x: np.dot(x,w[1:])+w[0])
```

    Trained weights: tensor([ 0.1500,  0.5651, -0.0613], dtype=torch.float64)



    
![png](/docs/notes/sml/worksheet06_solutions_files/worksheet06_solutions_37_1.png)
    


## Bonus tasks (optional)

### Comparison with logistic regression
We've seen that the perceptron is not robust to binary classification problems with overlapping classes. 
But how does logistic regression fare in this case?

Adapt the code block above to fit a logistic regression model. To do so you will need to change the loss function to the cross entropy loss from the logistic regression model. 


```python
def train_logistic_regression_SGD_pytorch(X, y, n_epochs=100, eta=0.1):
    
    # Create iterable dataset in Torch format - technical details, 
    # don't worry too much about this!
    X = torch.cat([torch.ones(X.shape[0],1), X], dim=1)
    train_ds = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1)
    
    # Create trainable weight vector, tell Torch to track operations on w
    w = torch.zeros(X.shape[1]).double().requires_grad_()
    
    # Setup the optimizer. This implements the basic gradient descent update
    optimizer = torch.optim.SGD([w], lr=eta)
    
    for _ in range(n_epochs):
        for i, (xi,yi) in enumerate(train_loader):
            xi = torch.squeeze(xi)
            if yi == -1:
                yi = 0

            p = torch.sigmoid(torch.dot(xi,w))
            loss = - (yi * torch.log(p) + (1. - yi) * torch.log(1.-p))
            
            loss.backward()               # Backward pass (compute parameter gradients)
            optimizer.step()              # Update weight parameter using SGD
            optimizer.zero_grad()         # Reset gradients to zero for next ieration
        
    w = w.detach()  # Tell Torch to stop tracking operations for autograd
    print('Trained weights:', w)
    return w.numpy()
```


```python
w = train_logistic_regression_SGD_pytorch(torch.from_numpy(X_train), torch.from_numpy(y_train),n_epochs = 5)

plot_results(X_train, y_train, X_test, y_test, score_fn=lambda x: np.dot(x,w[1:])+w[0])
```

    Trained weights: tensor([ 0.1589,  3.1016, -0.1826], dtype=torch.float64)



    
![png](/docs/notes/sml/worksheet06_solutions_files/worksheet06_solutions_40_1.png)
    

