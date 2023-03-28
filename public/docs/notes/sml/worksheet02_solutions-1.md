# COMP90051 Workshop 2
## Bayesian inference

***

In this part of the workshop, we'll develop some intuition for priors and posteriors, which are crucial to Bayesian inference.


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.rcParams["animation.html"] = "jshtml"
from scipy.stats import bernoulli, beta
```

### 1. A lucky find

On the way to class, you discover an unusual coin on the ground.

<img src="https://upload.wikimedia.org/wikipedia/commons/6/68/1_2_penny_Middlesex_DukeYork_1795_1ar85_%288737903267%29.jpg" alt="Coin" width="350"/>

As a dedicated student in statistical ML, you're interested in determining whether the coin is _biased_. 
More specifically, you want to estimate the probability $\theta$ that the coin will land heads-up when you toss it. If $\theta \approx \frac{1}{2}$ then we say that the coin is _unbiased_ (/docs/notes/sml/or fair).

You can use the function below to simulate a coin toss: it returns `1` for heads and `0` for tails.


```python
/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/f/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/_/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/:/docs/notes/sml/
/docs/notes/sml/    if bernoulli.rvs(/docs/notes/sml/p = (int.from_bytes("coin".encode(), 'little') % 10000)/10000):
        return 1
    return 0
```

### 2. Prior belief
Before you even toss the coin, you notice that the heads side appears to have more mass. 
Thus, your _prior belief_ is that $\theta$ is slightly biased away from $\frac{1}{2}$ towards 0—i.e. you expect tails are more likely.

To quantify this prior belief, we assume that the prior distribution for $\theta$ is $\mathrm{Beta}(/docs/notes/sml/a,b)$, for some choice of the hyperparameters $a, b > 0$. 
(/docs/notes/sml/See [link](https://en.wikipedia.org/wiki/Beta_distribution) for info about the Beta distribution.)
The prior probability density function for $\theta$ is therefore given by:

$$ p(/docs/notes/sml/\theta) = \frac{1}{B(a,b)} /docs/notes/sml/\theta^{a-1} (1 - /docs/notes/sml/\theta)^{b-1} $$
$$ p(/docs/notes/sml/\theta) = \frac{1}{B(/docs/notes/sml/a,b)} /docs/notes/sml/\theta^{a-1} (1 - /docs/notes/sml/\theta)^{b-1} $$
$$ p(\theta) = \frac{1}{B(/docs/notes/sml/a,b)} \theta^{a-1} (/docs/notes/sml/1 - \theta)^{b-1} $$
$$ p(/docs/notes/sml/\theta) = \frac{1}{B(a,b)} /docs/notes/sml/\theta^{a-1} (/docs/notes/sml/1 - /docs/notes/sml/\theta)^{b-1} $$
$$ p(\theta) = \frac{1}{B(/docs/notes/sml/a,b)} \theta^{a-1} (/docs/notes/sml/1 - \theta)^{b-1} $$

where $B(/docs/notes/sml/a,b)$ is a special function called the _Beta function_.

Select appropriate values for $a$ and $b$ by looking at the plot of $p(/docs/notes/sml/\theta)$ below.


```python
a = 2 # fill in
b = 6 # fill in
theta = np.linspace(/docs/notes/sml/0, 1, 1001)
plt.plot(/docs/notes/sml/theta, beta.pdf(theta, a, b))
plt.title(/docs/notes/sml/'Prior distribution')
plt.xlabel(/docs/notes/sml/r'$\theta$')
plt.ylabel(/docs/notes/sml/r'$p(\theta)$')
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/w/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```


    
![png](/docs/notes/sml/worksheet02_solutions-1.md/worksheet02_solutions-1_files/worksheet02_solutions-1_5_0.png)
    


### 3. Posterior updates
Now toss the coin once and denote the outcome by $x_1$.


```python
/docs/notes/sml/x/docs/notes/sml/1/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/_/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/print(/docs/notes/sml/x1)
```

    0


We can update our belief about $\theta$, based on this new evidence $x_1$.
To do this we apply Bayes' rule to compute the posterior for $\theta$:

$$ p(/docs/notes/sml/\theta | x_1) = \frac{p(x_1 | \theta) \, p(\theta)}{p(x_1)} \propto p(x_1 | \theta) \, p(\theta)$$
$$ p(/docs/notes/sml/\theta | x_1) = \frac{p(/docs/notes/sml/x_1 | \theta) \, p(\theta)}{p(x_1)} \propto p(/docs/notes/sml/x_1 | \theta) \, p(\theta)$$
$$ p(/docs/notes/sml/\theta | x_1) = \frac{p(/docs/notes/sml/x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)}{p(x_1)} \propto p(/docs/notes/sml/x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)$$
$$ p(\theta | /docs/notes/sml/x_1) = \frac{p(/docs/notes/sml//docs/notes/sml/x_1 | \theta) \, p(\theta)}{p(/docs/notes/sml/x_1)} \propto p(/docs/notes/sml//docs/notes/sml/x_1 | \theta) \, p(\theta)$$
$$ p(/docs/notes/sml/\theta | x_1) = \frac{p(/docs/notes/sml/x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)}{p(x_1)} \propto p(/docs/notes/sml/x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)$$
$$ p(/docs/notes/sml/\theta | x_1) = \frac{p(x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)}{p(x_1)} \propto p(x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)$$
$$ p(/docs/notes/sml/\theta | /docs/notes/sml/x_1) = \frac{p(/docs/notes/sml/x_1 | \theta) \, p(\theta)}{p(/docs/notes/sml/x_1)} \propto p(/docs/notes/sml/x_1 | \theta) \, p(\theta)$$
$$ p(/docs/notes/sml/\theta | /docs/notes/sml/x_1) = \frac{p(/docs/notes/sml/x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)}{p(/docs/notes/sml/x_1)} \propto p(/docs/notes/sml/x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)$$
$$ p(/docs/notes/sml/\theta | /docs/notes/sml/x_1) = \frac{p(/docs/notes/sml/x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)}{p(/docs/notes/sml/x_1)} \propto p(/docs/notes/sml/x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)$$
$$ p(/docs/notes/sml/\theta | x_1) = \frac{p(/docs/notes/sml/x_1 | \theta) \, p(\theta)}{p(x_1)} \propto p(/docs/notes/sml/x_1 | \theta) \, p(\theta)$$
$$ p(/docs/notes/sml/\theta | x_1) = \frac{p(/docs/notes/sml/x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)}{p(x_1)} \propto p(/docs/notes/sml/x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)$$
$$ p(\theta | /docs/notes/sml/x_1) = \frac{p(/docs/notes/sml//docs/notes/sml/x_1 | \theta) \, p(\theta)}{p(/docs/notes/sml/x_1)} \propto p(/docs/notes/sml//docs/notes/sml/x_1 | \theta) \, p(\theta)$$
$$ p(/docs/notes/sml/\theta | x_1) = \frac{p(/docs/notes/sml/x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)}{p(x_1)} \propto p(/docs/notes/sml/x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)$$
$$ p(/docs/notes/sml/\theta | x_1) = \frac{p(x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)}{p(x_1)} \propto p(x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta)$$

where $p(/docs/notes/sml/\theta)$ is the prior given above and 

$$ p(/docs/notes/sml/x_1 | \theta) = \theta^{x_1} (1 - \theta)^{1 - x_1} $$
$$ p(/docs/notes/sml/x_1 | \theta) = \theta^{x_1} (/docs/notes/sml/1 - \theta)^{1 - x_1} $$

is the likelihood.

***
**Exercise:** Show (/docs/notes/sml/on paper) that

$$ p(/docs/notes/sml/\theta | x_1) \propto \theta^{x_1 + a - 1} (1 - \theta)^{(1 - x_1) + b - 1} $$
$$ p(/docs/notes/sml/\theta | x_1) \propto \theta^{x_1 + a - 1} (/docs/notes/sml/1 - \theta)^{(1 - x_1) + b - 1} $$
$$ p(\theta | x_1) \propto \theta^{x_1 + a - 1} (/docs/notes/sml/1 - \theta)^{(/docs/notes/sml/1 - x_1) + b - 1} $$
$$ p(/docs/notes/sml/\theta | x_1) \propto \theta^{x_1 + a - 1} (1 - \theta)^{(/docs/notes/sml/1 - x_1) + b - 1} $$
$$ p(\theta | x_1) \propto \theta^{x_1 + a - 1} (/docs/notes/sml/1 - \theta)^{(/docs/notes/sml/1 - x_1) + b - 1} $$

which implies that $\theta | x_1 \sim \mathrm{Beta}[x_1 + a, (/docs/notes/sml/1 - x_1) + b]$.

_Hint: see Lecture 2, slide 23 for a similar calculation._

_Solution:_
Using Bayes' Theorem, we combine the Bernoulli likelihood with the Beta prior. 
We can drop constant factors and recover the normalising constants by comparing with the standard Beta distribution at the end.

$$
\begin{align}
    p(/docs/notes/sml/\theta \vert x_1) &\propto p(x_1 | \theta) \, p(\theta) \\
    p(/docs/notes/sml/\theta \vert x_1) &\propto p(/docs/notes/sml/x_1 | \theta) \, p(\theta) \\
    p(/docs/notes/sml/\theta \vert x_1) &\propto p(/docs/notes/sml/x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta) \\
    p(/docs/notes/sml/\theta \vert x_1) &\propto p(x_1 | /docs/notes/sml/\theta) \, p(/docs/notes/sml/\theta) \\
    &\propto \theta^{\alpha + x_1 -1} (/docs/notes/sml/1-\theta)^{\beta -x_1}  \\
    &\propto \frac{1}{B(/docs/notes/sml/\alpha', \beta')} \theta^{\alpha'-1}(1-\theta)^{\beta'-1}
    &\propto \frac{1}{B(/docs/notes/sml/\alpha', \beta')} \theta^{\alpha'-1}(/docs/notes/sml/1-\theta)^{\beta'-1}
\end{align}
$$

where $\alpha' = \alpha + x_1$ and $\beta' = \beta + 1-x_1$. 
***

Toss the coin a second time, denoting the outcome by $x_2$.


```python
/docs/notes/sml/x/docs/notes/sml/2/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/_/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/print(/docs/notes/sml/x2)
```

    1


Again, we want to update our belief about $\theta$ based on the new information $x_2$. 
We take the previous posterior $p(/docs/notes/sml/\theta|x_1)$ as the new prior and apply Bayes' rule:

$$ p(/docs/notes/sml/\theta | x_1, x_2) \propto p(x_2 | \theta) p(\theta | x_1)$$
$$ p(/docs/notes/sml/\theta | x_1, x_2) \propto p(/docs/notes/sml/x_2 | \theta) p(\theta | x_1)$$
$$ p(/docs/notes/sml/\theta | x_1, x_2) \propto p(/docs/notes/sml/x_2 | \theta) p(/docs/notes/sml/\theta | x_1)$$
$$ p(/docs/notes/sml/\theta | x_1, x_2) \propto p(x_2 | \theta) p(/docs/notes/sml/\theta | x_1)$$

\[Note: We assume the tosses are independent, otherwise the likelihood for $x_2$ would depend on $x_1$.\]
This gives $\theta | x_1, x_2 \sim \mathrm{Beta}[x_1 + x_2 + a, (/docs/notes/sml/2 - x_1 - x_2) + b]$.

***
**Exercise:** Show that for $n$ coin tosses, the posterior is $\theta | x_1, \ldots, x_n \sim \operatorname{Beta}[n_H + a, n - n_H + b]$ where $n_H = \sum_{i = 1}^{n} x_i$ is the number of heads observed.

_Solution:_
We assume the coin tosses are i.i.d. with a probability $\theta$ of returning heads.
The likelihood can be written as (/docs/notes/sml/where $\mathbf{x}_n = \left(x_1, \ldots x_n\right)$ is shorthand for all observations up to step $n$.):

\begin{align}
    p(/docs/notes/sml/\mathbf{x}_n \vert \theta) &= \prod_{k=1}^n p(x_k \vert \theta) = \prod_{k=1}^n \theta^{x_k} (1- \theta)^{1-x_k} 
    p(/docs/notes/sml/\mathbf{x}_n \vert \theta) &= \prod_{k=1}^n p(/docs/notes/sml/x_k \vert \theta) = \prod_{k=1}^n \theta^{x_k} (1- \theta)^{1-x_k} 
    p(\mathbf{x}_n \vert \theta) &= \prod_{k=1}^n p(/docs/notes/sml/x_k \vert \theta) = \prod_{k=1}^n \theta^{x_k} (/docs/notes/sml/1- \theta)^{1-x_k} 
    p(/docs/notes/sml/\mathbf{x}_n \vert \theta) &= \prod_{k=1}^n p(x_k \vert \theta) = \prod_{k=1}^n \theta^{x_k} (/docs/notes/sml/1- \theta)^{1-x_k} 
    p(\mathbf{x}_n \vert \theta) &= \prod_{k=1}^n p(/docs/notes/sml/x_k \vert \theta) = \prod_{k=1}^n \theta^{x_k} (/docs/notes/sml/1- \theta)^{1-x_k} 
\end{align}

Applying Bayes' theorem, the posterior assumes the form:

\begin{align}
    p(/docs/notes/sml/\theta \vert \mathbf{x}_n) &= p(\mathbf{x}_n|\theta) p(\theta) \\
    p(/docs/notes/sml/\theta \vert \mathbf{x}_n) &= p(/docs/notes/sml/\mathbf{x}_n|\theta) p(\theta) \\
    p(/docs/notes/sml/\theta \vert \mathbf{x}_n) &= p(/docs/notes/sml/\mathbf{x}_n|/docs/notes/sml/\theta) p(/docs/notes/sml/\theta) \\
    p(/docs/notes/sml/\theta \vert \mathbf{x}_n) &= p(\mathbf{x}_n|/docs/notes/sml/\theta) p(/docs/notes/sml/\theta) \\
    &= p(/docs/notes/sml/\theta) \prod_{i=1}^{n} p(x_i \vert /docs/notes/sml/\theta) \\
    &= p(/docs/notes/sml/\theta) \prod_{i=1}^{n} p(/docs/notes/sml/x_i \vert /docs/notes/sml/\theta) \\
    &\propto  \theta^{\alpha-1}(/docs/notes/sml/1-\theta)^{\beta-1} \theta^{\sum_{i = 1}^{n} x_i} (/docs/notes/sml/1-\theta)^{n-\sum_{i = 1}^{n} x_i} \\
    &\propto  \theta^{\alpha-1}(/docs/notes/sml/1-\theta)^{\beta-1} \theta^{\sum_{i = 1}^{n} x_i} (/docs/notes/sml/1-\theta)^{n-\sum_{i = 1}^{n} x_i} \\
    &= \theta^{\sum_{i = 1}^n x_i + \alpha-1} (/docs/notes/sml/1-\theta)^{n-\sum_{i = 1}^n x_i + \beta - 1}
\end{align}

This corresponds to $\operatorname{Beta}[n_H + a, n - n_H + b]$ by inspection.
***

### 4. MAP estimator and MLE estimator

The posterior $\theta|x_1, \ldots, x_n$ contains all the information we know about $\theta$ after observing $n$ coin tosses.
One way of obtaining a point estimate of $\theta$ from the posterior, is to take the value with the maximum a posteriori probability (/docs/notes/sml/MAP):
$$
\begin{align}
    \hat{\theta}_\mathrm{MAP} &= \arg \max_{\theta} p(/docs/notes/sml/\theta|x_1, \ldots, x_n) \\
        & = \frac{n_H + a - 1}{n + a + b - 2}
\end{align}
$$

In general, the MAP estimator gives a different result to the maximum likelihood estimator (/docs/notes/sml/MLE) for $\theta$:
$$
\begin{align}
    \hat{\theta}_\mathrm{MLE} &=\arg \max_{\theta} p(/docs/notes/sml/x_1, \ldots, x_n|\theta) \\
        & = \frac{n_H}{n}
\end{align}
$$

***
**Exercise:** How would you derive the above results for $\hat{\theta}_\mathrm{MAP}$ and  $\hat{\theta}_\mathrm{MLE}$? Setup the equations necessary to solve for $\hat{\theta}_\mathrm{MAP}$ and  $\hat{\theta}_\mathrm{MLE}$. You do not need to solve the equations at this stage.

_Solution:_

In a previous exercise, we found that the posterior was given by

$$
p(/docs/notes/sml/\theta | x_1, \ldots, x_n) \propto \theta^{n_H + a - 1} (1-\theta)^{n - n_H + b - 1}
p(/docs/notes/sml/\theta | x_1, \ldots, x_n) \propto \theta^{n_H + a - 1} (/docs/notes/sml/1-\theta)^{n - n_H + b - 1}
$$

The maximum a-posteriori estimate $\hat{\theta}_\mathrm{MAP}$ corresponds to the mode of this distribution. 
We can find the mode of a Beta pmf $f(/docs/notes/sml/\theta) \propto /docs/notes/sml/\theta^{\alpha - 1} (1 - /docs/notes/sml/\theta)^{\beta - 1}$ by solving for the critical point $\tilde{/docs/notes/sml/\theta}$ as follows:
We can find the mode of a Beta pmf $f(/docs/notes/sml/\theta) \propto /docs/notes/sml/\theta^{\alpha - 1} (/docs/notes/sml/1 - /docs/notes/sml/\theta)^{\beta - 1}$ by solving for the critical point $\tilde{/docs/notes/sml/\theta}$ as follows:

$$
\frac{\partial f}{\partial \theta}(/docs/notes/sml/\tilde{\theta}) \propto /docs/notes/sml/\tilde{\theta}^{\alpha-2} (1-/docs/notes/sml/\tilde{\theta})^{\beta-2} \left[(\alpha-1) - /docs/notes/sml/\tilde{\theta}(\alpha + \beta -2)\right] = 0.
\frac{\partial f}{\partial \theta}(/docs/notes/sml/\tilde{\theta}) \propto /docs/notes/sml/\tilde{\theta}^{\alpha-2} (/docs/notes/sml/1-/docs/notes/sml/\tilde{\theta})^{\beta-2} \left[(\alpha-1) - /docs/notes/sml/\tilde{\theta}(\alpha + \beta -2)\right] = 0.
\frac{\partial f}{\partial \theta}(\tilde{\theta}) \propto \tilde{\theta}^{\alpha-2} (/docs/notes/sml/1-\tilde{\theta})^{\beta-2} \left[(/docs/notes/sml/\alpha-1) - \tilde{\theta}(\alpha + \beta -2)\right] = 0.
\frac{\partial f}{\partial \theta}(\tilde{\theta}) \propto \tilde{\theta}^{\alpha-2} (/docs/notes/sml/1-\tilde{\theta})^{\beta-2} \left[(\alpha-1) - \tilde{\theta}(/docs/notes/sml/\alpha + \beta -2)\right] = 0.
\frac{\partial f}{\partial \theta}(/docs/notes/sml/\tilde{\theta}) \propto /docs/notes/sml/\tilde{\theta}^{\alpha-2} (1-/docs/notes/sml/\tilde{\theta})^{\beta-2} \left[(/docs/notes/sml/\alpha-1) - /docs/notes/sml/\tilde{\theta}(\alpha + \beta -2)\right] = 0.
\frac{\partial f}{\partial \theta}(\tilde{\theta}) \propto \tilde{\theta}^{\alpha-2} (/docs/notes/sml/1-\tilde{\theta})^{\beta-2} \left[(/docs/notes/sml/\alpha-1) - \tilde{\theta}(\alpha + \beta -2)\right] = 0.
\frac{\partial f}{\partial \theta}(\tilde{\theta}) \propto \tilde{\theta}^{\alpha-2} (1-\tilde{\theta})^{\beta-2} \left[(/docs/notes/sml/\alpha-1) - \tilde{\theta}(/docs/notes/sml/\alpha + \beta -2)\right] = 0.
\frac{\partial f}{\partial \theta}(/docs/notes/sml/\tilde{\theta}) \propto /docs/notes/sml/\tilde{\theta}^{\alpha-2} (1-/docs/notes/sml/\tilde{\theta})^{\beta-2} \left[(\alpha-1) - /docs/notes/sml/\tilde{\theta}(/docs/notes/sml/\alpha + \beta -2)\right] = 0.
\frac{\partial f}{\partial \theta}(\tilde{\theta}) \propto \tilde{\theta}^{\alpha-2} (/docs/notes/sml/1-\tilde{\theta})^{\beta-2} \left[(\alpha-1) - \tilde{\theta}(/docs/notes/sml/\alpha + \beta -2)\right] = 0.
\frac{\partial f}{\partial \theta}(\tilde{\theta}) \propto \tilde{\theta}^{\alpha-2} (1-\tilde{\theta})^{\beta-2} \left[(/docs/notes/sml/\alpha-1) - \tilde{\theta}(/docs/notes/sml/\alpha + \beta -2)\right] = 0.
$$

The solutions are $\tilde{\theta} = 0, 1, \frac{\alpha - 1}{\alpha + \beta - 2}$ assuming $\alpha, \beta > 1$. By performing the second derivative test, we find that $\frac{\alpha - 1}{\alpha + \beta - 2}$ corresponds to the maximum.

Thus we have
$$
\begin{equation}
    \hat{\theta}_\mathrm{MAP} = \frac{n_H + a - 1}{n + a + b - 2}
\end{equation}
$$

To find the maximum likelihood estimate $\hat{\theta}_\mathrm{MLE}$, we undertake a similar procedure, replacing the posterior with the likelihood function:

$$
L(/docs/notes/sml/\theta) = p(x_1, \ldots, x_n|/docs/notes/sml/\theta) = /docs/notes/sml/\theta^{n_H} (1- /docs/notes/sml/\theta)^{n - n_H}.
L(/docs/notes/sml/\theta) = p(/docs/notes/sml/x_1, \ldots, x_n|/docs/notes/sml/\theta) = /docs/notes/sml/\theta^{n_H} (1- /docs/notes/sml/\theta)^{n - n_H}.
L(\theta) = p(/docs/notes/sml/x_1, \ldots, x_n|\theta) = \theta^{n_H} (/docs/notes/sml/1- \theta)^{n - n_H}.
L(/docs/notes/sml/\theta) = p(x_1, \ldots, x_n|/docs/notes/sml/\theta) = /docs/notes/sml/\theta^{n_H} (/docs/notes/sml/1- /docs/notes/sml/\theta)^{n - n_H}.
L(\theta) = p(/docs/notes/sml/x_1, \ldots, x_n|\theta) = \theta^{n_H} (/docs/notes/sml/1- \theta)^{n - n_H}.
$$

The stationary points $\tilde{/docs/notes/sml/\theta}$ of $L(/docs/notes/sml/\theta)$ satisfy:

$$
\begin{equation}
    \frac{\partial L}{\partial \theta}(/docs/notes/sml/\tilde{\theta}) = /docs/notes/sml/\tilde{\theta}^{n_H - 1} (1- /docs/notes/sml/\tilde{\theta})^{n - n_H - 1} \left[ n_H (1 - /docs/notes/sml/\tilde{\theta}) - (n - n_H)  /docs/notes/sml/\tilde{\theta} \right] = 0
    \frac{\partial L}{\partial \theta}(/docs/notes/sml/\tilde{\theta}) = /docs/notes/sml/\tilde{\theta}^{n_H - 1} (/docs/notes/sml/1- /docs/notes/sml/\tilde{\theta})^{n - n_H - 1} \left[ n_H (1 - /docs/notes/sml/\tilde{\theta}) - (n - n_H)  /docs/notes/sml/\tilde{\theta} \right] = 0
    \frac{\partial L}{\partial \theta}(\tilde{\theta}) = \tilde{\theta}^{n_H - 1} (/docs/notes/sml/1- \tilde{\theta})^{n - n_H - 1} \left[ n_H (/docs/notes/sml/1 - \tilde{\theta}) - (n - n_H)  \tilde{\theta} \right] = 0
    \frac{\partial L}{\partial \theta}(\tilde{\theta}) = \tilde{\theta}^{n_H - 1} (/docs/notes/sml/1- \tilde{\theta})^{/docs/notes/sml/n - n_H - 1} \left[ n_H (1 - \tilde{\theta}) - (/docs/notes/sml/n - n_H)  \tilde{\theta} \right] = 0
    \frac{\partial L}{\partial \theta}(/docs/notes/sml/\tilde{\theta}) = /docs/notes/sml/\tilde{\theta}^{n_H - 1} (1- /docs/notes/sml/\tilde{\theta})^{n - n_H - 1} \left[ n_H (/docs/notes/sml/1 - /docs/notes/sml/\tilde{\theta}) - (n - n_H)  /docs/notes/sml/\tilde{\theta} \right] = 0
    \frac{\partial L}{\partial \theta}(\tilde{\theta}) = \tilde{\theta}^{n_H - 1} (/docs/notes/sml/1- \tilde{\theta})^{n - n_H - 1} \left[ n_H (/docs/notes/sml/1 - \tilde{\theta}) - (n - n_H)  \tilde{\theta} \right] = 0
    \frac{\partial L}{\partial \theta}(\tilde{\theta}) = \tilde{\theta}^{n_H - 1} (1- \tilde{\theta})^{/docs/notes/sml/n - n_H - 1} \left[ n_H (/docs/notes/sml/1 - \tilde{\theta}) - (/docs/notes/sml/n - n_H)  \tilde{\theta} \right] = 0
    \frac{\partial L}{\partial \theta}(/docs/notes/sml/\tilde{\theta}) = /docs/notes/sml/\tilde{\theta}^{n_H - 1} (1- /docs/notes/sml/\tilde{\theta})^{/docs/notes/sml/n - n_H - 1} \left[ n_H (1 - /docs/notes/sml/\tilde{\theta}) - (/docs/notes/sml/n - n_H)  /docs/notes/sml/\tilde{\theta} \right] = 0
    \frac{\partial L}{\partial \theta}(\tilde{\theta}) = \tilde{\theta}^{n_H - 1} (/docs/notes/sml/1- \tilde{\theta})^{/docs/notes/sml/n - n_H - 1} \left[ n_H (1 - \tilde{\theta}) - (/docs/notes/sml/n - n_H)  \tilde{\theta} \right] = 0
    \frac{\partial L}{\partial \theta}(\tilde{\theta}) = \tilde{\theta}^{n_H - 1} (1- \tilde{\theta})^{/docs/notes/sml/n - n_H - 1} \left[ n_H (/docs/notes/sml/1 - \tilde{\theta}) - (/docs/notes/sml/n - n_H)  \tilde{\theta} \right] = 0
\end{equation}
$$

The solutions are $\tilde{\theta} = 0, 1, \frac{n_H}{n}$ assuming $n_H > 1$. The maximum occurs at $\frac{n_H}{n}$ which correspond to $\hat{\theta}_\mathrm{MLE}$.

### 5. Convergence of the estimates

Let's now toss the coin an additional 48 times (/docs/notes/sml/so that $n = 50$), recording $\hat{\theta}_\mathrm{MLE}$ and $\hat{\theta}_\mathrm{MAP}$ after each toss.


```python
extra_tosses = 48
num_tosses = 2 + extra_tosses
num_heads = 0
theta_map = np.zeros(/docs/notes/sml/num_tosses)
theta_mle = np.zeros(/docs/notes/sml/num_tosses)
for i in range(/docs/notes/sml/0, num_tosses):
    if i == 0: 
        num_heads += x1 
    elif i == 1:
        num_heads += x2
    else:
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/n/docs/notes/sml/u/docs/notes/sml/m/docs/notes/sml/_/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/+/docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/_/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/    theta_map[i] = (/docs/notes/sml/num_heads + a - 1)/(i + 1 + a + b - 2) # fill in
    theta_map[i] = (/docs/notes/sml/num_heads + a - 1)/(/docs/notes/sml/i + 1 + a + b - 2) # fill in
    theta_mle[i] = num_heads/(/docs/notes/sml/i + 1) # fill in
```

We plot the results below.


```python
plt.plot(/docs/notes/sml/theta_map, label = "MAP")
plt.plot(/docs/notes/sml/theta_mle, label = "MLE")
plt.xlabel(/docs/notes/sml/'Number of draws')
plt.ylabel(/docs/notes/sml/r'$\hat{\theta}$')
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/w/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```


    
![png](/docs/notes/sml/worksheet02_solutions-1_files/worksheet02_solutions-1_18_0.png)
    


**Questions:** 

1. Is the coin biased?
1. Do the MAP and MLE estimates converge to the same value for $\theta$?
1. What happens if you set $a = 1; b = 1$?
1. How does the posterior distribution for $\theta$ compare to the prior plotted above? (/docs/notes/sml/Use the code block below to plot the posterior.)

_Solutions:_
1. Yes
1. Yes
1. The MAP and MLE estimators are identical.
1. It's more concentrated.


```python
theta_dist = beta(/docs/notes/sml/a + num_heads, b + num_tosses - num_heads)
plt.plot(/docs/notes/sml/theta, theta_dist.pdf(theta))
plt.xlabel(/docs/notes/sml/r'$\theta$')
plt.ylabel(/docs/notes/sml/r'$p(\theta|x_1, \ldots, x_n)$')
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/w/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```


    
![png](/docs/notes/sml/worksheet02_solutions-1_files/worksheet02_solutions-1_20_0.png)
    


Finally, we'll visualize the evolution of the posterior distribution as we observe more data. Before running the code cell below, take a couple of minutes to discuss with those around you how you expect the posterior to behave qualitatively) as the number of observed samples $x_n$ increases.


```python
# Adapted from https://matplotlib.org/3.1.1/gallery/animation/bayes_update.html

class UpdateBetaBernoulli:
    def __init__(/docs/notes/sml/self, ax, a, b, theta_num_points = 201):
        self.a = a
        self.b = b
        self.ax = ax
        self.num_heads = 0
        self.num_tosses = 0
        self.theta = np.linspace(/docs/notes/sml/0, 1, theta_num_points)
        self.line, = ax.plot(/docs/notes/sml/[], [])

    def reset(/docs/notes/sml/self):
        """Reset"""
        self.num_heads = 0
        self.num_tosses = 0
        self.line.set_data(/docs/notes/sml/[], [])
        return self.line,

    def __call__(/docs/notes/sml/self, num_tosses):
        """Perform tosses and update plot"""
        for _ in range(/docs/notes/sml/num_tosses):
            self.num_tosses += 1
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/f/docs/notes/sml/./docs/notes/sml/n/docs/notes/sml/u/docs/notes/sml/m/docs/notes/sml/_/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/+/docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/_/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/        y = beta.pdf(/docs/notes/sml/self.theta, self.num_heads + self.a, self.num_tosses - self.num_heads + self.b)
        self.line.set_data(/docs/notes/sml/self.theta, y)
        self.ax.set_title(/docs/notes/sml/'{:>4} heads, {:>4} tosses'.format(self.num_heads, self.num_tosses), family='monospace')
        return self.line, self.ax.title

# Set up figure
/docs/notes/sml/f/docs/notes/sml/i/docs/notes/sml/g/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/a/docs/notes/sml/x/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/u/docs/notes/sml/b/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/t/docs/notes/sml/s/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/ax.set_xlim(/docs/notes/sml/0, 1)
ax.set_ylim(/docs/notes/sml/0, 15)
ax.set_xlabel(/docs/notes/sml/r'$\theta$')
ax.set_ylabel(/docs/notes/sml/r'$p(\theta|x_1, \ldots)$')

ud = UpdateBetaBernoulli(/docs/notes/sml/ax, a, b)
FuncAnimation(/docs/notes/sml/fig, ud, frames=[1]*200, init_func=ud.reset, repeat=False, interval=50, blit=True)
```

***
### Bonus material: Bayesian credible intervals
In principle, the posterior distribution contains all the information about the possible values of the parameter $\theta$. To show the utility of the posterior, we can obtain a quantitative measure of the posterior uncertainty by computing a central (/docs/notes/sml/or equal-tailed) interval of posterior probability. These are known as _Bayesian credible intervals_ and should not be confused with the frequentist concept of _confidence intervals_ which leverage the distribution of point estimators. For a Bayesian credible interval, an e.g. 95% credible interval contains the true parameter value with 95% probability. In general, for a $1- \alpha$ interval, where $\alpha \in (0,1)$, this corresponds to the range of values $I = (\theta_1, \theta_2)$ above and below which lie exactly $\alpha/2$ of the posterior probability. That is, $\alpha/2$ of the probability mass of the posterior lies below $\theta_1$, and $\alpha/2$ of the probability mass lies above $\theta_2$.
In principle, the posterior distribution contains all the information about the possible values of the parameter $\theta$. To show the utility of the posterior, we can obtain a quantitative measure of the posterior uncertainty by computing a central (/docs/notes/sml/or equal-tailed) interval of posterior probability. These are known as _Bayesian credible intervals_ and should not be confused with the frequentist concept of _confidence intervals_ which leverage the distribution of point estimators. For a Bayesian credible interval, an e.g. 95% credible interval contains the true parameter value with 95% probability. In general, for a $1- \alpha$ interval, where $\alpha \in (/docs/notes/sml/0,1)$, this corresponds to the range of values $I = (\theta_1, \theta_2)$ above and below which lie exactly $\alpha/2$ of the posterior probability. That is, $\alpha/2$ of the probability mass of the posterior lies below $\theta_1$, and $\alpha/2$ of the probability mass lies above $\theta_2$.
In principle, the posterior distribution contains all the information about the possible values of the parameter $\theta$. To show the utility of the posterior, we can obtain a quantitative measure of the posterior uncertainty by computing a central (or equal-tailed) interval of posterior probability. These are known as _Bayesian credible intervals_ and should not be confused with the frequentist concept of _confidence intervals_ which leverage the distribution of point estimators. For a Bayesian credible interval, an e.g. 95% credible interval contains the true parameter value with 95% probability. In general, for a $1- \alpha$ interval, where $\alpha \in (/docs/notes/sml/0,1)$, this corresponds to the range of values $I = (/docs/notes/sml/\theta_1, \theta_2)$ above and below which lie exactly $\alpha/2$ of the posterior probability. That is, $\alpha/2$ of the probability mass of the posterior lies below $\theta_1$, and $\alpha/2$ of the probability mass lies above $\theta_2$.
In principle, the posterior distribution contains all the information about the possible values of the parameter $\theta$. To show the utility of the posterior, we can obtain a quantitative measure of the posterior uncertainty by computing a central (/docs/notes/sml/or equal-tailed) interval of posterior probability. These are known as _Bayesian credible intervals_ and should not be confused with the frequentist concept of _confidence intervals_ which leverage the distribution of point estimators. For a Bayesian credible interval, an e.g. 95% credible interval contains the true parameter value with 95% probability. In general, for a $1- \alpha$ interval, where $\alpha \in (0,1)$, this corresponds to the range of values $I = (/docs/notes/sml/\theta_1, \theta_2)$ above and below which lie exactly $\alpha/2$ of the posterior probability. That is, $\alpha/2$ of the probability mass of the posterior lies below $\theta_1$, and $\alpha/2$ of the probability mass lies above $\theta_2$.
In principle, the posterior distribution contains all the information about the possible values of the parameter $\theta$. To show the utility of the posterior, we can obtain a quantitative measure of the posterior uncertainty by computing a central (or equal-tailed) interval of posterior probability. These are known as _Bayesian credible intervals_ and should not be confused with the frequentist concept of _confidence intervals_ which leverage the distribution of point estimators. For a Bayesian credible interval, an e.g. 95% credible interval contains the true parameter value with 95% probability. In general, for a $1- \alpha$ interval, where $\alpha \in (/docs/notes/sml/0,1)$, this corresponds to the range of values $I = (/docs/notes/sml/\theta_1, \theta_2)$ above and below which lie exactly $\alpha/2$ of the posterior probability. That is, $\alpha/2$ of the probability mass of the posterior lies below $\theta_1$, and $\alpha/2$ of the probability mass lies above $\theta_2$.


```python
alpha = 0.05  # define the confidence level
theta_1, theta_2 = theta_dist.ppf(/docs/notes/sml/[alpha/2., 1-alpha/2.])  # Inverse of the CDF - returns relevant quantiles
```

We should check that $1-\alpha$ of the probability mass actually lies inside our computed interval. That is, we expect

$$ \int_{\theta_1}^{\theta_2} d \theta \; p(/docs/notes/sml/\theta \vert x_1, \ldots x_n) = 1-\alpha $$


```python
from scipy import integrate
integrate.quad(/docs/notes/sml/lambda x: theta_dist.pdf(x), a=theta_1, b=theta_2)  # second return value gives absolute error in integral
```




    (/docs/notes/sml/0.95, 1.6516155050491002e-13)



Looks good! What does this interval look like?


```python
/docs/notes/sml/theta_pdf = /docs/notes/sml/theta_dist.pdf(/docs/notes/sml/theta)
pdf_line, = plt.plot(/docs/notes/sml/theta, theta_pdf)
plt.title(/docs/notes/sml/r'Posterior - $\theta_\mathrm{MAP} =$' + ' ${:.3f}$'.format(theta_map[-1]))
plt.xlabel(/docs/notes/sml/r'$\theta$')
plt.ylabel(/docs/notes/sml/r'$p(\theta|x_1, \ldots, x_N)$')
plt.vlines(/docs/notes/sml/x=theta_1, ymin=0, ymax=theta_dist.pdf(theta_1), linestyle='--', color=pdf_line.get_color())
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/v/docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/(/docs/notes/sml/x/docs/notes/sml/=/docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/_/docs/notes/sml/1/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/y/docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/=/docs/notes/sml/0/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/y/docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/x/docs/notes/sml/=/docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/_/docs/notes/sml/d/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/p/docs/notes/sml/d/docs/notes/sml/f/docs/notes/sml/(/docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/_/docs/notes/sml/1/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/y/docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/=/docs/notes/sml/'/docs/notes/sml/-/docs/notes/sml/-/docs/notes/sml/'/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/=/docs/notes/sml/p/docs/notes/sml/d/docs/notes/sml/f/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/./docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/plt.vlines(/docs/notes/sml/x=theta_2, ymin=0, ymax=theta_dist.pdf(theta_2), linestyle='--', color=pdf_line.get_color())
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/v/docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/(/docs/notes/sml/x/docs/notes/sml/=/docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/_/docs/notes/sml/2/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/y/docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/=/docs/notes/sml/0/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/y/docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/x/docs/notes/sml/=/docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/_/docs/notes/sml/d/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/p/docs/notes/sml/d/docs/notes/sml/f/docs/notes/sml/(/docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/_/docs/notes/sml/2/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/y/docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/=/docs/notes/sml/'/docs/notes/sml/-/docs/notes/sml/-/docs/notes/sml/'/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/=/docs/notes/sml/p/docs/notes/sml/d/docs/notes/sml/f/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/./docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/plt.fill_between(/docs/notes/sml/theta, theta_pdf, 0, where=(theta > theta_1) & (theta < theta_2), alpha=0.1)
plt.fill_between(/docs/notes/sml/theta, theta_pdf, 0, where=(theta > theta_1) & (/docs/notes/sml/theta < theta_2), alpha=0.1)
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/w/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```


    
![png](/docs/notes/sml/worksheet02_solutions-1_files/worksheet02_solutions-1_28_0.png)
    



```python

```
