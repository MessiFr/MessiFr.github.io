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
More specifically, you want to estimate the probability $\theta$ that the coin will land heads-up when you toss it. If $\theta \approx \frac{1}{2}$ then we say that the coin is _unbiased_ (or fair).

You can use the function below to simulate a coin toss: it returns `1` for heads and `0` for tails.


```python
def toss_coin():
    if bernoulli.rvs(p = (int.from_bytes("coin".encode(), 'little') % 10000)/10000):
        return 1
    return 0
```

### 2. Prior belief
Before you even toss the coin, you notice that the heads side appears to have more mass. 
Thus, your _prior belief_ is that $\theta$ is slightly biased away from $\frac{1}{2}$ towards 0—i.e. you expect tails are more likely.

To quantify this prior belief, we assume that the prior distribution for $\theta$ is $\mathrm{Beta}(a,b)$, for some choice of the hyperparameters $a, b > 0$. 
(See [link](https://en.wikipedia.org/wiki/Beta_distribution) for info about the Beta distribution.)
The prior probability density function for $\theta$ is therefore given by:

$$ p(\theta) = \frac{1}{B(a,b)} \theta^{a-1} (1 - \theta)^{b-1} $$

where $B(a,b)$ is a special function called the _Beta function_.

Select appropriate values for $a$ and $b$ by looking at the plot of $p(\theta)$ below.


```python
a = 2 # fill in
b = 6 # fill in
theta = np.linspace(0, 1, 1001)
plt.plot(theta, beta.pdf(theta, a, b))
plt.title('Prior distribution')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta)$')
plt.show()
```


    
![png](/docs/notes/sml/worksheet02_solutions-1_files/worksheet02_solutions-1_5_0.png)
    


### 3. Posterior updates
Now toss the coin once and denote the outcome by $x_1$.


```python
x1 = toss_coin()
print(x1)
```

    0


We can update our belief about $\theta$, based on this new evidence $x_1$.
To do this we apply Bayes' rule to compute the posterior for $\theta$:

$$ p(\theta | x_1) = \frac{p(x_1 | \theta) \, p(\theta)}{p(x_1)} \propto p(x_1 | \theta) \, p(\theta)$$

where $p(\theta)$ is the prior given above and 

$$ p(x_1 | \theta) = \theta^{x_1} (1 - \theta)^{1 - x_1} $$

is the likelihood.

***
**Exercise:** Show (on paper) that

$$ p(\theta | x_1) \propto \theta^{x_1 + a - 1} (1 - \theta)^{(1 - x_1) + b - 1} $$

which implies that $\theta | x_1 \sim \mathrm{Beta}[x_1 + a, (1 - x_1) + b]$.

_Hint: see Lecture 2, slide 23 for a similar calculation._

_Solution:_
Using Bayes' Theorem, we combine the Bernoulli likelihood with the Beta prior. 
We can drop constant factors and recover the normalising constants by comparing with the standard Beta distribution at the end.

$$
\begin{align}
    p(\theta \vert x_1) &\propto p(x_1 | \theta) \, p(\theta) \\
    &\propto \theta^{\alpha + x_1 -1} (1-\theta)^{\beta -x_1}  \\
    &\propto \frac{1}{B(\alpha', \beta')} \theta^{\alpha'-1}(1-\theta)^{\beta'-1}
\end{align}
$$

where $\alpha' = \alpha + x_1$ and $\beta' = \beta + 1-x_1$. 
***

Toss the coin a second time, denoting the outcome by $x_2$.


```python
x2 = toss_coin()
print(x2)
```

    1


Again, we want to update our belief about $\theta$ based on the new information $x_2$. 
We take the previous posterior $p(\theta|x_1)$ as the new prior and apply Bayes' rule:

$$ p(\theta | x_1, x_2) \propto p(x_2 | \theta) p(\theta | x_1)$$

\[Note: We assume the tosses are independent, otherwise the likelihood for $x_2$ would depend on $x_1$.\]
This gives $\theta | x_1, x_2 \sim \mathrm{Beta}[x_1 + x_2 + a, (2 - x_1 - x_2) + b]$.

***
**Exercise:** Show that for $n$ coin tosses, the posterior is $\theta | x_1, \ldots, x_n \sim \operatorname{Beta}[n_H + a, n - n_H + b]$ where $n_H = \sum_{i = 1}^{n} x_i$ is the number of heads observed.

_Solution:_
We assume the coin tosses are i.i.d. with a probability $\theta$ of returning heads.
The likelihood can be written as (where $\mathbf{x}_n = \left(x_1, \ldots x_n\right)$ is shorthand for all observations up to step $n$.):

\begin{align}
    p(\mathbf{x}_n \vert \theta) &= \prod_{k=1}^n p(x_k \vert \theta) = \prod_{k=1}^n \theta^{x_k} (1- \theta)^{1-x_k} 
\end{align}

Applying Bayes' theorem, the posterior assumes the form:

\begin{align}
    p(\theta \vert \mathbf{x}_n) &= p(\mathbf{x}_n|\theta) p(\theta) \\
    &= p(\theta) \prod_{i=1}^{n} p(x_i \vert \theta) \\
    &\propto  \theta^{\alpha-1}(1-\theta)^{\beta-1} \theta^{\sum_{i = 1}^{n} x_i} (1-\theta)^{n-\sum_{i = 1}^{n} x_i} \\
    &= \theta^{\sum_{i = 1}^n x_i + \alpha-1} (1-\theta)^{n-\sum_{i = 1}^n x_i + \beta - 1}
\end{align}

This corresponds to $\operatorname{Beta}[n_H + a, n - n_H + b]$ by inspection.
***

### 4. MAP estimator and MLE estimator

The posterior $\theta|x_1, \ldots, x_n$ contains all the information we know about $\theta$ after observing $n$ coin tosses.
One way of obtaining a point estimate of $\theta$ from the posterior, is to take the value with the maximum a posteriori probability (MAP):
$$
\begin{align}
    \hat{\theta}_\mathrm{MAP} &= \arg \max_{\theta} p(\theta|x_1, \ldots, x_n) \\
        & = \frac{n_H + a - 1}{n + a + b - 2}
\end{align}
$$

In general, the MAP estimator gives a different result to the maximum likelihood estimator (MLE) for $\theta$:
$$
\begin{align}
    \hat{\theta}_\mathrm{MLE} &=\arg \max_{\theta} p(x_1, \ldots, x_n|\theta) \\
        & = \frac{n_H}{n}
\end{align}
$$

***
**Exercise:** How would you derive the above results for $\hat{\theta}_\mathrm{MAP}$ and  $\hat{\theta}_\mathrm{MLE}$? Setup the equations necessary to solve for $\hat{\theta}_\mathrm{MAP}$ and  $\hat{\theta}_\mathrm{MLE}$. You do not need to solve the equations at this stage.

_Solution:_

In a previous exercise, we found that the posterior was given by

$$
p(\theta | x_1, \ldots, x_n) \propto \theta^{n_H + a - 1} (1-\theta)^{n - n_H + b - 1}
$$

The maximum a-posteriori estimate $\hat{\theta}_\mathrm{MAP}$ corresponds to the mode of this distribution. 
We can find the mode of a Beta pmf $f(\theta) \propto \theta^{\alpha - 1} (1 - \theta)^{\beta - 1}$ by solving for the critical point $\tilde{\theta}$ as follows:

$$
\frac{\partial f}{\partial \theta}(\tilde{\theta}) \propto \tilde{\theta}^{\alpha-2} (1-\tilde{\theta})^{\beta-2} \left[(\alpha-1) - \tilde{\theta}(\alpha + \beta -2)\right] = 0.
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
L(\theta) = p(x_1, \ldots, x_n|\theta) = \theta^{n_H} (1- \theta)^{n - n_H}.
$$

The stationary points $\tilde{\theta}$ of $L(\theta)$ satisfy:

$$
\begin{equation}
    \frac{\partial L}{\partial \theta}(\tilde{\theta}) = \tilde{\theta}^{n_H - 1} (1- \tilde{\theta})^{n - n_H - 1} \left[ n_H (1 - \tilde{\theta}) - (n - n_H)  \tilde{\theta} \right] = 0
\end{equation}
$$

The solutions are $\tilde{\theta} = 0, 1, \frac{n_H}{n}$ assuming $n_H > 1$. The maximum occurs at $\frac{n_H}{n}$ which correspond to $\hat{\theta}_\mathrm{MLE}$.

### 5. Convergence of the estimates

Let's now toss the coin an additional 48 times (so that $n = 50$), recording $\hat{\theta}_\mathrm{MLE}$ and $\hat{\theta}_\mathrm{MAP}$ after each toss.


```python
extra_tosses = 48
num_tosses = 2 + extra_tosses
num_heads = 0
theta_map = np.zeros(num_tosses)
theta_mle = np.zeros(num_tosses)
for i in range(0, num_tosses):
    if i == 0: 
        num_heads += x1 
    elif i == 1:
        num_heads += x2
    else:
        num_heads += toss_coin()
    theta_map[i] = (num_heads + a - 1)/(i + 1 + a + b - 2) # fill in
    theta_mle[i] = num_heads/(i + 1) # fill in
```

We plot the results below.


```python
plt.plot(theta_map, label = "MAP")
plt.plot(theta_mle, label = "MLE")
plt.xlabel('Number of draws')
plt.ylabel(r'$\hat{\theta}$')
plt.legend()
plt.show()
```


    
![png](/docs/notes/sml/worksheet02_solutions-1_files/worksheet02_solutions-1_18_0.png)
    


**Questions:** 

1. Is the coin biased?
1. Do the MAP and MLE estimates converge to the same value for $\theta$?
1. What happens if you set $a = 1; b = 1$?
1. How does the posterior distribution for $\theta$ compare to the prior plotted above? (Use the code block below to plot the posterior.)

_Solutions:_
1. Yes
1. Yes
1. The MAP and MLE estimators are identical.
1. It's more concentrated.


```python
theta_dist = beta(a + num_heads, b + num_tosses - num_heads)
plt.plot(theta, theta_dist.pdf(theta))
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta|x_1, \ldots, x_n)$')
plt.show()
```


    
![png](/docs/notes/sml/worksheet02_solutions-1_files/worksheet02_solutions-1_20_0.png)
    


Finally, we'll visualize the evolution of the posterior distribution as we observe more data. Before running the code cell below, take a couple of minutes to discuss with those around you how you expect the posterior to behave qualitatively) as the number of observed samples $x_n$ increases.


```python
# Adapted from https://matplotlib.org/3.1.1/gallery/animation/bayes_update.html

class UpdateBetaBernoulli:
    def __init__(self, ax, a, b, theta_num_points = 201):
        self.a = a
        self.b = b
        self.ax = ax
        self.num_heads = 0
        self.num_tosses = 0
        self.theta = np.linspace(0, 1, theta_num_points)
        self.line, = ax.plot([], [])

    def reset(self):
        """Reset"""
        self.num_heads = 0
        self.num_tosses = 0
        self.line.set_data([], [])
        return self.line,

    def __call__(self, num_tosses):
        """Perform tosses and update plot"""
        for _ in range(num_tosses):
            self.num_tosses += 1
            self.num_heads += toss_coin()
        y = beta.pdf(self.theta, self.num_heads + self.a, self.num_tosses - self.num_heads + self.b)
        self.line.set_data(self.theta, y)
        self.ax.set_title('{:>4} heads, {:>4} tosses'.format(self.num_heads, self.num_tosses), family='monospace')
        return self.line, self.ax.title

# Set up figure
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 15)
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$p(\theta|x_1, \ldots)$')

ud = UpdateBetaBernoulli(ax, a, b)
FuncAnimation(fig, ud, frames=[1]*200, init_func=ud.reset, repeat=False, interval=50, blit=True)
```





<link rel="stylesheet"
href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">
<script language="javascript">
  function isInternetExplorer() {
    ua = navigator.userAgent;
    /* MSIE used to detect old browsers and Trident used to newer ones*/
    return ua.indexOf("MSIE ") > -1 || ua.indexOf("Trident/") > -1;
  }

</script>

<style>
.animation {
    display: inline-block;
    text-align: center;
}
input[type=range].anim-slider {
    width: 374px;
    margin-left: auto;
    margin-right: auto;
}
.anim-buttons {
    margin: 8px 0px;
}
.anim-buttons button {
    padding: 0;
    width: 36px;
}
.anim-state label {
    margin-right: 8px;
}
.anim-state input {
    margin: 0;
    vertical-align: middle;
}
</style>

<div class="animation">
  <img id="_anim_imgc8cc733b2868431b81039491982ca9f9">
  <div class="anim-controls">
    <input id="_anim_sliderc8cc733b2868431b81039491982ca9f9" type="range" class="anim-slider"
           name="points" min="0" max="1" step="1" value="0"
           oninput="animc8cc733b2868431b81039491982ca9f9.set_frame(parseInt(this.value));">
    <div class="anim-buttons">
      <button title="Decrease speed" aria-label="Decrease speed" onclick="animc8cc733b2868431b81039491982ca9f9.slower()">
          <i class="fa fa-minus"></i></button>
      <button title="First frame" aria-label="First frame" onclick="animc8cc733b2868431b81039491982ca9f9.first_frame()">
        <i class="fa fa-fast-backward"></i></button>
      <button title="Previous frame" aria-label="Previous frame" onclick="animc8cc733b2868431b81039491982ca9f9.previous_frame()">
          <i class="fa fa-step-backward"></i></button>
      <button title="Play backwards" aria-label="Play backwards" onclick="animc8cc733b2868431b81039491982ca9f9.reverse_animation()">
          <i class="fa fa-play fa-flip-horizontal"></i></button>
      <button title="Pause" aria-label="Pause" onclick="animc8cc733b2868431b81039491982ca9f9.pause_animation()">
          <i class="fa fa-pause"></i></button>
      <button title="Play" aria-label="Play" onclick="animc8cc733b2868431b81039491982ca9f9.play_animation()">
          <i class="fa fa-play"></i></button>
      <button title="Next frame" aria-label="Next frame" onclick="animc8cc733b2868431b81039491982ca9f9.next_frame()">
          <i class="fa fa-step-forward"></i></button>
      <button title="Last frame" aria-label="Last frame" onclick="animc8cc733b2868431b81039491982ca9f9.last_frame()">
          <i class="fa fa-fast-forward"></i></button>
      <button title="Increase speed" aria-label="Increase speed" onclick="animc8cc733b2868431b81039491982ca9f9.faster()">
          <i class="fa fa-plus"></i></button>
    </div>
    <form title="Repetition mode" aria-label="Repetition mode" action="#n" name="_anim_loop_selectc8cc733b2868431b81039491982ca9f9"
          class="anim-state">
      <input type="radio" name="state" value="once" id="_anim_radio1_c8cc733b2868431b81039491982ca9f9"
             checked>
      <label for="_anim_radio1_c8cc733b2868431b81039491982ca9f9">Once</label>
      <input type="radio" name="state" value="loop" id="_anim_radio2_c8cc733b2868431b81039491982ca9f9"
             >
      <label for="_anim_radio2_c8cc733b2868431b81039491982ca9f9">Loop</label>
      <input type="radio" name="state" value="reflect" id="_anim_radio3_c8cc733b2868431b81039491982ca9f9"
             >
      <label for="_anim_radio3_c8cc733b2868431b81039491982ca9f9">Reflect</label>
    </form>
  </div>
</div>





    
![png](/docs/notes/sml/worksheet02_solutions-1_files/worksheet02_solutions-1_22_1.png)
    


***
### Bonus material: Bayesian credible intervals
In principle, the posterior distribution contains all the information about the possible values of the parameter $\theta$. To show the utility of the posterior, we can obtain a quantitative measure of the posterior uncertainty by computing a central (or equal-tailed) interval of posterior probability. These are known as _Bayesian credible intervals_ and should not be confused with the frequentist concept of _confidence intervals_ which leverage the distribution of point estimators. For a Bayesian credible interval, an e.g. 95% credible interval contains the true parameter value with 95% probability. In general, for a $1- \alpha$ interval, where $\alpha \in (0,1)$, this corresponds to the range of values $I = (\theta_1, \theta_2)$ above and below which lie exactly $\alpha/2$ of the posterior probability. That is, $\alpha/2$ of the probability mass of the posterior lies below $\theta_1$, and $\alpha/2$ of the probability mass lies above $\theta_2$.


```python
alpha = 0.05  # define the confidence level
theta_1, theta_2 = theta_dist.ppf([alpha/2., 1-alpha/2.])  # Inverse of the CDF - returns relevant quantiles
```

We should check that $1-\alpha$ of the probability mass actually lies inside our computed interval. That is, we expect

$$ \int_{\theta_1}^{\theta_2} d \theta \; p(\theta \vert x_1, \ldots x_n) = 1-\alpha $$


```python
from scipy import integrate
integrate.quad(lambda x: theta_dist.pdf(x), a=theta_1, b=theta_2)  # second return value gives absolute error in integral
```




    (0.95, 1.6516155050491002e-13)



Looks good! What does this interval look like?


```python
theta_pdf = theta_dist.pdf(theta)
pdf_line, = plt.plot(theta, theta_pdf)
plt.title(r'Posterior - $\theta_\mathrm{MAP} =$' + ' ${:.3f}$'.format(theta_map[-1]))
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta|x_1, \ldots, x_N)$')
plt.vlines(x=theta_1, ymin=0, ymax=theta_dist.pdf(theta_1), linestyle='--', color=pdf_line.get_color())
plt.vlines(x=theta_2, ymin=0, ymax=theta_dist.pdf(theta_2), linestyle='--', color=pdf_line.get_color())
plt.fill_between(theta, theta_pdf, 0, where=(theta > theta_1) & (theta < theta_2), alpha=0.1)
plt.show()
```


    
![png](/docs/notes/sml/worksheet02_solutions-1_files/worksheet02_solutions-1_28_0.png)
    



```python

```
