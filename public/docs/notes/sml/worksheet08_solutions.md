# COMP90051 Workshop 8
## Convolutional Neural Networks / Convolutional Autoencoders

Last week we trained an MLP model for the MNIST dataset. In this workshop, we will look at a more challenging image classification dataset, `CIFAR-10`, and implement a convolutional neural network (/docs/notes/sml/CNN) in PyTorch

## Image Classification on CIFAR-10
Here we will tackle a supervised learning problem on a canonical image dataset, `CIFAR-10`. This consists of $32 \times 32$ 3-channel color images arriving in 10 classes of various animals and vehicles. Lets take a look at some randomly sampled images below, using a handy convenience function from Torch to download and preprocess the data.


```python
import torch
from torch.utils import data

import numpy as np
import time, os
```


```python
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(/docs/notes/sml/'darkgrid')
%matplotlib inline

/docs/notes/sml/c/docs/notes/sml/i/docs/notes/sml/f/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/1/docs/notes/sml/0/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/n/docs/notes/sml/s/docs/notes/sml/f/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/m/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/n/docs/notes/sml/s/docs/notes/sml/f/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/m/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/T/docs/notes/sml/o/docs/notes/sml/T/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/s/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/
trainset = torchvision.datasets.CIFAR10(/docs/notes/sml/root='./data', train=True, download=True, transform=cifar10_transform)
testset = torchvision.datasets.CIFAR10(/docs/notes/sml/root='./data', train=False, download=True, transform=cifar10_transform)

train_loader = torch.utils.data.DataLoader(/docs/notes/sml/trainset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(/docs/notes/sml/testset, batch_size=128, shuffle=False)

classes = (/docs/notes/sml/'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
label_class_map = dict(/docs/notes/sml/zip(range(10), classes))
```

    Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz



    HBox(/docs/notes/sml/children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))
    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(/docs/notes/sml/value='')))


    Failed download. Trying https -> http instead. Downloading http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz



    HBox(/docs/notes/sml/children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(value='')))
    HBox(children=(FloatProgress(value=1.0, bar_style='info', max=1.0), HTML(/docs/notes/sml/value='')))


    Extracting ./data/cifar-10-python.tar.gz to ./data
    Files already downloaded and verified



```python
FIGURE_RESOLUTION = 256
plt.rcParams['figure.dpi'] = FIGURE_RESOLUTION

def imshow(/docs/notes/sml/img, title=None):
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/n/docs/notes/sml/p/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/g/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/g/docs/notes/sml/./docs/notes/sml/c/docs/notes/sml/p/docs/notes/sml/u/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/n/docs/notes/sml/u/docs/notes/sml/m/docs/notes/sml/p/docs/notes/sml/y/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/n/docs/notes/sml/p/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/g/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/g/docs/notes/sml/./docs/notes/sml/c/docs/notes/sml/p/docs/notes/sml/u/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/n/docs/notes/sml/u/docs/notes/sml/m/docs/notes/sml/p/docs/notes/sml/y/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/    plt.imshow(/docs/notes/sml/np.transpose(npimg, (1, 2, 0)))
    plt.axis(/docs/notes/sml/'off')
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/g/docs/notes/sml/h/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/y/docs/notes/sml/o/docs/notes/sml/u/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/    
    if title is not None:
        plt.title(/docs/notes/sml/str(title))
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/w/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/
images, labels = iter(/docs/notes/sml/train_loader).next()
/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/b/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/(/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/x/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/images = images[:8]
labels = labels[:8]

imshow(/docs/notes/sml/torchvision.utils.make_grid(images))
print(/docs/notes/sml/[label_class_map[c.item()] for c in labels])
```


    
![png](/docs/notes/sml/worksheet08_solutions_files/worksheet08_solutions_3_0.png)
    


    ['bird', 'ship', 'ship', 'truck', 'car', 'frog', 'dog', 'bird']


## Multilayer Perceptron

You should immediately notice that `CIFAR-10` is similar to `MNIST`, they both are _multi-calss_ classification problems, with the same number of labels, but different shapes of image. We'll first tackle the problem by reusing MLP from the last workshop.


```python
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_DIM1 = 256
HIDDEN_DIM2 = 100

class MultilayerPerceptronModel(/docs/notes/sml/nn.Module):
    
    def __init__(/docs/notes/sml/self, n_features, n_classes, hidden_dim1 = HIDDEN_DIM1, hidden_dim2 = HIDDEN_DIM2):
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/u/docs/notes/sml/p/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/_/docs/notes/sml/_/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/_/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/u/docs/notes/sml/p/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/_/docs/notes/sml/_/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/_/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/        
        self.input_layer = nn.Linear(/docs/notes/sml/n_features, hidden_dim1)
        self.hidden_layer = nn.Linear(/docs/notes/sml/hidden_dim1, hidden_dim2)
        self.output_layer = nn.Linear(/docs/notes/sml/hidden_dim2, n_classes)
        self.dropout = nn.Dropout(/docs/notes/sml/0.25)

    def forward(/docs/notes/sml/self, x):

        batch_size = x.shape[0]
        
        x = x.view(/docs/notes/sml/batch_size, -1)  # Flatten image into vector, retaining batch dimension
        
        x = F.relu(/docs/notes/sml/self.input_layer(x))
        /docs/notes/sml/x = self.dropout(/docs/notes/sml/x)
        
        
        self.rep1 = x #for plotting purpose
        
        x = F.relu(/docs/notes/sml/self.hidden_layer(x))
        /docs/notes/sml/x = self.dropout(/docs/notes/sml/x)
        
        self.rep2 = x #for plotting purpose
        
        out = self.output_layer(/docs/notes/sml/x)
        
        return out
        
```

We need `train` and `test` functions for MLP and CNN models, below code is identical to what you encountered last week.


```python
def test(/docs/notes/sml/model, criterion, test_loader):
    test_loss = 0.
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/p/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/d/docs/notes/sml/s/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/b/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/p/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/d/docs/notes/sml/s/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/b/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/    for i, data in enumerate(/docs/notes/sml/test_loader):
        x, labels = data

/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/w/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/./docs/notes/sml/n/docs/notes/sml/o/docs/notes/sml/_/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/:/docs/notes/sml/
/docs/notes/sml/            logits = model(/docs/notes/sml/x)  # Compute scores
            predictions = torch.argmax(/docs/notes/sml/logits, dim=1)
            test_loss += criterion(/docs/notes/sml/input=logits, target=labels).item()
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/+/docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/c/docs/notes/sml/r/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/p/docs/notes/sml/u/docs/notes/sml/t/docs/notes/sml/=/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/g/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/s/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/=/docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/b/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/s/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/m/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/            test_preds.append(/docs/notes/sml/predictions)
            test_/docs/notes/sml/labels.append(/docs/notes/sml/labels)

    /docs/notes/sml/test_preds = torch.cat(/docs/notes/sml/test_preds)
    /docs/notes/sml/test_labels = torch.cat(/docs/notes/sml/test_labels)

    test_accuracy = torch.eq(/docs/notes/sml/test_preds, test_labels).float().mean().item()
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/c/docs/notes/sml/u/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/y/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/./docs/notes/sml/e/docs/notes/sml/q/docs/notes/sml/(/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/p/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/d/docs/notes/sml/s/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/b/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/s/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/f/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/m/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/c/docs/notes/sml/u/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/y/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/./docs/notes/sml/e/docs/notes/sml/q/docs/notes/sml/(/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/p/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/d/docs/notes/sml/s/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/b/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/s/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/f/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/m/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/c/docs/notes/sml/u/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/y/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/./docs/notes/sml/e/docs/notes/sml/q/docs/notes/sml/(/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/p/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/d/docs/notes/sml/s/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/b/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/s/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/f/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/m/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/
    print(/docs/notes/sml/'[TEST] Mean loss {:.4f} | Accuracy {:.4f}'.format(test_loss/len(test_loader), test_accuracy))

def train(/docs/notes/sml/model, train_loader, test_loader, optimizer, n_epochs=10):
    """
    Generic training loop for supervised multiclass learning
    """
    LOG_INTERVAL = 250
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/r/docs/notes/sml/u/docs/notes/sml/n/docs/notes/sml/n/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/g/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/r/docs/notes/sml/u/docs/notes/sml/n/docs/notes/sml/n/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/g/docs/notes/sml/_/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/c/docs/notes/sml/u/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/y/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/r/docs/notes/sml/u/docs/notes/sml/n/docs/notes/sml/n/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/g/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/r/docs/notes/sml/u/docs/notes/sml/n/docs/notes/sml/n/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/g/docs/notes/sml/_/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/c/docs/notes/sml/u/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/y/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/./docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/c/docs/notes/sml/r/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/./docs/notes/sml/n/docs/notes/sml/n/docs/notes/sml/./docs/notes/sml/C/docs/notes/sml/r/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/E/docs/notes/sml/n/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/o/docs/notes/sml/p/docs/notes/sml/y/docs/notes/sml/L/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/
    for epoch in range(/docs/notes/sml/n_epochs):  # Loop over training dataset `/docs/notes/sml/n_epochs` times

        epoch_loss = 0.

        for i, data in enumerate(/docs/notes/sml/train_loader):  # Loop over elements in training set

            x, labels = data

            logits = model(/docs/notes/sml/x)

            predictions = torch.argmax(/docs/notes/sml/logits, dim=1)
            train_acc = torch.mean(/docs/notes/sml/torch.eq(predictions, labels).float()).item()
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/c/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/./docs/notes/sml/e/docs/notes/sml/q/docs/notes/sml/(/docs/notes/sml/p/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/d/docs/notes/sml/i/docs/notes/sml/c/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/s/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/b/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/s/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/f/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/m/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/_/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/c/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/./docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/./docs/notes/sml/e/docs/notes/sml/q/docs/notes/sml/(/docs/notes/sml/p/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/d/docs/notes/sml/i/docs/notes/sml/c/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/s/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/b/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/s/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/f/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/m/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/
            loss = criterion(/docs/notes/sml/input=logits, target=labels)

/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/b/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/k/docs/notes/sml/w/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/#/docs/notes/sml/ /docs/notes/sml/B/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/k/docs/notes/sml/w/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/d/docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/(/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/m/docs/notes/sml/p/docs/notes/sml/u/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/i/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/t/docs/notes/sml/s/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/            loss.backward()               # Backward pass (/docs/notes/sml/compute parameter gradients)
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/o/docs/notes/sml/p/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/z/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/p/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/#/docs/notes/sml/ /docs/notes/sml/U/docs/notes/sml/p/docs/notes/sml/d/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/w/docs/notes/sml/e/docs/notes/sml/i/docs/notes/sml/g/docs/notes/sml/h/docs/notes/sml/t/docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/u/docs/notes/sml/s/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/g/docs/notes/sml/ /docs/notes/sml/S/docs/notes/sml/G/docs/notes/sml/D/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/o/docs/notes/sml/p/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/z/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/./docs/notes/sml/z/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/o/docs/notes/sml/_/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/#/docs/notes/sml/ /docs/notes/sml/R/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/ /docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/i/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/t/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/ /docs/notes/sml/z/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/o/docs/notes/sml/ /docs/notes/sml/f/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/x/docs/notes/sml/t/docs/notes/sml/ /docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/
/docs/notes/sml/

            # ============================================================================
            # You can safely ignore the boilerplate code below - just reports metrics over
            # training and test sets

            running_loss.append(/docs/notes/sml/loss.item())
            running_accuracy.append(/docs/notes/sml/train_acc)

/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/e/docs/notes/sml/p/docs/notes/sml/o/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/+/docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/m/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/
            if i % LOG_INTERVAL == 0:  # Log training stats
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/T/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/./docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/ /docs/notes/sml/-/docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/
/docs/notes/sml/                mean_loss = epoch_loss / (/docs/notes/sml/i+1)
                print('[TRAIN] Epoch {} [{}/{}]| Mean loss {:.4f} | Train accuracy {:.5f} | Time {:.2f} s'.format(epoch, 
                    i, len(/docs/notes/sml/train_loader), mean_loss, train_acc, deltaT))

        print(/docs/notes/sml/'Epoch complete! Mean loss: {:.4f}'.format(epoch_loss/len(train_loader)))

        test(/docs/notes/sml/model, criterion, test_loader)
        
    return running_loss, running_accuracy
```


```python
n_features, n_classes = 32*32*3, 10  # Here we flatten the 3D image into a 1D vector
mlp_model = MultilayerPerceptronModel(/docs/notes/sml/n_features, n_classes)
optimizer = torch.optim.SGD(/docs/notes/sml/mlp_model.parameters(), lr=1e-2, momentum=0.9)
mlp_loss, mlp_acc = train(/docs/notes/sml/mlp_model, train_loader, test_loader, optimizer)
```

    [TRAIN] Epoch 0 [0/391]| Mean loss 2.3087 | Train accuracy 0.10938 | Time 0.10 s
    [TRAIN] Epoch 0 [250/391]| Mean loss 2.0595 | Train accuracy 0.25781 | Time 7.59 s
    Epoch complete! Mean loss: 1.9966
    [TEST] Mean loss 1.8652 | Accuracy 0.3288
    [TRAIN] Epoch 1 [0/391]| Mean loss 1.7463 | Train accuracy 0.35156 | Time 13.33 s
    [TRAIN] Epoch 1 [250/391]| Mean loss 1.8364 | Train accuracy 0.34375 | Time 20.01 s
    Epoch complete! Mean loss: 1.8148
    [TEST] Mean loss 1.7806 | Accuracy 0.3601
    [TRAIN] Epoch 2 [0/391]| Mean loss 1.7453 | Train accuracy 0.32031 | Time 25.54 s
    [TRAIN] Epoch 2 [250/391]| Mean loss 1.7609 | Train accuracy 0.37500 | Time 31.91 s
    Epoch complete! Mean loss: 1.7494
    [TEST] Mean loss 1.7113 | Accuracy 0.3795
    [TRAIN] Epoch 3 [0/391]| Mean loss 1.7166 | Train accuracy 0.38281 | Time 36.91 s
    [TRAIN] Epoch 3 [250/391]| Mean loss 1.7133 | Train accuracy 0.36719 | Time 43.26 s
    Epoch complete! Mean loss: 1.7055
    [TEST] Mean loss 1.6988 | Accuracy 0.3903
    [TRAIN] Epoch 4 [0/391]| Mean loss 1.7458 | Train accuracy 0.40625 | Time 48.24 s
    [TRAIN] Epoch 4 [250/391]| Mean loss 1.6747 | Train accuracy 0.45312 | Time 54.58 s
    Epoch complete! Mean loss: 1.6653
    [TEST] Mean loss 1.6466 | Accuracy 0.4097
    [TRAIN] Epoch 5 [0/391]| Mean loss 1.5158 | Train accuracy 0.43750 | Time 59.65 s
    [TRAIN] Epoch 5 [250/391]| Mean loss 1.6369 | Train accuracy 0.48438 | Time 66.42 s
    Epoch complete! Mean loss: 1.6313
    [TEST] Mean loss 1.6223 | Accuracy 0.4221
    [TRAIN] Epoch 6 [0/391]| Mean loss 1.7862 | Train accuracy 0.35938 | Time 71.70 s
    [TRAIN] Epoch 6 [250/391]| Mean loss 1.6175 | Train accuracy 0.32031 | Time 78.17 s
    Epoch complete! Mean loss: 1.6137
    [TEST] Mean loss 1.6034 | Accuracy 0.4266
    [TRAIN] Epoch 7 [0/391]| Mean loss 1.7263 | Train accuracy 0.40625 | Time 83.63 s
    [TRAIN] Epoch 7 [250/391]| Mean loss 1.5926 | Train accuracy 0.52344 | Time 90.53 s
    Epoch complete! Mean loss: 1.5990
    [TEST] Mean loss 1.6114 | Accuracy 0.4276
    [TRAIN] Epoch 8 [0/391]| Mean loss 1.6061 | Train accuracy 0.50000 | Time 96.10 s
    [TRAIN] Epoch 8 [250/391]| Mean loss 1.5787 | Train accuracy 0.41406 | Time 102.73 s
    Epoch complete! Mean loss: 1.5760
    [TEST] Mean loss 1.6087 | Accuracy 0.4185
    [TRAIN] Epoch 9 [0/391]| Mean loss 1.5539 | Train accuracy 0.46094 | Time 108.58 s
    [TRAIN] Epoch 9 [250/391]| Mean loss 1.5544 | Train accuracy 0.49219 | Time 115.35 s
    Epoch complete! Mean loss: 1.5600
    [TEST] Mean loss 1.5848 | Accuracy 0.4305


`CIFAR-10` is much more challenging than `MNIST`, you should be getting a test accuracy of around 40% at the conclusion of training, which is much lower than the accuracy on `MNIST` dataset (/docs/notes/sml/you may want to try playing around with the optimizer later on to see if you can get better results). While this is significantly better than random chance, this is still a bit of a silly approach because we ignore all spatial structure of the input images. We are also _a priori_ treating all pixels in the image identically, irrespective of their separation/proximity to other pixels. Lets now consider using convolutional networks for the same task.

## Convolutional Networks

Convolutional networks exploit the "translation invariance" property seen in natural images. A representation useful at a certain spatial location is typically useful across the whole image. The convolution operation applies the same linear transformation in different local regions across the image, allowing the model to learn local features depending only on small regions of the image. For a more thorough discussion one may refer to Bishop, Section 5.5.6. We will implement a small convolutional network described below:

1. *Convolutional Layer #1* | 8 5×5 filters with a stride of 1, ReLU activation function.
2. *Max Pooling #1*         | Kernel size 2 with a stride of 2.
3. *Convolutional Layer #2* | 16 5×5 filters with a stride of 1, ReLU activation function.
4. *Max Pooling #2*         | Kernel size 2 with a stride of 2.
5. *Fully Connected Layer #1* | 400 input units (/docs/notes/sml/flattened convolutional output), 256 output units.
5. *Fully Connected Layer #2* | 256 input units, 10 output units - yields logits for classification.

In the interests of training time, we keep the number of parameters low by restricting the size of the output channels. Increasing this would most likely increase the classification performance of our network at the expense of additional compute. If you have time you may want to experiment with the architecture. Some questions you may want to consider - is a larger number of parameters always better? How should we adjust the learning rate if we increase the number of parameters?


```python
OUT_C1 = 8
OUT_C2 = 16
DENSE_UNITS = 256

class BasicConvNet(/docs/notes/sml/nn.Module):
    def __init__(/docs/notes/sml/self, out_c1, out_c2, dense_units, n_classes=10):
        super(/docs/notes/sml/BasicConvNet, self).__init__()
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/u/docs/notes/sml/p/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/(/docs/notes/sml/B/docs/notes/sml/a/docs/notes/sml/s/docs/notes/sml/i/docs/notes/sml/c/docs/notes/sml/C/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/v/docs/notes/sml/N/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/f/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/_/docs/notes/sml/_/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/_/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/        self.conv1 = nn.Conv2d(/docs/notes/sml/in_channels=3, out_channels=out_c1, kernel_size=5)
        self.pool = nn.MaxPool2d(/docs/notes/sml/kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(/docs/notes/sml/in_channels=out_c1, out_channels=out_c2, kernel_size=5)
        self.fc1 = nn.Linear(/docs/notes/sml/16 * 5 * 5, dense_units)
        self.logits = nn.Linear(/docs/notes/sml/dense_units, n_classes)

    def forward(/docs/notes/sml/self, x):
        x = self.pool(/docs/notes/sml/F.relu(self.conv1(x)))
        x = self.pool(/docs/notes/sml/F.relu(self.conv2(x)))
        x = x.view(/docs/notes/sml/-1, 16 * 5 * 5)
        x = F.relu(/docs/notes/sml/self.fc1(x))
        out = self.logits(/docs/notes/sml/x)
        return out


conv2D_model = BasicConvNet(/docs/notes/sml/OUT_C1, OUT_C2, DENSE_UNITS)
```

The architecture for the default setting is shown below (/docs/notes/sml/generated using http://alexlenail.me/NN-SVG/LeNet.html).

<img src="https://i.imgur.com/vNd3Lkg.png" alt="CNN arch" width="1000"/>


/docs/notes/sml/*/docs/notes/sml/*/docs/notes/sml/Q/docs/notes/sml/u/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/:/docs/notes/sml/*/docs/notes/sml/*/docs/notes/sml/ /docs/notes/sml/C/docs/notes/sml/a/docs/notes/sml/l/docs/notes/sml/c/docs/notes/sml/u/docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/n/docs/notes/sml/u/docs/notes/sml/m/docs/notes/sml/b/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/o/docs/notes/sml/f/docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/m/docs/notes/sml/u/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/y/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/e/docs/notes/sml/p/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/ /docs/notes/sml/m/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/ /docs/notes/sml/a/docs/notes/sml/n/docs/notes/sml/d/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/a/docs/notes/sml/b/docs/notes/sml/o/docs/notes/sml/v/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/v/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/ /docs/notes/sml/T/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/d/docs/notes/sml/i/docs/notes/sml/a/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/m/docs/notes/sml/ /docs/notes/sml/a/docs/notes/sml/b/docs/notes/sml/o/docs/notes/sml/v/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/y/docs/notes/sml/ /docs/notes/sml/b/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/a/docs/notes/sml/ /docs/notes/sml/u/docs/notes/sml/s/docs/notes/sml/e/docs/notes/sml/f/docs/notes/sml/u/docs/notes/sml/l/docs/notes/sml/ /docs/notes/sml/g/docs/notes/sml/u/docs/notes/sml/i/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/./docs/notes/sml/ /docs/notes/sml/Y/docs/notes/sml/o/docs/notes/sml/u/docs/notes/sml/ /docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/y/docs/notes/sml/ /docs/notes/sml/a/docs/notes/sml/l/docs/notes/sml/s/docs/notes/sml/o/docs/notes/sml/ /docs/notes/sml/w/docs/notes/sml/a/docs/notes/sml/n/docs/notes/sml/t/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/o/docs/notes/sml/k/docs/notes/sml/ /docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/`/docs/notes/sml/m/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/./docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/s/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/`/docs/notes/sml/ /docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/ /docs/notes/sml/f/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/e/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/ /docs/notes/sml/`/docs/notes/sml/m/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/`/docs/notes/sml/./docs/notes/sml/
/docs/notes/sml/

**Answer:** (/docs/notes/sml/Note max-pooling is a parameter-free operation.)

#### Multilayer Perceptron
* Dense I: $32 \times 32 \times 3 \times 256$ (/docs/notes/sml/Flatten $(3,32,32)$ tensor into vector, 256 hidden dimention).
* Bias term: $256$
* Dense II: $256 \times 100$ (/docs/notes/sml/256 input units and 100 output units)
* Bias term: $100$
* Dense III: $100 \times 10$ (/docs/notes/sml/10 output classes)
* Bias term: $10$
* **Total**: $813398$

#### Convolution
* Convolution I: $3 \times 8 \times 5 \times 5 +8$ (/docs/notes/sml/3 channels in, 8 channels out, $5 \times 5$ convolutional kernel). 
* Convolution II: $8 \times 16 \times 5 \times 5 + 16$ (/docs/notes/sml/8 channels in, 16 channels out, $5 \times 5$ convolutional kernel).
* Dense I: $16 \times 5 \times 5 \times 256 + 256$ (/docs/notes/sml/Flatten image and feed to fully connected layer w/ 256 units).
* Dense II: $256 \times 10 + 10$ (/docs/notes/sml/10 output classes)
* **Total**: $109050$

We see that the convolutional parameters are relatively lightweight and most of the parameters are attributed to the fully connected layers.

As before, we have to tell the chosen optimizer which parameters to learn, we'll use the same optimizer for the sake of a fair comparison.


```python
optimizer = torch.optim.SGD(/docs/notes/sml/conv2D_model.parameters(), lr=1e-2, momentum=0.9)

conv_loss, conv_acc = train(/docs/notes/sml/conv2D_model, train_loader, test_loader, optimizer)
```

    [TRAIN] Epoch 0 [0/391]| Mean loss 2.3015 | Train accuracy 0.15625 | Time 0.09 s
    [TRAIN] Epoch 0 [250/391]| Mean loss 2.2532 | Train accuracy 0.25781 | Time 9.92 s
    Epoch complete! Mean loss: 2.1371
    [TEST] Mean loss 1.8449 | Accuracy 0.3376
    [TRAIN] Epoch 1 [0/391]| Mean loss 1.8223 | Train accuracy 0.35156 | Time 16.66 s
    [TRAIN] Epoch 1 [250/391]| Mean loss 1.7255 | Train accuracy 0.43750 | Time 25.46 s
    Epoch complete! Mean loss: 1.6751
    [TEST] Mean loss 1.5233 | Accuracy 0.4462
    [TRAIN] Epoch 2 [0/391]| Mean loss 1.6047 | Train accuracy 0.45312 | Time 32.36 s
    [TRAIN] Epoch 2 [250/391]| Mean loss 1.4672 | Train accuracy 0.49219 | Time 41.06 s
    Epoch complete! Mean loss: 1.4484
    [TEST] Mean loss 1.3978 | Accuracy 0.4881
    [TRAIN] Epoch 3 [0/391]| Mean loss 1.4137 | Train accuracy 0.45312 | Time 47.90 s
    [TRAIN] Epoch 3 [250/391]| Mean loss 1.3583 | Train accuracy 0.49219 | Time 56.26 s
    Epoch complete! Mean loss: 1.3402
    [TEST] Mean loss 1.2881 | Accuracy 0.5367
    [TRAIN] Epoch 4 [0/391]| Mean loss 1.3351 | Train accuracy 0.53906 | Time 63.04 s
    [TRAIN] Epoch 4 [250/391]| Mean loss 1.2605 | Train accuracy 0.54688 | Time 72.22 s
    Epoch complete! Mean loss: 1.2453
    [TEST] Mean loss 1.2360 | Accuracy 0.5581
    [TRAIN] Epoch 5 [0/391]| Mean loss 1.2861 | Train accuracy 0.53125 | Time 79.04 s
    [TRAIN] Epoch 5 [250/391]| Mean loss 1.1799 | Train accuracy 0.66406 | Time 87.30 s
    Epoch complete! Mean loss: 1.1764
    [TEST] Mean loss 1.2058 | Accuracy 0.5743
    [TRAIN] Epoch 6 [0/391]| Mean loss 1.1569 | Train accuracy 0.61719 | Time 93.91 s
    [TRAIN] Epoch 6 [250/391]| Mean loss 1.1104 | Train accuracy 0.53906 | Time 102.28 s
    Epoch complete! Mean loss: 1.1146
    [TEST] Mean loss 1.1933 | Accuracy 0.5850
    [TRAIN] Epoch 7 [0/391]| Mean loss 1.2411 | Train accuracy 0.50000 | Time 108.61 s
    [TRAIN] Epoch 7 [250/391]| Mean loss 1.0572 | Train accuracy 0.55469 | Time 116.76 s
    Epoch complete! Mean loss: 1.0554
    [TEST] Mean loss 1.1715 | Accuracy 0.5913
    [TRAIN] Epoch 8 [0/391]| Mean loss 0.9186 | Train accuracy 0.67188 | Time 123.05 s
    [TRAIN] Epoch 8 [250/391]| Mean loss 0.9970 | Train accuracy 0.69531 | Time 131.07 s
    Epoch complete! Mean loss: 0.9999
    [TEST] Mean loss 1.1048 | Accuracy 0.6100
    [TRAIN] Epoch 9 [0/391]| Mean loss 0.9965 | Train accuracy 0.64062 | Time 137.29 s
    [TRAIN] Epoch 9 [250/391]| Mean loss 0.9464 | Train accuracy 0.68750 | Time 145.36 s
    Epoch complete! Mean loss: 0.9557
    [TEST] Mean loss 1.1318 | Accuracy 0.6044


You should be seeing a ~17 % increase in test performance just by swapping out the multilayer perceptron architecture for our modest 2D convolutional net! Here the exact convolutional architecture is of secondary importance to our modular approach to model design. Plase ask your tutor if anything is unclear on this front. We'll plot the loss curves below and move on. The downward slope of the loss curve at termination suggests we have not reached full convergence yet.


```python
from scipy.signal import savgol_filter  # Smooth spiky curves
smooth_mlp_loss, smooth_conv_loss = savgol_filter(/docs/notes/sml/mlp_loss, 21, 3), savgol_filter(conv_loss, 21, 3)
smooth_mlp_loss, smooth_conv_loss = savgol_filter(mlp_loss, 21, 3), savgol_filter(/docs/notes/sml/conv_loss, 21, 3)
smooth_mlp_acc, smooth_conv_acc = savgol_filter(/docs/notes/sml/mlp_acc, 21, 3), savgol_filter(conv_acc, 21, 3)
smooth_mlp_acc, smooth_conv_acc = savgol_filter(mlp_acc, 21, 3), savgol_filter(/docs/notes/sml/conv_acc, 21, 3)
plt.rcParams['figure.dpi'] = 128
```


```python
plt.plot(/docs/notes/sml/smooth_mlp_loss, label='Multilayer Perceptron')
plt.plot(/docs/notes/sml/smooth_conv_loss, label='Conv Net')
plt.xlabel(/docs/notes/sml/'Iterations')
plt.ylabel(/docs/notes/sml/'Cross-entropy Loss (Train)')
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```




    <matplotlib.legend.Legend at 0x148243550>




    
![png](/docs/notes/sml/worksheet08_solutions_files/worksheet08_solutions_17_1.png)
    



```python
plt.plot(/docs/notes/sml/smooth_mlp_acc, label='Multilayer Perceptron')
plt.plot(/docs/notes/sml/smooth_conv_acc, label='Conv Net')
plt.xlabel(/docs/notes/sml/'Iterations')
plt.ylabel(/docs/notes/sml/'Accuracy (Train)')
/docs/notes/sml/p/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/./docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/```




    <matplotlib.legend.Legend at 0x147e85588>




    
![png](/docs/notes/sml/worksheet08_solutions_files/worksheet08_solutions_18_1.png)
    


# Convolutional Autoencoders

## Autoencoders
Recall from lectures that an autoencoder is used to learn an 'efficient' or lower dimensional coding of the input $\mathbf{x} \in \mathbb{R}^n$ to some latent code $\mathbf{z} \in \mathbb{R}^d$. The intuitive idea is that we wish to recover the high-dimensional, potentially sparse data signal represented by $\mathbf{x}$ from some low-dimensional projection $\mathbf{z}$. Natural images are a good example of data which is potentially very high-dimensional (for an $m \times n$ color image, $\mathbf{x} \in \mathbb{R}^{3mn}$ naively, but we would only expect valid inputs to lie in some relatively small subspace of the original space, which can be spanned using fewer 'meaningful' dimensions. By enforcing a lower-dimensional projection, we would like our representation to discard redundant dimensions while retaining dimensions which correspond to intrinsic aspects of the input space. Autoencoding-based models have many applications such as image/speech synthesis, super-resolution and compressed-sensing. 

The autoencoder is trained in an unsupervised manner. It is composed of two components:
* The _encoder_ $f$ from the original space to the latent space, $f(/docs/notes/sml/\mathbf{x}) = \mathbf{z}$
* The _decoder_ $g$ from the latent space to the original space, $g(/docs/notes/sml/\mathbf{z}) = \mathbf{\hat{x}}$

The autoencoder parameters are learnt such that $g \circ f$ is close to the identity when averaged over the training set. As the latent space is typically much lower dimensional than the original space, the encoder needs to learn a compact representation of the original data that contains sufficient information for the decoder to reconstruct.


The simplest autoencoder model occurs when both the encoder and decoder are linear functions. It is well known (/docs/notes/sml/Bourlard and Kamp 1988) that for a linear autoencoder with encoder and decoder functions represented by matrices:
* $Y \in \mathbb{R}^{d \times n}$
* $Y' \in \mathbb{R}^{n \times d}$

respectively, then the quadratic error objective 

$$ \min_{Y, Y'} \sum_k \Vert \mathbf{x}_k - YY'\mathbf{x}_k \Vert^2 $$
is minimized by the PCA decomposition (/docs/notes/sml/as you'll see in week 9).

For more general mappings we minimize an empirical estimate of the expected quadratic loss:

$$ \min_{f,g} \sum_k \Vert /docs/notes/sml/\mathbf{x}_k - g \circ f(/docs/notes/sml/\mathbf{x}_k) \Vert^2 $$

We will use a fully convolutional network for the encoder and a decoder composed of the reciprocal transposed convolution layers (/docs/notes/sml/essentially the inverse of the convolution operator - needed to upsample the compressed image).

<img src="https://i.imgur.com/Q69VB3H.png" alt="AE" width="1000"/>

Your task is to define the convolutional encoder we'll be using through `torch.nn.Module`. We'll define the decoder for you, which you can use as a template to build the encoder. Note that it is conventional for the decoder/encoder to mirror one another in terms of the nonlinearities, kernel sizes and strides used at each stage. The basic structure of our architecture is very simple, and you should implement the encoder structure.

* Encoder 
    1. *Convolutional Layer #1* | 8 4×4 filters with a stride of 2, padding 1, ReLU activation function.
    2. *Convolutional Layer #2* | 16 4×4 filters with a stride of 2, padding 1, ReLU activation function.
    3. *Convolutional Layer #3* | 32 4×4 filters with a stride of 2, padding 1 - the output of this layer is the latent code.


* Decoder (/docs/notes/sml/mirror encoder structure)
    4. *Transposed Convolution #1* | 32 4×4 filters with a stride of 2, padding 1, ReLU activation function.
    5. *Transposed Convolution #2* | 16 4×4 filters with a stride of 2, padding 1, ReLU activation function.
    6. *Transposed Convolution #3* | 8 4×4 filters with a stride of 2, padding 1, sigmoid activation function.

We apply a sigmoid activation function at the last layer to keep the output within the range $[0,1]$. 

You may find the `torch.nn.Module` reference to be useful here: https://pytorch.org/docs/stable/nn.html. 


```python
OUT_C1 = 8
OUT_C2 = 16
OUT_C3 = 32
OUT_C4 = 32
EMBEDDING_DIMENSION = 32

class ConvEncoder(/docs/notes/sml/nn.Module):
    def __init__(/docs/notes/sml/self, out_c1, out_c2, out_c3, out_c4, embedding_dim):
        super(/docs/notes/sml/ConvEncoder, self).__init__()
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/u/docs/notes/sml/p/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/(/docs/notes/sml/C/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/v/docs/notes/sml/E/docs/notes/sml/n/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/f/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/_/docs/notes/sml/_/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/_/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/        
        self.conv1 = nn.Conv2d(/docs/notes/sml/in_channels=3, out_channels=out_c1, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(/docs/notes/sml/in_channels=out_c1, out_channels=out_c2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(/docs/notes/sml/in_channels=out_c2, out_channels=embedding_dim, kernel_size=4, stride=2, padding=1)
        
    def forward(/docs/notes/sml/self, x):
        x = F.relu(/docs/notes/sml/self.conv1(x))
        x = F.relu(/docs/notes/sml/self.conv2(x))
        out = self.conv3(/docs/notes/sml/x)

        return out
    
class ConvDecoder(/docs/notes/sml/nn.Module):
    def __init__(/docs/notes/sml/self, out_c1, out_c2, out_c3, out_c4, embedding_dim):
        super(/docs/notes/sml/ConvDecoder, self).__init__()
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/u/docs/notes/sml/p/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/(/docs/notes/sml/C/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/v/docs/notes/sml/D/docs/notes/sml/e/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/f/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/_/docs/notes/sml/_/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/_/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/        
        self.deconv1 = nn.ConvTranspose2d(/docs/notes/sml/in_channels=embedding_dim, out_channels=out_c2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(/docs/notes/sml/in_channels=out_c2, out_channels=out_c1, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(/docs/notes/sml/in_channels=out_c1, out_channels=3, kernel_size=4, stride=2, padding=1)
        
    def forward(/docs/notes/sml/self, x):
        x = F.relu(/docs/notes/sml/self.deconv1(x))
        x = F.relu(/docs/notes/sml/self.deconv2(x))
        out = torch.sigmoid(/docs/notes/sml/self.deconv3(x))

        return out
```

Note that we could have defined a single `Autoencoder(/docs/notes/sml/nn.Module)` subclass with `encode(...)` and `decode(...)` defined as distinct methods as well. Since we're considering an unsupervised setup the training loop is slightly different. Training is conducted using the above quadratic loss.
Note that we could have defined a single `Autoencoder(nn.Module)` subclass with `encode(/docs/notes/sml/...)` and `decode(/docs/notes/sml/...)` defined as distinct methods as well. Since we're considering an unsupervised setup the training loop is slightly different. Training is conducted using the above quadratic loss.
Note that we could have defined a single `Autoencoder(nn.Module)` subclass with `encode(/docs/notes/sml/...)` and `decode(/docs/notes/sml/...)` defined as distinct methods as well. Since we're considering an unsupervised setup the training loop is slightly different. Training is conducted using the above quadratic loss.


```python
def train_autoencoder(/docs/notes/sml/encoder, decoder, train_loader, test_loader, optimizer, n_epochs=10, cuda=False):
    """
    Generic training loop for supervised multiclass learning
    """
    LOG_INTERVAL = 250
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/r/docs/notes/sml/u/docs/notes/sml/n/docs/notes/sml/n/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/g/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/./docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/    # criterion = torch.nn.BCELoss(/docs/notes/sml/reduction="sum")
    criterion = torch.nn.MSELoss(/docs/notes/sml/reduction="sum")

    for epoch in range(/docs/notes/sml/n_epochs):  # Loop over training dataset `/docs/notes/sml/n_epochs` times

        epoch_loss = 0.

        for i, data in enumerate(/docs/notes/sml/train_loader):  # Loop over elements in training set
            
            x, _ = data
            batch_size = x.shape[0]
            
            if cuda is True:  # Send to GPU
                x = x.to(/docs/notes/sml/device)

            code = encoder(/docs/notes/sml/x)
            reconstructed_x = de/docs/notes/sml/coder(/docs/notes/sml/code)

            loss = criterion(/docs/notes/sml/input=reconstructed_x, target=x) / batch_size

/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/b/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/k/docs/notes/sml/w/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/#/docs/notes/sml/ /docs/notes/sml/B/docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/k/docs/notes/sml/w/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/d/docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/(/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/m/docs/notes/sml/p/docs/notes/sml/u/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/i/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/t/docs/notes/sml/s/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/            loss.backward()               # Backward pass (/docs/notes/sml/compute parameter gradients)
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/o/docs/notes/sml/p/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/z/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/./docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/p/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/#/docs/notes/sml/ /docs/notes/sml/U/docs/notes/sml/p/docs/notes/sml/d/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/w/docs/notes/sml/e/docs/notes/sml/i/docs/notes/sml/g/docs/notes/sml/h/docs/notes/sml/t/docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/u/docs/notes/sml/s/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/g/docs/notes/sml/ /docs/notes/sml/S/docs/notes/sml/G/docs/notes/sml/D/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/o/docs/notes/sml/p/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/z/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/./docs/notes/sml/z/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/o/docs/notes/sml/_/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/#/docs/notes/sml/ /docs/notes/sml/R/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/ /docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/i/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/t/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/ /docs/notes/sml/z/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/o/docs/notes/sml/ /docs/notes/sml/f/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/x/docs/notes/sml/t/docs/notes/sml/ /docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/
/docs/notes/sml/

            # ============================================================================
            # You can safely ignore the boilerplate code below - just reports metrics over
            # training and test sets

            running_loss.append(/docs/notes/sml/loss.item())

/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/e/docs/notes/sml/p/docs/notes/sml/o/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/+/docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/m/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/
            if i % LOG_INTERVAL == 0:  # Log training stats
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/T/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/./docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/ /docs/notes/sml/-/docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/
/docs/notes/sml/                mean_loss = epoch_loss / (/docs/notes/sml/i+1)
                print('[TRAIN] Epoch {} [{}/{}]| Mean loss {:.4f} | Time {:.2f} s'.format(epoch, 
                    i, len(/docs/notes/sml/train_loader), mean_loss, deltaT))

        print(/docs/notes/sml/'Epoch complete! Mean training loss: {:.4f}'.format(epoch_loss/len(train_loader)))

        test_loss = 0.
        
        for i, data in enumerate(/docs/notes/sml/test_loader):
            x, _ = data

            if cuda is True:  # Send to GPU
                x = x.to(/docs/notes/sml/device)
            
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/w/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/./docs/notes/sml/n/docs/notes/sml/o/docs/notes/sml/_/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/:/docs/notes/sml/
/docs/notes/sml/                code = encoder(/docs/notes/sml/x)
                reconstructed_x = de/docs/notes/sml/coder(/docs/notes/sml/code)

                test_loss += criterion(/docs/notes/sml/input=reconstructed_x, target=x).item() / batch_size
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/l/docs/notes/sml/o/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/+/docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/c/docs/notes/sml/r/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/p/docs/notes/sml/u/docs/notes/sml/t/docs/notes/sml/=/docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/u/docs/notes/sml/c/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/d/docs/notes/sml/_/docs/notes/sml/x/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/=/docs/notes/sml/x/docs/notes/sml/)/docs/notes/sml/./docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/m/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/ /docs/notes/sml///docs/notes/sml/ /docs/notes/sml/b/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/_/docs/notes/sml/s/docs/notes/sml/i/docs/notes/sml/z/docs/notes/sml/e/docs/notes/sml/
/docs/notes/sml/
        print(/docs/notes/sml/'[TEST] Mean loss {:.4f}'.format(test_loss/len(test_loader)))
```

/docs/notes/sml/F/docs/notes/sml/i/docs/notes/sml/r/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/ /docs/notes/sml/w/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/a/docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/z/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/a/docs/notes/sml/n/docs/notes/sml/d/docs/notes/sml/ /docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/m/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/u/docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/./docs/notes/sml/ /docs/notes/sml/W/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/s/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/o/docs/notes/sml/f/docs/notes/sml/ /docs/notes/sml/b/docs/notes/sml/o/docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/ /docs/notes/sml/m/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/u/docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/ /docs/notes/sml/o/docs/notes/sml/u/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/o/docs/notes/sml/p/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/z/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/./docs/notes/sml/ /docs/notes/sml/S/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/c/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/`/docs/notes/sml/m/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/./docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/s/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/`/docs/notes/sml/ /docs/notes/sml/r/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/u/docs/notes/sml/r/docs/notes/sml/n/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/a/docs/notes/sml/ /docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/w/docs/notes/sml/e/docs/notes/sml/'/docs/notes/sml/l/docs/notes/sml/l/docs/notes/sml/ /docs/notes/sml/`/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/`/docs/notes/sml/ /docs/notes/sml/b/docs/notes/sml/o/docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/ /docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/./docs/notes/sml/ /docs/notes/sml/W/docs/notes/sml/e/docs/notes/sml/'/docs/notes/sml/l/docs/notes/sml/l/docs/notes/sml/ /docs/notes/sml/u/docs/notes/sml/s/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/a/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/S/docs/notes/sml/G/docs/notes/sml/D/docs/notes/sml/ /docs/notes/sml/w/docs/notes/sml///docs/notes/sml/ /docs/notes/sml/m/docs/notes/sml/o/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/t/docs/notes/sml/u/docs/notes/sml/m/docs/notes/sml/ /docs/notes/sml/o/docs/notes/sml/p/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/z/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/e/docs/notes/sml/x/docs/notes/sml/c/docs/notes/sml/e/docs/notes/sml/p/docs/notes/sml/t/docs/notes/sml/ /docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/d/docs/notes/sml/ /docs/notes/sml/N/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/o/docs/notes/sml/v/docs/notes/sml/ /docs/notes/sml/m/docs/notes/sml/o/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/t/docs/notes/sml/u/docs/notes/sml/m/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/ /docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/a/docs/notes/sml/b/docs/notes/sml/i/docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/z/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/g/docs/notes/sml/ /docs/notes/sml/a/docs/notes/sml/n/docs/notes/sml/d/docs/notes/sml/ /docs/notes/sml/a/docs/notes/sml/c/docs/notes/sml/c/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/ /docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/v/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/c/docs/notes/sml/e/docs/notes/sml/./docs/notes/sml/
/docs/notes/sml/

```python
use_cuda = True  # Set this to true if using a GPU
device = torch.device(/docs/notes/sml/"cuda" if torch.cuda.is_available() else "cpu")
encoder = ConvEncoder(/docs/notes/sml/OUT_C1, OUT_C2, OUT_C3, OUT_C4, EMBEDDING_DIMENSION)
decoder = ConvDecoder(/docs/notes/sml/OUT_C1, OUT_C2, OUT_C3, OUT_C4, EMBEDDING_DIMENSION)

if use_cuda is True:
    encoder.to(/docs/notes/sml/device)
    decoder.to(/docs/notes/sml/device)
```


```python
from itertools import chain
# optimizer = torch.optim.SGD(/docs/notes/sml/chain(encoder.parameters(), decoder.parameters()), lr=1e-3, momentum=0.9, nesterov=True)
/docs/notes/sml/#/docs/notes/sml/ /docs/notes/sml/o/docs/notes/sml/p/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/z/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/./docs/notes/sml/o/docs/notes/sml/p/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/./docs/notes/sml/S/docs/notes/sml/G/docs/notes/sml/D/docs/notes/sml/(/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/./docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/s/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/./docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/s/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/r/docs/notes/sml/=/docs/notes/sml/1/docs/notes/sml/e/docs/notes/sml/-/docs/notes/sml/3/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/m/docs/notes/sml/o/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/t/docs/notes/sml/u/docs/notes/sml/m/docs/notes/sml/=/docs/notes/sml/0/docs/notes/sml/./docs/notes/sml/9/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/o/docs/notes/sml/v/docs/notes/sml/=/docs/notes/sml/T/docs/notes/sml/r/docs/notes/sml/u/docs/notes/sml/e/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/optimizer = torch.optim.Adam(/docs/notes/sml/chain(encoder.parameters(), decoder.parameters()), lr=1e-3)
/docs/notes/sml/o/docs/notes/sml/p/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/i/docs/notes/sml/z/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/./docs/notes/sml/o/docs/notes/sml/p/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/./docs/notes/sml/A/docs/notes/sml/d/docs/notes/sml/a/docs/notes/sml/m/docs/notes/sml/(/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/a/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/(/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/./docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/s/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/c/docs/notes/sml/o/docs/notes/sml/d/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/./docs/notes/sml/p/docs/notes/sml/a/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/m/docs/notes/sml/e/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/s/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/)/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/r/docs/notes/sml/=/docs/notes/sml/1/docs/notes/sml/e/docs/notes/sml/-/docs/notes/sml/3/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/
train_autoencoder(/docs/notes/sml/encoder, decoder, train_loader, test_loader, optimizer, cuda=use_cuda, n_epochs=48)
```

    [TRAIN] Epoch 0 [0/391]| Mean loss 208.3372 | Time 0.09 s
    [TRAIN] Epoch 0 [250/391]| Mean loss 94.8893 | Time 12.66 s
    Epoch complete! Mean training loss: 77.8003
    [TEST] Mean loss 70.1171
    [TRAIN] Epoch 1 [0/391]| Mean loss 44.0567 | Time 22.29 s
    [TRAIN] Epoch 1 [250/391]| Mean loss 40.4253 | Time 34.09 s
    Epoch complete! Mean training loss: 37.0801
    [TEST] Mean loss 45.2664
    [TRAIN] Epoch 2 [0/391]| Mean loss 27.4138 | Time 43.86 s
    [TRAIN] Epoch 2 [250/391]| Mean loss 26.7709 | Time 55.10 s
    Epoch complete! Mean training loss: 25.9491
    [TEST] Mean loss 37.1040
    [TRAIN] Epoch 3 [0/391]| Mean loss 24.0313 | Time 63.60 s
    [TRAIN] Epoch 3 [250/391]| Mean loss 22.7646 | Time 75.18 s
    Epoch complete! Mean training loss: 22.2536
    [TEST] Mean loss 32.7586
    [TRAIN] Epoch 4 [0/391]| Mean loss 19.8443 | Time 85.67 s
    [TRAIN] Epoch 4 [250/391]| Mean loss 20.3198 | Time 98.60 s
    Epoch complete! Mean training loss: 19.9769
    [TEST] Mean loss 29.9883
    [TRAIN] Epoch 5 [0/391]| Mean loss 19.2427 | Time 108.80 s
    [TRAIN] Epoch 5 [250/391]| Mean loss 18.8016 | Time 122.53 s
    Epoch complete! Mean training loss: 18.5441
    [TEST] Mean loss 28.2635
    [TRAIN] Epoch 6 [0/391]| Mean loss 18.3570 | Time 130.77 s
    [TRAIN] Epoch 6 [250/391]| Mean loss 17.8450 | Time 142.64 s
    Epoch complete! Mean training loss: 17.6936
    [TEST] Mean loss 27.1193
    [TRAIN] Epoch 7 [0/391]| Mean loss 18.0666 | Time 150.74 s
    [TRAIN] Epoch 7 [250/391]| Mean loss 17.0982 | Time 161.62 s
    Epoch complete! Mean training loss: 16.9661
    [TEST] Mean loss 26.1699
    [TRAIN] Epoch 8 [0/391]| Mean loss 17.3159 | Time 169.72 s
    [TRAIN] Epoch 8 [250/391]| Mean loss 16.3378 | Time 181.67 s
    Epoch complete! Mean training loss: 16.2300
    [TEST] Mean loss 24.8617
    [TRAIN] Epoch 9 [0/391]| Mean loss 15.2612 | Time 190.08 s
    [TRAIN] Epoch 9 [250/391]| Mean loss 15.4977 | Time 203.64 s
    Epoch complete! Mean training loss: 15.3763
    [TEST] Mean loss 23.3931
    [TRAIN] Epoch 10 [0/391]| Mean loss 14.5567 | Time 213.25 s
    [TRAIN] Epoch 10 [250/391]| Mean loss 14.7226 | Time 225.36 s
    Epoch complete! Mean training loss: 14.6158
    [TEST] Mean loss 22.5085
    [TRAIN] Epoch 11 [0/391]| Mean loss 13.8157 | Time 233.84 s
    [TRAIN] Epoch 11 [250/391]| Mean loss 14.2130 | Time 244.73 s
    Epoch complete! Mean training loss: 14.1055
    [TEST] Mean loss 22.2759
    [TRAIN] Epoch 12 [0/391]| Mean loss 14.7946 | Time 252.87 s
    [TRAIN] Epoch 12 [250/391]| Mean loss 13.8031 | Time 263.83 s
    Epoch complete! Mean training loss: 13.6709
    [TEST] Mean loss 21.0270
    [TRAIN] Epoch 13 [0/391]| Mean loss 15.2164 | Time 273.73 s
    [TRAIN] Epoch 13 [250/391]| Mean loss 13.2989 | Time 284.90 s
    Epoch complete! Mean training loss: 13.1886
    [TEST] Mean loss 20.1871
    [TRAIN] Epoch 14 [0/391]| Mean loss 12.5808 | Time 293.68 s
    [TRAIN] Epoch 14 [250/391]| Mean loss 12.6492 | Time 306.65 s
    Epoch complete! Mean training loss: 12.5898
    [TEST] Mean loss 19.2112
    [TRAIN] Epoch 15 [0/391]| Mean loss 11.9092 | Time 315.21 s
    [TRAIN] Epoch 15 [250/391]| Mean loss 12.0940 | Time 326.46 s
    Epoch complete! Mean training loss: 11.9206
    [TEST] Mean loss 18.1343
    [TRAIN] Epoch 16 [0/391]| Mean loss 10.6183 | Time 335.91 s
    [TRAIN] Epoch 16 [250/391]| Mean loss 11.4574 | Time 348.02 s
    Epoch complete! Mean training loss: 11.3807
    [TEST] Mean loss 18.1311
    [TRAIN] Epoch 17 [0/391]| Mean loss 11.0397 | Time 357.17 s
    [TRAIN] Epoch 17 [250/391]| Mean loss 11.1012 | Time 369.37 s
    Epoch complete! Mean training loss: 11.0298
    [TEST] Mean loss 17.7110
    [TRAIN] Epoch 18 [0/391]| Mean loss 10.7478 | Time 379.34 s
    [TRAIN] Epoch 18 [250/391]| Mean loss 10.8051 | Time 392.83 s
    Epoch complete! Mean training loss: 10.7741
    [TEST] Mean loss 16.6655
    [TRAIN] Epoch 19 [0/391]| Mean loss 10.8944 | Time 402.90 s
    [TRAIN] Epoch 19 [250/391]| Mean loss 10.5352 | Time 417.13 s
    Epoch complete! Mean training loss: 10.5058
    [TEST] Mean loss 16.4519
    [TRAIN] Epoch 20 [0/391]| Mean loss 9.7003 | Time 427.47 s
    [TRAIN] Epoch 20 [250/391]| Mean loss 10.3680 | Time 441.69 s
    Epoch complete! Mean training loss: 10.3173
    [TEST] Mean loss 16.3692
    [TRAIN] Epoch 21 [0/391]| Mean loss 10.6555 | Time 452.25 s
    [TRAIN] Epoch 21 [250/391]| Mean loss 10.1660 | Time 464.30 s
    Epoch complete! Mean training loss: 10.1611
    [TEST] Mean loss 16.0705
    [TRAIN] Epoch 22 [0/391]| Mean loss 10.5038 | Time 473.62 s
    [TRAIN] Epoch 22 [250/391]| Mean loss 10.0791 | Time 486.34 s
    Epoch complete! Mean training loss: 10.0416
    [TEST] Mean loss 15.5117
    [TRAIN] Epoch 23 [0/391]| Mean loss 10.0509 | Time 495.49 s
    [TRAIN] Epoch 23 [250/391]| Mean loss 9.9140 | Time 507.45 s
    Epoch complete! Mean training loss: 9.8896
    [TEST] Mean loss 15.6631
    [TRAIN] Epoch 24 [0/391]| Mean loss 10.3269 | Time 516.24 s
    [TRAIN] Epoch 24 [250/391]| Mean loss 9.7931 | Time 528.66 s
    Epoch complete! Mean training loss: 9.7858
    [TEST] Mean loss 15.0958
    [TRAIN] Epoch 25 [0/391]| Mean loss 9.3127 | Time 537.67 s
    [TRAIN] Epoch 25 [250/391]| Mean loss 9.6828 | Time 549.23 s
    Epoch complete! Mean training loss: 9.6455
    [TEST] Mean loss 14.9579
    [TRAIN] Epoch 26 [0/391]| Mean loss 9.2963 | Time 558.06 s
    [TRAIN] Epoch 26 [250/391]| Mean loss 9.5321 | Time 569.42 s
    Epoch complete! Mean training loss: 9.5516
    [TEST] Mean loss 14.9632
    [TRAIN] Epoch 27 [0/391]| Mean loss 9.4864 | Time 577.93 s
    [TRAIN] Epoch 27 [250/391]| Mean loss 9.4016 | Time 592.16 s
    Epoch complete! Mean training loss: 9.4133
    [TEST] Mean loss 14.5871
    [TRAIN] Epoch 28 [0/391]| Mean loss 9.3130 | Time 602.63 s
    [TRAIN] Epoch 28 [250/391]| Mean loss 9.2483 | Time 614.75 s
    Epoch complete! Mean training loss: 9.2133
    [TEST] Mean loss 14.2763
    [TRAIN] Epoch 29 [0/391]| Mean loss 9.0446 | Time 623.37 s
    [TRAIN] Epoch 29 [250/391]| Mean loss 9.0621 | Time 635.94 s
    Epoch complete! Mean training loss: 9.0406
    [TEST] Mean loss 13.9915
    [TRAIN] Epoch 30 [0/391]| Mean loss 8.6813 | Time 646.40 s
    [TRAIN] Epoch 30 [250/391]| Mean loss 8.9101 | Time 660.48 s
    Epoch complete! Mean training loss: 8.8462
    [TEST] Mean loss 13.6720
    [TRAIN] Epoch 31 [0/391]| Mean loss 8.6117 | Time 670.09 s
    [TRAIN] Epoch 31 [250/391]| Mean loss 8.7267 | Time 682.99 s
    Epoch complete! Mean training loss: 8.7118
    [TEST] Mean loss 13.4889
    [TRAIN] Epoch 32 [0/391]| Mean loss 8.6301 | Time 692.35 s
    [TRAIN] Epoch 32 [250/391]| Mean loss 8.5925 | Time 705.23 s
    Epoch complete! Mean training loss: 8.5281
    [TEST] Mean loss 13.1926
    [TRAIN] Epoch 33 [0/391]| Mean loss 8.4958 | Time 716.01 s
    [TRAIN] Epoch 33 [250/391]| Mean loss 8.4249 | Time 731.18 s
    Epoch complete! Mean training loss: 8.3947
    [TEST] Mean loss 13.0265
    [TRAIN] Epoch 34 [0/391]| Mean loss 8.0744 | Time 742.60 s
    [TRAIN] Epoch 34 [250/391]| Mean loss 8.3237 | Time 756.70 s
    Epoch complete! Mean training loss: 8.2801
    [TEST] Mean loss 13.1524
    [TRAIN] Epoch 35 [0/391]| Mean loss 8.3317 | Time 767.56 s
    [TRAIN] Epoch 35 [250/391]| Mean loss 8.1546 | Time 780.35 s
    Epoch complete! Mean training loss: 8.1490
    [TEST] Mean loss 12.6697
    [TRAIN] Epoch 36 [0/391]| Mean loss 7.4025 | Time 790.75 s
    [TRAIN] Epoch 36 [250/391]| Mean loss 8.0515 | Time 804.47 s
    Epoch complete! Mean training loss: 8.0544
    [TEST] Mean loss 12.5254
    [TRAIN] Epoch 37 [0/391]| Mean loss 7.8947 | Time 815.28 s
    [TRAIN] Epoch 37 [250/391]| Mean loss 7.9385 | Time 828.18 s
    Epoch complete! Mean training loss: 7.9803
    [TEST] Mean loss 12.3448
    [TRAIN] Epoch 38 [0/391]| Mean loss 7.3315 | Time 837.72 s
    [TRAIN] Epoch 38 [250/391]| Mean loss 7.8698 | Time 855.99 s
    Epoch complete! Mean training loss: 7.8548
    [TEST] Mean loss 12.1854
    [TRAIN] Epoch 39 [0/391]| Mean loss 7.8473 | Time 865.39 s
    [TRAIN] Epoch 39 [250/391]| Mean loss 7.7839 | Time 877.51 s
    Epoch complete! Mean training loss: 7.7564
    [TEST] Mean loss 12.1228
    [TRAIN] Epoch 40 [0/391]| Mean loss 7.5617 | Time 885.87 s
    [TRAIN] Epoch 40 [250/391]| Mean loss 7.7080 | Time 898.66 s
    Epoch complete! Mean training loss: 7.6857
    [TEST] Mean loss 11.9632
    [TRAIN] Epoch 41 [0/391]| Mean loss 7.7279 | Time 913.00 s
    [TRAIN] Epoch 41 [250/391]| Mean loss 7.6127 | Time 935.12 s
    Epoch complete! Mean training loss: 7.5839
    [TEST] Mean loss 12.5520
    [TRAIN] Epoch 42 [0/391]| Mean loss 7.5665 | Time 944.33 s
    [TRAIN] Epoch 42 [250/391]| Mean loss 7.5212 | Time 955.53 s
    Epoch complete! Mean training loss: 7.4981
    [TEST] Mean loss 11.5966
    [TRAIN] Epoch 43 [0/391]| Mean loss 7.5877 | Time 964.02 s
    [TRAIN] Epoch 43 [250/391]| Mean loss 7.4320 | Time 975.21 s
    Epoch complete! Mean training loss: 7.4018
    [TEST] Mean loss 11.6008
    [TRAIN] Epoch 44 [0/391]| Mean loss 7.3054 | Time 984.13 s
    [TRAIN] Epoch 44 [250/391]| Mean loss 7.2841 | Time 995.91 s
    Epoch complete! Mean training loss: 7.2731
    [TEST] Mean loss 11.5502
    [TRAIN] Epoch 45 [0/391]| Mean loss 7.3638 | Time 1005.49 s
    [TRAIN] Epoch 45 [250/391]| Mean loss 7.1471 | Time 1018.46 s
    Epoch complete! Mean training loss: 7.1578
    [TEST] Mean loss 11.2064
    [TRAIN] Epoch 46 [0/391]| Mean loss 7.2483 | Time 1028.04 s
    [TRAIN] Epoch 46 [250/391]| Mean loss 7.0616 | Time 1040.56 s
    Epoch complete! Mean training loss: 7.0292
    [TEST] Mean loss 10.8063
    [TRAIN] Epoch 47 [0/391]| Mean loss 7.4787 | Time 1050.26 s
    [TRAIN] Epoch 47 [250/391]| Mean loss 6.9371 | Time 1062.53 s
    Epoch complete! Mean training loss: 6.9239
    [TEST] Mean loss 10.8036


Below we'll grab 8 random images from the test set and plot the images along with their reconstructions.


```python
plt.rcParams['figure.dpi'] = 256
test_im_generator = iter(/docs/notes/sml/test_loader)
```


```python
/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/a/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/,/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/b/docs/notes/sml/e/docs/notes/sml/l/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/i/docs/notes/sml/m/docs/notes/sml/_/docs/notes/sml/g/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/./docs/notes/sml/n/docs/notes/sml/e/docs/notes/sml/x/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/images = images[:8]
labels = labels[:8]

/docs/notes/sml/w/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/./docs/notes/sml/n/docs/notes/sml/o/docs/notes/sml/_/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/:/docs/notes/sml/
/docs/notes/sml/    code = encoder(/docs/notes/sml/images.to(device))
    reconstructed_images = de/docs/notes/sml/coder(/docs/notes/sml/code)

imshow(/docs/notes/sml/torchvision.utils.make_grid(images), title='Original')
imshow(/docs/notes/sml/torchvision.utils.make_grid(reconstructed_images), title='Reconstructed')
print(/docs/notes/sml/[label_class_map[c.item()] for c in labels])
```


    
![png](/docs/notes/sml/worksheet08_solutions_files/worksheet08_solutions_29_0.png)
    



    
![png](/docs/notes/sml/worksheet08_solutions_files/worksheet08_solutions_29_1.png)
    


    ['cat', 'ship', 'ship', 'plane', 'frog', 'frog', 'car', 'frog']


Not bad! Looks like we were able to recover most of the salient features from our lower dimensional projection. The reconstructed images are somewhat recognizable as their original classes. You may want to try adjusting the architecture / adding layers, or changing the latent code dimension to see the effect on the reconstruction loss.

## Extension: Interpolations
A popular application of autoencoder models is the generation of a semantically meaningful latent space. To test if our autoencoder has learnt a well-structured latent space, one can decode a convex combination of the latent codes for datapoints, $\mathbf{x}$ and $\mathbf{x}'$. If interpolating between two points in latent space produces a meaningful image in data space that is not a pixelwise admixture of $\mathbf{x}$ and $\mathbf{x'}$, this suggests that nearby points in latent space are semanatically similar - i.e. the different classes are clustered together in latent space. This property of an 'interpretable' latent space may be a useful representation for downstream tasks.

Concretely, we achieve this by interpolating the samples $x$ and $x'$ along a line in latent space.

$$ I(/docs/notes/sml/x, x'; \alpha) = g\left((1-\alpha) f\left(x\right) + \alpha f(x')\right)$$
$$ I(x, x'; \alpha) = g\left(/docs/notes/sml/(1-\alpha) f\left(x\right) + \alpha f(x')\right)$$
$$ I(x, x'; \alpha) = g\left((1-\alpha) f\left(/docs/notes/sml/x\right) + \alpha f(x')\right)$$
$$ I(x, /docs/notes/sml/x'; \alpha) = g\left((1-\alpha) f\left(x\right) + \alpha f(/docs/notes/sml/x')\right)$$

<img src="https://i.imgur.com/QgCkUTa.png" alt="AE_int" width="750"/>


```python
# Choose images to interpolate between, add batch dimension
x_i, x_f = torch.unsqueeze(/docs/notes/sml/images[0], dim=0), torch.unsqueeze(images[3], dim=0)
x_i, x_f = torch.unsqueeze(images[0], dim=0), torch.unsqueeze(/docs/notes/sml/images[3], dim=0)
encoder.to(/docs/notes/sml/'cpu');
decoder.to(/docs/notes/sml/'cpu');
```


```python
/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/p/docs/notes/sml/o/docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml//docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/n/docs/notes/sml/t/docs/notes/sml/_/docs/notes/sml/i/docs/notes/sml/n/docs/notes/sml/t/docs/notes/sml/e/docs/notes/sml/r/docs/notes/sml/p/docs/notes/sml/o/docs/notes/sml/l/docs/notes/sml/a/docs/notes/sml/t/docs/notes/sml/i/docs/notes/sml/o/docs/notes/sml/n/docs/notes/sml/s/docs/notes/sml/ /docs/notes/sml/=/docs/notes/sml/ /docs/notes/sml/l/docs/notes/sml/i/docs/notes/sml/s/docs/notes/sml/t/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/
/docs/notes/sml/line = torch.linspace(/docs/notes/sml/0,1,steps=8)

# Get latent representations of images
/docs/notes/sml/w/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/./docs/notes/sml/n/docs/notes/sml/o/docs/notes/sml/_/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/:/docs/notes/sml/
/docs/notes/sml/    code_i = encoder(/docs/notes/sml/x_i)
    code_f = encoder(/docs/notes/sml/x_f)

for alpha in line:
    
    # Interpolate along line in data space
    x_interpolated = alpha * x_f + (/docs/notes/sml/1-alpha) * x_i
    
/docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/ /docs/notes/sml/w/docs/notes/sml/i/docs/notes/sml/t/docs/notes/sml/h/docs/notes/sml/ /docs/notes/sml/t/docs/notes/sml/o/docs/notes/sml/r/docs/notes/sml/c/docs/notes/sml/h/docs/notes/sml/./docs/notes/sml/n/docs/notes/sml/o/docs/notes/sml/_/docs/notes/sml/g/docs/notes/sml/r/docs/notes/sml/a/docs/notes/sml/d/docs/notes/sml/(/docs/notes/sml/)/docs/notes/sml/:/docs/notes/sml/
/docs/notes/sml/        # Interpolate along line in latent space
        reconstruction = decoder(/docs/notes/sml/alpha * code_f + (1-alpha) * code_i)
    
    interpolations.append(/docs/notes/sml/x_interpolated)
    latent_interpolations.append(/docs/notes/sml/reconstruction)

/docs/notes/sml/interpolations = torch.cat(/docs/notes/sml/interpolations)
/docs/notes/sml/latent_interpolations = torch.cat(/docs/notes/sml/latent_interpolations)

imshow(/docs/notes/sml/torchvision.utils.make_grid(interpolations), title='Interpolation in original space')
imshow(/docs/notes/sml/torchvision.utils.make_grid(latent_interpolations), title='Interpolation in latent space')
```


    
![png](/docs/notes/sml/worksheet08_solutions_files/worksheet08_solutions_33_0.png)
    



    
![png](/docs/notes/sml/worksheet08_solutions_files/worksheet08_solutions_33_1.png)
    


You should find that interpolations in the latent space tends to be more structured than interpolation in the original pixel space - in the sense that it is more than just pixelwise interpolation. Each reconstructed image should resemble one class instead of an admixture of two classes. Conventional autoencoders aren't strictly the best task for this because they don't enforce any structure on the latent space. The [variational autoencoder](/docs/notes/sml/http://paulrubenstein.co.uk/variational-autoencoders-are-not-autoencoders/) explicitly enforces some sort of user-specified structure on the latent space and typically leads to superior latent space representations. It is one of the most prominent marriages of deep learning with traditional Bayesian methods. But this is out of the scope of the subject and we're out of time - that's all for this week!
