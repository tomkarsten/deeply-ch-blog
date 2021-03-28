---
title: Deep Learning Fundamentals - How Backpropagation Works 
tags: ["deep learning"]
date: "2021-03-23"
---
![](backpropagatin-cover.png)


> “I have no special talents. I am only passionately curious.”
― Albert Einstein


A typical *deep learning model* consists of many layers, each containing many adjustable parameters (also called *weights*). From the deep learning perspective the outcomes or *predictions* are nothing more than a sequence of *data transformations*. 

![](backpropagation_intro.png)

During the training phase each data transformation (or to be more precise its weights) are tuned to make more accurate predictions. 

**But how to tune these weights and by how much?** This question remained for a long time unanswered until Rumelhart, Hinton & Williams made a revolutionary breakthrough with the *Backpropagation algorithm*.

Simply put, backpropagation allows efficient computation of prediction contribution for each individual weight. Without knowing who contributes and how much it is nearly impossible to adjust weights and to *learn* how to make better predictions.

At its core, the backpropagation consists of following three steps:
1. take data input and calculate forward through the layers (Note: typically, the last layer is special as it calculates the error, see *how to learn from data*),
2. on the way down in the computational graph calculate rate of change (*gradients*) for each operation, then
3. run from the end of the graph up and calculate individual weight contribution using forward pass results (*backward pass*).

Sounds a bit more complicated than it is. To make the backpropagation algorithm stick, we go a bit further and consider an example from several perspectives by: 
* introducing step-by-step visualization,
* implementing it in pseudocode, then
* implementing concise versions using `PyTorch` and `TensorFlow` deep learning frameworks, and last but not least, 
* providing mathematical symbolic notation for all the steps above.

### Backpropagation visualized


![](backpropagation_comp_graph.png)
Let's a with very simple data transformation $(3x)^2+5$ where the transformation of input $x$ can be visualized as a series of steps in a computational graph. The first step is multiplying input by 3: $3*x$, then applying $(..)^2$ and adding $5$ to the final result.

![](backpropagation_forward_pass1.png)
To see the computational graph in motion we use data input $x=2$ as an example. Passing the input through to the end of the computational graph yields $41$.
![](backpropagation_forward_pass2.png)
Before we move on there is an additional step to. For every data transformation (in our case three in total) we save additional information about the *rate of change*. Rate of change is just another name for *how fast the outcome of an operation is responding to an unit of input*. Let's take an example: if the data transformation is $3x$, where $x$ is the input, then for every unit $x=1$ the change is: $3*1=3$. We keep this as additional information in our graph and move to the next step. The rate of change for $x^2$ is $2*x$ and for the last step $x+5$ it is just $1$. 
![](backpropagation_backward_pass.png)
Now, after being done with the *forward phase*, we run so called *backward pass* where all our previous efforts are starting bearing fruits. We begin backward pass with the last operation's *rate of change* and move up the graph by multiplying all intermediate results from the forward pass.

### Backpropagation using pseudocode

```python
# define individual functions
def f(x):
    return 3*x

def g(x):
    return x**2

def h(x):
    return x+5
```
We start by defining data transformations as simple functions.
```python
# define composite function
def hgf(x):
    x1 = f(x)
    x2 = g(x1)
    x3 = h(x2)
    return x3
```
Next, we stack individual functions into one function.

```python
# define derivatives for each individual function
def grad_f(x):
    return 3

def grad_g(x):
    return 2*x

def grad_h(x):
    return 1
```

Each individual function has a shadow *rate of change* function. 

```python
# define derivatives for the compsite function
def grad_hgf(x):
    x1 = f(x)
    x2 = g(x1)
    return grad_h(x2)*grad_g(x1)*grad_f(x)

grad_hgf(2) 
# 36
```
For the composite function, *the rate of change* is the multiplication of *rate of change* functions.


### Backpropagation using PyTorch

`PyTorch` records all *forward pass* information automatically for any data which has `.requires_grad` set to `True`. This can be done by providing `requires_grad=True` during initialization.

Calling `.backward()` method of the predictor trigger the *backward pass*.
```python
# import libraries
import torch

# data
x = torch.tensor(2.0,requires_grad=True)
x.requires_grad # True

# forward pass
y = hgf(x)

# backward pass
y.backward()
x.grad 
# tensor(36.)
```

### Backpropagation using TensorFlow

```python
# import libraries
import tensorflow as tf

# data
x = tf.Variable(2.0)

# forward pass
with tf.GradientTape() as tape:
  y = hgf(x)

# backward pass
tape.gradient(y, x) 
# <tf.Tensor: shape=(), dtype=float32, numpy=36.0>
```
TensorFlow provides `tf.GradientTape()` to record all required information during the *forward pass*.

To run the *backward pass* use the `gradient(prediction,input)` method of `tf.GradientTape()`.
### Backpropagation using mathematical notation

Mathematically, the workhorse behind backpropagation is a *chain rule*. The name comes from the fact that the end result is actually chain of gradient multiplications:

$$(h(g(f(x)))'=(h \circ g \circ f)(x)'=h'(g(f(x)))*g'(f(x))*f'(x)$$

Four our example consider $h(g(f(x)))=(3x)^2+5$, which can be also decomposed as:

$$f(x) = 3x$$

$$g(x)=x^2$$

$$h(x)=x+5$$

and the respective gradients:

$$f'(x) = 3$$

$$g'(x)=2x$$

$$h'(x)=1$$

Applying the chain rule yields:

$$h(g(f(x)))'=1*6x*3=18x$$ 

Placing $x=2$ yields:

$$h(g(f(2)))'=1*6x*3=18*x=36$$ 

## What's next?
Actually, backpropagation refers only to the method for computing individual weight contributions. 

Further algorithms are needed to perform the actual weight adjustments. Among the most prominent are: *Stochastic Gradient Descent*, *Momentum*, *Adagrad* and *RMSProp*.

### References
[Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams. "Learning representations by back-propagating errors." nature 323.6088 (1986): 533-536.](http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)

[Deep Learning, 2016. Ian Goodfellow, Yoshua Bengio, Aaron Courville. The reference book for deep learning models: 197-214](https://www.deeplearningbook.org/)

[Autograd Mechanics in PyTorch](https://pytorch.org/docs/stable/notes/autograd.html)

[TensorFlow Differentiation](https://www.tensorflow.org/guide/advanced_autodiff)