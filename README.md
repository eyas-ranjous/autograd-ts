# autograd-ts

A tiny Typescript library that implements **reverse-mode automatic differentiation** (autograd) and neural network training.

The goal of the project is exploring the fundamental mechanism that allows neural networks to learn.

At its core, a neural network is a system of interconnected computations.

Producing an output is straightforward. However, learning requires determining how each value influenced that output and how those values should be adjusted when the prediction is wrong.

This library explores the mechanism that makes that possible.

---

# Understanding Learning

A neural network can make a prediction by evaluating a series of mathematical operations.

For example, consider the simplified model:

```text
z = x * y + x
```

Here, x and y are inputs to the model and z is the resulting prediction.

Given the inputs:

```
x = 2
y = 3
```

the model produces:
```
z = 8
```

Suppose the expected value was:

```text
target = 10
```

Now we need to answer:

```text
1. Which values contributed to the error?
2. How much did they contribute?
3. How should they change?
```

For a simple expression this may seem manageable. For a neural network containing thousands or millions of parameters, it becomes a much more difficult problem.

This leads to the central question behind the project:

> When a neural network makes a prediction, how does it know which weights to adjust?

A model must be able to trace a prediction back through the computations that produced it, measure how each value contributed to the final result, and determine how those values should change when the prediction is wrong.

Modern machine learning systems solve this problem through a sequence of concepts that build on one another:

```text
Computation Graph
        ↓
Automatic Differentiation
        ↓
Backpropagation
        ↓
Gradient Descent
        ↓
Learning
```

---

# Layer 1: Computation Graphs

Everything begins with mathematical expressions.

Consider:

```text
z = x * y + x
```

Expanded:

```text
a = x * y
z = a + x
```

This expression can be represented as a directed acyclic graph (DAG):

```text
x -----------\
              * ---> a ----\
y -----------/             \
                             + ---> z
x -------------------------/
```

Each node represents a computed value.

| Node | Meaning | Value |
| ---- | ------- | ----- |
| x    | Input   | 2     |
| y    | Input   | 3     |
| a    | x * y   | 6     |
| z    | a + x   | 8     |

Each edge represents a dependency relationship.

```text
x → a
y → a
a → z
x → z
```

The graph gives us a structured representation of the computation.

Once that structure exists, we can begin reasoning about how changes in one value affect others.

---

# Layer 2: Automatic Differentiation

As we have the computation graph, we can now ask:

> How much did each node contribute to the final output?

For the expression:

```text
z = x * y + x
```

we want:

```text
∂z/∂x
∂z/∂y
```
(∂ denotes a partial derivative).

The autograd engine computes these derivatives automatically using reverse graph traversal and the chain rule.

So after backpropagation they become:

```text
x.grad = 4
y.grad = 2
```

These gradients measure how sensitive the final output is to each input.

For example:

```
∂z/∂x = 4
```

which means that a small increase of 1 in x would increase z by approximately 4.

This process is known as **reverse-mode automatic differentiation**, the same technique used by modern machine learning frameworks.

---

# Layer 3: Neural Networks

Neural networks are built directly on top of the computation graph.

A single neuron computes:

```text
w1*x1 + w2*x2 + b
```

which is simply a larger graph.

```text
x1 --*
     |
w1 --|

x2 --*
     + ---> tanh ---> output
w2 --|

b -------------|
```

The weights (`w1`, `w2`) and bias (`b`) are themselves graph nodes.

The autograd engine computes gradients for those weights automatically.

Nothing special is added for neural networks, the same graph and differentiation system powers everything.

---

# Layer 4: Learning

Once gradients are available, training becomes possible.

A typical training loop looks like:

```text
Forward Pass
      ↓
Compute Loss
      ↓
Backward Pass
      ↓
Update Weights
      ↓
Repeat
```

Over time the network adjusts its weights to reduce prediction error.

This feedback loop is the mechanism that allows neural networks to learn.

---

# Project Scope

Version 0.1.0 focuses on a minimal learning system.

## Core Autograd Engine

* Value
* add
* sub
* mul
* div
* pow
* tanh
* relu
* exp
* backward

## Neural Network Components

* Neuron
* Layer
* MLP (Multi-Layer Perceptron)

## Training Utilities

* Mean Squared Error (MSE)
* Stochastic Gradient Descent (SGD)

## Examples
```
examples/
├── basic.ts
└── xor.ts
```

**Basic computation graph example**
```
z: 8
dz/dx: 4
dz/dy: 2
```

**XOR training example**
```
XOR Training
epoch=0   loss=1.824531
epoch=100 loss=0.712943
epoch=200 loss=0.403817
epoch=300 loss=0.182691
epoch=400 loss=0.087432
epoch=500 loss=0.041827
epoch=600 loss=0.021554
epoch=700 loss=0.012115
epoch=800 loss=0.007083
epoch=900 loss=0.004291

XOR Predictions
[0, 0] => 0.0213
[0, 1] => 0.9834
[1, 0] => 0.9772
[1, 1] => 0.0189
```
---
