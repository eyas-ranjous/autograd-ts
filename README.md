# autograd-ts
A tiny TypeScript library implementing **reverse-mode automatic differentiation** (autograd) and neural network training from scratch.

The goal of the project is exploring the fundamental mechanism that allows neural networks to learn.

At its core, a neural network is a system of interconnected computations.

Producing an output is straightforward. However, learning requires determining how each value influenced that output and how those values should be adjusted when the prediction is wrong.

This library explores the mechanism that makes that possible.

## Installation

```bash
npm install autograd-ts
```

## Import
```ts
import { Value, val, MLP, mse, sgd } from 'autograd-ts';
```

---

# Understanding Learning

A neural network can make a prediction by evaluating a series of mathematical operations.

For example, consider the simplified model:

```text
z = x * y + x
```

Given the inputs:

```text
x = 2
y = 3
```

the model produces:

```text
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

Each edge represents a dependency. This structure is what makes it possible to reason about how a change in one node affects any other.

---

# Layer 2: Automatic Differentiation

Every operation in this library doesn't just compute a result, it also records how to differentiate through itself. Each `Value` node stores a `_backward` closure that knows how to propagate gradients to its inputs using the chain rule.

```ts
// When you write:
const z = x.mul(y).add(x);

// mul records: x.grad += y.data * out.grad
//              y.grad += x.data * out.grad
// add records: a.grad += out.grad
//              x.grad += out.grad
```

Calling `z.backward()` topologically sorts the computation graph, sets `z.grad = 1`, then traverses in reverse — each node firing its `_backward` to accumulate gradients up the chain.

For `z = x * y + x` with `x = 2, y = 3`:

```text
x.grad = 4   (∂z/∂x)
y.grad = 2   (∂z/∂y)
```

These gradients measure how sensitive the output is to each input. `x.grad = 4` means a small increase in `x` increases `z` by approximately 4.

This process is known as reverse-mode automatic differentiation, the same technique used by modern machine learning frameworks.

---

# Layer 3: Neural Networks

A single neuron computes:

```text
w1*x1 + w2*x2 + b
```

which is simply a larger computation graph:

```text
x1 --*
     |
w1 --|

x2 --*
     + ---> activation ---> output
w2 --|

b ---|
```

The weights (`w1`, `w2`) and bias (`b`) are `Value` nodes. The autograd engine computes their gradients automatically — nothing special is added to support neural networks. The same graph and differentiation system powers everything.

---

# Layer 4: Learning

Once gradients are available, training becomes possible. Each step follows the same pattern:

```ts
for (let epoch = 0; epoch < epochs; epoch++) {
  const prediction = model.forward(inputs);  // build the computation graph
  const loss = mse(prediction, targets);     // measure how wrong the prediction is

  model.zeroGrad();                          // clear gradients from the previous step
  loss.backward();                           // propagate gradients back through the graph

  sgd(model.parameters(), learningRate);     // nudge each weight to reduce loss
}
```

> **Important:** `zeroGrad()` must be called before each `backward()`. Gradients accumulate with `+=` — skipping this causes gradients from previous steps to corrupt the update.

Over many epochs, the network adjusts its weights to reduce prediction error. This feedback loop is the mechanism that allows neural networks to learn.

---

# API

## `Value`

The core scalar node. Every computation builds on `Value` objects.

| Operation      | Method        | Notes                          |
| -------------- | ------------- | ------------------------------ |
| Addition       | `a.add(b)`    | `b` can be `Value` or `number` |
| Subtraction    | `a.sub(b)`    | `b` can be `Value` or `number` |
| Multiplication | `a.mul(b)`    | `b` can be `Value` or `number` |
| Division       | `a.div(b)`    | `b` can be `Value` or `number` |
| Power          | `a.pow(n)`    | `n` is a plain `number`        |
| Tanh           | `a.tanh()`    | Activation function            |
| ReLU           | `a.relu()`    | Activation function            |
| Backprop       | `a.backward()` | Computes all gradients        |

`val(n)` is a shorthand for `new Value(n)`.

## `MLP`

A Multi-Layer Perceptron. a fully connected neural network composed of stacked layers of neurons.

```ts
new MLP(inputSize, layerSizes, options?)
```

| Parameter                  | Description                                      |
| -------------------------- | ------------------------------------------------ |
| `inputSize`                | Number of input features                         |
| `layerSizes`               | Output size of each layer — e.g. `[4, 1]` creates a hidden layer of 4 neurons and an output layer of 1 |
| `options.hiddenActivation` | `'tanh'` (default), `'relu'`, or `'none'`        |
| `options.outputActivation` | `'none'` (default), `'tanh'`, or `'relu'`        |

```ts
model.forward(inputs)              // returns Value[] — runs the forward pass
model.parameters()                 // returns all trainable Value nodes
model.zeroGrad()                   // resets all parameter gradients to zero
```

## `mse(predictions, targets)`

Computes mean squared error. Returns a `Value` that is part of the computation graph and supports `backward()`.

## `sgd(parameters, learningRate)`

Updates each parameter in-place: `param.data -= learningRate * param.grad`.

---

# Examples

```
examples/
├── basic.ts   — computation graph, forward pass, and gradients
└── xor.ts     — training a network to learn XOR
```

**Basic**
```
z: 8
dz/dx: 4
dz/dy: 2
```

**XOR training**
```
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

[0, 0] => 0.0213
[0, 1] => 0.9834
[1, 0] => 0.9772
[1, 1] => 0.0189
```

---

# Project Scope

A minimal foundation for training neural networks from scratch.

**Core autograd engine:** `Value` — add, sub, mul, div, pow, tanh, relu, backward

**Neural network components:** Neuron, Layer, MLP

**Training utilities:** MSE loss, SGD optimizer
