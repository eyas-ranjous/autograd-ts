# autograd-ts

A small reverse-mode automatic differentiation engine written in TypeScript.

This project implements the core mechanism used to train neural networks: **reverse-mode automatic differentiation (autograd)**.

## Example

```ts
import { val } from "autograd-ts"

const x = val(2)
const y = val(3)

const z = x.mul(y).add(x)

z.backward()

console.log(z.data) // 8
console.log(x.grad) // 4
console.log(y.grad) // 2
