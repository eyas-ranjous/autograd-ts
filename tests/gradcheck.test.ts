import { describe, expect, it } from 'vitest';
import { MLP, Value, mse, val } from '../src';

const epsilon = 1e-5;

function checkGradients(compute: (...inputs: Value[]) => Value, values: number[]): void {
  const inputs = values.map(val);
  compute(...inputs).backward();

  for (let i = 0; i < inputs.length; i++) {
    const above = values.map(val);
    const below = values.map(val);
    above[i]!.data += epsilon;
    below[i]!.data -= epsilon;

    const numerical = (compute(...above).data - compute(...below).data) / (2 * epsilon);
    expect(inputs[i]!.grad, `input ${i}`).toBeCloseTo(numerical, 6);
  }
}

describe('numerical gradient checks', () => {
  const binaryOperations = [
    { name: 'add', compute: (x: Value, y: Value) => x.add(y) },
    { name: 'sub', compute: (x: Value, y: Value) => x.sub(y) },
    { name: 'mul', compute: (x: Value, y: Value) => x.mul(y) },
    { name: 'div', compute: (x: Value, y: Value) => x.div(y) },
  ];

  it.each(binaryOperations)('checks $name for both operands', ({ compute }) => {
    checkGradients(compute, [1.3, -0.7]);
    checkGradients(compute, [-2.1, 1.8]);
  });

  it.each([
    { name: 'add', compute: (x: Value) => x.add(1.7) },
    { name: 'sub', compute: (x: Value) => x.sub(1.7) },
    { name: 'mul', compute: (x: Value) => x.mul(1.7) },
    { name: 'div', compute: (x: Value) => x.div(1.7) },
    { name: 'neg', compute: (x: Value) => x.neg() },
    { name: 'tanh', compute: (x: Value) => x.tanh() },
    { name: 'relu', compute: (x: Value) => x.relu() },
  ])('checks $name with one differentiable input', ({ compute }) => {
    checkGradients(compute, [-0.8]);
    checkGradients(compute, [1.2]);
  });

  it.each([-2, 0, 0.5, 1, 3])('checks pow(%s) at a positive base', (exponent) => {
    checkGradients((x) => x.pow(exponent), [1.4]);
  });

  it('checks powers at a negative base', () => {
    checkGradients((x) => x.pow(3), [-1.4]);
  });

  it('checks shared operands and branching intermediate nodes', () => {
    checkGradients((x, y) => {
      const shared = x.mul(y).tanh();
      return shared.mul(shared).add(shared.mul(x)).sub(y).div(2);
    }, [0.6, -0.9]);
  });

  it('checks every MLP parameter through a batched MSE loss', () => {
    const model = new MLP(2, [2, 1]);
    const parameters = model.parameters();
    parameters.forEach((parameter, index) => {
      parameter.data = (index - 4) * 0.1;
    });

    const loss = () => mse([
      model.forward([0.3, -0.7])[0]!,
      model.forward([-0.8, 0.5])[0]!,
    ], [0.4, -0.2]);

    model.zeroGrad();
    loss().backward();

    for (let i = 0; i < parameters.length; i++) {
      const parameter = parameters[i]!;
      const original = parameter.data;
      parameter.data = original + epsilon;
      const above = loss().data;
      parameter.data = original - epsilon;
      const below = loss().data;
      parameter.data = original;

      const numerical = (above - below) / (2 * epsilon);
      expect(parameter.grad, `parameter ${i}`).toBeCloseTo(numerical, 6);
    }
  });
});
