import { describe, expect, it } from 'vitest';
import { MLP, mse, sgd } from '../src';

describe('nn', () => {
  it('runs a forward pass through an MLP', () => {
    const model = new MLP(2, [3, 1]);

    const output = model.forward([1, 2]);

    expect(output).toHaveLength(1);
    expect(Number.isFinite(output[0]!.data)).toBe(true);
  });

  it('returns trainable parameters', () => {
    const model = new MLP(2, [3, 1]);

    const params = model.parameters();

    // Layer 1: 3 neurons * (2 weights + 1 bias) = 9
    // Layer 2: 1 neuron * (3 weights + 1 bias) = 4
    // Total = 13
    expect(params).toHaveLength(13);
  });

  it('computes MSE loss', () => {
    const model = new MLP(2, [1]);

    const prediction = model.forward([1, 2])[0]!;
    const loss = mse([prediction], [1]);

    expect(Number.isFinite(loss.data)).toBe(true);
  });

  it('computes gradients for model parameters', () => {
    const model = new MLP(2, [2, 1]);

    const prediction = model.forward([1, 0])[0]!;
    const loss = mse([prediction], [1]);

    model.zeroGrad();
    loss.backward();

    const hasGradient = model.parameters().some((p) => p.grad !== 0);

    expect(hasGradient).toBe(true);
  });

  it('updates parameters with SGD', () => {
    const model = new MLP(2, [2, 1]);

    const prediction = model.forward([1, 0])[0]!;
    const loss = mse([prediction], [1]);

    model.zeroGrad();
    loss.backward();

    const paramsBefore = model.parameters().map((p) => p.data);

    sgd(model.parameters(), 0.01);

    const paramsAfter = model.parameters().map((p) => p.data);

    expect(paramsAfter).not.toEqual(paramsBefore);
  });

  it('can reduce loss over a few training steps', () => {
    const model = new MLP(2, [4, 1], {
      hiddenActivation: 'tanh',
      outputActivation: 'none',
    });

    const x = [1, 1];
    const y = [1];

    const firstPrediction = model.forward(x)[0]!;
    const firstLoss = mse([firstPrediction], y).data;

    for (let i = 0; i < 50; i++) {
      const prediction = model.forward(x)[0]!;
      const loss = mse([prediction], y);

      model.zeroGrad();
      loss.backward();
      sgd(model.parameters(), 0.05);
    }

    const finalPrediction = model.forward(x)[0]!;
    const finalLoss = mse([finalPrediction], y).data;

    expect(finalLoss).toBeLessThan(firstLoss);
  });
});
