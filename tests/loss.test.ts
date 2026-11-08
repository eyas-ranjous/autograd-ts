import { describe, expect, it } from 'vitest';
import { mse, val } from '../src';

describe('mse', () => {
  it('rejects empty inputs', () => {
    expect(() => mse([], [])).toThrow('MSE requires at least one prediction and target');
  });

  it.each([
    { predictions: [], targets: [1] },
    { predictions: [val(1)], targets: [] },
    { predictions: [val(1), val(2)], targets: [1] },
  ])('rejects mismatched lengths: $predictions.length and $targets.length', ({ predictions, targets }) => {
    expect(() => mse(predictions, targets)).toThrow(
      `Expected ${predictions.length} targets, received ${targets.length}`,
    );
  });

  it('computes the mean loss and prediction gradients', () => {
    const predictions = [val(2), val(5)];
    const loss = mse(predictions, [1, 2]);

    loss.backward();

    expect(loss.data).toBe(5);
    expect(predictions.map((prediction) => prediction.grad)).toEqual([1, 3]);
  });

  it('computes loss and gradient for one prediction', () => {
    const prediction = val(2);
    const loss = mse([prediction], [4]);

    loss.backward();

    expect(loss.data).toBe(4);
    expect(prediction.grad).toBe(-4);
  });
});
