import { describe, expect, it } from 'vitest';
import { dataset, trainXor } from '../examples/xor';

describe('XOR training example', () => {
  it('learns all four XOR cases and repeats the same training result', () => {
    const model = trainXor();
    let squaredError = 0;

    for (const sample of dataset) {
      const prediction = model.forward(sample.inputs)[0]!.data;
      expect(Number.isFinite(prediction)).toBe(true);
      expect(Math.abs(prediction - sample.target)).toBeLessThan(0.1);
      squaredError += (prediction - sample.target) ** 2;
    }

    expect(squaredError / dataset.length).toBeLessThan(0.01);
    expect(trainXor().parameters().map(parameter => parameter.data))
      .toEqual(model.parameters().map(parameter => parameter.data));
  });
});
