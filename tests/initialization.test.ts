import { describe, expect, it } from 'vitest';
import { MLP, mse, sgd } from '../src';

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

describe('reproducible initialization', () => {
  it('reproduces parameters and predictions with the same seed', () => {
    const first = new MLP(2, [3, 1], { random: seededRandom(42) });
    const second = new MLP(2, [3, 1], { random: seededRandom(42) });
    expect(first.parameters().map(p => p.data)).toEqual(second.parameters().map(p => p.data));
    expect(first.forward([1, 2]).map(p => p.data)).toEqual(second.forward([1, 2]).map(p => p.data));
  });

  it('produces different parameters for different seeds', () => {
    const first = new MLP(2, [3, 1], { random: seededRandom(42) });
    const second = new MLP(2, [3, 1], { random: seededRandom(43) });
    expect(first.parameters().map(p => p.data)).not.toEqual(second.parameters().map(p => p.data));
  });

  it('uses one random stream for all weights and keeps biases zero', () => {
    let calls = 0;
    const model = new MLP(2, [2, 1], { random: () => ++calls / 10 });
    expect(calls).toBe(6);
    const expected = [-0.8, -0.6, 0, -0.4, -0.2, 0, 0, 0.2, 0];
    model.parameters().forEach((p, i) => expect(p.data).toBeCloseTo(expected[i]!));
  });

  it('reproduces training with the same seed and sample order', () => {
    const train = () => {
      const model = new MLP(2, [3, 1], { random: seededRandom(42) });
      for (let step = 0; step < 20; step++) {
        const loss = mse(model.forward([1, 2]), [1]);
        model.zeroGrad();
        loss.backward();
        sgd(model.parameters(), 0.01);
      }
      return model.parameters().map(p => p.data);
    };
    expect(train()).toEqual(train());
  });
});
