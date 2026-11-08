import { describe, expect, it } from 'vitest';
import { MLP, val } from '../src';
import { Layer } from '../src/Layer';
import { Neuron } from '../src/Neuron';

describe('network dimensions', () => {
  it.each([0, -1, 0.5, 1.5, NaN, Infinity, -Infinity])('rejects invalid size %s', (size) => {
    expect(() => new Neuron(size)).toThrow('inputSize must be a positive integer');
    expect(() => new Layer(size, 1)).toThrow('inputSize must be a positive integer');
    expect(() => new Layer(1, size)).toThrow('outputSize must be a positive integer');
    expect(() => new MLP(size, [1])).toThrow('inputSize must be a positive integer');
    expect(() => new MLP(1, [size])).toThrow('outputSize must be a positive integer');
    expect(() => new MLP(1, [2, size, 1])).toThrow('outputSize must be a positive integer');
  });

  it('rejects a network without layers', () => {
    expect(() => new MLP(1, [])).toThrow('MLP must have at least one layer');
  });

  it('accepts a single input and neuron', () => {
    const neuron = new Neuron(1);
    const layer = new Layer(1, 1);
    const model = new MLP(1, [1]);

    expect(neuron.parameters()).toHaveLength(2);
    expect(Number.isFinite(neuron.forward([val(1)]).data)).toBe(true);
    expect(layer.parameters()).toHaveLength(2);
    expect(layer.forward([val(1)])).toHaveLength(1);
    expect(model.parameters()).toHaveLength(2);
    expect(model.forward([1])).toHaveLength(1);
  });

  it('preserves dimensions across multiple layers', () => {
    const model = new MLP(2, [3, 2]);

    expect(model.parameters()).toHaveLength(17);
    expect(model.forward([1, 2])).toHaveLength(2);
  });
});
