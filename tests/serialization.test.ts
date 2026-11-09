import { describe, expect, it, vi } from 'vitest';
import { Activation, MLP, MLPSnapshot, mse, sgd } from '../src';

function snapshot(): MLPSnapshot {
  return new MLP(2, [2, 1], { random: () => 0.75 }).toJSON();
}

describe('MLP serialization', () => {
  it('restores trained predictions and can continue training', () => {
    const model = new MLP(2, [2, 1], { random: () => 0.75 });
    const initialLoss = mse(model.forward([1, 0]), [0]).data;
    for (let step = 0; step < 20; step++) {
      model.zeroGrad();
      mse(model.forward([1, 0]), [0]).backward();
      sgd(model.parameters(), 0.05);
    }
    expect(mse(model.forward([1, 0]), [0]).data).toBeLessThan(initialLoss);
    expect(model.parameters().some((parameter) => parameter.grad !== 0)).toBe(true);

    const restored = MLP.fromJSON(JSON.parse(JSON.stringify(model)));
    expect(restored.toJSON()).toEqual(model.toJSON());
    for (const inputs of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
      expect(restored.forward(inputs).map((value) => value.data))
        .toEqual(model.forward(inputs).map((value) => value.data));
    }
    expect(restored.parameters().every((parameter) => parameter.grad === 0)).toBe(true);
    const before = restored.toJSON();
    mse(restored.forward([0, 1]), [1]).backward();
    sgd(restored.parameters(), 0.05);
    expect(restored.toJSON()).not.toEqual(before);
  });

  const activations: Activation[] = ['none', 'tanh', 'relu'];
  for (const hiddenActivation of activations) {
    for (const outputActivation of activations) {
      it(`preserves ${hiddenActivation}/${outputActivation} activations and multiple outputs`, () => {
        const model = new MLP(2, [3, 2], { hiddenActivation, outputActivation, random: () => 0.8 });
        const restored = MLP.fromJSON(JSON.parse(JSON.stringify(model)));
        expect(restored.forward([-2, 0.5]).map((value) => value.data))
          .toEqual(model.forward([-2, 0.5]).map((value) => value.data));
      });
    }
  }

  it('uses stable names for weights and biases in a single layer', () => {
    const model = new MLP(2, [1], { random: () => 0.75 });
    expect(model.toJSON().parameters).toEqual({
      'layers.0.neurons.0.weights.0': 0.5,
      'layers.0.neurons.0.weights.1': 0.5,
      'layers.0.neurons.0.bias': 0,
    });
    expect(MLP.fromJSON(model.toJSON()).forward([1, 2])[0]!.data).toBe(1.5);
  });

  it('does not share architecture or parameter storage', () => {
    const sizes = [2, 1];
    const model = new MLP(2, sizes);
    const saved = model.toJSON();
    const restored = MLP.fromJSON(saved);
    const expected = restored.toJSON();
    sizes[0] = 7;
    saved.layerSizes[0] = 8;
    saved.parameters['layers.0.neurons.0.bias'] = 99;
    model.parameters()[0]!.data = 123;
    expect(restored.toJSON()).toEqual(expected);
    expect(model.toJSON().layerSizes).toEqual([2, 1]);
  });

  it.each([null, [], {}, { version: 2 }])('rejects invalid versions: %j', (value) => {
    expect(() => MLP.fromJSON(value)).toThrow(/version/);
  });

  it.each([0, -1, 1.5, Infinity, NaN, '2', null, Number.MAX_SAFE_INTEGER + 1])(
    'rejects invalid dimensions: %s', (size) => {
      expect(() => MLP.fromJSON({ ...snapshot(), inputSize: size })).toThrow();
      expect(() => MLP.fromJSON({ ...snapshot(), layerSizes: [size] })).toThrow();
    },
  );

  it.each([[], [1, , 1], null, '2'])('rejects invalid layer arrays: %j', (layerSizes) => {
    expect(() => MLP.fromJSON({ ...snapshot(), layerSizes })).toThrow();
  });

  it.each(['sigmoid', null, 1])('rejects invalid activation: %s', (activation) => {
    expect(() => MLP.fromJSON({ ...snapshot(), hiddenActivation: activation })).toThrow(/activation/);
    expect(() => MLP.fromJSON({ ...snapshot(), outputActivation: activation })).toThrow(/activation/);
  });

  it('rejects mismatched architectures before initializing a model', () => {
    const random = vi.spyOn(Math, 'random').mockImplementation(() => { throw new Error('unexpected initialization'); });
    try {
      expect(() => MLP.fromJSON({ ...snapshot(), inputSize: Number.MAX_SAFE_INTEGER }))
        .toThrow(/parameter count/);
      expect(() => MLP.fromJSON({ ...snapshot(), layerSizes: [1000000000] }))
        .toThrow(/parameter count/);
      expect(random).not.toHaveBeenCalled();
      expect(() => MLP.fromJSON(snapshot())).not.toThrow();
    } finally {
      random.mockRestore();
    }
  });

  it('rejects missing, extra, and renamed parameters', () => {
    const saved = snapshot();
    delete saved.parameters['layers.0.neurons.0.bias'];
    expect(() => MLP.fromJSON(saved)).toThrow(/parameter count/);
    saved.parameters.unexpected = 0;
    expect(() => MLP.fromJSON(saved)).toThrow(/finite parameter/);
    saved.parameters['layers.0.neurons.0.bias'] = 0;
    expect(() => MLP.fromJSON(saved)).toThrow(/parameter count/);
  });

  it.each([NaN, Infinity, -Infinity, '0', null])('rejects invalid parameter: %s', (value) => {
    const saved = snapshot();
    const parameters = { ...saved.parameters, 'layers.0.neurons.0.bias': value };
    expect(() => MLP.fromJSON({ ...saved, parameters })).toThrow(/finite parameter/);
  });

  it.each([NaN, Infinity, -Infinity])('refuses to serialize non-finite parameters: %s', (value) => {
    const model = new MLP(1, [1]);
    model.parameters()[0]!.data = value;
    expect(() => model.toJSON()).toThrow(/non-finite/);
  });
});
