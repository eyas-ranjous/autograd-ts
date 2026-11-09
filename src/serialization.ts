import { Activation } from './Neuron';

/** JSON-compatible model architecture and learned parameters. */
export interface MLPSnapshot {
  version: 1;
  inputSize: number;
  layerSizes: number[];
  hiddenActivation: Activation;
  outputActivation: Activation;
  parameters: Record<string, number>;
}

export function* parameterNames(inputSize: number, layerSizes: number[]): Generator<string> {
  for (let layer = 0; layer < layerSizes.length; layer++) {
    for (let neuron = 0; neuron < layerSizes[layer]!; neuron++) {
      const prefix = `layers.${layer}.neurons.${neuron}`;
      for (let weight = 0; weight < inputSize; weight++) {
        yield `${prefix}.weights.${weight}`;
      }
      yield `${prefix}.bias`;
    }
    inputSize = layerSizes[layer]!;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isSize(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0;
}

function isActivation(value: unknown): value is Activation {
  return value === 'tanh' || value === 'relu' || value === 'none';
}

export function validateSnapshot(value: unknown): asserts value is MLPSnapshot {
  if (!isRecord(value) || value.version !== 1) {
    throw new Error('Unsupported MLP snapshot version; expected version 1');
  }
  if (!isSize(value.inputSize) || !Array.isArray(value.layerSizes) || value.layerSizes.length === 0) {
    throw new Error('MLP snapshot must contain positive integer dimensions and at least one layer');
  }
  let inputSize = value.inputSize;
  let parameterCount = 0;
  for (const size of value.layerSizes) {
    if (!isSize(size)) {
      throw new Error('MLP snapshot layer sizes must be positive safe integers');
    }
    parameterCount += size * (inputSize + 1);
    inputSize = size;
  }
  if (!isActivation(value.hiddenActivation) || !isActivation(value.outputActivation)) {
    throw new Error('MLP snapshot activation must be tanh, relu, or none');
  }
  if (!isRecord(value.parameters) || !Number.isSafeInteger(parameterCount)
    || Object.keys(value.parameters).length !== parameterCount) {
    throw new Error('MLP snapshot parameter count does not match its architecture');
  }
  for (const name of parameterNames(value.inputSize, value.layerSizes)) {
    const parameter = value.parameters[name];
    if (!Object.prototype.hasOwnProperty.call(value.parameters, name)
      || typeof parameter !== 'number' || !Number.isFinite(parameter)) {
      throw new Error(`MLP snapshot requires a finite parameter: ${name}`);
    }
  }
}
