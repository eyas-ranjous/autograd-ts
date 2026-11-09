import { Value, val } from './Value';
import { Activation } from './Neuron';
import { Layer } from './Layer';
import { MLPSnapshot, parameterNames, validateSnapshot } from './serialization';

/**
 * Multi-layer perceptron composed of one or more fully connected layers.
 */
export class MLP {
  private layers: Layer[];
  private architecture: Omit<MLPSnapshot, 'version' | 'parameters'>;

  /**
   * Creates a multi-layer perceptron.
   * @param inputSize Positive integer number of input features.
   * @param layerSizes Positive integer output size of each layer.
   * @param options Network activation configuration.
   */
  constructor(
    inputSize: number,
    layerSizes: number[],
    options?: {
      hiddenActivation?: Activation;
      outputActivation?: Activation;
      random?: () => number;
    },
  ) {
    if (layerSizes.length === 0) {
      throw new Error('MLP must have at least one layer');
    }

    const hiddenActivation = options?.hiddenActivation ?? 'tanh';
    const outputActivation = options?.outputActivation ?? 'none';

    this.architecture = {
      inputSize,
      layerSizes: [...layerSizes],
      hiddenActivation,
      outputActivation,
    };
    const sizes = [inputSize, ...layerSizes];

    this.layers = [];

    for (let i = 0; i < layerSizes.length; i++) {
      const isOutputLayer = i === layerSizes.length - 1;

      this.layers.push(
        new Layer(
          sizes[i]!,
          sizes[i + 1]!,
          isOutputLayer ? outputActivation : hiddenActivation,
          options?.random,
        ),
      );
    }
  }

  /**
   * Copies architecture and finite parameter values into a versioned JSON snapshot.
   * @returns Model state without gradients or computation graphs.
   */
  public toJSON(): MLPSnapshot {
    const parameters: Record<string, number> = {};
    const values = this.parameters();
    let index = 0;
    for (const name of parameterNames(this.architecture.inputSize, this.architecture.layerSizes)) {
      const value = values[index++]!.data;
      if (!Number.isFinite(value)) {
        throw new Error(`Cannot serialize non-finite MLP parameter: ${name}`);
      }
      parameters[name] = value;
    }
    return {
      version: 1,
      ...this.architecture,
      layerSizes: [...this.architecture.layerSizes],
      parameters,
    };
  }

  /**
   * Restores a model from a version 1 snapshot, validating before allocation.
   * @param snapshot Parsed JSON model state.
   * @returns Independent model with restored parameters and zero gradients.
   */
  public static fromJSON(snapshot: unknown): MLP {
    validateSnapshot(snapshot);
    const model = new MLP(snapshot.inputSize, snapshot.layerSizes, {
      hiddenActivation: snapshot.hiddenActivation,
      outputActivation: snapshot.outputActivation,
      random: () => 0.5,
    });
    const parameters = model.parameters();
    let index = 0;
    for (const name of parameterNames(snapshot.inputSize, snapshot.layerSizes)) {
      parameters[index++]!.data = snapshot.parameters[name]!;
    }
    return model;
  }

  /**
   * Runs a forward pass through the network.
   * @param inputs Input feature values.
   * @returns Network predictions.
   */
  public forward(inputs: number[]): Value[] {
    let output = inputs.map((input) => val(input));

    for (const layer of this.layers) {
      output = layer.forward(output);
    }

    return output;
  }

  /**
   * Runs a forward pass through the network.
   * @param inputs Input feature values.
   * @returns Network predictions.
   */
  public parameters(): Value[] {
    return this.layers.flatMap((layer) => layer.parameters());
  }

  /**
   * Runs a forward pass through the network.
   * @param inputs Input feature values.
   * @returns Network predictions.
   */
  public zeroGrad(): void {
    for (const parameter of this.parameters()) {
      parameter.zeroGrad();
    }
  }
}
