import { Value, val } from '../core/Value';
import { Activation } from './Neuron';
import { Layer } from './Layer';

/**
 * Multi-layer perceptron composed of one or more fully connected layers.
 */
export class MLP {
  private layers: Layer[];

  /**
   * Creates a multi-layer perceptron.
   * @param inputSize Number of input features.
   * @param layerSizes Output size of each layer.
   * @param options Network activation configuration.
   */
  constructor(
    inputSize: number,
    layerSizes: number[],
    options?: {
      hiddenActivation?: Activation;
      outputActivation?: Activation;
    },
  ) {
    if (layerSizes.length === 0) {
      throw new Error('MLP must have at least one layer');
    }

    const hiddenActivation = options?.hiddenActivation ?? 'tanh';
    const outputActivation = options?.outputActivation ?? 'none';

    const sizes = [inputSize, ...layerSizes];

    this.layers = [];

    for (let i = 0; i < layerSizes.length; i++) {
      const isOutputLayer = i === layerSizes.length - 1;

      this.layers.push(
        new Layer(
          sizes[i]!,
          sizes[i + 1]!,
          isOutputLayer ? outputActivation : hiddenActivation,
        ),
      );
    }
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
