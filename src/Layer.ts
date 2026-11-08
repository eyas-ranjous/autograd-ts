import { Value } from './Value';
import { Activation, Neuron } from './Neuron';

/**
 * Fully connected layer containing multiple neurons.
 */
export class Layer {
  private neurons: Neuron[];

  /**
   * Creates a layer of neurons.
   * @param inputSize Positive integer number of inputs per neuron.
   * @param outputSize Positive integer number of neurons in the layer.
   * @param activation Activation function used by each neuron.
   * @param random Random source shared across the layer.
   */
  constructor(inputSize: number, outputSize: number, activation: Activation = "tanh", random: () => number = Math.random) {
    if (!Number.isInteger(outputSize) || outputSize <= 0) {
      throw new Error('outputSize must be a positive integer');
    }

    this.neurons = Array.from(
      { length: outputSize },
      () => new Neuron(inputSize, activation, random),
    );
  }

  /**
   * Computes outputs for all neurons in the layer.
   * @param inputs Input values.
   * @returns Layer outputs.
   */
  public forward(inputs: Value[]): Value[] {
    return this.neurons.map((neuron) => neuron.forward(inputs));
  }

  /**
   * Returns all trainable parameters in the layer.
   * @returns Flattened list of neuron parameters.
   */
  public parameters(): Value[] {
    return this.neurons.flatMap((neuron) => neuron.parameters());
  }
}
