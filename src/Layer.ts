import { Value } from './Value';
import { Activation, Neuron } from './Neuron';

/**
 * Fully connected layer containing multiple neurons.
 */
export class Layer {
  private neurons: Neuron[];

  /**
   * Creates a layer of neurons.
   * @param inputSize Number of inputs per neuron.
   * @param outputSize Number of neurons in the layer.
   * @param activation Activation function used by each neuron.
   */
  constructor(inputSize: number, outputSize: number, activation: Activation = "tanh") {
    this.neurons = Array.from(
      { length: outputSize },
      () => new Neuron(inputSize, activation),
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
