import { Value, val } from '../core/Value';

export type Activation = 'tanh' | 'relu' | 'none';

/**
 * Fully connected neuron with trainable weights and bias.
 */
export class Neuron {
  /** Trainable weights applied to each input feature. */
  private weights: Value[];

  /** Trainable bias added to the weighted sum. */
  private bias: Value;

  /** Activation function applied to the neuron output. */
  private activation: Activation;

  /**
   * Creates a neuron with randomly initialized weights.
   * @param inputSize Number of expected input features.
   * @param activation Activation function applied to the output.
   */
  constructor(inputSize: number, activation: Activation = 'tanh') {
    this.weights = Array.from({ length: inputSize }, () => val(randomWeight()));
    this.bias = val(0);
    this.activation = activation;
  }

  /**
   * Computes the neuron output for a given set of inputs.
   * @param inputs Input values.
   * @returns Activated neuron output.
   */
  public forward(inputs: Value[]): Value {
    if (inputs.length !== this.weights.length) {
      throw new Error(
        `Expected ${this.weights.length} inputs, received ${inputs.length}`,
      );
    }

    let output = this.bias;

    for (let i = 0; i < this.weights.length; i++) {
      output = output.add(this.weights[i]!.mul(inputs[i]!));
    }

    if (this.activation === 'tanh') return output.tanh();
    if (this.activation === 'relu') return output.relu();

    return output;
  }

  /**
   * Returns all trainable parameters of the neuron.
   * @returns Weights and bias.
   */
  public parameters(): Value[] {
    return [...this.weights, this.bias];
  }
}

function randomWeight(): number {
  return Math.random() * 2 - 1;
}
