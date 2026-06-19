import { Value } from '../core/Value';

/**
 * Updates parameters using stochastic gradient descent.
 * @param parameters Trainable parameters.
 * @param learningRate Optimization step size.
 */
export function sgd(parameters: Value[], learningRate: number): void {
  for (const parameter of parameters) {
    parameter.data -= learningRate * parameter.grad;
  }
}
