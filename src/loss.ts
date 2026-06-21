import { Value, val } from './Value';

/**
 * Computes mean squared error between predictions and targets.
 * @param predictions Model predictions.
 * @param targets Expected values.
 * @returns Scalar loss value.
 */
export function mse(predictions: Value[], targets: number[]): Value {
  if (predictions.length !== targets.length) {
    throw new Error(
      `Expected ${predictions.length} targets, received ${targets.length}`,
    );
  }

  let loss = val(0);

  for (let i = 0; i < predictions.length; i++) {
    const diff = predictions[i]!.sub(targets[i]!);
    loss = loss.add(diff.pow(2));
  }

  return loss.div(predictions.length);
}
