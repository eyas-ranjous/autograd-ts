import { MLP, mse, sgd } from '../src';

export const dataset = [
  { inputs: [0, 0], target: 0 },
  { inputs: [0, 1], target: 1 },
  { inputs: [1, 0], target: 1 },
  { inputs: [1, 1], target: 0 },
];

/** Trains XOR with a fixed initialization and sample order. */
export function trainXor(): MLP {
  let randomState = 42;
  const model = new MLP(2, [4, 1], {
    hiddenActivation: 'tanh',
    outputActivation: 'none',
    random: () => {
      randomState = (Math.imul(1664525, randomState) + 1013904223) >>> 0;
      return randomState / 4294967296;
    },
  });
  const parameters = model.parameters();

  for (let epoch = 0; epoch < 1000; epoch++) {
    for (const sample of dataset) {
      const loss = mse(model.forward(sample.inputs), [sample.target]);
      model.zeroGrad();
      loss.backward();
      sgd(parameters, 0.1);
    }
  }

  return model;
}

if (require.main === module) {
  const model = trainXor();
  console.log('XOR predictions (seed=42, epochs=1000)');

  for (const sample of dataset) {
    const prediction = model.forward(sample.inputs)[0]!;
    console.log(`${sample.inputs.join(',')} => ${prediction.data.toFixed(4)}`);
  }
}
