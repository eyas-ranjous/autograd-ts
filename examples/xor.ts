import { MLP, mse, sgd } from '../src/nn';

const dataset = [
  { inputs: [0, 0], target: 0 },
  { inputs: [0, 1], target: 1 },
  { inputs: [1, 0], target: 1 },
  { inputs: [1, 1], target: 0 },
];

const model = new MLP(2, [4, 1], {
  hiddenActivation: "tanh",
  outputActivation: "none",
});

const learningRate = 0.1;
const epochs = 1000;

for (let epoch = 0; epoch < epochs; epoch++) {
  let totalLoss = 0;

  for (const sample of dataset) {
    const prediction = model.forward(sample.inputs);

    const loss = mse(prediction, [sample.target]);

    model.zeroGrad();
    loss.backward();

    sgd(model.parameters(), learningRate);

    totalLoss += loss.data;
  }

  if (epoch % 100 === 0) {
    console.log(
      `epoch=${epoch} loss=${totalLoss.toFixed(6)}`,
    );
  }
}

console.log("\nXOR Predictions\n");

for (const sample of dataset) {
  const prediction = model.forward(sample.inputs)[0];

  console.log(
    `${sample.inputs.join(",")} => ${prediction?.data.toFixed(4)}`,
  );
}

/*

epoch=0 loss=1.824531
epoch=100 loss=0.712943
epoch=200 loss=0.403817
epoch=300 loss=0.182691
epoch=400 loss=0.087432
epoch=500 loss=0.041827
epoch=600 loss=0.021554
epoch=700 loss=0.012115
epoch=800 loss=0.007083
epoch=900 loss=0.004291

XOR Predictions
0,0 => 0.0213
0,1 => 0.9834
1,0 => 0.9772
1,1 => 0.0189

*/
