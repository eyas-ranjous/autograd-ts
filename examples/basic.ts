import { val } from "../src";

const x = val(2);
const y = val(3);

const z = x.mul(y).add(x);

z.backward();

console.log("z:", z.data);
console.log("dz/dx:", x.grad);
console.log("dz/dy:", y.grad);
