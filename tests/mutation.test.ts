import { describe, expect, it } from 'vitest';
import { val } from '../src';

describe('backward after data mutation', () => {
  it('uses both original multiplication operands', () => {
    const x = val(2);
    const y = val(3);
    const output = x.mul(y);

    x.data = 7;
    y.data = 11;
    output.backward();

    expect(output.data).toBe(6);
    expect(x.grad).toBe(3);
    expect(y.grad).toBe(2);
  });

  it('uses the original operand when multiplying a value by itself', () => {
    const x = val(2);
    const output = x.mul(x);

    x.data = 3;
    output.backward();

    expect(output.data).toBe(4);
    expect(x.grad).toBe(4);
  });

  it('uses the original base for powers', () => {
    const x = val(2);
    const output = x.pow(3);

    x.data = 5;
    output.backward();

    expect(output.data).toBe(8);
    expect(x.grad).toBe(12);
  });

  it.each([
    [2, -2, 2, 1],
    [-2, 2, 0, 0],
    [0, 2, 0, 0],
  ])('preserves the ReLU derivative at %s after changing it to %s', (initial, updated, result, gradient) => {
    const x = val(initial);
    const output = x.relu();

    x.data = updated;
    output.backward();

    expect(output.data).toBe(result);
    expect(x.grad).toBe(gradient);
  });

  it('preserves derivatives through mutated intermediate nodes', () => {
    const x = val(2);
    const square = x.mul(x);
    const output = square.pow(2).mul(3);

    x.data = 10;
    square.data = 20;
    output.backward();

    expect(output.data).toBe(48);
    expect(x.grad).toBe(96);
  });

  it('uses the original denominator for division', () => {
    const x = val(6);
    const y = val(2);
    const output = x.div(y);

    x.data = 10;
    y.data = 4;
    output.backward();

    expect(output.data).toBe(3);
    expect(x.grad).toBe(0.5);
    expect(y.grad).toBe(-1.5);
  });
});
