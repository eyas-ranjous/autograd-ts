import { describe, expect, it } from 'vitest';
import { val } from '../src';

describe('backward accumulation', () => {
  it('accumulates leaf gradients while recomputing intermediate gradients', () => {
    const x = val(2);
    const square = x.mul(x);
    const output = square.mul(3);

    output.backward();
    output.backward();

    expect(x.grad).toBe(24);
    expect(square.grad).toBe(3);
    expect(output.grad).toBe(1);

    x.zeroGrad();
    output.backward();

    expect(x.grad).toBe(12);
    expect(square.grad).toBe(3);
  });

  it('accumulates gradients across losses with a shared intermediate', () => {
    const x = val(2);
    const square = x.mul(x);
    const firstLoss = square.mul(3);
    const secondLoss = square.mul(4);

    firstLoss.backward();
    secondLoss.backward();

    expect(x.grad).toBe(28);
    expect(square.grad).toBe(4);
    expect(firstLoss.grad).toBe(1);
    expect(secondLoss.grad).toBe(1);
  });

  it('accumulates gradients across separate graphs sharing a leaf', () => {
    const x = val(2);

    x.mul(3).backward();
    x.pow(2).backward();

    expect(x.grad).toBe(7);
  });

  it('recomputes gradients through shared branches on every call', () => {
    const x = val(2);
    const square = x.mul(x);
    const output = square.mul(3).add(square.mul(4));

    output.backward();
    output.backward();

    expect(x.grad).toBe(56);
    expect(square.grad).toBe(7);
  });

  it('accumulates a unit gradient when the root is a leaf', () => {
    const x = val(2);

    x.backward();
    x.backward();

    expect(x.grad).toBe(2);

    x.zeroGrad();
    x.backward();

    expect(x.grad).toBe(1);
  });

  it('recomputes a former output when it becomes an intermediate', () => {
    const x = val(2);
    const square = x.mul(x);

    square.backward();
    square.mul(3).backward();

    expect(x.grad).toBe(16);
    expect(square.grad).toBe(3);
  });
});
