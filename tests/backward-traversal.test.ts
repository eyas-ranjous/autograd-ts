import { describe, expect, it } from 'vitest';
import { val } from '../src';

describe('backward traversal', () => {
  it('differentiates a chain of 20,000 operations without overflowing the stack', () => {
    const x = val(2);
    let output = x;

    for (let i = 0; i < 20_000; i++) {
      output = output.add(1);
    }

    output.backward();

    expect(output.data).toBe(20_002);
    expect(x.grad).toBe(1);
  });

  it('combines both branches before differentiating their shared intermediate', () => {
    const x = val(2);
    const square = x.mul(x);
    const output = square.mul(3).add(square.mul(4));

    output.backward();

    expect(output.data).toBe(28);
    expect(square.grad).toBe(7);
    expect(x.grad).toBe(28);
  });

  it('orders a shared intermediate before a branch that also uses it directly', () => {
    const x = val(2);
    const square = x.mul(x);
    const output = square.add(square.mul(3));

    output.backward();

    expect(output.data).toBe(16);
    expect(square.grad).toBe(4);
    expect(x.grad).toBe(16);
  });
});
