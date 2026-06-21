import { describe, expect, it } from 'vitest';
import { val } from '../src';

describe('Value', () => {
  it('computes gradients for z = x*y + x', () => {
    const x = val(2);
    const y = val(3);

    const z = x.mul(y).add(x);

    z.backward();

    expect(z.data).toBe(8);
    expect(x.grad).toBe(4);
    expect(y.grad).toBe(2);
  });

  it('accumulates gradients when a value is used more than once', () => {
    const x = val(3);

    const z = x.mul(x).add(x); // z = x^2 + x

    z.backward();

    expect(z.data).toBe(12);
    expect(x.grad).toBe(7); // dz/dx = 2x + 1 = 7
  });

  it('computes gradients through tanh', () => {
    const x = val(0.5);

    const y = x.tanh();

    y.backward();

    const expected = 1 - Math.tanh(0.5) ** 2;

    expect(y.data).toBeCloseTo(Math.tanh(0.5));
    expect(x.grad).toBeCloseTo(expected);
  });

  it('computes gradients through relu', () => {
    const x = val(2);

    const y = x.relu();

    y.backward();

    expect(y.data).toBe(2);
    expect(x.grad).toBe(1);
  });

  it('computes zero gradient for relu when input is negative', () => {
    const x = val(-2);

    const y = x.relu();

    y.backward();

    expect(y.data).toBe(0);
    expect(x.grad).toBe(0);
  });
});
