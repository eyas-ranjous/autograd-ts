export type BackwardFn = () => void;

interface ValueOptions {
  prev?: Value[];
  op?: string;
  backward?: BackwardFn;
}

/**
 * Scalar node in a computation graph supporting reverse-mode automatic differentiation.
 */
export class Value {
  public data: number;
  public grad = 0;

  private _prev: Value[];
  private _op?: string;
  private _backward: BackwardFn;

  /**
   * Creates a new computation graph node.
   * @param data Numeric value stored in this node.
   * @param opts Internal metadata for graph construction.
   */
  constructor(data: number, opts?: ValueOptions) {
    this.data = data;
    this._prev = opts?.prev ?? [];
    this._op = opts?.op;
    this._backward = opts?.backward ?? (() => {});
  }

  /**
   * Adds another value to this value.
   * @param v Value or number to add.
   * @returns New Value representing the sum.
   */
  public add(v: Value | number): Value {
    const o = v instanceof Value ? v : val(v);

    const out = new Value(this.data + o.data, {
      prev: [this, o],
      op: "+",
      backward: () => {
        this.grad += out.grad;
        o.grad += out.grad;
      },
    });

    return out;
  }

  /**
   * Multiplies this value by another.
   * @param v Value or number to multiply.
   * @returns New Value representing the product.
   */
  public mul(v: Value | number): Value {
    const o = v instanceof Value ? v : val(v);

    const out = new Value(this.data * o.data, {
      prev: [this, o],
      op: "*",
      backward: () => {
        this.grad += o.data * out.grad;
        o.grad += this.data * out.grad;
      },
    });

    return out;
  }

  /**
   * Raises this value to a numeric power.
   * @param n Exponent.
   * @returns New Value representing x^n.
   */
  public pow(n: number): Value {
    const out = new Value(this.data ** n, {
      prev: [this],
      op: `pow(${n})`,
      backward: () => {
        this.grad += n * this.data ** (n - 1) * out.grad;
      },
    });

    return out;
  }

  /**
   * Applies the hyperbolic tangent activation.
   * @returns New Value representing tanh(x).
   */
  public tanh(): Value {
    const t = Math.tanh(this.data);

    const out = new Value(t, {
      prev: [this],
      op: "tanh",
      backward: () => {
        this.grad += (1 - t * t) * out.grad;
      },
    });

    return out;
  }

  /**
   * Applies the ReLU activation function.
   * @returns New Value representing max(0, x).
   */
  public relu(): Value {
    const out = new Value(this.data > 0 ? this.data : 0, {
      prev: [this],
      op: "relu",
      backward: () => {
        this.grad += (this.data > 0 ? 1 : 0) * out.grad;
      },
    });

    return out;
  }

  /**
   * Negates this value.
   * @returns New Value representing -x.
   */
  public neg(): Value {
    return this.mul(-1);
  }

  /**
   * Subtracts another value from this value.
   * @param v Value or number to subtract.
   * @returns New Value representing x - y.
   */
  public sub(v: Value | number): Value {
    return this.add(v instanceof Value ? v.neg() : -v);
  }

  /**
   * Divides this value by another.
   * @param v Value or number to divide by.
   * @returns New Value representing x / y.
   */
  public div(v: Value | number): Value {
    const o = v instanceof Value ? v : val(v);
    return this.mul(o.pow(-1));
  }

  /**
   * Resets the gradient stored on this node.
   */
  public zeroGrad(): void {
    this.grad = 0;
  }

  /**
   * Performs reverse-mode automatic differentiation to compute gradients.
   */
  public backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    const buildTopo = (v: Value): void => {
      if (visited.has(v)) return;

      visited.add(v);

      for (const parent of v._prev) {
        buildTopo(parent);
      }

      topo.push(v);
    };

    buildTopo(this);

    this.grad = 1;

    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }

  /**
   * Returns a human-readable representation of the Value.
   * @returns String representation of this node.
   */
  public toString(): string {
    return `Value(data=${this.data}, grad=${this.grad})`;
  }
}

/**
 * A helper for creating a Value from a number.
 * @param n Numeric value.
 * @returns Value
 */
export function val(n: number): Value {
  return new Value(n);
}
