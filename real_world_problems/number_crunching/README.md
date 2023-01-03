# Number Crunching

## Motivation

Use dynamic memory to use self found rules.

## Goal

The algorithm should be able to perform addition, subtraction and multiplication on neutral numbers up to 100.

## Rules

### Define the numbers

Either each number explicit:

`<m> = <n>+1`

or via digits

$$d_1d_0 = 10\cdot d_1+d_0$$

Pro:

* Scalable

Contra:

* Needs more features in the [core library](../../libcore)
* Complicated to map a symbol to multiple inputs.

Might it be a good choice to use a [Integer factorization](https://en.wikipedia.org/wiki/Integer_factorization) later?

### Define multiplication

`a*b = b + (a-1)*b`