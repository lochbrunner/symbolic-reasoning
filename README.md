# Symbolic reasoning

## Definitions

Each term is defined by its nodes `n_{i_1, ..., i_l}` where `l` is the level of the node.
The term `n` with the content `a+b` is then represented by `n = +`, `n_0 = a` and `n_1 = b`

Further more one has to distinguish if a sub-term `n_{i_1, ..., i_l}` is fixed one.
Which means can we use it as variable or is a constant or pre-defined operator.

## Example

Consider the following equation

```latex
b*(cd-cd)=e
```

and the following replacement rule

```latex
a-a => 0
```

which comes directly out of the equation

```latex
a-a = 0
```

When trying to apply this rule on the equation above results in the fitting try of the abstract term `a-a` in one of the nodes of the formula `b*(cd-cd)=e`.

The fitting result would be that the rule fits in the node `cd-cd` and the mapping `a -> cd`.

Applying this fitting result on the initial equation would transform it to

```latex
b*(a-a)=e
```

where `a = cd`.

with the conclusion this results in

```latex
b*0=e
```

.

You can find this example as a [e2e test](./libcore/src/apply.rs#L148-L167) from line 149.

## Abstract

Rule

F(a,b) -> G(a)

## Minimal examples with Derivations

### Polynomial

```math
D(x^n, x) => n*x^(n-1)
```

## Implementation

This project contains.

* [Calculation Generator](./generator)
* [Machine Learning](./ml)