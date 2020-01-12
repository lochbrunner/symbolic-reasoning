[![GitHub Actions status](https://github.com/lochbrunner/symbolic-reasoning/workflows/core/badge.svg)](https://github.com/lochbrunner/symbolic-reasoning/actions?workflow=core)
[![GitHub Actions status](https://github.com/lochbrunner/symbolic-reasoning/workflows/pycore/badge.svg)](https://github.com/lochbrunner/symbolic-reasoning/actions?workflow=pycore)[![GitHub Actions status](https://github.com/lochbrunner/symbolic-reasoning/workflows/ml/badge.svg)](https://github.com/lochbrunner/symbolic-reasoning/actions?workflow=ml)


# Symbolic reasoning

## Outline

* Motivation
* Definitions
* Basics
  * fit and apply
  * Brut-force solver
* Networks for pattern recognition
  * Network with deep capacity
  * Bias, FC, LSTM, ...
* Fits vs best fit and generation of synthetic trainings/test data
* Using memory
* Applications to [real world problems](./real_world_problems/README.md)
* Results
* Outlook

## Definitions

Each term is defined by its nodes $n_{i_1},\cdots,n_{i_\ell}$  where $\ell$ is the level of the node.
The term $n$ with the content $a+b$ is then represented by $n=+; n_0=a; n_1=b$

Further more one has to distinguish if a sub-term $n_{i_1,\cdots,i_\ell}$  is fixed one.
Which means can we use it as variable or is a constant or pre-defined operator.

## Example

Consider the following equation

$$b\cdot\left(cd-cd\right)=e$$

and the following replacement rule

$$a-a \Rightarrow 0$$

which comes directly out of the equation

$$a-a = 0$$

When trying to apply this rule on the equation above results in the fitting try of the abstract term $a-a$ in one of the nodes of the formula $b\cdot\left(cd-cd\right)=e$.

The fitting result would be that the rule fits in the node $cd-cd$ and the mapping $a\rightarrow cd$.

Applying this fitting result on the initial equation would transform it to

$$b\cdot\left(a-a\right)=e$$

where $a = cd$.

with the conclusion this results in

$$b\cdot0=e$$

.

You can find this example as a [e2e test](./libcore/src/apply.rs#L328-L347) from line 149.

## Abstract

Rule

$$F(a,b) \rightarrow G(a)$$

## Minimal examples with Derivations

### Polynomial

$$D\left( x^n\right), x) \Rightarrow n\cdot x^{n-1} $$


## Implementation

This project contains.

* [Calculation Generator](./generator)
* [Machine Learning](./ml)


## Test

Run

```zsh
cargo t --workspace --exclude pycore
```

## Profiling

```zsh
perf record -g cargo bench no_variable_deep --workspace --exclude pycore
perf script | stackcollapse-perf.pl | rust-unmangle | flamegraph.pl > flame_no_variable_deep.svg
firefox flame_no_variable_deep.svg
```
