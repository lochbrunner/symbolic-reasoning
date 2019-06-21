# Symbolic reasoning

## Outline

* Motivation
* Definitions
* Basics
  * fit and apply
  * Brut-force solver
* Networks for pattern recognition
  * Bias
  * FC
  * LSTM
  * ...
* Fits vs best fit and generation of synthetic trainings/test data
* Using memory
* Applications to [real world problems](./real_world_problems/README.md)
* Results
* Outlook

## Definitions

Each term is defined by its nodes <img style="margin:0 0 -7px 0" src="docs/formulas/symbol.png" /> where <img style="margin:0 0 -2px 0" src="docs/formulas/ell.png" /> is the level of the node.
The term <img style="margin:0 0 -1x 0" src="docs/formulas/n.png" /> with the content <img style="margin:0 0 -3px 0" src="docs/formulas/a+b.png" /> is then represented by <img style="margin:0 0 -4px 0" src="docs/formulas/parts-of-a+b.png" />

Further more one has to distinguish if a sub-term <img style="margin:0 0 -7px 0" src="docs/formulas/symbol.png" /> is fixed one.
Which means can we use it as variable or is a constant or pre-defined operator.

## Example

Consider the following equation

![b*(cd-cd)=e](docs/formulas/b*(cd-cd)=e.png)

and the following replacement rule

![a-a => 0](docs/formulas/a-a=>0.png)

which comes directly out of the equation

![a-a = 0](docs/formulas/a-a=0_300.png)

When trying to apply this rule on the equation above results in the fitting try of the abstract term <img style="margin:0 0 -1px 0" src="docs/formulas/a-a_150.png" /> in one of the nodes of the formula <img style="margin:0 0 -5px 0" src="docs/formulas/b*(cd-cd)=e_150.png" />.

The fitting result would be that the rule fits in the node <img style="margin:0 0 -1px 0" src="docs/formulas/cd-cd_150.png" /> and the mapping <img style="margin:0 0 -1px 0" src="docs/formulas/a->cd_150.png" />.

Applying this fitting result on the initial equation would transform it to

![b*(a-a)=e](docs/formulas/b*(a-a)=e_300.png)

where `a = cd`.

with the conclusion this results in

![b*0=e](docs/formulas/b*0=e_300.png)

.

You can find this example as a [e2e test](./libcore/src/apply.rs#L328-L347) from line 149.

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