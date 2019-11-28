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
# LSTM Evaluation

## Setup

```zsh
python3 -m venv <venv name>
. <venv name>/bin/activate

pip install -r requirements.txt
```

## Standard Sequence

```zsh
./flat_main.py -n 60 -s all --use own torch rebuilt optimized torch-cell && ./summary
```

## Tree Data

Requirements:

* Self-similarity

```zsh
./deep_main.py
```

## Unit Tests


```zsh
PYTHONPATH=`pwd` deep/generate.specs.py
```

## Road-Map

1. Use [DataSets](https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel) :heavy_check_mark:
1. Use packed data (boost of factor 25) :heavy_check_mark:
1. Evaluate performance on pattern in the noise (needs output at each node)
1. Evaluate published Tree LSTM networks
1. Try to find better networks (needs hyper parameter search (using [scikit-optimize](https://scikit-optimize.github.io/notebooks/bayesian-optimization.html)))
1. Integrate into main repo