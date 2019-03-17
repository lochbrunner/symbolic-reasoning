# Symbolic reasoning

## Simple Example

> Note: Variables of rules are written in upper case. Concrete variables in lower case. 

Rule

```latex
A*B+A*C -> A*(B+C)
```

Can be applied to

```latex
r*s+r*t
```

which transforms it to

```latex
r*(s+t)
```

## Abstract

Rule

F(a,b) -> G(a)

## Minimal examples with Derivations

### Polynomial

```math
D(x^n, x) => n*x^(n-1)
```

## Roadmap

* Fitting :soon:
* Parsing of binary operators :x:
* Randomly generating of calculations :x:
* Adapter to TensorFlow :x:
* Modeling and Training :x:
* Using inferencing for solving manual and generated calculations :x:
* Latex loading/dumping :x: