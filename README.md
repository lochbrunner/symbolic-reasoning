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