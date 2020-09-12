# Concepts

## Dynamic Rule Learning

Let $n$ be the number of rules in per node.

1. Select $n$ rules and train with them
1. Select the next rule and infer that samples
2. Get the rule and the node of which was mostly predicted by that samples 
3. Either:
  1. Create a new rule if the best matching node is full with the best matching rule
  1. Insert the new rule in the best matching node

### Hierarchical Multi-head Network

## Value Network

Alpha Go uses a linear layer to compute the scalar.
Not feasible for dynamic hidden layer size.

Alternatives:

* RNN
* Max
* Avg