# Machine Learning approaches

## Setup

```zsh
python3 -m venv <venv name>
. <venv name>/bin/activate

pip install -r requirements.txt
```

## Standard Sequence

```zsh
./deep_main.py
```

## Tree Data

Requirements:

* Self-similarity

```zsh
./deep_main.py
```

## Unit Tests


```zsh
./run_tests.py
```

## Road-Map

1. Use [DataSets](https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel) :heavy_check_mark:
1. Use packed data (boost of factor 25) :heavy_check_mark:
1. Evaluate performance on pattern in the noise :heavy_check_mark:
1. Evaluate published Tree LSTM networks (skip for now)
1. Try to find better networks (skip for now) (needs hyper parameter search (using [scikit-optimize](https://scikit-optimize.github.io/notebooks/bayesian-optimization.html)))
1. Padding smaller trees :heavy_check_mark:
1. Integrate into main repo