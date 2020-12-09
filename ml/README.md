# Machine Learning approaches

## Setup

```zsh
python3 -m venv <venv name>
. <venv name>/bin/activate

pip install -r requirements.txt
```

## Training

```zsh
./train.py
```

## Try-Trace-Train Loop

Given a set of problems, the search trace train loop is

* Search for a solution
* Generate training data from search results
* Train

```zsh
./ml/t3_loop.py -c real_world_problems/number_crunching/config.yaml
```

### Try-Trace-Train-Trance Loop

*Not implemented yet*

Extension of t3

* Search for a solution
* Generate training data from search results
* Add new rules reorder and forget (sleep trance)
* Train


## Unit Tests

```zsh
PYTHONPATH=ml ./run_tests.py
```

## Road-Map

1. Use [DataSets](https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel) :heavy_check_mark:
1. Use packed data (boost of factor 25) :heavy_check_mark:
1. Evaluate performance on pattern in the noise :heavy_check_mark:
1. Evaluate published Tree LSTM networks :heavy_check_mark: (bad performance)
1. Try to find better networks :heavy_check_mark: Done with Azure (skip for now) (needs hyper parameter search (using [scikit-optimize](https://scikit-optimize.github.io/notebooks/bayesian-optimization.html)))
1. Padding smaller trees :heavy_check_mark:
1. Integrate into main repo :heavy_check_mark: