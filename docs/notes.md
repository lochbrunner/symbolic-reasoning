# Decisions

* Use Cnn instead of LSTM: Much more simpler to bring it on speed
* Use Index tensor per sample instead of global: Simpler and possibility to reduce unrolled vector size. Size per mini-batch possible.
* Use CPU with SIMD instead of GPU: Hard to reduce with CUDA.
* Not use torch.index_select: torch.gather is more suitable
* Use own iconv implementation: Much more faster that torch.gather

# Technical Depth

## Generation

Walk randomly from simple terms to complex ones.  

### Filter

* Depth of the intermediate term
* Density of the intermediate term
* Blacklisted sub-terms
* Banal repetitions (e.g. a^0^0^0^0^0 or (1*1)*(1*1))

### Augmentation

* Permutation of free variable names 


# Paper ideas

* Rule based symbolic reasoning using indexed convolution networks
* Using hierarchical Network for self-learned abstraction
* Similarity modeling