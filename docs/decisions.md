# Decisions

* Use Cnn instead of LSTM: Much more simpler to bring it on speed
* Use Index tensor per sample instead of global: Simpler and possibility to reduce unrolled vector size. Size per mini-batch possible.
* Use CPU with SIMD instead of GPU: Hard to reduce with CUDA.
* Not use torch.index_select: torch.gather is more suitable
* Use own iconv implementation: Much more faster that torch.gather