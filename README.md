# dlib activations benchmark

Profiling results on a GeForce GTX 1080 Ti for a 64x3x224x224 tensor, over 100 runs after 10 warmup runs:

- `relu fwd: 221.732 ± 1.39986 us`
- `relu bwd: 435.857 ± 2.80692 us`
- `mish fwd: 235.555 ± 1.90957 us`
- `mish bwd: 446.959 ± 2.46593 us`
