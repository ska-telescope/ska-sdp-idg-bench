# SKA SDP IDG Bench

This benchmark is a simplified version of the [Image Domain Gridding repository](https://git.astron.nl/RD/idg).
It mainly contains a CUDA and HIP IDG implementation for benchmark purposes.

# Installation

Clone the repository and cd the folder:

```
git clone https://gitlab.com/ska-telescope/sdp/ska-sdp-idg-bench.git
cd ska-sdp-idg-bench
```

Make sure to have CUDA/HIP support on your system.
Please follow this [guidelines](https://git.astron.nl/RD/schaap-spack/-/wikis/Reproducible-SW-environment-with-Spack).

Source the setup file in the **share** folder:

```
source ./share/setup-env.sh
```

On an NVIDIA system source also the HIP-patch:
```
source ./share/setup-hip-nvidia.sh
```

```
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install
```

Check the benchmark you would like to build with the cmake GUI:

```
ccmake .
```

Then install:
```
make -j
make install
```

**NOTE**: there are also some scripts that can be used in the **scripts** folder.

To build the benchmark for couda you could simply run this in the main folder:
```
./scripts/install_nvidia.sh
```

### Run benchmark
cuda-c_vadd  cuda-p_vadd  hip-c_vadd  hip-p_vadd


Simply run:
```
./<programming-model>-<test-version>_<benchmark-name>
```

where:
- `<programming-model>` can be `cuda` or `hip`.
- `<test-version>` can be `p` (performance) or `c` (correctness).
- `<benchmark-name>` can be `vadd`, `gridder`, `degridder`, etc.

### Environment variables

- `OUTPUT_PATH`: when running the performance version of the benchmark (`p`), you may want to specify an output folder where to store your **csv** output.
- `NR_WARM_UP_RUNS`: number of warm-up runs per benchmark.
- `NR_ITERATIONS`: number of iterations per benchmark (in addition to the warm up runs).

### Results

Some results on different GPUs are reported in the **res** folder.


### Based on:
- [Image Domain Gridding micro-benchmark](https://gitlab.com/astron-idg/idg-cuda-bench).
- [Image Domain Gridding main repository](https://git.astron.nl/RD/idg).
