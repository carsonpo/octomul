# OctoMul — CUDA Int8 Tensor Core GEMM

![OctoMul Logo](assets/octomul-logo.jpg)

This int8 gemm kernel gets speeds faster than cuBLAS FP16 (and occasionally cuBLAS Int8). It's also relatively easy to read, hack, fuse, and do whatever you want with.

Hopper and Ada GPUs have fp8 support, but GPUs going back to even the Turing gen have int8 tensor core support.
Things like SmoothQuant are also possible which would enable a 2x inference speedup on the prefill rather than just on the AR decoding (what things like Marlin, FLUTE, and GGML quants typically focus on).

You can install it for yourself with `pip install .` and then run the tests with `python test.py`.

I tuned for kernels for an A40, so if you're using another device the configs might be less than optimal and you'd need to autotune them for said device.

Talk is cheap though, so here are the benchmarks:

| M    | N     | K     | cuBLAS int8 (TOP/s)  | OctoMul (TOP/s)     | cuBLAS fp16 (TOP/s)  | Speedup over int8 | Speedup over FP16 |
|------|-------|-------|----------------------|---------------------|----------------------|-------------------|-------------------|
| 512  | 10240 | 8192  | 137.85               | 152.71              | 117.60               | 1.11x             | 1.30x             |
| 1024 | 10240 | 8192  | 232.84               | 173.46              | 105.80               | 0.74x             | 1.64x             |
| 2048 | 10240 | 8192  | 226.35               | 163.70              | 110.35               | 0.72x             | 1.48x             |
| 4096 | 10240 | 8192  | 227.96               | 163.58              | 111.16               | 0.72x             | 1.47x             |
| 512  | 8192  | 8192  | 136.33               | 132.47              | 90.26                | 0.97x             | 1.47x             |
| 1024 | 8192  | 8192  | 196.60               | 150.70              | 102.34               | 0.77x             | 1.47x             |
| 2048 | 8192  | 8192  | 212.42               | 163.71              | 106.84               | 0.77x             | 1.53x             |
| 4096 | 8192  | 8192  | 223.64               | 148.46              | 112.26               | 0.66x             | 1.32x             |
| 512  | 57344 | 8192  | 220.94               | 147.43              | 105.28               | 0.67x             | 1.40x             |
| 1024 | 57344 | 8192  | 225.33               | 163.06              | 108.05               | 0.72x             | 1.51x             |
| 2048 | 57344 | 8192  | 230.42               | 164.05              | 116.18               | 0.71x             | 1.41x             |
| 4096 | 57344 | 8192  | 228.53               | 150.87              | 105.27               | 0.66x             | 1.43x             |
| 512  | 8192  | 28672 | 123.20               | 136.92              | 74.35                | 1.11x             | 1.84x             |
| 1024 | 8192  | 28672 | 204.14               | 148.46              | 111.43               | 0.73x             | 1.33x             |
| 2048 | 8192  | 28672 | 224.47               | 167.06              | 113.97               | 0.74x             | 1.47x             |
| 4096 | 8192  | 28672 | 238.00               | 163.32              | 114.75               | 0.69x             | 1.42x             |
| 512  | 6144  | 4096  | 142.71               | 132.36              | 95.01                | 0.93x             | 1.39x             |
| 1024 | 6144  | 4096  | 179.30               | 149.64              | 117.38               | 0.83x             | 1.27x             |
| 2048 | 6144  | 4096  | 206.32               | 140.75              | 105.50               | 0.68x             | 1.33x             |
| 4096 | 6144  | 4096  | 201.50               | 143.11              | 105.98               | 0.71x             | 1.35x             |
| 512  | 4096  | 4096  | 131.54               | 120.72              | 104.94               | 0.92x             | 1.15x             |
| 1024 | 4096  | 4096  | 150.41               | 138.37              | 98.04                | 0.92x             | 1.41x             |
| 2048 | 4096  | 4096  | 180.55               | 144.51              | 102.02               | 0.80x             | 1.42x             |
| 4096 | 4096  | 4096  | 199.89               | 143.97              | 107.02               | 0.72x             | 1.35x             |
| 512  | 28672 | 4096  | 202.00               | 154.45              | 105.10               | 0.76x             | 1.47x             |
| 1024 | 28672 | 4096  | 208.70               | 160.34              | 106.09               | 0.77x             | 1.51x             |
| 2048 | 28672 | 4096  | 208.89               | 165.06              | 112.70               | 0.79x             | 1.46x             |
| 4096 | 28672 | 4096  | 213.60               | 152.57              | 103.54               | 0.71x             | 1.47x             |
| 512  | 4096  | 14336 | 132.86               | 143.12              | 103.94               | 1.08x             | 1.38x             |
| 1024 | 4096  | 14336 | 130.37               | 139.53              | 108.19               | 1.07x             | 1.29x             |
| 2048 | 4096  | 14336 | 200.29               | 156.05              | 102.77               | 0.78x             | 1.52x             |
| 4096 | 4096  | 14336 | 219.94               | 158.80              | 109.93               | 0.72x             | 1.44x             |


I'm still working on improving the kernel so I hope to reach parity with cuBLAS on int8 fully.

---

## License

MIT