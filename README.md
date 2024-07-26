# CUDA Int8 Tensor Core GEMM

This int8 gemm kernel gets speeds faster than CuBLAS FP16. It's also relatively easy to read, hack, fuse, and do whatever you want with.

The meat of the kernel (everything save the helpers) is 150 LoC.

On an A40 it gets 124 TOPS, versus 112 for CuBLAS fp16 through torch (problem size MNK=4096)

For some other problem sizes the difference is larger. For M=512, NK=8192 (large batch LLaMA 70b), it gets 115 TOPS versus 85 for CuBLAS FP16

---

## License

MIT