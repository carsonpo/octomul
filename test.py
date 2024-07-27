import torch
import octomul
import time

octomul.init_cublas()


def test_correctness_and_benchmark(
    M, N, K, dtype=torch.float16, num_runs=50, num_warmup=5
):
    # Generate random input data
    A = torch.randint(-4, 4, (M, K), dtype=torch.int8, device="cuda")
    B = torch.randint(-4, 4, (N, K), dtype=torch.int8, device="cuda")
    C = torch.zeros((M, N), dtype=torch.int32, device="cuda")
    C_ref = torch.zeros((M, N), dtype=torch.int32, device="cuda")

    # Generate half-precision tensors
    A_half = torch.randn((M, K), dtype=torch.float16, device="cuda")
    B_half = torch.randn((N, K), dtype=torch.float16, device="cuda")
    C_half = torch.zeros((M, N), dtype=torch.float16, device="cuda")

    torch.matmul(A_half, B_half.t(), out=C_half)

    # correctness check
    octomul.cublas_igemm(A, B, C_ref, M, N, K)
    octomul.igemm(A, B, C, M, N, K)
    max_diff = torch.max(torch.abs(C_ref - C))
    is_correct = max_diff <= 1

    if not is_correct:
        print(f"Incorrect: {M}, {N}, {K}, Max diff: {max_diff.item()}")
        return {
            "is_correct": is_correct,
            "max_diff": max_diff.item(),
            "pytorch_time": 0,
            "cuda_time": 0,
            "pytorch_tops": 0,
            "cuda_tops": 0,
            "speedup": 0,
            "half_time": 0,
            "half_tops": 0,
            "half_speedup": 0,
        }

    octomul.cublas_igemm(A, B, C_ref, M, N, K)

    # PyTorch reference implementation (int8)
    pytorch_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(pytorch_graph):
        for _ in range(num_runs):
            octomul.cublas_igemm(A, B, C_ref, M, N, K)

    # Warmup
    pytorch_graph.replay()
    torch.cuda.synchronize()

    # Benchmark PyTorch (int8)
    start = time.time()
    pytorch_graph.replay()
    torch.cuda.synchronize()
    end = time.time()
    pytorch_time = (end - start) / num_runs

    # Your CUDA kernel implementation
    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        for _ in range(num_runs):
            octomul.igemm(A, B, C, M, N, K)

    cuda_graph.replay()
    torch.cuda.synchronize()

    # Benchmark CUDA kernel
    start = time.time()
    cuda_graph.replay()
    torch.cuda.synchronize()
    end = time.time()
    cuda_time = (end - start) / num_runs

    # PyTorch half-precision implementation
    half_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(half_graph):
        for _ in range(num_runs):
            torch.matmul(A_half, B_half.t(), out=C_half)

    half_graph.replay()
    torch.cuda.synchronize()

    # Benchmark PyTorch half-precision
    start = time.time()
    half_graph.replay()
    torch.cuda.synchronize()
    end = time.time()
    half_time = (end - start) / num_runs

    # Calculate FLOPS
    ops = 2 * M * N * K  # multiply-add is 2 operations
    pytorch_tops = ops / pytorch_time / 1e12
    cuda_tops = ops / cuda_time / 1e12
    half_tops = ops / half_time / 1e12

    return {
        "is_correct": is_correct,
        "max_diff": max_diff.item(),
        "pytorch_time": pytorch_time,
        "cuda_time": cuda_time,
        "pytorch_tops": pytorch_tops,
        "cuda_tops": cuda_tops,
        "speedup": pytorch_time / cuda_time,
        "half_time": half_time,
        "half_tops": half_tops,
        "half_speedup": half_time / cuda_time,
    }


def run_tests():
    test_cases = [
        # LLaMA 70b QKV Proj
        (512, 10240, 8192),
        (1024, 10240, 8192),
        (2048, 10240, 8192),
        (4096, 10240, 8192),
        # LLaMA 70b Attn Out Proj
        (512, 8192, 8192),
        (1024, 8192, 8192),
        (2048, 8192, 8192),
        (4096, 8192, 8192),
        # LLaMA 70b MLP In
        (512, 28672 * 2, 8192),
        (1024, 28672 * 2, 8192),
        (2048, 28672 * 2, 8192),
        (4096, 28672 * 2, 8192),
        # LLaMA 70b MLP Out
        (512, 8192, 28672),
        (1024, 8192, 28672),
        (2048, 8192, 28672),
        (4096, 8192, 28672),
        # LLaMA 8b QKV Proj
        (512, 6144, 4096),
        (1024, 6144, 4096),
        (2048, 6144, 4096),
        (4096, 6144, 4096),
        # LLaMA 8b Attn Out
        (512, 4096, 4096),
        (1024, 4096, 4096),
        (2048, 4096, 4096),
        (4096, 4096, 4096),
        # LLaMA 8b MLP In
        (512, 14336 * 2, 4096),
        (1024, 14336 * 2, 4096),
        (2048, 14336 * 2, 4096),
        (4096, 14336 * 2, 4096),
        # LLaMA 8b MLP Out
        (512, 4096, 14336),
        (1024, 4096, 14336),
        (2048, 4096, 14336),
        (4096, 4096, 14336),
    ]

    for M, N, K in test_cases:
        print(f"\nTesting M={M}, N={N}, K={K}")
        result = test_correctness_and_benchmark(M, N, K)

        if result["is_correct"]:
            print("Results correct!")
            print(f"PyTorch int8 performance: {result['pytorch_tops']:.2f} Tops")
            print(f"CUDA kernel performance: {result['cuda_tops']:.2f} Tops")
            print(f"PyTorch half performance: {result['half_tops']:.2f} Tops")
            print(f"Speedup over int8: {result['speedup']:.2f}x")
            print(f"Speedup over half: {result['half_speedup']:.2f}x")
        else:
            print(f"Incorrect: {M}, {N}, {K}, Max diff: {result['max_diff']:.2f}")


if __name__ == "__main__":
    run_tests()
    octomul.destroy_cublas()
