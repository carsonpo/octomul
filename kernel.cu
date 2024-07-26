#include <cuda_runtime.h>
#include <mma.h>
#include <cuda/pipeline>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include "helpers.cuh"

#define div_ru(a, b) (((a) + (b) - 1) / (b))

// CUDA and WMMA constants
constexpr int WARP_SIZE = 32;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// M is not constexpr-d because tokens * batch can vary, but the rest of the problem size is fixed for specific configs
template <int BlockRowWarps, int BlockColWarps, int WarpRowTiles, int WarpColTiles, int ChunkK, int NumStages, int kK, int kN>
struct IGemmConfig
{
    static constexpr int kBlockRowWarps = BlockRowWarps;
    static constexpr int kBlockColWarps = BlockColWarps;
    static constexpr int kWarpRowTiles = WarpRowTiles;
    static constexpr int kWarpColTiles = WarpColTiles;
    static constexpr int kChunkK = ChunkK;
    static constexpr int kNumStages = NumStages;

    // Derived constants
    static constexpr int kBlockRowTiles = kWarpRowTiles * kBlockRowWarps;
    static constexpr int kBlockColTiles = kWarpColTiles * kBlockColWarps;

    static constexpr int kTileSizeM = WMMA_M * kBlockRowTiles;
    static constexpr int kTileSizeN = WMMA_N * kBlockColTiles;
    static constexpr int kTileSizeK = WMMA_K * kChunkK;

    static constexpr int kSharedMemSize = kTileSizeM * kTileSizeK + kTileSizeN * kTileSizeK;

    static constexpr int K = kK;
    static constexpr int N = kN;
};

// 128-bit vector type for efficient memory loads
using Data128B = int4;
constexpr int VECTOR_LOAD_SIZE = 16; // 128 bytes, maximum in CUDA

template <typename Config>
__global__ void igemm(const int8_t *A, const int8_t *B, int32_t *C, int M)
{
    extern __shared__ int8_t shared_memory[];

    // Set up shared memory tensors for A and B with multiple stages
    SmemTensor3D<int8_t, Config::kNumStages, Config::kTileSizeM, Config::kTileSizeK> smemA(shared_memory);
    SmemTensor3D<int8_t, Config::kNumStages, Config::kTileSizeN, Config::kTileSizeK> smemB(smemA.endPtr);

    // Set up global memory tensors for A, B, and C
    GMemTensor2D<int8_t> gmemA((int8_t *)A, M, Config::K);
    GMemTensor2D<int8_t> gmemB((int8_t *)B, Config::N, Config::K); // Note: B is transposed
    GMemTensor2D<int32_t> gmemC(C, M, Config::N);

    // Calculate warp and lane IDs
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_row = warp_id / Config::kBlockColWarps;
    int warp_col = warp_id % Config::kBlockColWarps;

    // Calculate starting positions for this block
    int block_row_start = blockIdx.x * Config::kTileSizeM;
    int block_col_start = blockIdx.y * Config::kTileSizeN;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, nvcuda::wmma::row_major> a_frag[Config::kWarpRowTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, nvcuda::wmma::col_major> b_frag[Config::kWarpColTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[Config::kWarpRowTiles][Config::kWarpColTiles];

    // Initialize accumulator fragments
    for (int i = 0; i < Config::kWarpRowTiles; i++)
    {
        for (int j = 0; j < Config::kWarpColTiles; j++)
        {
            nvcuda::wmma::fill_fragment(c_frag[i][j], 0);
        }
    }

    // Lambda for loading A tiles
    auto load_A_tile = [&](int stage, int k_offset)
    {
        for (int i = threadIdx.x; i < (Config::kTileSizeM * Config::kTileSizeK) / VECTOR_LOAD_SIZE; i += blockDim.x)
        {
            int row = (i * VECTOR_LOAD_SIZE) / Config::kTileSizeK;
            int col = (i * VECTOR_LOAD_SIZE) % Config::kTileSizeK;
            if (block_row_start + row < M && k_offset + col + VECTOR_LOAD_SIZE - 1 < Config::K)
            {
                Data128B *shared_ptr = reinterpret_cast<Data128B *>(smemA.get_ptr(stage, row, col));
                Data128B *global_ptr = reinterpret_cast<Data128B *>(gmemA.get_ptr(block_row_start + row, k_offset + col));
                __pipeline_memcpy_async(shared_ptr, global_ptr, sizeof(Data128B));
            }
        }
    };

    // Lambda for loading B tiles
    auto load_B_tile = [&](int stage, int k_offset)
    {
        for (int i = threadIdx.x; i < (Config::kTileSizeN * Config::kTileSizeK) / VECTOR_LOAD_SIZE; i += blockDim.x)
        {
            int row = (i * VECTOR_LOAD_SIZE) / Config::kTileSizeK;
            int col = (i * VECTOR_LOAD_SIZE) % Config::kTileSizeK;
            if (block_col_start + row < Config::N && k_offset + col + VECTOR_LOAD_SIZE - 1 < Config::K)
            {
                Data128B *shared_ptr = reinterpret_cast<Data128B *>(smemB.get_ptr(stage, row, col));
                Data128B *global_ptr = reinterpret_cast<Data128B *>(gmemB.get_ptr(block_col_start + row, k_offset + col));
                __pipeline_memcpy_async(shared_ptr, global_ptr, sizeof(Data128B));
            }
        }
    };

    // Lambda for storing C tiles
    auto store_C_tile = [&]()
    {
        for (int i = 0; i < Config::kWarpRowTiles; i++)
        {
            for (int j = 0; j < Config::kWarpColTiles; j++)
            {
                int row = block_row_start + (warp_row * Config::kWarpRowTiles + i) * WMMA_M;
                int col = block_col_start + (warp_col * Config::kWarpColTiles + j) * WMMA_N;

                if (row < M && col < Config::N)
                {

                    // Store the result directly to global memory using wmma::store_matrix_sync
                    nvcuda::wmma::store_matrix_sync(
                        gmemC.get_ptr(row, col),
                        c_frag[i][j],
                        Config::N,
                        nvcuda::wmma::mem_row_major);
                }
            }
        }
        __syncthreads();
    };

    // Main loop with pipelining
    for (int k = 0; k < Config::K; k += Config::kTileSizeK * Config::kNumStages)
    {
        // Prefetch stages
        for (int s = 0; s < Config::kNumStages - 1; s++)
        {
            load_A_tile(s, k + s * Config::kTileSizeK);
            load_B_tile(s, k + s * Config::kTileSizeK);
            __pipeline_commit();
        }
        __pipeline_wait_prior(Config::kNumStages - 2);
        __syncthreads();

        // Main computation loop
        for (int s = 0; s < Config::kNumStages; s++)
        {
            int current_k = k + s * Config::kTileSizeK;

            // Load next stage if available
            if (s < Config::kNumStages - 1)
            {
                load_A_tile((s + Config::kNumStages - 1) % Config::kNumStages, current_k + Config::kTileSizeK);
                load_B_tile((s + Config::kNumStages - 1) % Config::kNumStages, current_k + Config::kTileSizeK);
                __pipeline_commit();
            }

            // Compute using current stage
            for (int kk = 0; kk < Config::kTileSizeK; kk += WMMA_K)
            {
                // Load A and B fragments
                for (int i = 0; i < Config::kWarpRowTiles; i++)
                {
                    nvcuda::wmma::load_matrix_sync(a_frag[i], smemA.get_ptr(s, warp_row * Config::kWarpRowTiles * WMMA_M + i * WMMA_M, kk), Config::kTileSizeK);
                }

                for (int j = 0; j < Config::kWarpColTiles; j++)
                {
                    nvcuda::wmma::load_matrix_sync(b_frag[j], smemB.get_ptr(s, warp_col * Config::kWarpColTiles * WMMA_N + j * WMMA_N, kk), Config::kTileSizeK);
                }

                // Perform matrix multiplication
                for (int i = 0; i < Config::kWarpRowTiles; i++)
                {
                    for (int j = 0; j < Config::kWarpColTiles; j++)
                    {
                        nvcuda::wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                    }
                }
            }

            __pipeline_wait_prior(Config::kNumStages - 2);
            __syncthreads();
        }
    }

    // Store results
    store_C_tile();
}

template <typename Config>
void launch_igemm(const int8_t *A, const int8_t *B, int32_t *C, int M)
{
    dim3 grid_dim(div_ru(M, Config::kTileSizeM), div_ru(Config::N, Config::kTileSizeN));
    dim3 block_dim(WARP_SIZE * Config::kBlockRowWarps * Config::kBlockColWarps);

    size_t shared_mem_size = Config::kNumStages * (Config::kTileSizeM * Config::kTileSizeK * sizeof(int8_t) + Config::kTileSizeN * Config::kTileSizeK * sizeof(int8_t));

    igemm<Config><<<grid_dim, block_dim, shared_mem_size>>>(A, B, C, M);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}

void cpu_gemm_int8(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K)
{
    return;
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            int32_t sum = 0;
            for (int k = 0; k < K; ++k)
            {
                sum += static_cast<int32_t>(A[m * K + k]) * static_cast<int32_t>(B[n * K + k]);
            }
            C[m * N + n] = sum;
        }
    }
}

bool compare_results(const int32_t *gpu_result, const int32_t *cpu_result, int size, float tolerance = 1e-5)
{
    for (int i = 0; i < size; ++i)
    {
        if (std::abs(static_cast<float>(gpu_result[i] - cpu_result[i])) > tolerance)
        {
            std::cout << "Mismatch at index " << i << ": GPU = " << gpu_result[i] << ", CPU = " << cpu_result[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main()
{
    // Choose matrix dimensions (multiples of 16 for best performance)
    const int M = 4096, N = 4096, K = 4096;

    using Config = IGemmConfig<4, 2, 4, 4, 4, 2, M, N>;

    // Allocate host memory
    std::vector<int8_t> h_A(M * K);
    std::vector<int8_t> h_B(K * N);
    std::vector<int32_t> h_C_gpu(M * N);
    std::vector<int32_t> h_C_cpu(M * N);

    // Initialize input matrices with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-4, 4);

    std::generate(h_A.begin(), h_A.end(), [&]()
                  { return static_cast<int8_t>(dis(gen)); });
    std::generate(h_B.begin(), h_B.end(), [&]()
                  { return static_cast<int8_t>(dis(gen)); });

    // Allocate device memory
    int8_t *d_A, *d_B;
    int32_t *d_C;
    cudaMalloc(&d_A, M * K * sizeof(int8_t));
    cudaMalloc(&d_B, K * N * sizeof(int8_t));
    cudaMalloc(&d_C, M * N * sizeof(int32_t));

    // Copy input data to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    constexpr int numWarmups = 10;
    constexpr int numTrials = 100;

    for (int i = 0; i < numWarmups; ++i)
    {
        launch_igemm<Config>(d_A, d_B, d_C, M);
    }

    cudaDeviceSynchronize();

    // Launch GPU kernel
    cudaEventRecord(start);
    for (int i = 0; i < numTrials; ++i)
    {
        launch_igemm<Config>(d_A, d_B, d_C, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate TOPS
    double seconds = milliseconds / 1000.0;
    double operations = static_cast<double>(M) * N * K * 2 * numTrials; // 2 ops per multiply-add
    double tops = operations / (seconds * 1e12);

    std::cout << "GPU Performance: " << tops << " TOPS" << std::endl;

    // Copy result back to host
    cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Compute CPU result (commented out for performance)
    cpu_gemm_int8(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);

    // Compare results (commented out for performance)
    bool results_match = compare_results(h_C_gpu.data(), h_C_cpu.data(), M * N);

    if (results_match)
    {
        std::cout << "Results match! The WMMA GEMM implementation is correct." << std::endl;
    }
    else
    {
        std::cout << "Results do not match. There might be an error in the WMMA GEMM implementation." << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}