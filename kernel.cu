#include <cuda_runtime.h>
#include <mma.h>
#include <cuda/pipeline>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include "helpers.cu"
#include <iostream>
#include <vector>
#include <cstdint>
#include "configs.cu"
#include "cublas_v2.h"

#define div_ru(a, b) (((a) + (b) - 1) / (b))

#define div_ru(a, b) (((a) + (b) - 1) / (b))

#define WARP_SIZE 32
#define DEBUG false

// M is not constexpr-d because tokens * batch can vary, but the rest of the problem size is fixed for specific configs
template <int BlockRowWarps, int BlockColWarps, int WarpRowTiles, int WarpColTiles, int ChunkK, int NumStages, int PipelineStrategy, int kWMMA_M, int kWMMA_N, int kWMMA_K, int kN, int kK>
struct IGemmConfig
{
    static constexpr int kBlockRowWarps = BlockRowWarps;
    static constexpr int kBlockColWarps = BlockColWarps;
    static constexpr int kWarpRowTiles = WarpRowTiles;
    static constexpr int kWarpColTiles = WarpColTiles;
    static constexpr int kChunkK = ChunkK;
    static constexpr int kNumStages = NumStages;
    static constexpr int kPipelineStrategy = PipelineStrategy;

    // Derived constants
    static constexpr int kBlockRowTiles = kWarpRowTiles * kBlockRowWarps;
    static constexpr int kBlockColTiles = kWarpColTiles * kBlockColWarps;

    static constexpr int kTileSizeM = kWMMA_M * kBlockRowTiles;
    static constexpr int kTileSizeN = kWMMA_N * kBlockColTiles;
    static constexpr int kTileSizeK = kWMMA_K * kChunkK;

    static constexpr int kSharedMemSize = kTileSizeM * kTileSizeK + kTileSizeN * kTileSizeK;

    static constexpr int K = kK;
    static constexpr int N = kN;
    static constexpr int WMMA_M = kWMMA_M;
    static constexpr int WMMA_N = kWMMA_N;
    static constexpr int WMMA_K = kWMMA_K;
};

// 128-bit vector type for efficient memory loads
using Data128B = int4;
constexpr int VECTOR_LOAD_SIZE = 16; // 128 bytes, maximum in CUDA

template <typename Config>
__global__ void igemm(const int8_t *A, const int8_t *B, int32_t *C, int M)
{
    extern __shared__ int8_t shared_memory[];

    using FragA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, int8_t, nvcuda::wmma::row_major>;
    using FragB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, int8_t, nvcuda::wmma::col_major>;

    // Set up shared memory tensors for A and B with multiple stages
    SmemTensor3D<int8_t, Config::kNumStages, Config::kTileSizeM, Config::kTileSizeK> smemA(shared_memory);
    SmemTensor3D<int8_t, Config::kNumStages, Config::kTileSizeN, Config::kTileSizeK> smemB(smemA.endPtr);

    // Set up global memory tensors for A, B, and C
    GMemTensor2D<int8_t> gmemA((int8_t *)A, M, Config::K);
    GMemTensor2D<int8_t> gmemB((int8_t *)B, Config::N, Config::K); // Note: B is transposed
    GMemTensor2D<int32_t> gmemC(C, M, Config::N);

    // Calculate warp and lane IDs
    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_row = warp_id / Config::kBlockColWarps;
    int warp_col = warp_id % Config::kBlockColWarps;

    // Calculate starting positions for this block
    int block_row_start = blockIdx.x * Config::kTileSizeM;
    int block_col_start = blockIdx.y * Config::kTileSizeN;

    FragA a_frag[Config::kWarpRowTiles];
    FragB b_frag[Config::kWarpColTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, Config::WMMA_M, Config::WMMA_N, Config::WMMA_K, int32_t> c_frag[Config::kWarpRowTiles][Config::kWarpColTiles];

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
                int row = block_row_start + (warp_row * Config::kWarpRowTiles + i) * Config::WMMA_M;
                int col = block_col_start + (warp_col * Config::kWarpColTiles + j) * Config::WMMA_N;

                if (row < M && col < Config::N)
                {
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

    auto pipeline_strategy_0 = [&]()
    {
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
                for (int kk = 0; kk < Config::kTileSizeK; kk += Config::WMMA_K)
                {
                    // Load A and B fragments
                    for (int i = 0; i < Config::kWarpRowTiles; i++)
                    {
                        nvcuda::wmma::load_matrix_sync(a_frag[i], smemA.get_ptr(s, warp_row * Config::kWarpRowTiles * Config::WMMA_M + i * Config::WMMA_M, kk), Config::kTileSizeK);
                    }

                    for (int j = 0; j < Config::kWarpColTiles; j++)
                    {
                        nvcuda::wmma::load_matrix_sync(b_frag[j], smemB.get_ptr(s, warp_col * Config::kWarpColTiles * Config::WMMA_N + j * Config::WMMA_N, kk), Config::kTileSizeK);
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
    };

    // Strategy 1: Overlapped computation and loading
    auto pipeline_strategy_1 = [&]()
    {
        // Load first stage
        load_A_tile(0, 0);
        load_B_tile(0, 0);
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        int current_stage = 0;
        for (int k = 0; k < Config::K; k += Config::kTileSizeK)
        {
            // Start loading next stage if available
            if (k + Config::kTileSizeK < Config::K)
            {
                int next_stage = 1 - current_stage;
                load_A_tile(next_stage, k + Config::kTileSizeK);
                load_B_tile(next_stage, k + Config::kTileSizeK);
                __pipeline_commit();
            }

            // Compute using current stage
            for (int kk = 0; kk < Config::kTileSizeK; kk += Config::WMMA_K)
            {
                // Load A and B fragments
                for (int i = 0; i < Config::kWarpRowTiles; i++)
                {
                    nvcuda::wmma::load_matrix_sync(a_frag[i], smemA.get_ptr(current_stage, warp_row * Config::kWarpRowTiles * Config::WMMA_M + i * Config::WMMA_M, kk), Config::kTileSizeK);
                    // ldsm8(a_frag[i], smemA.get_ptr(current_stage, warp_row * Config::kWarpRowTiles * WMMA_M + i * WMMA_M, kk), Config::kTileSizeK);
                }

                for (int j = 0; j < Config::kWarpColTiles; j++)
                {
                    nvcuda::wmma::load_matrix_sync(b_frag[j], smemB.get_ptr(current_stage, warp_col * Config::kWarpColTiles * Config::WMMA_N + j * Config::WMMA_N, kk), Config::kTileSizeK);
                    // ldsm8(b_frag[j], smemB.get_ptr(current_stage, warp_col * Config::kWarpColTiles * WMMA_N + j * WMMA_N, kk), Config::kTileSizeK);
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

            // Wait for next stage to finish loading
            __pipeline_wait_prior(0);
            __syncthreads();

            // Swap stages
            current_stage = 1 - current_stage;
        }
    };

    auto pipeline_strategy_2 = [&]()
    {
        // Prefetch first stage
        load_A_tile(0, 0);
        load_B_tile(0, 0);
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        for (int k = 0; k < Config::K; k += Config::kTileSizeK)
        {
            int current_stage = k / Config::kTileSizeK % Config::kNumStages;
            int next_stage = (current_stage + 1) % Config::kNumStages;

            // Prefetch next stage if available
            if (k + Config::kTileSizeK < Config::K)
            {
                load_A_tile(next_stage, k + Config::kTileSizeK);
                load_B_tile(next_stage, k + Config::kTileSizeK);
                __pipeline_commit();
            }

            // Compute using current stage
            for (int kk = 0; kk < Config::kTileSizeK; kk += Config::WMMA_K)
            {
                for (int i = 0; i < Config::kWarpRowTiles; i++)
                {
                    nvcuda::wmma::load_matrix_sync(a_frag[i], smemA.get_ptr(current_stage, warp_row * Config::kWarpRowTiles * Config::WMMA_M + i * Config::WMMA_M, kk), Config::kTileSizeK);
                }

                for (int j = 0; j < Config::kWarpColTiles; j++)
                {
                    nvcuda::wmma::load_matrix_sync(b_frag[j], smemB.get_ptr(current_stage, warp_col * Config::kWarpColTiles * Config::WMMA_N + j * Config::WMMA_N, kk), Config::kTileSizeK);
                }

                for (int i = 0; i < Config::kWarpRowTiles; i++)
                {
                    for (int j = 0; j < Config::kWarpColTiles; j++)
                    {
                        nvcuda::wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                    }
                }
            }

            // Wait for next stage to finish loading
            __pipeline_wait_prior(0);
            __syncthreads();
        }
    };

    auto pipeline_strategy_3 = [&]()
    {
        for (int k = 0; k < Config::K; k += Config::kTileSizeK)
        {
            int current_stage = k / Config::kTileSizeK % Config::kNumStages;

            // Load current stage
            load_A_tile(current_stage, k);
            load_B_tile(current_stage, k);
            __pipeline_commit();

            // Compute using previous stage while loading current stage
            if (k > 0)
            {
                int prev_stage = (current_stage - 1 + Config::kNumStages) % Config::kNumStages;
                for (int kk = 0; kk < Config::kTileSizeK; kk += Config::WMMA_K)
                {
                    for (int i = 0; i < Config::kWarpRowTiles; i++)
                    {
                        nvcuda::wmma::load_matrix_sync(a_frag[i], smemA.get_ptr(prev_stage, warp_row * Config::kWarpRowTiles * Config::WMMA_M + i * Config::WMMA_M, kk), Config::kTileSizeK);
                    }

                    for (int j = 0; j < Config::kWarpColTiles; j++)
                    {
                        nvcuda::wmma::load_matrix_sync(b_frag[j], smemB.get_ptr(prev_stage, warp_col * Config::kWarpColTiles * Config::WMMA_N + j * Config::WMMA_N, kk), Config::kTileSizeK);
                    }

                    for (int i = 0; i < Config::kWarpRowTiles; i++)
                    {
                        for (int j = 0; j < Config::kWarpColTiles; j++)
                        {
                            nvcuda::wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                        }
                    }
                }
            }

            // Wait for current stage to finish loading
            __pipeline_wait_prior(0);
            __syncthreads();
        }

        // Compute final stage
        int final_stage = (Config::K / Config::kTileSizeK - 1) % Config::kNumStages;
        for (int kk = 0; kk < Config::kTileSizeK; kk += Config::WMMA_K)
        {
            for (int i = 0; i < Config::kWarpRowTiles; i++)
            {
                nvcuda::wmma::load_matrix_sync(a_frag[i], smemA.get_ptr(final_stage, warp_row * Config::kWarpRowTiles * Config::WMMA_M + i * Config::WMMA_M, kk), Config::kTileSizeK);
            }

            for (int j = 0; j < Config::kWarpColTiles; j++)
            {
                nvcuda::wmma::load_matrix_sync(b_frag[j], smemB.get_ptr(final_stage, warp_col * Config::kWarpColTiles * Config::WMMA_N + j * Config::WMMA_N, kk), Config::kTileSizeK);
            }

            for (int i = 0; i < Config::kWarpRowTiles; i++)
            {
                for (int j = 0; j < Config::kWarpColTiles; j++)
                {
                    nvcuda::wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }
    };

    switch (Config::kPipelineStrategy)
    {
    case 0:
        pipeline_strategy_0();
        break;
    case 1:
        pipeline_strategy_1();
        break;
    case 2:
        pipeline_strategy_2();
        break;
    case 3:
        pipeline_strategy_3();
        break;
    default:
        pipeline_strategy_0();
        break;
    }

    // Store results
    store_C_tile();
}

template <typename Config>
void launch_igemm(const int8_t *A, const int8_t *B, int32_t *C, int M, cudaStream_t stream)
{
    dim3 grid_dim(div_ru(M, Config::kTileSizeM), div_ru(Config::N, Config::kTileSizeN));
    dim3 block_dim(WARP_SIZE * Config::kBlockRowWarps * Config::kBlockColWarps);

    // printf("grid_dim x: %d, block_dim x: %d, grid_dim y: %d, block_dim y: %d\n", grid_dim.x, block_dim.x, grid_dim.y, block_dim.y);
    // printf("M: %d, N: %d, K: %d\n", M, Config::N, Config::K);

    size_t shared_mem_size = Config::kNumStages * (Config::kTileSizeM * Config::kTileSizeK * sizeof(int8_t) + Config::kTileSizeN * Config::kTileSizeK * sizeof(int8_t));

    igemm<Config><<<grid_dim, block_dim, shared_mem_size, stream>>>(A, B, C, M);
    // cudaDeviceSynchronize();

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    // }
}

void cpu_gemm_int8(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K)
{
    // return;
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

#define LAUNCH_KERNEL_IF_CONDITION(config, mCond, nCond, kCond)                                                                                                       \
    else if (n == nCond && m == mCond && k == kCond)                                                                                                                  \
    {                                                                                                                                                                 \
        using ThisConfig = IGemmConfig<config.BlockRowWarps, config.BlockColWarps, config.WarpRowTiles, config.WarpColTiles,                                          \
                                       config.ChunkK, config.NumStages, config.PipelineStrategy, config.kWMMA_M, config.kWMMA_N, config.kWMMA_K, config.N, config.K>; \
        launch_igemm<ThisConfig>(A_ptr, B_ptr, C_ptr, m, stream);                                                                                                     \
        return;                                                                                                                                                       \
    }

void wrapper(void *A, void *B, void *C, const int m, const int n, const int k, cudaStream_t stream)
{
    const int8_t *A_ptr = reinterpret_cast<const int8_t *>(A);
    const int8_t *B_ptr = reinterpret_cast<const int8_t *>(B);
    int32_t *C_ptr = reinterpret_cast<int32_t *>(C);

    if (false)
    {
    }
    LAUNCH_KERNEL_IF_CONDITION(octomul_4096_57344_8192, 4096, 57344, 8192)
    LAUNCH_KERNEL_IF_CONDITION(octomul_4096_8192_8192, 4096, 8192, 8192)
    LAUNCH_KERNEL_IF_CONDITION(octomul_4096_28672_4096, 4096, 28672, 4096)
    LAUNCH_KERNEL_IF_CONDITION(octomul_4096_10240_8192, 4096, 10240, 8192)
    LAUNCH_KERNEL_IF_CONDITION(octomul_4096_6144_4096, 4096, 6144, 4096)
    LAUNCH_KERNEL_IF_CONDITION(octomul_4096_4096_4096, 4096, 4096, 4096)
    LAUNCH_KERNEL_IF_CONDITION(octomul_2048_8192_28672, 2048, 8192, 28672)
    LAUNCH_KERNEL_IF_CONDITION(octomul_2048_10240_8192, 2048, 10240, 8192)
    LAUNCH_KERNEL_IF_CONDITION(octomul_2048_28672_4096, 2048, 28672, 4096)
    LAUNCH_KERNEL_IF_CONDITION(octomul_2048_6144_4096, 2048, 6144, 4096)
    LAUNCH_KERNEL_IF_CONDITION(octomul_2048_4096_4096, 2048, 4096, 4096)
    LAUNCH_KERNEL_IF_CONDITION(octomul_1024_8192_28672, 1024, 8192, 28672)
    LAUNCH_KERNEL_IF_CONDITION(octomul_1024_8192_8192, 1024, 8192, 8192)
    LAUNCH_KERNEL_IF_CONDITION(octomul_1024_6144_4096, 1024, 6144, 4096)
    LAUNCH_KERNEL_IF_CONDITION(octomul_512_10240_8192, 512, 10240, 8192)
    LAUNCH_KERNEL_IF_CONDITION(octomul_4096_4096_14336, 4096, 4096, 14336)
    LAUNCH_KERNEL_IF_CONDITION(octomul_1024_10240_8192, 1024, 10240, 8192)
    LAUNCH_KERNEL_IF_CONDITION(octomul_1024_4096_4096, 1024, 4096, 4096)
    LAUNCH_KERNEL_IF_CONDITION(octomul_256_28672_4096, 256, 28672, 4096)
    LAUNCH_KERNEL_IF_CONDITION(octomul_512_6144_4096, 512, 6144, 4096)
    LAUNCH_KERNEL_IF_CONDITION(octomul_2048_8192_8192, 2048, 8192, 8192)
    LAUNCH_KERNEL_IF_CONDITION(octomul_256_6144_4096, 256, 6144, 4096)
    LAUNCH_KERNEL_IF_CONDITION(octomul_256_10240_8192, 256, 10240, 8192)
    LAUNCH_KERNEL_IF_CONDITION(octomul_1024_4096_14336, 1024, 4096, 14336)
    LAUNCH_KERNEL_IF_CONDITION(octomul_1024_28672_4096, 1024, 28672, 4096)
    LAUNCH_KERNEL_IF_CONDITION(octomul_256_57344_8192, 256, 57344, 8192)
    LAUNCH_KERNEL_IF_CONDITION(octomul_256_4096_4096, 256, 4096, 4096)
    LAUNCH_KERNEL_IF_CONDITION(octomul_256_4096_14336, 256, 4096, 14336)
    LAUNCH_KERNEL_IF_CONDITION(octomul_2048_57344_8192, 2048, 57344, 8192)
    LAUNCH_KERNEL_IF_CONDITION(octomul_1024_57344_8192, 1024, 57344, 8192)
    LAUNCH_KERNEL_IF_CONDITION(octomul_256_8192_8192, 256, 8192, 8192)
    LAUNCH_KERNEL_IF_CONDITION(octomul_2048_4096_14336, 2048, 4096, 14336)
    LAUNCH_KERNEL_IF_CONDITION(octomul_512_57344_8192, 512, 57344, 8192)
    LAUNCH_KERNEL_IF_CONDITION(octomul_256_8192_28672, 256, 8192, 28672)
    LAUNCH_KERNEL_IF_CONDITION(octomul_512_8192_28672, 512, 8192, 28672)
    LAUNCH_KERNEL_IF_CONDITION(octomul_512_4096_4096, 512, 4096, 4096)
    LAUNCH_KERNEL_IF_CONDITION(octomul_512_28672_4096, 512, 28672, 4096)
    LAUNCH_KERNEL_IF_CONDITION(octomul_512_4096_14336, 512, 4096, 14336)
    LAUNCH_KERNEL_IF_CONDITION(octomul_512_8192_8192, 512, 8192, 8192)
}

cublasHandle_t g_cublas_handle = nullptr;

void init_cublas()
{
    if (g_cublas_handle == nullptr)
    {
        cublasStatus_t status = cublasCreate(&g_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            printf("cuBLAS initialization failed with error code %d\n", status);
        }
    }
}

void destroy_cublas()
{
    if (g_cublas_handle != nullptr)
    {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
}

void cublas_igemm(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K, cudaStream_t stream)
{
    if (g_cublas_handle == nullptr)
    {
        printf("cuBLAS handle not initialized\n");
        return;
    }

    cublasStatus_t status = cublasSetStream(g_cublas_handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("cuBLAS set stream failed with error code %d\n", status);
        return;
    }

    const int32_t alpha = 1;
    const int32_t beta = 0;

    status = cublasGemmEx(g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                          N, M, K,
                          &alpha,
                          B, CUDA_R_8I, K,
                          A, CUDA_R_8I, K,
                          &beta,
                          C, CUDA_R_32I, N,
                          CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("cuBLAS GEMM failed with error code %d\n", status);
    }
}