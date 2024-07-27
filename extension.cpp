#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void init_cublas();
void destroy_cublas();
void cublas_igemm(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K, cudaStream_t stream);
void wrapper(void *A, void *B, void *C, const int m, const int n, const int k, cudaStream_t stream);

void igemm(torch::Tensor a, torch::Tensor b, torch::Tensor c, int m, int n, int k)
{

    TORCH_CHECK(a.device() == b.device() && a.device() == c.device(), "All tensors must be on the same device");
    TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8, "A and B tensors must be of dtype int8");
    TORCH_CHECK(c.dtype() == torch::kInt32, "C tensor must be of dtype int32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "All tensors must be 2D");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), "All tensors must be contiguous");

    wrapper(a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            m,
            n,
            k,
            at::cuda::getCurrentCUDAStream(a.get_device()));
}

void cublas_igemm_wrapper(torch::Tensor a, torch::Tensor b, torch::Tensor c, int m, int n, int k)
{
    TORCH_CHECK(a.device() == b.device() && a.device() == c.device(), "All tensors must be on the same device");
    TORCH_CHECK(a.dtype() == torch::kInt8 && b.dtype() == torch::kInt8, "A and B tensors must be of dtype int8");
    TORCH_CHECK(c.dtype() == torch::kInt32, "C tensor must be of dtype int32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "All tensors must be 2D");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), "All tensors must be contiguous");

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    cublas_igemm(static_cast<const int8_t *>(a.data_ptr()),
                 static_cast<const int8_t *>(b.data_ptr()),
                 static_cast<int32_t *>(c.data_ptr()),
                 m, n, k,
                 stream);

    C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("igemm", &igemm, "Int8xInt8 Matrix Multiplication Kernel");
    m.def("init_cublas", &init_cublas, "Initialize cuBLAS handle");
    m.def("destroy_cublas", &destroy_cublas, "Destroy cuBLAS handle");
    m.def("cublas_igemm", &cublas_igemm_wrapper, "cuBLAS Int8xInt8 Matrix Multiplication");
}