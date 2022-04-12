#include "util.cuh"

namespace cuda {

__global__ void kernel_vadd(float *a, float *b, float *c, int size) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < size) {
    c[i] = a[i] + b[i];
  }
}

void p_run_vadd() {
  unsigned long size = get_env_var("VADD_SIZE", 1000000000);
  std::vector<int> dim = {get_cu_nr(), get_max_threads()};

  std::cout << "Number of elements: " << size << std::endl;
  if (size < dim[1]) {
    dim[0] = 1;
    dim[1] = size;
  } else {
    dim[0] = (size + dim[1] - 1) / dim[1];
  }

  std::string func_name = "vadd";
  double gflops = size * 1e-9;
  double gbytes = size * sizeof(float) * 3 * 1e-9;

  float *d_a, *d_b, *d_c;
  cudaCheck(cudaMalloc(&d_a, size * sizeof(float)));
  cudaCheck(cudaMalloc(&d_b, size * sizeof(float)));
  cudaCheck(cudaMalloc(&d_c, size * sizeof(float)));

  void *args[] = {&d_a, &d_b, &d_c, &size};

  p_run_kernel((void *)kernel_vadd, dim3(dim[0]), dim3(dim[1]), args, func_name,
               gflops, gbytes);

  cudaCheck(cudaFree(d_a));
  cudaCheck(cudaFree(d_b));
  cudaCheck(cudaFree(d_c));
}

void c_run_vadd(std::vector<float> &a, std::vector<float> &b,
                std::vector<float> &c, int size) {
  std::vector<int> dim = {get_cu_nr(), get_max_threads()};

  if (size < dim[1]) {
    dim[0] = 1;
    dim[1] = size;
  } else {
    dim[0] = (size + dim[1] - 1) / dim[1];
  }

  float *d_a, *d_b, *d_c;
  cudaCheck(cudaMalloc(&d_a, a.size() * sizeof(float)));
  cudaCheck(cudaMalloc(&d_b, b.size() * sizeof(float)));
  cudaCheck(cudaMalloc(&d_c, c.size() * sizeof(float)));

  cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);

  void *args[] = {&d_a, &d_b, &d_c, &size};

  c_run_kernel((void *)kernel_vadd, dim3(dim[0]), dim3(dim[1]), args);

  cudaMemcpy(c.data(), d_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost);

  cudaCheck(cudaFree(d_a));
  cudaCheck(cudaFree(d_b));
  cudaCheck(cudaFree(d_c));
}

} // namespace cuda
