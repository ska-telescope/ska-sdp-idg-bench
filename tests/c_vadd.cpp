#include "lib-cpu.hpp"

#if defined(BUILD_CUDA)
#include "lib-cuda.hpp"
#elif defined(BUILD_HIP)
#include "lib-hip.hpp"
#endif

#include "test_util.hpp"

int main() {
  std::cout << ">>> Correctness Vector Addition test" << std::endl;
#if defined(BUILD_CUDA)
  cuda::extern_print_device_info();
  cuda::print_benchmark();
#elif defined(BUILD_HIP)
  hip::extern_print_device_info();
  hip::print_benchmark();
#endif
  int size = get_env_var("VADD_SIZE", 1000);

  std::cout << "Number of elements: " << size << std::endl;

  std::vector<float> cpu_a(size), cpu_b(size), cpu_c(size, 0);
  std::vector<float> gpu_a(size), gpu_b(size), gpu_c(size, 0);

  srand(time(0));
  fill_vector_rand(cpu_a);
  fill_vector_rand(cpu_b);

  gpu_a = copy_vector(cpu_a);
  gpu_b = copy_vector(cpu_b);

  cpu::c_run_vadd(cpu_a, cpu_b, cpu_c, size);

#if defined(BUILD_CUDA)
  cuda::c_run_vadd(gpu_a, gpu_b, gpu_c, size);
#elif defined(BUILD_HIP)
  hip::c_run_vadd(gpu_a, gpu_b, gpu_c, size);
#endif

  bool equal = true;
  for (int i = 0; i < size; i++) {
    if (cpu_c[i] != gpu_c[i]) {
      std::cout << ">>> Error" << std::endl;
      std::cout << "  Index " << i << " cpu_c = " << cpu_c[i]
                << ", gpu_c = " << gpu_c[i] << std::endl;
      std::cout << ">>> Result NOT PASSED" << std::endl;
      equal = false;
      break;
    }
  }

  if (equal) {
    std::cout << ">>> Result PASSED" << std::endl;
  }
}
