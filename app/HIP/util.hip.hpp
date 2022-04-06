#pragma once

#include <hip/hip_runtime.h>
#include "lib-common.hpp"

namespace hip {

#define hipCheck(ans)                                                         \
  { hipAssert((ans), __FILE__, __LINE__); }
inline void hipAssert(hipError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != hipSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

inline std::string get_device_name() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  std::string device_name = prop.name;
  std::replace(device_name.begin(), device_name.end(), ' ', '_');
  return device_name;
}

inline void print_device_info() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  std::cout << std::endl;
  std::cout << "Device Name " << prop.name << std::endl;
  std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate
            << std::endl;
  std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth
            << std::endl;
  std::cout << "  Peak Memory Bandwidth (GB/s): "
            << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6
            << std::endl;
  std::cout << "  Memory size (GB): " << prop.totalGlobalMem / 1e9 << std::endl;
  std::cout << "  Streaming MultiProcessors (SMs): " << prop.multiProcessorCount
            << std::endl;
  std::cout << std::endl;
}

inline std::vector<int> get_launch_kernel_dimensions() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  return {prop.multiProcessorCount, prop.maxThreadsPerBlock};
}

inline int get_cu_nr() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  return prop.multiProcessorCount;
}

inline int get_max_threads() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  return prop.maxThreadsPerBlock;
}

inline int get_gmem_size() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  return prop.totalGlobalMem;
}

inline int get_cu_freq() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  return prop.clockRate;
}

inline void print_dimensions(dim3 gridDim, dim3 blockDim) {
  std::cout << "Dimensions: (";
  std::cout << gridDim.x << "," << gridDim.y << "," << gridDim.z;
  std::cout << ") - (";
  std::cout << blockDim.x << "," << blockDim.y << "," << blockDim.z;
  std::cout << ")" << std::endl << std::endl;
}

template <typename T>
void p_run_kernel(const T *func, dim3 gridDim, dim3 blockDim, void **args,
                  std::string func_name = "", double gflops = 0,
                  double gbytes = 0, double mvis = 0) {

  int nr_warm_up_runs = get_env_var("NR_WARM_UP_RUNS", 2);
  int nr_iterations = get_env_var("NR_ITERATIONS", 5);

#ifdef DEBUG
  print_device_info();
  print_dimensions(gridDim, blockDim);

  std::cout << "NR_WARM_UP_RUNS: " << nr_warm_up_runs << std::endl;
  std::cout << "NR_ITERATIONS: " << nr_iterations << std::endl;
#endif

  hipEvent_t start, stop;
  hipCheck(hipEventCreate(&start));
  hipCheck(hipEventCreate(&stop));

  std::vector<float> ex_time;

  for (int i = 0; i < nr_iterations + nr_warm_up_runs; i++) {
    hipCheck(hipEventRecord(start));

    hipLaunchKernel(func, gridDim, blockDim, args, 0, 0);

    hipCheck(hipEventRecord(stop));

    hipCheck(hipEventSynchronize(stop));

    float milliseconds = 0;
    hipCheck(hipEventElapsedTime(&milliseconds, start, stop));

    ex_time.push_back(milliseconds);
  }
  float avg_milliseconds =
      std::accumulate(ex_time.begin() + nr_warm_up_runs, ex_time.end(), 0.0) /
      (ex_time.size() - nr_warm_up_runs);

  double joules = 0;
  report(func_name, avg_milliseconds * 1e-3, gflops, gbytes, mvis, joules);
  report_csv(func_name, get_device_name(), "-hip.csv", avg_milliseconds * 1e-3, gflops, gbytes, mvis, joules);
}

template <typename T>
void c_run_kernel(const T *func, dim3 gridDim, dim3 blockDim, void **args) {

#ifdef DEBUG
  print_device_info();
  print_dimensions(gridDim, blockDim);
#endif

  hipLaunchKernel(func, gridDim, blockDim, args, 0, 0);
}

} // namespace hip