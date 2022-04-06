#include "util.cuh"

namespace cuda {
std::string extern_get_device_name() { return get_device_name(); }

void extern_print_device_info() { print_device_info(); }

void print_benchmark() { std::cout << ">>> CUDA IDG BENCHMARK" << std::endl; }

} // namespace cuda