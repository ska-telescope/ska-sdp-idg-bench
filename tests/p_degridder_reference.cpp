#include "lib-cpu.hpp"

#if defined(BUILD_CUDA)
#include "lib-cuda.hpp"
using namespace cuda;
#elif defined(BUILD_HIP)
#include "lib-hip.hpp"
using namespace hip;
#endif

int main() {
  std::cout << ">>> Performance IDG-Gridder test" << std::endl;

  extern_print_device_info();
  print_benchmark();
#if defined(BUILD_CUDA) || defined(BUILD_HIP)
  p_run_degridder_reference();
#endif
}