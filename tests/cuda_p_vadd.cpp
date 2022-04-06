#include "lib-cpu.hpp"
#include "lib-cuda.hpp"

int main() {
  std::cout << ">>> Performance Vector Addition test" << std::endl;
  cuda::extern_print_device_info();
  cuda::print_benchmark();

  cuda::p_run_vadd();
}