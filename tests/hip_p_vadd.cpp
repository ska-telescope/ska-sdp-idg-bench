#include "lib-cpu.hpp"
#include "lib-hip.hpp"

int main() {
  std::cout << ">>> Performance Vector Addition test" << std::endl;
  hip::extern_print_device_info();
  hip::print_benchmark();

  hip::p_run_vadd();
}