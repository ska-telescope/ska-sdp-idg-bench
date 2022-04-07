#include "cpu.hpp"
namespace cpu {
void c_run_vadd(std::vector<float> &a, std::vector<float> &b,
                std::vector<float> &c, int size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}
} // namespace cpu