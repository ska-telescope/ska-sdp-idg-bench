#include "util.hpp"

namespace cpu {
void kernel_vadd(float *a, float *b, float *c, int size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

void c_run_vadd(std::vector<float> &a, std::vector<float> &b,
                std::vector<float> &c, int size) {
  kernel_vadd(a.data(), b.data(), c.data(), size);
}
} // namespace cpu
