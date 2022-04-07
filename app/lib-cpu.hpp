#pragma once

#include "lib-common.hpp"

namespace cpu {
void c_run_vadd(std::vector<float> &a, std::vector<float> &b,
                std::vector<float> &c, int size);

}