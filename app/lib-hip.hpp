#pragma once

#include "lib-common.hpp"

namespace hip {
extern void extern_print_device_info();
extern std::string extern_get_device_name();
void print_benchmark();

void c_run_vadd(std::vector<float> &a, std::vector<float> &b,
                std::vector<float> &c, int size);
void p_run_vadd();
} // namespace cuda