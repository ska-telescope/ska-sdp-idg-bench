#pragma once

#include "lib-common.hpp"

namespace cuda {
void print_device_info();
std::string extern_get_device_name();
void print_benchmark();
} // namespace cuda
