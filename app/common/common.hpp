#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <regex>

#include "init.hpp"
//#include "math.hpp"
#include "parameters.hpp"
#include "print.hpp"
//#include "types.hpp"

unsigned roundToPowOf2(unsigned number);

unsigned long get_env_var(const char *env_var, unsigned long default_value);

std::string get_env_var(const char *env_var, std::string default_value);

void report(std::string name, double seconds = 0, double gflops = 0,
            double gbytes = 0, double mvis = 0, double joules = 0);

void report_csv(std::string name, std::string device_name = "",
                std::string file_extension = "", double seconds = 0,
                double gflops = 0, double gbytes = 0, double mvis = 0,
                double joules = 0);

uint64_t flops_gridder(uint64_t nr_channels, uint64_t nr_timesteps,
                       uint64_t nr_subgrids, uint64_t subgrid_size,
                       uint64_t nr_correlations);

uint64_t bytes_gridder(uint64_t nr_channels, uint64_t nr_timesteps,
                       uint64_t nr_subgrids, uint64_t subgrid_size,
                       uint64_t nr_correlations);
