#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

unsigned roundToPowOf2(unsigned number);

unsigned long get_env_var(const char *env_var, unsigned long default_value);

void print_stats(std::string title, std::vector<double> stats, int sel);

void print_t_stats(std::string title, double stat, int sel);

void print_benchmark();

void report(std::string name, double seconds = 0, double gflops = 0,
            double gbytes = 0, double mvis = 0, double joules = 0);