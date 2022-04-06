#include "common.hpp"

unsigned roundToPowOf2(unsigned number) {
  double logd = log(number) / log(2);
  logd = floor(logd);

  return (unsigned)pow(2, (int)logd);
}

unsigned long get_env_var(const char *env_var, unsigned long default_value) {
  if (const char *env_p = std::getenv(env_var)) {
    return atoi(env_p);
  } else {
    return default_value;
  }
}

void print_stats(std::string title, std::vector<double> stats, int sel) {
  std::vector<std::string> units = {"GFLOP", "GB"};

  std::cout << std::fixed;
  std::cout << std::setprecision(2);

  std::cout << std::setw(6) << title << ": ";
  std::cout << std::setw(8) << stats[0] << " (ms), ";

  std::cout << std::setw(8) << stats[1];

  if (sel != 0 && sel != 1) {
    std::cerr << std::endl << "ERROR: Option not available" << std::endl;
    exit(0);
  }

  std::string unit = " (" + units[sel] + "), ";
  std::cout << std::setw(10) << unit;

  std::cout << std::setw(9) << stats[2];
  unit = " (" + units[sel] + "/s), ";
  std::cout << std::setw(12) << unit;

  std::cout << std::endl;
}

void print_t_stats(std::string title, double stat, int sel) {
  std::vector<std::string> units = {"GFLOP", "GB"};

  std::cout << std::fixed;
  std::cout << std::setprecision(2);

  std::cout << std::setw(6) << title << ": ";

  if (sel != 0 && sel != 1) {
    std::cerr << std::endl << "ERROR: Option not available" << std::endl;
    exit(0);
  }

  std::string unit = " (" + units[sel] + "), ";
  std::cout << std::setw(10) << unit;

  std::cout << std::setw(9) << stat;
  unit = " (" + units[sel] + "/s), ";
  std::cout << std::setw(12) << unit;

  std::cout << std::endl;
}

void report(std::string name, double seconds, double gflops, double gbytes,
            double mvis, double joules) {
  int w1 = 20;
  int w2 = 7;
  std::cout << std::setw(w1) << std::string(name) << ": ";
  std::cout << std::setprecision(2) << std::fixed;
  std::cout << std::setw(w2) << seconds << " ms";
  if (gflops != 0) {
    std::cout << ", " << std::setw(w2) << gflops / seconds << " GFLOP/s";
  }
  if (gbytes != 0) {
    std::cout << ", " << std::setw(w2) << gbytes / seconds << " GB/s";
  }
  if (gflops != 0 && gbytes != 0) {
    float arithmetic_intensity = gflops / gbytes;
    std::cout << ", " << std::setw(w2) << arithmetic_intensity << " FLOP/byte";
  }
  if (mvis != 0) {
    std::cout << ", " << std::setw(w2) << mvis / seconds << " MVis/s";
  }
  if (joules != 0) {
    double watt = joules / seconds;
    double efficiency = gflops / joules;
    std::cout << ", " << std::setw(w2) << watt << " W";
    std::cout << ", " << std::setw(w2) << efficiency << " GFLOP/W";
  }
  std::cout << std::endl;
  // add option to export to csv
}
