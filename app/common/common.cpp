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

std::string get_env_var(const char *env_var, std::string default_value) {
  if (const char *env_p = std::getenv(env_var)) {
    std::string env = env_p;
    return env;
  } else {
    return default_value;
  }
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
    std::cout << ", " << std::setw(w2) << efficiency << " GFLOP/s/W";
  }
  std::cout << std::endl;
}

void report_csv(std::string name, std::string device_name,
                std::string file_extension, double seconds, double gflops,
                double gbytes, double mvis, double joules) {
  if (device_name.empty() || file_extension.empty()) {
    std::cout << ">>> Device name or file extension not provided" << std::endl;
  } else {
    std::string file_path = get_env_var("OUTPUT_PATH", ".");
    std::cout << "Saving output in " << file_path << std::endl;

    std::ofstream output;
    std::cout << file_path + "/" + device_name + "-" + name + file_extension
              << std::endl;
    output.open(file_path + "/" + device_name + "-" + name + file_extension);
    output << std::fixed << std::setprecision(2);

    output << "ms," << seconds * 1e3 << "\n";
    if (gflops != 0) {
      output << "GFLOP/s," << gflops / seconds << "\n";
    }
    if (gbytes != 0) {
      output << "GB/s," << gbytes / seconds << "\n";
    }
    if (gflops != 0 && gbytes != 0) {
      float arithmetic_intensity = gflops / gbytes;
      output << "FLOP/Byte," << arithmetic_intensity << "\n";
    }
    if (mvis != 0) {
      output << "MVis/s," << mvis / seconds << "\n";
    }
    if (joules != 0) {
      double watt = joules / seconds;
      double efficiency = gflops / joules;
      output << "W," << watt << "\n";
      output << "GFLOP/s/W," << efficiency << "\n";
    }
    output.close();
  }
}
