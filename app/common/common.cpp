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
  std::cout << std::setw(w2) << seconds * 1e3 << " ms";
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
    double mvis_j = mvis / joules;
    std::cout << ", " << std::setw(w2) << watt << " W";
    std::cout << ", " << std::setw(w2) << efficiency << " GFLOP/s/W";
    std::cout << ", " << std::setw(w2) << mvis_j << " MVis/J";
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
    device_name = std::regex_replace(device_name, std::regex("/"), "-");
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
      double mvis_j = mvis / joules;
      output << "W," << watt << "\n";
      output << "GFLOP/s/W," << efficiency << "\n";
      output << "MVis/J," << mvis_j << "\n";
    }
    output.close();
  }
}

uint64_t flops_gridder(uint64_t nr_channels, uint64_t nr_timesteps,
                       uint64_t nr_subgrids, uint64_t subgrid_size,
                       uint64_t nr_correlations) {
  // Number of flops per visibility
  uint64_t flops_per_visibility = 0;
  flops_per_visibility += 5;                                 // phase index
  flops_per_visibility += 5;                                 // phase offset
  flops_per_visibility += nr_channels * 2;                   // phase
  flops_per_visibility += nr_channels * nr_correlations * 8; // update

  // Number of flops per subgrid
  uint64_t flops_per_subgrid = 0;
  flops_per_subgrid += 6; // shift

  // Total number of flops
  uint64_t flops_total = 0;
  flops_total +=
      nr_timesteps * subgrid_size * subgrid_size * flops_per_visibility;
  flops_total += nr_subgrids * subgrid_size * subgrid_size * flops_per_subgrid;
  return flops_total;
}

uint64_t bytes_gridder(uint64_t nr_channels, uint64_t nr_timesteps,
                       uint64_t nr_subgrids, uint64_t subgrid_size,
                       uint64_t nr_correlations) {
  // Number of bytes per uvw coordinate
  uint64_t bytes_per_uvw = 0;
  bytes_per_uvw += 1ULL * 3 * sizeof(float); // read uvw

  // Number of bytes per visibility
  uint64_t bytes_per_vis = 0;
  bytes_per_vis += 1ULL * nr_channels * nr_correlations * 2 *
                   sizeof(float); // read visibilities

  // Number of bytes per pixel
  uint64_t bytes_per_pix = 0;
  bytes_per_pix += 1ULL * nr_correlations * 2 * sizeof(float); // read pixel
  bytes_per_pix += 1ULL * nr_correlations * 2 * sizeof(float); // write pixel

  // Number of bytes per aterm
  uint64_t bytes_per_aterm = 0;
  bytes_per_aterm +=
      1ULL * 2 * nr_correlations * 2 * sizeof(float); // read aterm

  // Number of bytes per spheroidal
  uint64_t bytes_per_spheroidal = 0;
  bytes_per_spheroidal += 1ULL * sizeof(float); // read spheroidal

  // Total number of bytes
  uint64_t bytes_total = 0;
  bytes_total += 1ULL * nr_timesteps * bytes_per_uvw;
  bytes_total += 1ULL * nr_timesteps * bytes_per_vis;
  bytes_total +=
      1ULL * nr_subgrids * subgrid_size * subgrid_size * bytes_per_pix;
  bytes_total +=
      1ULL * nr_subgrids * subgrid_size * subgrid_size * bytes_per_aterm;
  bytes_total +=
      1ULL * nr_subgrids * subgrid_size * subgrid_size * bytes_per_spheroidal;
  return bytes_total;
}
