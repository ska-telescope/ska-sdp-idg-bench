#pragma once

#include "lib-common.hpp"

#ifdef ENABLE_POWERSENSOR
#include <powersensor/NVMLPowerSensor.h>
#endif

namespace cuda {

#define cudaCheck(ans)                                                         \
  { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

inline std::string get_device_name() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::string device_name = prop.name;
  std::replace(device_name.begin(), device_name.end(), ' ', '_');
  return device_name;
}

inline void print_device_info() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << std::endl;
  std::cout << "Device Name " << prop.name << std::endl;
  std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate
            << std::endl;
  std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth
            << std::endl;
  std::cout << "  Peak Memory Bandwidth (GB/s): "
            << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6
            << std::endl;
  std::cout << "  Memory size (GB): " << prop.totalGlobalMem / 1e9 << std::endl;
  std::cout << "  Streaming MultiProcessors (SMs): " << prop.multiProcessorCount
            << std::endl;
  std::cout << std::endl;
}

inline std::vector<int> get_launch_kernel_dimensions() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return {prop.multiProcessorCount, prop.maxThreadsPerBlock};
}

inline int get_cu_nr() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.multiProcessorCount;
}

inline int get_max_threads() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.maxThreadsPerBlock;
}

inline int get_gmem_size() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.totalGlobalMem;
}

inline int get_cu_freq() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.clockRate;
}

inline void print_dimensions(dim3 gridDim, dim3 blockDim) {
  std::cout << "Dimensions: (";
  std::cout << gridDim.x << "," << gridDim.y << "," << gridDim.z;
  std::cout << ") - (";
  std::cout << blockDim.x << "," << blockDim.y << "," << blockDim.z;
  std::cout << ")" << std::endl << std::endl;
}

template <typename T>
void p_run_kernel(const T *func, dim3 gridDim, dim3 blockDim, void **args,
                  std::string func_name = "", double gflops = 0,
                  double gbytes = 0, double mvis = 0) {
  float seconds;
  double avg_time, joules, avg_joules;
  std::vector<double> ex_joules, ex_time;
#ifdef ENABLE_POWERSENSOR
  std::unique_ptr<powersensor::PowerSensor> powersensor(
      powersensor::nvml::NVMLPowerSensor::create());
  powersensor::State start, end;
#else
  cudaEvent_t start, stop;
  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&stop));
#endif

  int nr_warm_up_runs = get_env_var("NR_WARM_UP_RUNS", 2);
  int nr_iterations = get_env_var("NR_ITERATIONS", 5);

#ifdef DEBUG
  print_device_info();
  print_dimensions(gridDim, blockDim);

  std::cout << "NR_WARM_UP_RUNS: " << nr_warm_up_runs << std::endl;
  std::cout << "NR_ITERATIONS: " << nr_iterations << std::endl;
#endif

  for (int i = 0; i < nr_iterations + nr_warm_up_runs; i++) {
#ifdef ENABLE_POWERSENSOR
    start = powersensor->read();
#else
    cudaCheck(cudaEventRecord(start));
#endif
    cudaLaunchKernel(func, gridDim, blockDim, args);

#ifdef ENABLE_POWERSENSOR
    cudaDeviceSynchronize();
    end = powersensor->read();
    seconds = powersensor->seconds(start, end);
    joules = powersensor->Joules(start, end);
    ex_joules.push_back(joules);
#else
    cudaCheck(cudaEventRecord(stop));

    cudaCheck(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaCheck(cudaEventElapsedTime(&milliseconds, start, stop));
    seconds = milliseconds * 1e-3;
#endif
    ex_time.push_back(seconds);
  }

#ifdef ENABLE_POWERSENSOR
  avg_joules = std::accumulate(ex_joules.begin() + nr_warm_up_runs,
                               ex_joules.end(), 0.0) /
               (ex_joules.size() - nr_warm_up_runs);
#endif

  avg_time =
      std::accumulate(ex_time.begin() + nr_warm_up_runs, ex_time.end(), 0.0) /
      (ex_time.size() - nr_warm_up_runs);

  report(func_name, avg_time, gflops, gbytes, mvis, avg_joules);
  report_csv(func_name, get_device_name(), "-cuda.csv", avg_time, gflops,
             gbytes, mvis, avg_joules);
}

template <typename T>
void c_run_kernel(const T *func, dim3 gridDim, dim3 blockDim, void **args) {

#ifdef DEBUG
  print_device_info();
  print_dimensions(gridDim, blockDim);
#endif
  cudaLaunchKernel(func, gridDim, blockDim, args);
}

template <typename T>
void p_run_gridder(const T *func, std::string func_name, int num_threads) {

  float image_size = IMAGE_SIZE;
  float w_step_in_lambda = W_STEP;

  int nr_correlations = get_env_var("NR_CORRELATIONS", 4);
  int grid_size = get_env_var("GRID_SIZE", 1024);
  int subgrid_size = get_env_var("SUBGRID_SIZE", 32);
  int nr_stations = get_env_var("NR_STATIONS", 50);
  int nr_timeslots = get_env_var("NR_TIMESLOTS", 20);
  int nr_timesteps = get_env_var("NR_TIMESTEPS_SUBGRID", 128);
  int nr_channels = get_env_var("NR_CHANNELS", 16);

  int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;
  int nr_subgrids = nr_baselines * nr_timeslots;
  int total_nr_timesteps = nr_subgrids * nr_timesteps;

  std::vector<int> dim = {nr_subgrids, num_threads};

  print_parameters(nr_stations, nr_channels, nr_timesteps, nr_correlations,
                   nr_timeslots, image_size, grid_size, subgrid_size,
                   w_step_in_lambda, nr_baselines, nr_subgrids,
                   total_nr_timesteps);

  auto gflops =
      1e-9 * flops_gridder(nr_channels, total_nr_timesteps, nr_subgrids,
                           subgrid_size, nr_correlations);
  auto gbytes =
      1e-9 * bytes_gridder(nr_channels, total_nr_timesteps, nr_subgrids,
                           subgrid_size, nr_correlations);
  auto mvis = 1e-6 * total_nr_timesteps * nr_channels;

  idg::Array1D<idg::Metadata> metadata(nr_subgrids);

  idg::UVWCoordinate<float> *d_uvw;
  float *d_wavenumbers, *d_spheroidal;
  float2 *d_visibilities, *d_aterms, *d_subgrids;
  idg::Array1D<idg::Baseline> baselines(nr_baselines);
  idg::Metadata *d_metadata;

  initialize_baselines(nr_stations, baselines);
  initialize_metadata(grid_size, nr_timeslots, nr_timesteps, baselines,
                      metadata);

  cudaCheck(cudaMalloc(&d_uvw,
                       3 * nr_subgrids * nr_timesteps * sizeof(float)));
  cudaCheck(cudaMalloc(&d_wavenumbers, nr_channels * sizeof(float)));
  cudaCheck(
      cudaMalloc(&d_spheroidal, subgrid_size * subgrid_size * sizeof(float)));
  cudaCheck(cudaMalloc(&d_visibilities, nr_subgrids * nr_timesteps *
                                            nr_channels * nr_correlations * sizeof(float2)));
  cudaCheck(cudaMalloc(&d_aterms, nr_timeslots * nr_stations * subgrid_size *
                                      subgrid_size * nr_correlations * sizeof(float2)));
  cudaCheck(cudaMalloc(&d_subgrids, nr_subgrids * nr_correlations *
                                        subgrid_size * subgrid_size *
                                        sizeof(float2)));
  cudaCheck(cudaMalloc(&d_metadata, metadata.bytes()));

  cudaMemcpy(d_metadata, metadata.data(), metadata.bytes(),
             cudaMemcpyHostToDevice);

  void *args[] = {
      &grid_size,      &subgrid_size, &image_size, &w_step_in_lambda,
      &nr_channels,    &nr_stations,  &d_uvw,      &d_wavenumbers,
      &d_visibilities, &d_spheroidal, &d_aterms,   &d_metadata,
      &d_subgrids};

  p_run_kernel((void *)func, dim3(dim[0]), dim3(dim[1]), args, func_name,
               gflops, gbytes, mvis);

  cudaCheck(cudaFree(d_uvw));
  cudaCheck(cudaFree(d_wavenumbers));
  cudaCheck(cudaFree(d_spheroidal));
  cudaCheck(cudaFree(d_visibilities));
  cudaCheck(cudaFree(d_aterms));
  cudaCheck(cudaFree(d_metadata));
  cudaCheck(cudaFree(d_subgrids));
}

template <typename T>
void c_run_gridder(
    int nr_subgrids, int grid_size, int subgrid_size, float image_size,
    float w_step_in_lambda, int nr_channels, int nr_stations,
    idg::Array2D<idg::UVWCoordinate<float>> &uvw,
    idg::Array1D<float> &wavenumbers,
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities,
    idg::Array2D<float> &spheroidal,
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> &aterms,
    idg::Array1D<idg::Metadata> &metadata,
    idg::Array4D<std::complex<float>> &subgrids, const T *func,
    int num_threads) {

  std::vector<int> dim = {nr_subgrids, num_threads};

  idg::UVWCoordinate<float> *d_uvw;
  float *d_wavenumbers, *d_spheroidal;
  float2 *d_visibilities, *d_aterms, *d_subgrids;
  idg::Metadata *d_metadata;

  cudaCheck(cudaMalloc(&d_uvw, uvw.bytes()));
  cudaCheck(cudaMalloc(&d_wavenumbers, wavenumbers.bytes()));
  cudaCheck(cudaMalloc(&d_spheroidal, spheroidal.bytes()));
  cudaCheck(cudaMalloc(&d_visibilities, visibilities.bytes()));
  cudaCheck(cudaMalloc(&d_aterms, aterms.bytes()));
  cudaCheck(cudaMalloc(&d_subgrids, subgrids.bytes()));
  cudaCheck(cudaMalloc(&d_metadata, metadata.bytes()));

  cudaMemcpy(d_uvw, uvw.data(), uvw.bytes(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wavenumbers, wavenumbers.data(), wavenumbers.bytes(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_spheroidal, spheroidal.data(), spheroidal.bytes(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_visibilities, visibilities.data(), visibilities.bytes(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_aterms, aterms.data(), aterms.bytes(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_metadata, metadata.data(), metadata.bytes(),
             cudaMemcpyHostToDevice);

  void *args[] = {
      &grid_size,      &subgrid_size, &image_size, &w_step_in_lambda,
      &nr_channels,    &nr_stations,  &d_uvw,      &d_wavenumbers,
      &d_visibilities, &d_spheroidal, &d_aterms,   &d_metadata,
      &d_subgrids};

  c_run_kernel((void *)func, dim3(dim[0]), dim3(dim[1]), args);

  cudaMemcpy(subgrids.data(), d_subgrids, subgrids.bytes(),
             cudaMemcpyDeviceToHost);

  cudaCheck(cudaFree(d_uvw));
  cudaCheck(cudaFree(d_wavenumbers));
  cudaCheck(cudaFree(d_spheroidal));
  cudaCheck(cudaFree(d_visibilities));
  cudaCheck(cudaFree(d_aterms));
  cudaCheck(cudaFree(d_metadata));
  cudaCheck(cudaFree(d_subgrids));
}

template <typename T>
void p_run_degridder(const T *func, std::string func_name, int num_threads) {

  float image_size = IMAGE_SIZE;
  float w_step_in_lambda = W_STEP;

  int nr_correlations = get_env_var("NR_CORRELATIONS", 4);
  int grid_size = get_env_var("GRID_SIZE", 1024);
  int subgrid_size = get_env_var("SUBGRID_SIZE", 32);
  int nr_stations = get_env_var("NR_STATIONS", 50);
  int nr_timeslots = get_env_var("NR_TIMESLOTS", 20);
  int nr_timesteps = get_env_var("NR_TIMESTEPS_SUBGRID", 128);
  int nr_channels = get_env_var("NR_CHANNELS", 16);

  int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;
  int nr_subgrids = nr_baselines * nr_timeslots;
  int total_nr_timesteps = nr_subgrids * nr_timesteps;

  std::vector<int> dim = {nr_subgrids, num_threads};

  print_parameters(nr_stations, nr_channels, nr_timesteps, nr_correlations,
                   nr_timeslots, image_size, grid_size, subgrid_size,
                   w_step_in_lambda, nr_baselines, nr_subgrids,
                   total_nr_timesteps);

  auto gflops =
      1e-9 * flops_gridder(nr_channels, total_nr_timesteps, nr_subgrids,
                           subgrid_size, nr_correlations);
  auto gbytes =
      1e-9 * bytes_gridder(nr_channels, total_nr_timesteps, nr_subgrids,
                           subgrid_size, nr_correlations);
  auto mvis = 1e-6 * total_nr_timesteps * nr_channels;

  idg::Array1D<idg::Metadata> metadata(nr_subgrids);

  idg::UVWCoordinate<float> *d_uvw;
  float *d_wavenumbers, *d_spheroidal;
  float2 *d_visibilities, *d_aterms, *d_subgrids;
  idg::Array1D<idg::Baseline> baselines(nr_baselines);
  idg::Metadata *d_metadata;

  initialize_baselines(nr_stations, baselines);
  initialize_metadata(grid_size, nr_timeslots, nr_timesteps, baselines,
                      metadata);

  cudaCheck(cudaMalloc(&d_uvw,
                       3 * nr_subgrids * nr_timesteps * sizeof(float)));
  cudaCheck(cudaMalloc(&d_wavenumbers, nr_channels * sizeof(float)));
  cudaCheck(
      cudaMalloc(&d_spheroidal, subgrid_size * subgrid_size * sizeof(float)));
  cudaCheck(cudaMalloc(&d_visibilities, nr_subgrids * nr_timesteps *
                                            nr_channels * nr_correlations * sizeof(float2)));
  cudaCheck(cudaMalloc(&d_aterms, nr_timeslots * nr_stations * subgrid_size *
                                      subgrid_size * nr_correlations * sizeof(float2)));
  cudaCheck(cudaMalloc(&d_subgrids, nr_subgrids * nr_correlations *
                                        subgrid_size * subgrid_size *
                                        sizeof(float2)));
  cudaCheck(cudaMalloc(&d_metadata, metadata.bytes()));

  cudaMemcpy(d_metadata, metadata.data(), metadata.bytes(),
             cudaMemcpyHostToDevice);

  void *args[] = {
      &grid_size,      &subgrid_size, &image_size, &w_step_in_lambda,
      &nr_channels,    &nr_stations,  &d_uvw,      &d_wavenumbers,
      &d_visibilities, &d_spheroidal, &d_aterms,   &d_metadata,
      &d_subgrids};

  p_run_kernel((void *)func, dim3(dim[0]), dim3(dim[1]), args, func_name,
               gflops, gbytes, mvis);

  cudaCheck(cudaFree(d_uvw));
  cudaCheck(cudaFree(d_wavenumbers));
  cudaCheck(cudaFree(d_spheroidal));
  cudaCheck(cudaFree(d_visibilities));
  cudaCheck(cudaFree(d_aterms));
  cudaCheck(cudaFree(d_metadata));
  cudaCheck(cudaFree(d_subgrids));
}

template <typename T>
void c_run_degridder(
    int nr_subgrids, int grid_size, int subgrid_size, float image_size,
    float w_step_in_lambda, int nr_channels, int nr_stations,
    idg::Array2D<idg::UVWCoordinate<float>> &uvw,
    idg::Array1D<float> &wavenumbers,
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities,
    idg::Array2D<float> &spheroidal,
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> &aterms,
    idg::Array1D<idg::Metadata> &metadata,
    idg::Array4D<std::complex<float>> &subgrids, const T *func,
    int num_threads) {

  std::vector<int> dim = {nr_subgrids, num_threads};

  idg::UVWCoordinate<float> *d_uvw;
  float *d_wavenumbers, *d_spheroidal;
  float2 *d_visibilities, *d_aterms, *d_subgrids;
  idg::Metadata *d_metadata;

  cudaCheck(cudaMalloc(&d_uvw, uvw.bytes()));
  cudaCheck(cudaMalloc(&d_wavenumbers, wavenumbers.bytes()));
  cudaCheck(cudaMalloc(&d_spheroidal, spheroidal.bytes()));
  cudaCheck(cudaMalloc(&d_visibilities, visibilities.bytes()));
  cudaCheck(cudaMalloc(&d_aterms, aterms.bytes()));
  cudaCheck(cudaMalloc(&d_subgrids, subgrids.bytes()));
  cudaCheck(cudaMalloc(&d_metadata, metadata.bytes()));

  cudaMemcpy(d_uvw, uvw.data(), uvw.bytes(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_wavenumbers, wavenumbers.data(), wavenumbers.bytes(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_spheroidal, spheroidal.data(), spheroidal.bytes(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_aterms, aterms.data(), aterms.bytes(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_metadata, metadata.data(), metadata.bytes(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_subgrids, subgrids.data(), subgrids.bytes(),
             cudaMemcpyHostToDevice);

  void *args[] = {
      &grid_size,      &subgrid_size, &image_size, &w_step_in_lambda,
      &nr_channels,    &nr_stations,  &d_uvw,      &d_wavenumbers,
      &d_visibilities, &d_spheroidal, &d_aterms,   &d_metadata,
      &d_subgrids};

  c_run_kernel((void *)func, dim3(dim[0]), dim3(dim[1]), args);

  cudaMemcpy(visibilities.data(), d_visibilities, visibilities.bytes(),
             cudaMemcpyDeviceToHost);

  cudaCheck(cudaFree(d_uvw));
  cudaCheck(cudaFree(d_wavenumbers));
  cudaCheck(cudaFree(d_spheroidal));
  cudaCheck(cudaFree(d_visibilities));
  cudaCheck(cudaFree(d_aterms));
  cudaCheck(cudaFree(d_metadata));
  cudaCheck(cudaFree(d_subgrids));
}

} // namespace cuda