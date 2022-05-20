#include "util.hpp"

namespace hip {

#define hipCheck(ans)                                                         \
  { hipAssert((ans), __FILE__, __LINE__); }
inline void hipAssert(hipError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != hipSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", hipGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

std::string get_device_name() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  std::string device_name = prop.name;
  std::replace(device_name.begin(), device_name.end(), ' ', '_');
  return device_name;
}

void print_device_info() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
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

std::vector<int> get_launch_kernel_dimensions() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  return {prop.multiProcessorCount, prop.maxThreadsPerBlock};
}

int get_cu_nr() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  return prop.multiProcessorCount;
}

int get_max_threads() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  return prop.maxThreadsPerBlock;
}

int get_gmem_size() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  return prop.totalGlobalMem;
}

int get_cu_freq() {
  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  return prop.clockRate;
}

void print_dimensions(dim3 gridDim, dim3 blockDim) {
  std::cout << "Dimensions: (";
  std::cout << gridDim.x << "," << gridDim.y << "," << gridDim.z;
  std::cout << ") - (";
  std::cout << blockDim.x << "," << blockDim.y << "," << blockDim.z;
  std::cout << ")" << std::endl << std::endl;
}

void p_run_kernel(const void *func, dim3 gridDim, dim3 blockDim, void **args,
                  std::string func_name, double gflops,
                  double gbytes, double mvis) {
  float seconds;
  double avg_time, avg_joules = 0;
  std::vector<double> ex_joules, ex_time;
#ifdef ENABLE_POWERSENSOR
  double joules;
  std::unique_ptr<powersensor::PowerSensor> powersensor(
      powersensor::nvml::NVMLPowerSensor::create());
  powersensor::State start, end;
  hipEvent_t stop;
#else
  hipEvent_t start, stop;
  hipCheck(hipEventCreate(&start));
#endif
  hipCheck(hipEventCreate(&stop));

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
    if (i == nr_warm_up_runs) {
      start = powersensor->read();
    }
#else
    hipCheck(hipEventRecord(start));
#endif

    hipLaunchKernel(func, gridDim, blockDim, args, 0, 0);
    hipCheck(hipEventRecord(stop));
    hipCheck(hipEventSynchronize(stop));

    if (nr_iterations > nr_warm_up_runs) {
#ifdef ENABLE_POWERSENSOR
    end = powersensor->read();
    bool is_last_iteration = i == (nr_iterations - 1);
    double tot_time = powersensor->seconds(start, end);
    double min_time = 5; // run for at least 5 seconds
    if (is_last_iteration && tot_time < min_time) {
      nr_iterations++;
    }
#else
      float milliseconds = 0;
      hipCheck(hipEventElapsedTime(&milliseconds, start, stop));
      seconds = milliseconds * 1e-3;
      ex_time.push_back(seconds);
#endif
    }
  }

    hipDeviceSynchronize();
#ifdef ENABLE_POWERSENSOR
    end = powersensor->read();
    seconds = powersensor->seconds(start, end);
    joules = powersensor->Joules(start, end);
    avg_joules = joules / nr_iterations;
    avg_time = seconds / nr_iterations;
#else
  avg_time =
      std::accumulate(ex_time.begin(), ex_time.end(), 0.0) / ex_time.size();
#endif

  report(func_name, avg_time, gflops, gbytes, mvis, avg_joules);
  report_csv(func_name, get_device_name(), "-hip.csv", avg_time, gflops,
             gbytes, mvis, avg_joules);
}

void c_run_kernel(const void *func, dim3 gridDim, dim3 blockDim, void **args) {

#ifdef DEBUG
  print_device_info();
  print_dimensions(gridDim, blockDim);
#endif
  hipLaunchKernel(func, gridDim, blockDim, args, 0, 0);
}

void p_run_gridder_(const void *func, std::string func_name, int num_threads) {

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

  hipCheck(hipMalloc(&d_uvw,
                       3 * nr_subgrids * nr_timesteps * sizeof(float)));
  hipCheck(hipMalloc(&d_wavenumbers, nr_channels * sizeof(float)));
  hipCheck(
      hipMalloc(&d_spheroidal, subgrid_size * subgrid_size * sizeof(float)));
  hipCheck(hipMalloc(&d_visibilities, nr_subgrids * nr_timesteps *
                                            nr_channels * nr_correlations * sizeof(float2)));
  hipCheck(hipMalloc(&d_aterms, nr_timeslots * nr_stations * subgrid_size *
                                      subgrid_size * nr_correlations * sizeof(float2)));
  hipCheck(hipMalloc(&d_subgrids, nr_subgrids * nr_correlations *
                                        subgrid_size * subgrid_size *
                                        sizeof(float2)));
  hipCheck(hipMalloc(&d_metadata, metadata.bytes()));

  hipMemcpy(d_metadata, metadata.data(), metadata.bytes(),
             hipMemcpyHostToDevice);

  void *args[] = {
      &grid_size,      &subgrid_size, &image_size, &w_step_in_lambda,
      &nr_channels,    &nr_stations,  &d_uvw,      &d_wavenumbers,
      &d_visibilities, &d_spheroidal, &d_aterms,   &d_metadata,
      &d_subgrids};

  p_run_kernel((void *)func, dim3(dim[0]), dim3(dim[1]), args, func_name,
               gflops, gbytes, mvis);

  hipCheck(hipFree(d_uvw));
  hipCheck(hipFree(d_wavenumbers));
  hipCheck(hipFree(d_spheroidal));
  hipCheck(hipFree(d_visibilities));
  hipCheck(hipFree(d_aterms));
  hipCheck(hipFree(d_metadata));
  hipCheck(hipFree(d_subgrids));
}

void c_run_gridder_(
    int nr_subgrids, int grid_size, int subgrid_size, float image_size,
    float w_step_in_lambda, int nr_channels, int nr_stations,
    idg::Array2D<idg::UVWCoordinate<float>> &uvw,
    idg::Array1D<float> &wavenumbers,
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities,
    idg::Array2D<float> &spheroidal,
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> &aterms,
    idg::Array1D<idg::Metadata> &metadata,
    idg::Array4D<std::complex<float>> &subgrids, const void *func,
    int num_threads) {

  std::vector<int> dim = {nr_subgrids, num_threads};

  idg::UVWCoordinate<float> *d_uvw;
  float *d_wavenumbers, *d_spheroidal;
  float2 *d_visibilities, *d_aterms, *d_subgrids;
  idg::Metadata *d_metadata;

  hipCheck(hipMalloc(&d_uvw, uvw.bytes()));
  hipCheck(hipMalloc(&d_wavenumbers, wavenumbers.bytes()));
  hipCheck(hipMalloc(&d_spheroidal, spheroidal.bytes()));
  hipCheck(hipMalloc(&d_visibilities, visibilities.bytes()));
  hipCheck(hipMalloc(&d_aterms, aterms.bytes()));
  hipCheck(hipMalloc(&d_subgrids, subgrids.bytes()));
  hipCheck(hipMalloc(&d_metadata, metadata.bytes()));

  hipMemcpy(d_uvw, uvw.data(), uvw.bytes(), hipMemcpyHostToDevice);
  hipMemcpy(d_wavenumbers, wavenumbers.data(), wavenumbers.bytes(),
             hipMemcpyHostToDevice);
  hipMemcpy(d_spheroidal, spheroidal.data(), spheroidal.bytes(),
             hipMemcpyHostToDevice);
  hipMemcpy(d_visibilities, visibilities.data(), visibilities.bytes(),
             hipMemcpyHostToDevice);
  hipMemcpy(d_aterms, aterms.data(), aterms.bytes(), hipMemcpyHostToDevice);
  hipMemcpy(d_metadata, metadata.data(), metadata.bytes(),
             hipMemcpyHostToDevice);

  void *args[] = {
      &grid_size,      &subgrid_size, &image_size, &w_step_in_lambda,
      &nr_channels,    &nr_stations,  &d_uvw,      &d_wavenumbers,
      &d_visibilities, &d_spheroidal, &d_aterms,   &d_metadata,
      &d_subgrids};

  c_run_kernel((void *)func, dim3(dim[0]), dim3(dim[1]), args);

  hipMemcpy(subgrids.data(), d_subgrids, subgrids.bytes(),
             hipMemcpyDeviceToHost);

  hipCheck(hipFree(d_uvw));
  hipCheck(hipFree(d_wavenumbers));
  hipCheck(hipFree(d_spheroidal));
  hipCheck(hipFree(d_visibilities));
  hipCheck(hipFree(d_aterms));
  hipCheck(hipFree(d_metadata));
  hipCheck(hipFree(d_subgrids));
}

void p_run_degridder_(const void *func, std::string func_name, int num_threads) {

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

  hipCheck(hipMalloc(&d_uvw,
                       3 * nr_subgrids * nr_timesteps * sizeof(float)));
  hipCheck(hipMalloc(&d_wavenumbers, nr_channels * sizeof(float)));
  hipCheck(
      hipMalloc(&d_spheroidal, subgrid_size * subgrid_size * sizeof(float)));
  hipCheck(hipMalloc(&d_visibilities, nr_subgrids * nr_timesteps *
                                            nr_channels * nr_correlations * sizeof(float2)));
  hipCheck(hipMalloc(&d_aterms, nr_timeslots * nr_stations * subgrid_size *
                                      subgrid_size * nr_correlations * sizeof(float2)));
  hipCheck(hipMalloc(&d_subgrids, nr_subgrids * nr_correlations *
                                        subgrid_size * subgrid_size *
                                        sizeof(float2)));
  hipCheck(hipMalloc(&d_metadata, metadata.bytes()));

  hipMemcpy(d_metadata, metadata.data(), metadata.bytes(),
             hipMemcpyHostToDevice);

  void *args[] = {
      &grid_size,      &subgrid_size, &image_size, &w_step_in_lambda,
      &nr_channels,    &nr_stations,  &d_uvw,      &d_wavenumbers,
      &d_visibilities, &d_spheroidal, &d_aterms,   &d_metadata,
      &d_subgrids};

  p_run_kernel((void *)func, dim3(dim[0]), dim3(dim[1]), args, func_name,
               gflops, gbytes, mvis);

  hipCheck(hipFree(d_uvw));
  hipCheck(hipFree(d_wavenumbers));
  hipCheck(hipFree(d_spheroidal));
  hipCheck(hipFree(d_visibilities));
  hipCheck(hipFree(d_aterms));
  hipCheck(hipFree(d_metadata));
  hipCheck(hipFree(d_subgrids));
}

void c_run_degridder_(
    int nr_subgrids, int grid_size, int subgrid_size, float image_size,
    float w_step_in_lambda, int nr_channels, int nr_stations,
    idg::Array2D<idg::UVWCoordinate<float>> &uvw,
    idg::Array1D<float> &wavenumbers,
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities,
    idg::Array2D<float> &spheroidal,
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> &aterms,
    idg::Array1D<idg::Metadata> &metadata,
    idg::Array4D<std::complex<float>> &subgrids, const void *func,
    int num_threads) {

  std::vector<int> dim = {nr_subgrids, num_threads};

  idg::UVWCoordinate<float> *d_uvw;
  float *d_wavenumbers, *d_spheroidal;
  float2 *d_visibilities, *d_aterms, *d_subgrids;
  idg::Metadata *d_metadata;

  hipCheck(hipMalloc(&d_uvw, uvw.bytes()));
  hipCheck(hipMalloc(&d_wavenumbers, wavenumbers.bytes()));
  hipCheck(hipMalloc(&d_spheroidal, spheroidal.bytes()));
  hipCheck(hipMalloc(&d_visibilities, visibilities.bytes()));
  hipCheck(hipMalloc(&d_aterms, aterms.bytes()));
  hipCheck(hipMalloc(&d_subgrids, subgrids.bytes()));
  hipCheck(hipMalloc(&d_metadata, metadata.bytes()));

  hipMemcpy(d_uvw, uvw.data(), uvw.bytes(), hipMemcpyHostToDevice);
  hipMemcpy(d_wavenumbers, wavenumbers.data(), wavenumbers.bytes(),
             hipMemcpyHostToDevice);
  hipMemcpy(d_spheroidal, spheroidal.data(), spheroidal.bytes(),
             hipMemcpyHostToDevice);
  hipMemcpy(d_aterms, aterms.data(), aterms.bytes(), hipMemcpyHostToDevice);
  hipMemcpy(d_metadata, metadata.data(), metadata.bytes(),
             hipMemcpyHostToDevice);
  hipMemcpy(d_subgrids, subgrids.data(), subgrids.bytes(),
             hipMemcpyHostToDevice);

  void *args[] = {
      &grid_size,      &subgrid_size, &image_size, &w_step_in_lambda,
      &nr_channels,    &nr_stations,  &d_uvw,      &d_wavenumbers,
      &d_visibilities, &d_spheroidal, &d_aterms,   &d_metadata,
      &d_subgrids};

  c_run_kernel((void *)func, dim3(dim[0]), dim3(dim[1]), args);

  hipMemcpy(visibilities.data(), d_visibilities, visibilities.bytes(),
             hipMemcpyDeviceToHost);

  hipCheck(hipFree(d_uvw));
  hipCheck(hipFree(d_wavenumbers));
  hipCheck(hipFree(d_spheroidal));
  hipCheck(hipFree(d_visibilities));
  hipCheck(hipFree(d_aterms));
  hipCheck(hipFree(d_metadata));
  hipCheck(hipFree(d_subgrids));
}

void print_benchmark() { std::cout << ">>> hip IDG BENCHMARK" << std::endl; }

} // namespace hip