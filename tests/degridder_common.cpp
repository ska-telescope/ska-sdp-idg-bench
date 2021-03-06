#include <iostream>

#include "lib-cpu.hpp"

#if defined(BUILD_CUDA)
#include "lib-cuda.hpp"
#elif defined(BUILD_HIP)
#include "lib-hip.hpp"
#endif

#include "test_util.hpp"

#if defined(BUILD_CUDA)
namespace cuda
#elif defined(BUILD_HIP)
namespace hip
#endif
{
void p_run_degridder();

void c_run_degridder(
    int nr_subgrids, int grid_size, int subgrid_size, float image_size,
    float w_step_in_lambda, int nr_channels, int nr_stations,
    idg::Array2D<idg::UVWCoordinate<float>> &uvw,
    idg::Array1D<float> &wavenumbers,
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities,
    idg::Array2D<float> &spheroidal,
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> &aterms,
    idg::Array1D<idg::Metadata> &metadata,
    idg::Array4D<std::complex<float>> &subgrids);
} // namespace cuda

void run_performance() {
#if defined(BUILD_CUDA)
  cuda::print_device_info();
  cuda::p_run_degridder();
#elif defined(BUILD_HIP)
  hip::print_device_info();
  hip::p_run_degridder();
#endif
}

void run_correctness() {
  std::cout << ">>> Correctness IDG-Degridder test" << std::endl;
#if defined(BUILD_CUDA)
  cuda::print_device_info();
  cuda::print_benchmark();
#elif defined(BUILD_HIP)
  hip::print_device_info();
  hip::print_benchmark();
#endif

  // IDG parameters
  int nr_correlations = get_env_var("NR_CORRELATIONS", 4);
  int grid_size = get_env_var("GRID_SIZE", 1024);
  int subgrid_size = get_env_var("SUBGRID_SIZE", 32);
  int nr_stations = get_env_var("NR_STATIONS", 2);
  int nr_timeslots = get_env_var("NR_TIMESLOTS", 2);
  int nr_timesteps = get_env_var("NR_TIMESTEPS_SUBGRID", 128);
  int nr_channels = get_env_var("NR_CHANNELS", 16);

  int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;
  int nr_subgrids = nr_baselines * nr_timeslots;
  int total_nr_timesteps = nr_subgrids * nr_timesteps;

  print_parameters(nr_stations, nr_channels, nr_timesteps, nr_correlations,
                   nr_timeslots, IMAGE_SIZE, grid_size, subgrid_size, W_STEP,
                   nr_baselines, nr_subgrids, total_nr_timesteps);

  // Allocate data structures on host
  std::cout << ">>> Allocate data structures on host" << std::endl;
  idg::Array2D<idg::UVWCoordinate<float>> uvw(nr_subgrids, nr_timesteps);
  idg::Array3D<idg::Visibility<std::complex<float>>> cpu_visibilities(
      nr_subgrids, nr_timesteps, nr_channels);
  idg::Array3D<idg::Visibility<std::complex<float>>> gpu_visibilities(
      nr_subgrids, nr_timesteps, nr_channels);
  idg::Array1D<idg::Baseline> baselines(nr_baselines);
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms(
      nr_timeslots, nr_stations, subgrid_size, subgrid_size);
  idg::Array1D<float> frequencies(nr_channels);
  idg::Array1D<float> wavenumbers(nr_channels);
  idg::Array2D<float> spheroidal(subgrid_size, subgrid_size);
  idg::Array4D<std::complex<float>> subgrids(nr_subgrids, nr_correlations,
                                             subgrid_size, subgrid_size);
  idg::Array1D<idg::Metadata> metadata(nr_subgrids);

  // Initialize random number generator
  srand(0);

  // Initialize data structures
  std::cout << ">>> Initialize data structures on host" << std::endl;
  initialize_uvw(grid_size, uvw);
  initialize_frequencies(frequencies);
  initialize_wavenumbers(frequencies, wavenumbers);
  initialize_baselines(nr_stations, baselines);
  initialize_spheroidal(spheroidal);
  initialize_aterms(spheroidal, aterms);
  initialize_subgrids(subgrids);
  initialize_metadata(grid_size, nr_timeslots, nr_timesteps, baselines,
                      metadata);

  // copy cpu to gpu vector
  std::cout << ">>> Run on cpu" << std::endl;
  cpu::c_run_degridder_reference(nr_subgrids, grid_size, subgrid_size,
                                 IMAGE_SIZE, W_STEP, nr_channels, nr_stations,
                                 uvw, wavenumbers, cpu_visibilities, spheroidal,
                                 aterms, metadata, subgrids);
  std::cout << ">>> Run on gpu" << std::endl;
#if defined(BUILD_CUDA)
  cuda::c_run_degridder(nr_subgrids, grid_size, subgrid_size, IMAGE_SIZE,
                        W_STEP, nr_channels, nr_stations, uvw, wavenumbers,
                        gpu_visibilities, spheroidal, aterms, metadata,
                        subgrids);
#elif defined(BUILD_HIP)
  hip::c_run_degridder(nr_subgrids, grid_size, subgrid_size, IMAGE_SIZE, W_STEP,
                       nr_channels, nr_stations, uvw, wavenumbers,
                       gpu_visibilities, spheroidal, aterms, metadata,
                       subgrids);
#endif

  std::cout << ">>> Checking" << std::endl;
  // check algorithm correctness (check visibilities values)
  compare_visibilities(cpu_visibilities, gpu_visibilities);
}

int main(int argc, char *argv[]) {

  if (argc == 1) {
    run_performance();
    return EXIT_SUCCESS;
  } else if (argc == 2) {
    if (std::strncmp(argv[1], "-c", 2) == 0) {
      run_correctness();
      return EXIT_SUCCESS;
    }
  }

  std::cerr << "Usage: " << argv[0] << " [-c]" << std::endl;
  return EXIT_FAILURE;
}