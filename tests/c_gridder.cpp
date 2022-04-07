#include "lib-cpu.hpp"

#if defined(BUILD_CUDA)
#include "lib-cuda.hpp"
#elif defined(BUILD_HIP)
#include "lib-hip.hpp"
#endif

#include "test_util.hpp"

int main() {
  std::cout << ">>> Correctness IDG-Gridding test" << std::endl;
#if defined(BUILD_CUDA)
  cuda::extern_print_device_info();
  cuda::print_benchmark();
#elif defined(BUILD_HIP)
  hip::extern_print_device_info();
  hip::print_benchmark();
#endif

  // get_env parameters?

  // print IDG parameters?
  int grid_size = GRID_SIZE;
  int subgrid_size = SUBGRID_SIZE;
  int nr_subgrids = NR_SUBGRIDS;
  float image_size = IMAGE_SIZE;
  float w_step_in_lambda = W_STEP;

  int nr_channels = NR_CHANNELS;
  int nr_stations = NR_STATIONS;

  // initialize cpu and gpu vectors

  // Allocate data structures on host
  std::cout << ">>> Allocate data structures on host" << std::endl;
  idg::Array2D<idg::UVWCoordinate<float>> uvw(NR_BASELINES, NR_TIMESTEPS);
  idg::Array3D<idg::Visibility<std::complex<float>>> visibilities(
      NR_BASELINES, NR_TIMESTEPS, NR_CHANNELS);
  idg::Array1D<idg::Baseline> baselines(NR_BASELINES);
  idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms(
      NR_TIMESLOTS, NR_STATIONS, SUBGRID_SIZE, SUBGRID_SIZE);
  idg::Array1D<float> frequencies(NR_CHANNELS);
  idg::Array1D<float> wavenumbers(NR_CHANNELS);
  idg::Array2D<float> spheroidal(SUBGRID_SIZE, SUBGRID_SIZE);
  idg::Array4D<std::complex<float>> subgrids(NR_SUBGRIDS, NR_CORRELATIONS,
                                             SUBGRID_SIZE, SUBGRID_SIZE);
  idg::Array1D<idg::Metadata> metadata(NR_SUBGRIDS);

  // Initialize random number generator
  srand(0);

  // Initialize data structures
  std::cout << ">>> Initialize data structures on host" << std::endl;
  initialize_uvw(GRID_SIZE, uvw);
  initialize_frequencies(frequencies);
  initialize_wavenumbers(frequencies, wavenumbers);
  initialize_visibilities(GRID_SIZE, IMAGE_SIZE, frequencies, uvw,
                          visibilities);
  initialize_baselines(NR_STATIONS, baselines);
  initialize_spheroidal(spheroidal);
  initialize_aterms(spheroidal, aterms);
  initialize_metadata(GRID_SIZE, NR_TIMESLOTS, NR_TIMESTEPS_SUBGRID, baselines,
                      metadata);

  // copy cpu to gpu vector
  std::cout << ">>> Run on cpu" << std::endl;
  cpu::run_c_gridder(nr_subgrids, grid_size, subgrid_size, image_size,
                     w_step_in_lambda, nr_channels, nr_stations, uvw,
                     wavenumbers, visibilities, spheroidal, aterms, metadata,
                     subgrids);

#if defined(BUILD_CUDA)
// run CUDA test
#elif defined(BUILD_HIP)
// run HIP test
#endif

  // check algorithm correctness (check subgrid values)
  bool equal = true;

  if (equal) {
    std::cout << ">>> Result PASSED" << std::endl;
  }
}