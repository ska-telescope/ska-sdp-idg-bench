#include "print.hpp"

void print_parameters(int nr_stations, int nr_channels, int nr_timesteps,
                      int nr_correlations, int nr_timeslots, float image_size,
                      int grid_size, int subgrid_size, float w_step,
                      int nr_baselines, int nr_subgrids,
                      int total_nr_timesteps) {
  const int fw1 = 30;
  const int fw2 = 10;

  std::cout << "-----------" << std::endl;
  std::cout << "PARAMETERS:" << std::endl;

  std::cout << std::setw(fw1) << std::left << "Number of stations"
            << "== " << std::setw(fw2) << std::right << nr_stations
            << std::endl;

  std::cout << std::setw(fw1) << std::left << "Number of channels"
            << "== " << std::setw(fw2) << std::right << nr_channels
            << std::endl;

  std::cout << std::setw(fw1) << std::left << "Number of timesteps"
            << "== " << std::setw(fw2) << std::right << nr_timesteps
            << std::endl;

  std::cout << std::setw(fw1) << std::left << "Number of correlations"
            << "== " << std::setw(fw2) << std::right << nr_correlations
            << std::endl;

  std::cout << std::setw(fw1) << std::left << "Number of timeslots"
            << "== " << std::setw(fw2) << std::right << nr_timeslots
            << std::endl;

  std::cout << std::setw(fw1) << std::left << "Imagesize"
            << "== " << std::setw(fw2) << std::right << image_size << std::endl;

  std::cout << std::setw(fw1) << std::left << "Grid size"
            << "== " << std::setw(fw2) << std::right << grid_size << std::endl;

  std::cout << std::setw(fw1) << std::left << "Subgrid size"
            << "== " << std::setw(fw2) << std::right << subgrid_size
            << std::endl;

  std::cout << std::setw(fw1) << std::left << "W step size"
            << "== " << std::setw(fw2) << std::right << w_step << std::endl;

  std::cout << std::setw(fw1) << std::left << "Number of baselines"
            << "== " << std::setw(fw2) << std::right << nr_baselines
            << std::endl;

  std::cout << std::setw(fw1) << std::left << "Number of subgrids"
            << "== " << std::setw(fw2) << std::right << nr_subgrids
            << std::endl;

  std::cout << std::setw(fw1) << std::left << "Total number of timesteps"
            << "== " << std::setw(fw2) << std::right << total_nr_timesteps
            << std::endl;

  std::cout << "-----------" << std::endl;
}

void print_subgrid_diff(idg::Array4D<std::complex<float>> &subgrids1,
                        idg::Array4D<std::complex<float>> &subgrids2,
                        unsigned i) {
  format_saver(&std::cout);

  assert(subgrids1.bytes() == subgrids2.bytes());

  unsigned nr_correlations = subgrids1.get_z_dim();
  unsigned width = subgrids1.get_x_dim();
  unsigned height = subgrids1.get_y_dim();

  std::cout << ">>> subgrid: " << i << std::endl;
  for (unsigned c = 0;
       c < std::min(nr_correlations, (unsigned)PRINT_MAX_NR_CORRELATIONS);
       c++) {
    std::cout << ">> correlation: " << c << std::endl;

    for (unsigned y = 0; y < std::min(height, (unsigned)PRINT_MAX_HEIGHT);
         y++) {
      for (unsigned x = 0; x < std::min(width, (unsigned)PRINT_MAX_WIDTH);
           x++) {
        std::complex<float> pixel1 = subgrids1(i, c, y, x);
        std::complex<float> pixel2 = subgrids2(i, c, y, x);
        std::cout << std::fixed;
        std::cout << std::setprecision(1);
        std::cout << pixel1 - pixel2 << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void print_subgrid(idg::Array4D<std::complex<float>> &subgrids, unsigned i) {
  format_saver(&std::cout);

  unsigned nr_correlations = subgrids.get_z_dim();
  unsigned width = subgrids.get_x_dim();
  unsigned height = subgrids.get_y_dim();

  std::cout << ">>> subgrid: " << i << std::endl;
  for (unsigned c = 0;
       c < std::min(nr_correlations, (unsigned)PRINT_MAX_NR_CORRELATIONS);
       c++) {
    std::cout << ">> correlation: " << c << std::endl;

    for (unsigned y = 0; y < std::min(height, (unsigned)PRINT_MAX_HEIGHT);
         y++) {
      for (unsigned x = 0; x < std::min(width, (unsigned)PRINT_MAX_WIDTH);
           x++) {
        std::complex<float> pixel = subgrids(i, c, y, x);
        std::cout << std::fixed;
        std::cout << std::setprecision(1);
        std::cout << pixel << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void print_visibilities(
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities,
    unsigned i) {
  format_saver(&std::cout);

  unsigned int nr_timesteps = visibilities.get_y_dim();
  unsigned int nr_channels = visibilities.get_x_dim();

  std::cout << ">>> baseline: " << i << std::endl;
  for (unsigned time = 0;
       time < std::min(nr_timesteps, (unsigned)PRINT_MAX_NR_TIMESTEPS);
       time++) {
    std::cout << "time: " << time << std::endl;
    for (unsigned chan = 0;
         chan < std::min(nr_channels, (unsigned)PRINT_MAX_NR_CHANNELS);
         chan++) {
      idg::Visibility<std::complex<float>> visibility =
          visibilities(i, time, chan);
      std::cout << std::fixed;
      std::cout << std::setprecision(1);
      std::cout << "chan " << chan << ": ";
      std::cout << visibility.xx << ", ";
      std::cout << visibility.xy << ", ";
      std::cout << visibility.yx << ", ";
      std::cout << visibility.yy << std::endl;
    }
  }
}

void print_visibilities_diff(
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities1,
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities2,
    unsigned i) {
  format_saver(&std::cout);

  assert(visibilities1.bytes() == visibilities2.bytes());

  unsigned int nr_timesteps = visibilities1.get_y_dim();
  unsigned int nr_channels = visibilities1.get_x_dim();

  std::cout << ">>> baseline: " << i << std::endl;
  for (unsigned time = 0;
       time < std::min(nr_timesteps, (unsigned)PRINT_MAX_NR_TIMESTEPS);
       time++) {
    std::cout << "time: " << time << std::endl;
    for (unsigned chan = 0;
         chan < std::min(nr_channels, (unsigned)PRINT_MAX_NR_CHANNELS);
         chan++) {
      idg::Visibility<std::complex<float>> visibility1 =
          visibilities1(i, time, chan);
      idg::Visibility<std::complex<float>> visibility2 =
          visibilities2(i, time, chan);
      std::cout << std::fixed;
      std::cout << std::setprecision(1);
      std::cout << "chan " << chan << ": ";
      std::cout << visibility1.xx - visibility2.xx << ", ";
      std::cout << visibility1.xy - visibility2.xy << ", ";
      std::cout << visibility1.yx - visibility2.yx << ", ";
      std::cout << visibility1.yy - visibility2.yy << std::endl;
    }
  }
}
