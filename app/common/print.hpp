#pragma once

#include <cassert>
#include <iomanip>
#include <iostream>

#include "types.hpp"

#ifndef PRINT_MAX_NR_CORRELATIONS
#define PRINT_MAX_NR_CORRELATIONS 4
#endif

#ifndef PRINT_MAX_HEIGHT
#define PRINT_MAX_HEIGHT 3
#endif

#ifndef PRINT_MAX_WIDTH
#define PRINT_MAX_WIDTH 3
#endif

#ifndef PRINT_MAX_NR_TIMESTEPS
#define PRINT_MAX_NR_TIMESTEPS 3
#endif

#ifndef PRINT_MAX_NR_CHANNELS
#define PRINT_MAX_NR_CHANNELS 4
#endif

class format_saver {
public:
  format_saver(std::ostream *s)
      : s_(s), flags(s->flags()), precision(s->precision()) {}

  ~format_saver() {
    s_->flags(flags);
    s_->precision(precision);
  }

private:
  std::ostream *s_;
  std::ios_base::fmtflags flags;
  std::streamsize precision;
};

void print_parameters(int nr_stations, int nr_channels, int nr_timesteps,
                      int nr_correlations, int nr_timeslots, float image_size,
                      int grid_size, int subgrid_size, float w_step,
                      int nr_baselines, int nr_subgrids,
                      int total_nr_timesteps);

void print_subgrid_diff(idg::Array4D<std::complex<float>> &subgrids1,
                        idg::Array4D<std::complex<float>> &subgrids2,
                        unsigned i);

void print_subgrid(idg::Array4D<std::complex<float>> &subgrids, unsigned i);

void print_visibilities(
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities,
    unsigned i);

void print_visibilities_diff(
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities1,
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities2,
    unsigned i);
