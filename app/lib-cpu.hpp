#pragma once

#include "lib-common.hpp"

namespace cpu {
void c_run_vadd(std::vector<float> &a, std::vector<float> &b,
                std::vector<float> &c, int size);

void run_c_gridder(const int nr_subgrids, const int grid_size,
                   const int subgrid_size, const float image_size,
                   const float w_step_in_lambda, const int nr_channels,
                   const int nr_stations, const idg::Array2D<idg::UVWCoordinate<float>> &uvw,
                   idg::Array1D<float> &wavenumbers,
                   idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities,
                   idg::Array2D<float> &spheroidal, idg::Array4D<idg::Matrix2x2<std::complex<float>>>  &aterms,
                   idg::Array1D<idg::Metadata> &metadata,
                   idg::Array4D<std::complex<float>> &subgrids);
}