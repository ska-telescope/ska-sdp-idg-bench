#pragma once

#include <cassert>

#include "math.hpp"
#include "types.hpp"

#define GRID_SIZE 1024
#define NR_CORRELATIONS 4
#define SUBGRID_SIZE 32
#define IMAGE_SIZE 0.01f
#define W_STEP 0
#define NR_CHANNELS 16 // number of channels per subgrid
#define NR_STATIONS 10
#define NR_TIMESLOTS 2
#define NR_TIMESTEPS_SUBGRID 128 // number of timesteps per subgrid
#define NR_TIMESTEPS                                                           \
  (NR_TIMESTEPS_SUBGRID * NR_TIMESLOTS) // number of timesteps per baseline
#define NR_BASELINES ((NR_STATIONS * (NR_STATIONS - 1)) / 2)
#define NR_SUBGRIDS (NR_BASELINES * NR_TIMESLOTS)

void initialize_uvw(unsigned int grid_size,
                    idg::Array2D<idg::UVWCoordinate<float>> &uvw);

void initialize_frequencies(idg::Array1D<float> &frequencies);

void initialize_wavenumbers(const idg::Array1D<float> &frequencies,
                            idg::Array1D<float> &wavenumbers);

void initialize_visibilities(
    unsigned int grid_size, float image_size,
    const idg::Array1D<float> &frequencies,
    const idg::Array2D<idg::UVWCoordinate<float>> &uvw,
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities);

void initialize_baselines(unsigned int nr_stations,
                          idg::Array1D<idg::Baseline> &baselines);

void initialize_spheroidal(idg::Array2D<float> &spheroidal);

void initialize_aterms(
    const idg::Array2D<float> &spheroidal,
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> &aterms);

void initialize_metadata(unsigned int grid_size, unsigned int nr_timeslots,
                         unsigned int nr_timesteps_subgrid,
                         const idg::Array1D<idg::Baseline> &baselines,
                         idg::Array1D<idg::Metadata> &metadata);

void initialize_subgrids(idg::Array4D<std::complex<float>> &subgrids);

void initialize_uvw_offsets(unsigned int subgrid_size, unsigned int grid_size,
                            float image_size, float w_step,
                            const idg::Array1D<idg::Metadata> &metadata,
                            idg::Array2D<float> &uvw_offsets);

void initialize_lmn(float image_size, idg::Array3D<float> &lmn);
