#pragma once

#include <cassert>

#include "parameters.hpp"
#include "types.hpp"

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
