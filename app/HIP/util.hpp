#pragma once

#include "lib-common.hpp"
#include <hip/hip_runtime.h>

#if defined(ENABLE_POWERSENSOR) && defined(__HIP_PLATFORM_NVIDIA__)
#include <powersensor/NVMLPowerSensor.h>
#elif defined(ENABLE_POWERSENSOR) && defined(__HIP_PLATFORM_AMD__)
#include <powersensor/ROCMPowerSensor.h>
#endif


namespace hip {

std::string get_device_name();

void print_device_info();

std::vector<int> get_launch_kernel_dimensions();

int get_cu_nr();

int get_max_threads();

int get_gmem_size();

int get_cu_freq();

void print_dimensions(dim3 gridDim, dim3 blockDim);

void p_run_kernel(const void *func, dim3 gridDim, dim3 blockDim, void **args,
                  std::string func_name = "", double gflops = 0,
                  double gbytes = 0, double mvis = 0);

void c_run_kernel(const void *func, dim3 gridDim, dim3 blockDim, void **args);

void p_run_gridder_(const void *func, std::string func_name, int num_threads);

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
    int num_threads);

void p_run_degridder_(const void *func, std::string func_name, int num_threads);

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
    int num_threads);

void print_benchmark();

} // namespace hip