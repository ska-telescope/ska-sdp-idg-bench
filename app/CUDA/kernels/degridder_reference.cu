#include "../common/math.hpp"
#include "math.cuh"
#include "util.cuh"

// not possible to pass it as input
#define SUBGRID_SIZE 32

namespace cuda {

__global__ void kernel_degridder_reference(
    const int grid_size, int subgrid_size, float image_size,
    float w_step_in_lambda, int nr_channels, // channel_offset? for the macro?
    int nr_stations, idg::UVWCoordinate<float> *uvw, float *wavenumbers,
    float2 *visibilities, float *spheroidal, float2 *aterms,
    idg::Metadata *metadata, float2 *subgrids) {
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int tid = tidx + tidy * blockDim.x;
  int nr_threads = blockDim.x * blockDim.y;
  int s = blockIdx.x;

  // Find offset of first subgrid
  const idg::Metadata m_0 = metadata[0];
  const int baseline_offset_1 = m_0.baseline_offset;

  // Load metadata
  const idg::Metadata m = metadata[s];
  const int time_offset =
      (m.baseline_offset - baseline_offset_1) + m.time_offset;
  const int nr_timesteps = m.nr_timesteps;
  const int aterm_index = m.aterm_index;
  const int station1 = m.baseline.station1;
  const int station2 = m.baseline.station2;
  const int x_coordinate = m.coordinate.x;
  const int y_coordinate = m.coordinate.y;
  const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);
  // Storage
  float2 pixels[SUBGRID_SIZE][SUBGRID_SIZE][NR_CORRELATIONS];

  // Apply aterm to subgrid
  for (int y = 0; y < subgrid_size; y++) {
    for (int x = 0; x < subgrid_size; x++) {
      // Load aterm for station1
      int station1_index = (aterm_index * nr_stations + station1) *
                               subgrid_size * subgrid_size * NR_CORRELATIONS +
                           y * subgrid_size * NR_CORRELATIONS +
                           x * NR_CORRELATIONS;
      const float2 *aterm1_ptr = &aterms[station1_index];

      // Load aterm for station2
      int station2_index = (aterm_index * nr_stations + station2) *
                               subgrid_size * subgrid_size * NR_CORRELATIONS +
                           y * subgrid_size * NR_CORRELATIONS +
                           x * NR_CORRELATIONS;
      const float2 *aterm2_ptr = &aterms[station2_index];

      // Load spheroidal
      float sph = spheroidal[y * subgrid_size + x];

      // Load uv values
      float2 pixels_[NR_CORRELATIONS];
      for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
        unsigned idx_subgrid =
            s * NR_CORRELATIONS * subgrid_size * subgrid_size +
            pol * subgrid_size * subgrid_size + y * subgrid_size + x;
        pixels_[pol] = sph * subgrids[idx_subgrid];
      }

      // Apply aterm
      apply_aterm_degridder(pixels_, aterm1_ptr, aterm2_ptr);

      // Store pixels
      for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
        pixels[y][x][pol] = pixels_[pol];
      }
    } // end x
  }   // end y

  // Compute u and v offset in wavelenghts
  const float u_offset = (x_coordinate + subgrid_size / 2 - grid_size / 2) *
                         (2 * M_PI / image_size);
  const float v_offset = (y_coordinate + subgrid_size / 2 - grid_size / 2) *
                         (2 * M_PI / image_size);
  const float w_offset = 2 * M_PI * w_offset_in_lambda;

  // Iterate all timesteps
  for (int time = 0; time < nr_timesteps; time++) {
    // Load UVW coordinates
    float u = uvw[time_offset + time].u;
    float v = uvw[time_offset + time].v;
    float w = uvw[time_offset + time].w;

    // Iterate all channels
    for (int chan = 0; chan < nr_channels; chan++) {

      // Update all polarizations
      float2 sum[NR_CORRELATIONS];
      for (int i = 0; i < NR_CORRELATIONS; i++) {
        sum[i] = make_float2(0, 0);
      }

      // Iterate all pixels in subgrid
      for (int y = 0; y < subgrid_size; y++) {
        for (int x = 0; x < subgrid_size; x++) {

          // Compute l,m,n
          const float l = compute_l(x, subgrid_size, image_size);
          const float m = compute_m(y, subgrid_size, image_size);
          const float n = compute_n(l, m);

          // Compute phase index
          float phase_index = u * l + v * m + w * n;

          // Compute phase offset
          float phase_offset = u_offset * l + v_offset * m + w_offset * n;

          // Compute phase
          float phase = (phase_index * wavenumbers[chan]) - phase_offset;

          // Compute phasor
          float2 phasor = make_float2(cos(phase), sin(phase));

          for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
            sum[pol] += pixels[y][x][pol] * phasor;
          }
        } // end for x
      }   // end for y

      size_t index = (time_offset + time) * nr_channels + chan;
      for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
        visibilities[index * NR_CORRELATIONS + pol] = sum[pol];
      }
    } // end for channel
  }   // end for time
}

void p_run_degridder_reference() {

  float image_size = IMAGE_SIZE;
  float w_step_in_lambda = W_STEP;

  int nr_correlations = get_env_var("NR_CORRELATIONS", 4);

  int grid_size = get_env_var("GRID_SIZE", 1024);
  int subgrid_size = get_env_var("SUBGRID_SIZE", 32);
  int nr_stations = get_env_var("NR_STATIONS", 20);
  int nr_timeslots = get_env_var("NR_TIMESLOTS", 4);
  int nr_timesteps = get_env_var("NR_TIMESTEPS_SUBGRID", 128);
  int nr_channels = get_env_var("NR_CHANNELS", 16);

  int nr_baselines = (nr_stations * (nr_stations - 1)) / 2;
  int nr_subgrids = nr_baselines * nr_timeslots;
  int total_nr_timesteps = nr_subgrids * nr_timesteps;

  print_parameters(nr_stations, nr_channels, nr_timesteps, nr_correlations,
                   nr_timeslots, image_size, grid_size, subgrid_size,
                   w_step_in_lambda, nr_baselines, nr_subgrids,
                   total_nr_timesteps);

  std::string func_name = "degridder_reference";
  auto gflops =
      1e-9 * flops_gridder(nr_channels, total_nr_timesteps, nr_subgrids,
                           subgrid_size, nr_correlations);
  auto gbytes =
      1e-9 * bytes_gridder(nr_channels, total_nr_timesteps, nr_subgrids,
                           subgrid_size, nr_correlations);
  auto mvis = 1e-6 * total_nr_timesteps * nr_channels;

  std::vector<int> dim = {nr_subgrids, 1};

  idg::Array1D<idg::Metadata> metadata(nr_subgrids);

  idg::UVWCoordinate<float> *d_uvw;
  float *d_wavenumbers, *d_spheroidal;
  float2 *d_visibilities, *d_aterms, *d_subgrids;
  idg::Metadata *d_metadata;
  initialize_metadata(grid_size, nr_timeslots, nr_timesteps, nr_baselines,
                      metadata);

  cudaCheck(cudaMalloc(&d_uvw,
                       3 * nr_baselines * total_nr_timesteps * sizeof(float)));
  cudaCheck(cudaMalloc(&d_wavenumbers, nr_channels * sizeof(float)));
  cudaCheck(
      cudaMalloc(&d_spheroidal, subgrid_size * subgrid_size * sizeof(float)));
  cudaCheck(cudaMalloc(&d_visibilities, nr_baselines * total_nr_timesteps *
                                            nr_channels * sizeof(float2)));
  cudaCheck(cudaMalloc(&d_aterms, nr_timeslots * nr_stations * subgrid_size *
                                      subgrid_size * sizeof(float2)));
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

  p_run_kernel((void *)kernel_degridder_reference, dim3(dim[0]), dim3(dim[1]),
               args, func_name, gflops, gbytes);

  cudaCheck(cudaFree(d_uvw));
  cudaCheck(cudaFree(d_wavenumbers));
  cudaCheck(cudaFree(d_spheroidal));
  cudaCheck(cudaFree(d_visibilities));
  cudaCheck(cudaFree(d_aterms));
  cudaCheck(cudaFree(d_metadata));
  cudaCheck(cudaFree(d_subgrids));
}

void c_run_degridder_reference(
    int nr_subgrids, int grid_size, int subgrid_size, float image_size,
    float w_step_in_lambda, int nr_channels, int nr_stations,
    idg::Array2D<idg::UVWCoordinate<float>> &uvw,
    idg::Array1D<float> &wavenumbers,
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities,
    idg::Array2D<float> &spheroidal,
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> &aterms,
    idg::Array1D<idg::Metadata> &metadata,
    idg::Array4D<std::complex<float>> &subgrids) {

  std::vector<int> dim = {nr_subgrids, 1};

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

  c_run_kernel((void *)kernel_degridder_reference, dim3(dim[0]), dim3(dim[1]),
               args);

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
