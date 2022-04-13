#include "../common/math.hpp"
#include "math.hip.hpp"
#include "util.hip.hpp"

namespace hip {

__global__ void kernel_gridder_reference(
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

  // Compute u and v offset in wavelenghts
  const float u_offset = (x_coordinate + subgrid_size / 2 - grid_size / 2) *
                         (2 * M_PI / image_size);
  const float v_offset = (y_coordinate + subgrid_size / 2 - grid_size / 2) *
                         (2 * M_PI / image_size);
  const float w_offset = 2 * M_PI * w_offset_in_lambda;

  // Iterate all pixels in subgrid
  for (int y = 0; y < subgrid_size; y++) {
    for (int x = 0; x < subgrid_size; x++) {
      // Initialize pixel for every polarization
      float2 pixels[NR_CORRELATIONS];
      for (int i = 0; i < NR_CORRELATIONS; i++) {
        pixels[i] = make_float2(0, 0);
      }

      // Compute l,m,n
      float l = compute_l(x, subgrid_size, image_size);
      float m = compute_m(y, subgrid_size, image_size);
      float n = compute_n(l, m);
      // Iterate all timesteps
      for (int time = 0; time < nr_timesteps; time++) {
        // Load UVW coordinates
        float u = uvw[time_offset + time].u;
        float v = uvw[time_offset + time].v;
        float w = uvw[time_offset + time].w;

        // Compute phase index
        float phase_index = u * l + v * m + w * n;

        // Compute phase offset
        float phase_offset = u_offset * l + v_offset * m + w_offset * n;

        // Update pixel for every channel
        for (int chan = 0; chan < nr_channels; chan++) {
          // Compute phase
          float phase = phase_offset - (phase_index * wavenumbers[chan]);

          // Compute phasor
          float2 phasor = make_float2(cos(phase), sin(phase));

          // Update pixel for every polarization

          size_t index = (time_offset + time) * nr_channels + chan;
          for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
            float2 visibility = visibilities[index * NR_CORRELATIONS + pol];
            int tmp = (index * NR_CORRELATIONS + pol);
            pixels[pol] += visibility * phasor;
          }
        }
      }

      // Load a term for station1
      int station1_index = (aterm_index * nr_stations + station1) *
                               subgrid_size * subgrid_size * NR_CORRELATIONS +
                           y * subgrid_size * NR_CORRELATIONS +
                           x * NR_CORRELATIONS;
      float2 *aterm1_ptr = &aterms[station1_index];

      // Load aterm for station2
      int station2_index = (aterm_index * nr_stations + station2) *
                               subgrid_size * subgrid_size * NR_CORRELATIONS +
                           y * subgrid_size * NR_CORRELATIONS +
                           x * NR_CORRELATIONS;
      float2 *aterm2_ptr = &aterms[station2_index];
      // Apply aterm
      apply_aterm_gridder(pixels, aterm1_ptr, aterm2_ptr);

      // Load spheroidal
      float sph = spheroidal[y * subgrid_size + x];

      // Set subgrid value
      for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
        unsigned idx_subgrid =
            s * NR_CORRELATIONS * subgrid_size * subgrid_size +
            pol * subgrid_size * subgrid_size + y * subgrid_size + x;
        subgrids[idx_subgrid] = pixels[pol] * sph;
      }
    }
  }
}

void p_run_gridder_reference() {

  float image_size = IMAGE_SIZE;
  float w_step_in_lambda = W_STEP;

  int nr_correlations = get_env_var("NR_CORRELATIONS", 4);
  ;
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

  std::string func_name = "gridder_reference";
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

  hipCheck(
      hipMalloc(&d_uvw, 3 * nr_baselines * total_nr_timesteps * sizeof(float)));
  hipCheck(hipMalloc(&d_wavenumbers, nr_channels * sizeof(float)));
  hipCheck(
      hipMalloc(&d_spheroidal, subgrid_size * subgrid_size * sizeof(float)));
  hipCheck(hipMalloc(&d_visibilities, nr_baselines * total_nr_timesteps *
                                          nr_channels * sizeof(float2)));
  hipCheck(hipMalloc(&d_aterms, nr_timeslots * nr_stations * subgrid_size *
                                    subgrid_size * sizeof(float2)));
  hipCheck(hipMalloc(&d_subgrids, nr_subgrids * nr_correlations * subgrid_size *
                                      subgrid_size * sizeof(float2)));
  hipCheck(hipMalloc(&d_metadata, metadata.bytes()));

  hipMemcpy(d_metadata, metadata.data(), metadata.bytes(),
            hipMemcpyHostToDevice);

  void *args[] = {
      &grid_size,      &subgrid_size, &image_size, &w_step_in_lambda,
      &nr_channels,    &nr_stations,  &d_uvw,      &d_wavenumbers,
      &d_visibilities, &d_spheroidal, &d_aterms,   &d_metadata,
      &d_subgrids};

  p_run_kernel((void *)kernel_gridder_reference, dim3(dim[0]), dim3(dim[1]),
               args, func_name, gflops, gbytes);

  hipCheck(hipFree(d_uvw));
  hipCheck(hipFree(d_wavenumbers));
  hipCheck(hipFree(d_spheroidal));
  hipCheck(hipFree(d_visibilities));
  hipCheck(hipFree(d_aterms));
  hipCheck(hipFree(d_metadata));
  hipCheck(hipFree(d_subgrids));
}

void c_run_gridder_reference(
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

  c_run_kernel((void *)kernel_gridder_reference, dim3(dim[0]), dim3(dim[1]),
               args);

  hipMemcpy(subgrids.data(), d_subgrids,
            subgrids.size() * sizeof(std::complex<float>),
            hipMemcpyDeviceToHost);

  hipCheck(hipFree(d_uvw));
  hipCheck(hipFree(d_wavenumbers));
  hipCheck(hipFree(d_spheroidal));
  hipCheck(hipFree(d_visibilities));
  hipCheck(hipFree(d_aterms));
  hipCheck(hipFree(d_metadata));
  hipCheck(hipFree(d_subgrids));
}

} // namespace hip
