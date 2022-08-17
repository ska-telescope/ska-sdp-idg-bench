#include "common/math.hpp"
#include "math.hip.hpp"
#include "util.hpp"

namespace hip {

__global__ void
kernel_gridder_v2(const int grid_size, int subgrid_size, float image_size,
                  float w_step_in_lambda, int nr_channels, int nr_stations,
                  idg::UVWCoordinate<float> *uvw, float *wavenumbers,
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
  for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
    int x = i % subgrid_size;
    int y = i / subgrid_size;

    // Initialize pixel for every polarization
    float2 pixels[NR_CORRELATIONS];
    for (int k = 0; k < NR_CORRELATIONS; k++) {
      pixels[k] = make_float2(0, 0);
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
        float2 phasor = make_float2(__cosf(phase), __sinf(phase));

        // Update pixel for every polarization
        size_t index = (time_offset + time) * nr_channels + chan;
        for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
          float2 visibility = visibilities[index * NR_CORRELATIONS + pol];
          pixels[pol] += cmul(visibility, phasor);
        } // end for pol
      } // end for chan
    } // end for time

    // Load aterm for station1
    int station1_index = (aterm_index * nr_stations + station1) * subgrid_size *
                             subgrid_size * NR_CORRELATIONS +
                         y * subgrid_size * NR_CORRELATIONS +
                         x * NR_CORRELATIONS;
    float2 *aterm1_ptr = &aterms[station1_index];

    // Load aterm for station2
    int station2_index = (aterm_index * nr_stations + station2) * subgrid_size *
                             subgrid_size * NR_CORRELATIONS +
                         y * subgrid_size * NR_CORRELATIONS +
                         x * NR_CORRELATIONS;
    float2 *aterm2_ptr = &aterms[station2_index];

    // Apply aterm
    apply_aterm_gridder(pixels, aterm1_ptr, aterm2_ptr);

    // Load spheroidal
    float sph = spheroidal[y * subgrid_size + x];

    // Set subgrid value
    for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
      unsigned idx_subgrid = s * NR_CORRELATIONS * subgrid_size * subgrid_size +
                             pol * subgrid_size * subgrid_size +
                             y * subgrid_size + x;
      subgrids[idx_subgrid] = pixels[pol] * sph;
    } // end for pol
  } // end for i
}

void p_run_gridder() {
  p_run_gridder_((void *)kernel_gridder_v2, "gridder_v2", 128);
}

void c_run_gridder(
    int nr_subgrids, int grid_size, int subgrid_size, float image_size,
    float w_step_in_lambda, int nr_channels, int nr_stations,
    idg::Array2D<idg::UVWCoordinate<float>> &uvw,
    idg::Array1D<float> &wavenumbers,
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities,
    idg::Array2D<float> &spheroidal,
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> &aterms,
    idg::Array1D<idg::Metadata> &metadata,
    idg::Array4D<std::complex<float>> &subgrids) {

  c_run_gridder_(nr_subgrids, grid_size, subgrid_size, image_size,
                 w_step_in_lambda, nr_channels, nr_stations, uvw, wavenumbers,
                 visibilities, spheroidal, aterms, metadata, subgrids,
                 (void *)kernel_gridder_v2, 128);
}

} // namespace hip
