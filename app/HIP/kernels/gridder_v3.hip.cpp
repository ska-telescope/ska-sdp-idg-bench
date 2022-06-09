#include "../common/math.hpp"
#include "math.hip.hpp"
#include "util.hpp"

namespace hip {

__global__ void
kernel_gridder_v3(const int grid_size, int subgrid_size, float image_size,
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
    float2 pixelXX = make_float2(0, 0);
    float2 pixelXY = make_float2(0, 0);
    float2 pixelYX = make_float2(0, 0);
    float2 pixelYY = make_float2(0, 0);

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
        int idx_time = time_offset + time;
        int indexXX = index_visibility(nr_channels, idx_time, chan, 0);
        int indexXY = index_visibility(nr_channels, idx_time, chan, 1);
        int indexYX = index_visibility(nr_channels, idx_time, chan, 2);
        int indexYY = index_visibility(nr_channels, idx_time, chan, 3);

        // Load visibilities
        float2 visXX = visibilities[indexXX];
        float2 visXY = visibilities[indexXY];
        float2 visYX = visibilities[indexYX];
        float2 visYY = visibilities[indexYY];

        // Update pixel for every polarization
        pixelXX.x += phasor.x * visXX.x;
        pixelXX.y += phasor.x * visXX.y;
        pixelXX.x -= phasor.y * visXX.y;
        pixelXX.y += phasor.y * visXX.x;

        pixelXY.x += phasor.x * visXY.x;
        pixelXY.y += phasor.x * visXY.y;
        pixelXY.x -= phasor.y * visXY.y;
        pixelXY.y += phasor.y * visXY.x;

        pixelYX.x += phasor.x * visYX.x;
        pixelYX.y += phasor.x * visYX.y;
        pixelYX.x -= phasor.y * visYX.y;
        pixelYX.y += phasor.y * visYX.x;

        pixelYY.x += phasor.x * visYY.x;
        pixelYY.y += phasor.x * visYY.y;
        pixelYY.x -= phasor.y * visYY.y;
        pixelYY.y += phasor.y * visYY.x;
      } // end for chan
    } // end for time

    // Load aterm for station1
    float2 aXX1, aXY1, aYX1, aYY1;
    read_aterm(subgrid_size, nr_stations, aterm_index, station1, y, x, aterms,
               &aXX1, &aXY1, &aYX1, &aYY1);

    // Load aterm for station2
    float2 aXX2, aXY2, aYX2, aYY2;
    read_aterm(subgrid_size, nr_stations, aterm_index, station2, y, x, aterms,
               &aXX2, &aXY2, &aYX2, &aYY2);

    // Apply aterm
    apply_aterm(conj(aXX1), conj(aYX1), conj(aXY1), conj(aYY1), conj(aXX2),
                conj(aYX2), conj(aXY2), conj(aYY2), pixelXX, pixelXY, pixelYX,
                pixelYY);

    // Load spheroidal
    float sph = spheroidal[y * subgrid_size + x];

    // Set subgrid value
    int idx_xx = index_subgrid(subgrid_size, s, 0, 0, i);
    int idx_xy = index_subgrid(subgrid_size, s, 1, 0, i);
    int idx_yx = index_subgrid(subgrid_size, s, 2, 0, i);
    int idx_yy = index_subgrid(subgrid_size, s, 3, 0, i);

    subgrids[idx_xx] = pixelXX * sph;
    subgrids[idx_xy] = pixelXY * sph;
    subgrids[idx_yx] = pixelYX * sph;
    subgrids[idx_yy] = pixelYY * sph;
  } // end for i
}

void p_run_gridder() {
  p_run_gridder_((void *)kernel_gridder_v3, "gridder_v3", 128);
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
                 (void *)kernel_gridder_v3, 128);
}

} // namespace hip
