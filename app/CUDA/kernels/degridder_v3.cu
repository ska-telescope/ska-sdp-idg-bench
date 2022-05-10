#include "../common/math.hpp"
#include "math.cuh"
#include "util.cuh"
// not possible to pass it as input
#define SUBGRID_SIZE 32

// Storage
__shared__ float2 pixels_v3[SUBGRID_SIZE][SUBGRID_SIZE][NR_CORRELATIONS];

namespace cuda {

__global__ void
kernel_degridder_v3(const int grid_size, int subgrid_size, float image_size,
                    float w_step_in_lambda,
                    int nr_channels, // channel_offset? for the macro?
                    int nr_stations, idg::UVWCoordinate<float> *uvw,
                    float *wavenumbers, float2 *visibilities, float *spheroidal,
                    float2 *aterms, idg::Metadata *metadata, float2 *subgrids) {
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
  const int time_offset_global =
      (m.baseline_offset - baseline_offset_1) + m.time_offset;
  const int nr_timesteps = m.nr_timesteps;
  const int aterm_index = m.aterm_index;
  const int station1 = m.baseline.station1;
  const int station2 = m.baseline.station2;
  const int x_coordinate = m.coordinate.x;
  const int y_coordinate = m.coordinate.y;
  const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

  // Iterate all pixels in subgrid
  for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
    // for (int x = 0; x < subgrid_size; x++) {
    //  Initialize pixel for every polarization
    int x = i % subgrid_size;
    int y = i / subgrid_size;

      // Load spheroidal
      float sph = spheroidal[y * subgrid_size + x];

      // Load uv values
      float2 pixels_[NR_CORRELATIONS];

      float2 pixelXX;
      float2 pixelXY;
      float2 pixelYX;
      float2 pixelYY;

      int idx_xx = index_subgrid(subgrid_size, s, 0, 0, i);
      int idx_xy = index_subgrid(subgrid_size, s, 1, 0, i);
      int idx_yx = index_subgrid(subgrid_size, s, 2, 0, i);
      int idx_yy = index_subgrid(subgrid_size, s, 3, 0, i);

      pixelXX = sph * subgrids[idx_xx];
      pixelXY = sph * subgrids[idx_xy];
      pixelYX = sph * subgrids[idx_yx];
      pixelYY = sph * subgrids[idx_yy];

              // Load aterm for station1
    float2 aXX1, aXY1, aYX1, aYY1;
    read_aterm(subgrid_size, nr_stations, aterm_index, station1, y, x, aterms,
               &aXX1, &aXY1, &aYX1, &aYY1);

    // Load aterm for station2
    float2 aXX2, aXY2, aYX2, aYY2;
    read_aterm(subgrid_size, nr_stations, aterm_index, station2, y, x, aterms,
               &aXX2, &aXY2, &aYX2, &aYY2);

    // Apply the conjugate transpose of the A-term
    apply_aterm(
                aXX1, aYX1, aXY1, aYY1,
                aXX2, aYX2, aXY2, aYY2,
                pixelXX, pixelXY, pixelYX, pixelYY);




    pixels_v3[y][x][0] = pixelXX;
    pixels_v3[y][x][1] = pixelXY;
    pixels_v3[y][x][2] = pixelYX;
    pixels_v3[y][x][3] = pixelYY;
    //} // end x
  }   // end y

  // Compute u and v offset in wavelenghts
  const float u_offset = (x_coordinate + subgrid_size / 2 - grid_size / 2) *
                         (2 * M_PI / image_size);
  const float v_offset = (y_coordinate + subgrid_size / 2 - grid_size / 2) *
                         (2 * M_PI / image_size);
  const float w_offset = 2 * M_PI * w_offset_in_lambda;

  for (int time_offset_local = tid; time_offset_local < nr_timesteps; time_offset_local+=nr_threads) {
    // Load UVW coordinates
    float u = uvw[time_offset_global + time_offset_local].u;
    float v = uvw[time_offset_global + time_offset_local].v;
    float w = uvw[time_offset_global + time_offset_local].w;
    // Iterate all channels
    for (int chan = 0; chan < nr_channels; chan++) {

      // Update all polarizations
      float2 sumXX = make_float2(0, 0);
      float2 sumXY = make_float2(0, 0);
      float2 sumYX = make_float2(0, 0);
      float2 sumYY = make_float2(0, 0);

    // Iterate all pixels in subgrid
  for (int i = 0; i < subgrid_size * subgrid_size; i ++) {
    // for (int x = 0; x < subgrid_size; x++) {
    //  Initialize pixel for every polarization
    int x = i % subgrid_size;
    int y = i / subgrid_size;

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
          float2 phasor = make_float2(__cosf(phase), __sinf(phase));

          sumXX.x += pixels_v3[y][x][0].x * phasor.x;
          sumXX.y += pixels_v3[y][x][0].x * phasor.y;
          sumXX.x -= pixels_v3[y][x][0].y * phasor.y;
          sumXX.y += pixels_v3[y][x][0].y * phasor.x;

          sumXY.x += pixels_v3[y][x][1].x * phasor.x;
          sumXY.y += pixels_v3[y][x][1].x * phasor.y;
          sumXY.x -= pixels_v3[y][x][1].y * phasor.y;
          sumXY.y += pixels_v3[y][x][1].y * phasor.x;

          sumYX.x += pixels_v3[y][x][2].x * phasor.x;
          sumYX.y += pixels_v3[y][x][2].x * phasor.y;
          sumYX.x -= pixels_v3[y][x][2].y * phasor.y;
          sumYX.y += pixels_v3[y][x][2].y * phasor.x;

          sumYY.x += pixels_v3[y][x][3].x * phasor.x;
          sumYY.y += pixels_v3[y][x][3].x * phasor.y;
          sumYY.x -= pixels_v3[y][x][3].y * phasor.y;
          sumYY.y += pixels_v3[y][x][3].y * phasor.x;


       // } // end for x
      }   // end for y

      int idx_time = time_offset_global + time_offset_local;

      int indexXX = index_visibility(nr_channels, idx_time, chan, 0);
      int indexXY = index_visibility(nr_channels, idx_time, chan, 1);
      int indexYX = index_visibility(nr_channels, idx_time, chan, 2);
      int indexYY = index_visibility(nr_channels, idx_time, chan, 3);

      visibilities[indexXX] += sumXX;
      visibilities[indexXY] += sumXY;
      visibilities[indexYX] += sumYX;
      visibilities[indexYY] += sumYY;
    } // end for channel
  }   // end for time
}

void p_run_degridder_v3() {
  p_run_degridder((void *)kernel_degridder_v3, "degridder_v3", 128);
}

void c_run_degridder_v3(
    int nr_subgrids, int grid_size, int subgrid_size, float image_size,
    float w_step_in_lambda, int nr_channels, int nr_stations,
    idg::Array2D<idg::UVWCoordinate<float>> &uvw,
    idg::Array1D<float> &wavenumbers,
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities,
    idg::Array2D<float> &spheroidal,
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> &aterms,
    idg::Array1D<idg::Metadata> &metadata,
    idg::Array4D<std::complex<float>> &subgrids) {

  c_run_degridder(nr_subgrids, grid_size, subgrid_size, image_size,
                  w_step_in_lambda, nr_channels, nr_stations, uvw, wavenumbers,
                  visibilities, spheroidal, aterms, metadata, subgrids,
                  (void *)kernel_degridder_v3, 128);
}

} // namespace cuda
