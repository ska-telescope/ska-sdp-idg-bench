#include "../common/math.hpp"
#include "math.cuh"
#include "util.cuh"

__shared__ float2 visibilities_v5_[BATCH_SIZE][NR_CORRELATIONS];
__shared__ float4 uvw_v5_[BATCH_SIZE];
__shared__ float wavenumbers_v5_[MAX_NR_CHANNELS];

template <int current_nr_channels>
__device__ void
kernel_gridder_v5_(const int grid_size, int subgrid_size, float image_size,
                   float w_step_in_lambda, int nr_channels,
                   int channel_offset, // channel_offset? for the macro?
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

  // Set subgrid to zero
  if (channel_offset == 0) {
    for (int i = tid; i < subgrid_size * subgrid_size; i += nr_threads) {
      int idx_xx = index_subgrid(subgrid_size, s, 0, 0, i);
      int idx_xy = index_subgrid(subgrid_size, s, 1, 0, i);
      int idx_yx = index_subgrid(subgrid_size, s, 2, 0, i);
      int idx_yy = index_subgrid(subgrid_size, s, 3, 0, i);
      subgrids[idx_xx] = make_float2(0, 0);
      subgrids[idx_xy] = make_float2(0, 0);
      subgrids[idx_yx] = make_float2(0, 0);
      subgrids[idx_yy] = make_float2(0, 0);
    }
  }

  for (int i = tid; i < current_nr_channels; i += nr_threads) {
    wavenumbers_v5_[i] = wavenumbers[i + channel_offset];
  }

  __syncthreads();

  // Compute u and v offset in wavelenghts
  const float u_offset = (x_coordinate + subgrid_size / 2 - grid_size / 2) *
                         (2 * M_PI / image_size);
  const float v_offset = (y_coordinate + subgrid_size / 2 - grid_size / 2) *
                         (2 * M_PI / image_size);
  const float w_offset = 2 * M_PI * w_offset_in_lambda;

  // Iterate all pixels in subgrid
  for (int i = tid; i < subgrid_size * subgrid_size;
       i += nr_threads * UNROLL_PIXELS) {
    // for (int x = 0; x < subgrid_size; x++) {
    //  Initialize pixel for every polarization
    // int x = i % subgrid_size;
    // int y = i / subgrid_size;

    float2 pixelXX[UNROLL_PIXELS];
    float2 pixelXY[UNROLL_PIXELS];
    float2 pixelYX[UNROLL_PIXELS];
    float2 pixelYY[UNROLL_PIXELS];

    for (int p = 0; p < UNROLL_PIXELS; p++) {
      pixelXX[p] = make_float2(0, 0);
      pixelXY[p] = make_float2(0, 0);
      pixelYX[p] = make_float2(0, 0);
      pixelYY[p] = make_float2(0, 0);
    }

    float l[UNROLL_PIXELS];
    float m[UNROLL_PIXELS];
    float n[UNROLL_PIXELS];
    float phase_offset[UNROLL_PIXELS];
    for (int p = 0; p < UNROLL_PIXELS; p++) {

      int x = (i + p * nr_threads) % subgrid_size;
      int y = (i + p * nr_threads) / subgrid_size;
      l[p] = compute_l(x, subgrid_size, image_size);
      m[p] = compute_m(y, subgrid_size, image_size);
      n[p] = compute_n(l[p], m[p]);
      phase_offset[p] = u_offset * l[p] + v_offset * m[p] + w_offset * n[p];
    }

    int current_nr_timesteps = BATCH_SIZE / MAX_NR_CHANNELS;
    // Iterate all timesteps
    for (int time_offset_local = 0; time_offset_local < nr_timesteps;
         time_offset_local += current_nr_timesteps) {
      current_nr_timesteps =
          nr_timesteps - time_offset_local < current_nr_timesteps
              ? nr_timesteps - time_offset_local
              : current_nr_timesteps;
      __syncthreads();
      for (int time = tid; time < current_nr_timesteps; time += nr_threads) {
        idg::UVWCoordinate<float> a =
            uvw[time_offset_global + time_offset_local + time];
        uvw_v5_[time] = make_float4(a.u, a.v, a.w, 0);
      }

      // Load visibilities
      for (int k = tid; k < current_nr_timesteps * current_nr_channels;
           k += nr_threads) {
        int idx_time =
            time_offset_global + time_offset_local + (k / current_nr_channels);
        int idx_chan = channel_offset + (k % current_nr_channels);
        int index = (time_offset_global + time_offset_local +
                     (k / current_nr_channels)) *
                        nr_channels +
                    channel_offset + (k % current_nr_channels);
        int indexXX = index_visibility(nr_channels, idx_time, idx_chan, 0);
        int indexXY = index_visibility(nr_channels, idx_time, idx_chan, 1);
        int indexYX = index_visibility(nr_channels, idx_time, idx_chan, 2);
        int indexYY = index_visibility(nr_channels, idx_time, idx_chan, 3);
        visibilities_v5_[k][0] = visibilities[indexXX];
        visibilities_v5_[k][1] = visibilities[indexXY];
        visibilities_v5_[k][2] = visibilities[indexYX];
        visibilities_v5_[k][3] = visibilities[indexYY];
      }
      __syncthreads();

      for (int time = 0; time < current_nr_timesteps; time++) {

        // Load UVW coordinates
        float u = uvw_v5_[time].x;
        float v = uvw_v5_[time].y;
        float w = uvw_v5_[time].z;
        for (int p = 0; p < UNROLL_PIXELS; p++) {
          // Compute phase index
          float phase_index = u * l[p] + v * m[p] + w * n[p];

          for (int chan = 0; chan < current_nr_channels; chan++) {
            // Compute phase
            float phase =
                phase_offset[p] - (phase_index * wavenumbers_v5_[chan]);

            // Compute phasor
            float2 phasor = make_float2(cosf(phase), sinf(phase));

            // Update pixel for every polarization

            float2 visXX =
                visibilities_v5_[time * current_nr_channels + chan][0];
            float2 visXY =
                visibilities_v5_[time * current_nr_channels + chan][1];
            float2 visYX =
                visibilities_v5_[time * current_nr_channels + chan][2];
            float2 visYY =
                visibilities_v5_[time * current_nr_channels + chan][3];

            pixelXX[p].x += phasor.x * visXX.x;
            pixelXX[p].y += phasor.x * visXX.y;
            pixelXX[p].x -= phasor.y * visXX.y;
            pixelXX[p].y += phasor.y * visXX.x;

            pixelXY[p].x += phasor.x * visXY.x;
            pixelXY[p].y += phasor.x * visXY.y;
            pixelXY[p].x -= phasor.y * visXY.y;
            pixelXY[p].y += phasor.y * visXY.x;

            pixelYX[p].x += phasor.x * visYX.x;
            pixelYX[p].y += phasor.x * visYX.y;
            pixelYX[p].x -= phasor.y * visYX.y;
            pixelYX[p].y += phasor.y * visYX.x;

            pixelYY[p].x += phasor.x * visYY.x;
            pixelYY[p].y += phasor.x * visYY.y;
            pixelYY[p].x -= phasor.y * visYY.y;
            pixelYY[p].y += phasor.y * visYY.x;
          }
        }
      }
    }
    for (int p = 0; p < UNROLL_PIXELS; p++) {

      int x = (i + p * nr_threads) % subgrid_size;
      int y = (i + p * nr_threads) / subgrid_size;

      // Load aterm for station1
      float2 aXX1, aXY1, aYX1, aYY1;
      read_aterm(subgrid_size, nr_stations, aterm_index, station1, y, x, aterms,
                 &aXX1, &aXY1, &aYX1, &aYY1);

      // Load aterm for station2
      float2 aXX2, aXY2, aYX2, aYY2;
      read_aterm(subgrid_size, nr_stations, aterm_index, station2, y, x, aterms,
                 &aXX2, &aXY2, &aYX2, &aYY2);

      // Apply the conjugate transpose of the A-term
      apply_aterm(conj(aXX1), conj(aYX1), conj(aXY1), conj(aYY1), conj(aXX2),
                  conj(aYX2), conj(aXY2), conj(aYY2), pixelXX[p], pixelXY[p],
                  pixelYX[p], pixelYY[p]);

      // Load a term for station1
      // Load spheroidal
      float sph = spheroidal[y * subgrid_size + x];

      int idx_xx = index_subgrid(subgrid_size, s, 0, 0, i + p * nr_threads);
      int idx_xy = index_subgrid(subgrid_size, s, 1, 0, i + p * nr_threads);
      int idx_yx = index_subgrid(subgrid_size, s, 2, 0, i + p * nr_threads);
      int idx_yy = index_subgrid(subgrid_size, s, 3, 0, i + p * nr_threads);

      subgrids[idx_xx] += pixelXX[p] * sph;
      subgrids[idx_xy] += pixelXY[p] * sph;
      subgrids[idx_yx] += pixelYX[p] * sph;
      subgrids[idx_yy] += pixelYY[p] * sph;
    }
    // }
  }
}

#define KERNEL_GRIDDER_TEMPLATE(current_nr_channels)                           \
  for (; (channel_offset + current_nr_channels) <= nr_channels;                \
       channel_offset += current_nr_channels) {                                \
    kernel_gridder_v5_<current_nr_channels>(                                   \
        grid_size, subgrid_size, image_size, w_step_in_lambda, nr_channels,    \
        channel_offset, nr_stations, uvw, wavenumbers, visibilities,           \
        spheroidal, aterms, metadata, subgrids);                               \
  }

namespace cuda {

__global__ void

kernel_gridder_v5(const int grid_size, int subgrid_size, float image_size,
                  float w_step_in_lambda,
                  int nr_channels, // channel_offset? for the macro?
                  int nr_stations, idg::UVWCoordinate<float> *uvw,
                  float *wavenumbers, float2 *visibilities, float *spheroidal,
                  float2 *aterms, idg::Metadata *metadata, float2 *subgrids) {
  int channel_offset = 0;
  KERNEL_GRIDDER_TEMPLATE(8);
  KERNEL_GRIDDER_TEMPLATE(7);
  KERNEL_GRIDDER_TEMPLATE(6);
  KERNEL_GRIDDER_TEMPLATE(5);
  KERNEL_GRIDDER_TEMPLATE(4);
  KERNEL_GRIDDER_TEMPLATE(3);
  KERNEL_GRIDDER_TEMPLATE(2);
  KERNEL_GRIDDER_TEMPLATE(1);
}

void p_run_gridder_v5() {
  p_run_gridder((void *)kernel_gridder_v5, "gridder_v5", 128);
}

void c_run_gridder_v5(
    int nr_subgrids, int grid_size, int subgrid_size, float image_size,
    float w_step_in_lambda, int nr_channels, int nr_stations,
    idg::Array2D<idg::UVWCoordinate<float>> &uvw,
    idg::Array1D<float> &wavenumbers,
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities,
    idg::Array2D<float> &spheroidal,
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> &aterms,
    idg::Array1D<idg::Metadata> &metadata,
    idg::Array4D<std::complex<float>> &subgrids) {

  c_run_gridder(nr_subgrids, grid_size, subgrid_size, image_size,
                w_step_in_lambda, nr_channels, nr_stations, uvw, wavenumbers,
                visibilities, spheroidal, aterms, metadata, subgrids,
                (void *)kernel_gridder_v5, 128);
}

} // namespace cuda
