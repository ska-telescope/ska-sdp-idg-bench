#include "../common/math.hpp"
#include "math.hip.hpp"
#include "util.hpp"

#define NUM_THREADS 128

// Storage
__shared__ float2 pixels_v3[NUM_THREADS][NR_CORRELATIONS];

namespace hip {

__global__ void
kernel_degridder_v3(int grid_size, int subgrid_size, float image_size,
                    float w_step_in_lambda, int nr_channels,
                    int nr_stations, idg::UVWCoordinate<float> *uvw,
                    float *wavenumbers, float2 *visibilities, float *spheroidal,
                    float2 *aterms, idg::Metadata *metadata, float2 *subgrids) {
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;
  int tid = tidx + tidy * blockDim.x;
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

  // Iterate all timesteps
  for (int time = 0; time < nr_timesteps; time += NUM_THREADS) {
    int index_time = time + tid;

    // Load UVW coordinates
    float u, v, w;
    if (index_time < nr_timesteps) {
      u = uvw[time_offset + index_time].u;
      v = uvw[time_offset + index_time].v;
      w = uvw[time_offset + index_time].w;
    }

    // Iterate all channels
    for (int chan = 0; chan < nr_channels; chan++) {

      // Update all polarizations
      float2 sum[NR_CORRELATIONS];
      for (int k = 0; k < NR_CORRELATIONS; k++) {
        sum[k] = make_float2(0, 0);
      }

      // Iterate all pixels
      for (int i = 0; i < subgrid_size*subgrid_size; i += NUM_THREADS) {
        int index_pixel = i + tid;
        int y = index_pixel / subgrid_size;
        int x = index_pixel % subgrid_size;

        __syncthreads();

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

        // Load pixels
        float2 pixels[NR_CORRELATIONS];
        for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
          unsigned idx_subgrid =
              s * NR_CORRELATIONS * subgrid_size * subgrid_size +
              pol * subgrid_size * subgrid_size + y * subgrid_size + x;
          pixels[pol] = sph * subgrids[idx_subgrid];
        }

        // Apply aterm
        apply_aterm_degridder(pixels, aterm1_ptr, aterm2_ptr);

        // Store pixel in shared memory
        for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
          pixels_v3[tid][pol] = pixels[pol];
        }

        __syncthreads();

        for (int j = 0; j < NUM_THREADS; j++) {
          int pixel_index = i + j;
          int y = pixel_index / subgrid_size;
          int x = pixel_index % subgrid_size;

          if (y >= subgrid_size) {
            break;
          }

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

          for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
            sum[pol] += pixels_v3[j][pol] * phasor;
          }
        } // end for j
      } // end for i (pixels)

      // Store visibility
      if (index_time < nr_timesteps) {
        size_t index = (time_offset + index_time) * nr_channels + chan;
        for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
          visibilities[index * NR_CORRELATIONS + pol] = sum[pol];
        }
      }
    } // end for channel
  }   // end for time
}

void p_run_degridder() {
  p_run_degridder_((void *)kernel_degridder_v3, "degridder_v3", NUM_THREADS);
}

void c_run_degridder(
    int nr_subgrids, int grid_size, int subgrid_size, float image_size,
    float w_step_in_lambda, int nr_channels, int nr_stations,
    idg::Array2D<idg::UVWCoordinate<float>> &uvw,
    idg::Array1D<float> &wavenumbers,
    idg::Array3D<idg::Visibility<std::complex<float>>> &visibilities,
    idg::Array2D<float> &spheroidal,
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> &aterms,
    idg::Array1D<idg::Metadata> &metadata,
    idg::Array4D<std::complex<float>> &subgrids) {

  c_run_degridder_(nr_subgrids, grid_size, subgrid_size, image_size,
                   w_step_in_lambda, nr_channels, nr_stations, uvw, wavenumbers,
                   visibilities, spheroidal, aterms, metadata, subgrids,
                   (void *)kernel_degridder_v3, NUM_THREADS);
}

} // namespace hip
