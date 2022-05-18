#include "../common/math.hpp"
#include "math.hip.hpp"
#include "util.hip.hpp"

#define ALIGN(N,A) (((N)+(A)-1)/(A)*(A))
#define NUM_THREADS 128

// Storage
__shared__ float4 shared_v6[3][NUM_THREADS];

namespace hip {

__device__ void prepare_shared(
    const int                  current_nr_pixels,
    const int                  pixel_offset,
    const int                  nr_polarizations,
    const int                  grid_size,
    const int                  subgrid_size,
    const float                image_size,
    const int                  nr_stations,
    const int                  aterm_idx,
    const idg::Metadata&       metadata,
    const float*  spheroidal,
    const float2* aterms,
    const float2* subgrid)
{
    int s           = blockIdx.x;
    int num_threads = blockDim.x;
    int tid         = threadIdx.x;

    // Load metadata for current subgrid
    const int x_coordinate = metadata.coordinate.x;
    const int y_coordinate = metadata.coordinate.y;
    const int station1 = metadata.baseline.station1;
    const int station2 = metadata.baseline.station2;

    // Compute u,v,w offset in wavelenghts
    const float u_offset = (x_coordinate + subgrid_size / 2 - grid_size / 2) *
                           (2 * M_PI / image_size);
    const float v_offset = (y_coordinate + subgrid_size / 2 - grid_size / 2) *
                           (2 * M_PI / image_size);
    const float w_offset = (metadata.coordinate.z + 0.5) * 2 * M_PI;

    for (int j = tid; j < current_nr_pixels; j += num_threads) {
        int y = (pixel_offset + j) / subgrid_size;
        int x = (pixel_offset + j) % subgrid_size;

        // Load spheroidal
        const float spheroidal_ = spheroidal[y * subgrid_size + x];

        // Load pixels
        float2 pixel[4];
        for (unsigned pol = 0; pol < nr_polarizations; pol++) {
            unsigned int pixel_idx = index_subgrid(subgrid_size, s, pol, y, x);
            pixel[pol] = subgrid[pixel_idx] * spheroidal_;
        }

        // Apply aterm
        int station1_idx = index_aterm(subgrid_size, nr_stations, aterm_idx, station1, y, x);
        int station2_idx = index_aterm(subgrid_size, nr_stations, aterm_idx, station2, y, x);
        float2 *aterm1 = (float2 *) &aterms[station1_idx];
        float2 *aterm2 = (float2 *) &aterms[station2_idx];
        apply_aterm_degridder(pixel, aterm1, aterm2);

        // Store pixels in shared memory
        shared_v6[0][j] = make_float4(pixel[0].x, pixel[0].y, pixel[1].x, pixel[1].y);
        shared_v6[1][j] = make_float4(pixel[2].x, pixel[2].y, pixel[3].x, pixel[3].y);

        // Compute l,m,n for phase offset and phase index
        const float l = compute_l(x, subgrid_size, image_size);
        const float m = compute_m(y, subgrid_size, image_size);
        const float n = compute_n(l, m);
        const float phase_offset = -(u_offset*l + v_offset*m + w_offset*n);

        // Store l_index,m_index,n and phase offset in shared memory
        shared_v6[2][j] = make_float4(l, m, n, phase_offset);
    } // end for j (pixels)
}

__device__ void cmac(float2 &a, float2 b, float2 c)
{
    a.x = fma(b.x, c.x, a.x);
    a.y = fma(b.x, c.y, a.y);
    a.x = fma(-b.y, c.y, a.x);
    a.y = fma(b.y, c.x, a.y);
}

__device__ void compute_visibility(
    const int     nr_polarizations,
    const int     current_nr_pixels,
    const int     channel,
    const float   u,
    const float   v,
    const float   w,
    const float*  wavenumbers,
          float2* visibility)
{
    for (int i = 0; i < current_nr_pixels; i++) {
        // Compute phase_offset and phase index
        float l = shared_v6[2][i].x;
        float m = shared_v6[2][i].y;
        float n = shared_v6[2][i].z;
        float phase_offset = shared_v6[2][i].w;
        float phase_index = u * l + v * m + w * n;

        // Compute visibility
        const float4 a = shared_v6[0][i];
        const float4 b = shared_v6[1][i];
        float phase = wavenumbers[channel] * phase_index + phase_offset;
        float2 phasor = make_float2(__cosf(phase), __sinf(phase));
        cmac(visibility[0], phasor, make_float2(a.x, a.y));
        cmac(visibility[1], phasor, make_float2(a.z, a.w));
        cmac(visibility[2], phasor, make_float2(b.x, b.y));
        cmac(visibility[3], phasor, make_float2(b.z, b.w));
    } // end for k (batch)
}

__global__ void
kernel_degridder_v6(int grid_size, int subgrid_size, float image_size,
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

    // Load metadata for current subgrid
    const idg::Metadata &m = metadata[s];
    const int time_offset =
        (m.baseline_offset - baseline_offset_1) + m.time_offset;
    const int nr_timesteps = m.nr_timesteps;
    const int aterm_index = m.aterm_index;

    // Iterate all timesteps
    for (int time = 0; time < nr_timesteps; time += NUM_THREADS) {
      int index_time = time + tid;

      // Iterate all channels
      for (int chan = 0; chan < nr_channels; chan++) {
          size_t index = (time_offset + index_time) * nr_channels + chan;
          for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
            visibilities[index * NR_CORRELATIONS + pol] = make_float2(0, 0);
          }
      } // end for chan
    } // end for time

    // Iterate pixels
    const int nr_pixels = subgrid_size * subgrid_size;
    const int batch_size = NUM_THREADS;
    int current_nr_pixels = batch_size;
    for (int pixel_offset = 0; pixel_offset < nr_pixels; pixel_offset += current_nr_pixels) {
        current_nr_pixels = nr_pixels - pixel_offset < min(NUM_THREADS, batch_size) ?
                            nr_pixels - pixel_offset : min(NUM_THREADS, batch_size);

        // Iterate timesteps
        for (int time = 0; time < nr_timesteps; time += NUM_THREADS) {

            __syncthreads();

            // Prepare data
            prepare_shared(
                current_nr_pixels, pixel_offset, 4, grid_size,
                subgrid_size, image_size, nr_stations,
                aterm_index, m, spheroidal, aterms, subgrids);

            __syncthreads();

            // Determine the first and last timestep to process
            int time_start = time_offset + time;
            int time_end = time_start + NUM_THREADS;

            for (int i = tid; i < ALIGN(NUM_THREADS * nr_channels, NUM_THREADS); i += NUM_THREADS) {
                int time = time_start + (i / nr_channels);
                int channel = (i % nr_channels);

                float2 visibility[4];

                for (int pol = 0; pol < 4; pol++) {
                    visibility[pol] = make_float2(0, 0);
                }

                float u = 0, v = 0, w = 0;

                if (time < time_end) {
                    u = uvw[time].u;
                    v = uvw[time].v;
                    w = uvw[time].w;
                }

                // Compute visibility
                compute_visibility(
                    4, current_nr_pixels, channel,
                    u, v, w, wavenumbers, visibility);

                // Update visibility
                if (time < time_end) {
                    size_t index = time * nr_channels + channel;
                    for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
                      visibilities[index * NR_CORRELATIONS + pol] += visibility[pol];
                    }
                }
            } // end for time
        } // end for time_offset_local
    } // end for pixel_offset
}

void p_run_degridder_v6() {
  p_run_degridder((void *)kernel_degridder_v6, "degridder_v6", NUM_THREADS);
}

void c_run_degridder_v6(
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
                  (void *)kernel_degridder_v6, NUM_THREADS);
}

} // namespace hip
