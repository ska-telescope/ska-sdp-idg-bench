#include "../common/math.hpp"
#include "util.hpp"

namespace cpu {

void kernel_degridder_reference(int nr_subgrids, int grid_size, int subgrid_size,
                              float image_size, float w_step_in_lambda,
                              int nr_channels, int nr_stations,
                              idg::UVWCoordinate<float> *uvw,
                              float *wavenumbers,
                              std::complex<float> *visibilities,
                              float *spheroidal, std::complex<float> *aterms,
                              idg::Metadata *metadata,
                              std::complex<float> *subgrids) {
     // Find offset of first subgrid
    const idg::Metadata m       = metadata[0];
    const int baseline_offset_1 = m.baseline_offset;

    // Iterate all subgrids
    #pragma omp parallel for
    for (int s = 0; s < nr_subgrids; s++) {

        // Load metadata
        const idg::Metadata m  = metadata[s];
        const int time_offset = (m.baseline_offset - baseline_offset_1) + m.time_offset;
        const int nr_timesteps = m.nr_timesteps;
        const int aterm_index  = m.aterm_index;
        const int station1     = m.baseline.station1;
        const int station2     = m.baseline.station2;
        const int x_coordinate = m.coordinate.x;
        const int y_coordinate = m.coordinate.y;
        const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

        // Storage
        std::complex<float> pixels[subgrid_size][subgrid_size][NR_CORRELATIONS];

        // Apply aterm to subgrid
        for (int y = 0; y < subgrid_size; y++) {
            for (int x = 0; x < subgrid_size; x++) {
                // Load aterm for station1
                int station1_index =
                    (aterm_index * nr_stations + station1) *
                    subgrid_size * subgrid_size * NR_CORRELATIONS +
                    y * subgrid_size * NR_CORRELATIONS + x * NR_CORRELATIONS;
                const std::complex<float>* aterm1_ptr = &aterms[station1_index];

                // Load aterm for station2
                int station2_index =
                    (aterm_index * nr_stations + station2) *
                    subgrid_size * subgrid_size * NR_CORRELATIONS +
                    y * subgrid_size * NR_CORRELATIONS + x * NR_CORRELATIONS;
                const std::complex<float>* aterm2_ptr = &aterms[station2_index];

                // Load spheroidal
                float sph = spheroidal[y * subgrid_size + x];

                // Load uv values
                std::complex<float> pixels_[NR_CORRELATIONS];
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
        } // end y

        // Compute u and v offset in wavelenghts
        const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) * (2*M_PI / image_size);
        const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) * (2*M_PI / image_size);
        const float w_offset = 2*M_PI * w_offset_in_lambda;

        // Iterate all timesteps
        for (int time = 0; time < nr_timesteps; time++) {
            // Load UVW coordinates
            float u = uvw[time_offset + time].u;
            float v = uvw[time_offset + time].v;
            float w = uvw[time_offset + time].w;

            // Iterate all channels
            for (int chan = 0; chan < nr_channels; chan++) {

                // Update all polarizations
                std::complex<float> sum[NR_CORRELATIONS];
                memset(sum, 0, NR_CORRELATIONS * sizeof(std::complex<float>));

                // Iterate all pixels in subgrid
                for (int y = 0; y < subgrid_size; y++) {
                    for (int x = 0; x < subgrid_size; x++) {

                        // Compute l,m,n
                        const float l = compute_l(x, subgrid_size, image_size);
                        const float m = compute_m(y, subgrid_size, image_size);
                        const float n = compute_n(l, m);

                        // Compute phase index
                        float phase_index = u*l + v*m + w*n;

                        // Compute phase offset
                        float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                        // Compute phase
                        float phase = (phase_index * wavenumbers[chan]) - phase_offset;

                        // Compute phasor
                        std::complex<float> phasor = {cosf(phase), sinf(phase)};

                        for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
                            sum[pol] += pixels[y][x][pol] * phasor;
                        }
                    } // end for x
                } // end for y

                size_t index = (time_offset + time)*nr_channels + chan;
                for (int pol = 0; pol < NR_CORRELATIONS; pol++) {
                    visibilities[index * NR_CORRELATIONS + pol] = sum[pol];
                }
            } // end for channel
        } // end for time
    } // end for s
} // end kernel_gridder

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
  kernel_degridder_reference(
      nr_subgrids, grid_size, subgrid_size, image_size, w_step_in_lambda,
      nr_channels, nr_stations, uvw.data(), wavenumbers.data(),
      (std::complex<float> *)visibilities.data(), (float *)spheroidal.data(),
      (std::complex<float> *)aterms.data(), metadata.data(), subgrids.data());
}
} // end namespace cpu
