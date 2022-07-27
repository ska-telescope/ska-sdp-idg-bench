#pragma once

#include "common/parameters.hpp"
#include <hip/hip_complex.h>

inline __device__ float2 conj(float2 a) { return hipConjf(a); }

inline __device__ float2 operator+(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ float2 operator-(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ float2 operator*(float2 a, float b) {
  return make_float2(a.x * b, a.y * b);
}

inline __device__ float2 operator*(float a, float2 b) {
  return make_float2(a * b.x, a * b.y);
}

inline __device__ float2 operator*(const float2 a, float2 b) {
  return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline __device__ float4 operator*(float4 a, float b) {
  return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline __device__ float4 operator*(float a, float4 b) {
  return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

inline __device__ void operator+=(float2 &a, float2 b) {
  a.x += b.x;
  a.y += b.y;
}

inline __device__ void operator+=(double2 &a, double2 b) {
  a.x += b.x;
  a.y += b.y;
}

inline __device__ void operator+=(float4 &a, float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}

// Optimized methods
#define UNROLL_PIXELS 4
#define BATCH_SIZE 128
#define MAX_NR_CHANNELS 8

inline __device__ void apply_aterm(const float2 aXX1, const float2 aXY1,
                                   const float2 aYX1, const float2 aYY1,
                                   const float2 aXX2, const float2 aXY2,
                                   const float2 aYX2, const float2 aYY2,
                                   float2 pixels[NR_CORRELATIONS]) {
  float2 pixelsXX = pixels[0];
  float2 pixelsXY = pixels[1];
  float2 pixelsYX = pixels[2];
  float2 pixelsYY = pixels[3];

  // Apply aterm to subgrid: P = A1 * P
  // [ pixels[0], pixels[1];  = [ aXX1, aXY1;  [ pixelsXX, pixelsXY;
  //   pixels[2], pixels[3] ]     aYX1, aYY1 ]   pixelsYX], pixelsYY ] *
  pixels[0] = (pixelsXX * aXX1);
  pixels[0] += (pixelsYX * aXY1);
  pixels[1] = (pixelsXY * aXX1);
  pixels[1] += (pixelsYY * aXY1);
  pixels[2] = (pixelsXX * aYX1);
  pixels[2] += (pixelsYX * aYY1);
  pixels[3] = (pixelsXY * aYX1);
  pixels[3] += (pixelsYY * aYY1);

  pixelsXX = pixels[0];
  pixelsXY = pixels[1];
  pixelsYX = pixels[2];
  pixelsYY = pixels[3];

  // Apply aterm to subgrid: P = P * A2^H
  //    [ pixels[0], pixels[1];  =   [ pixelsXX, pixelsXY;  *  [ conj(aXX2),
  //    conj(aYX2);
  //      pixels[2], pixels[3] ]       pixelsYX, pixelsYY ]      conj(aXY2),
  //      conj(aYY2) ]
  pixels[0] = (pixelsXX * conj(aXX2));
  pixels[0] += (pixelsXY * conj(aXY2));
  pixels[1] = (pixelsXX * conj(aYX2));
  pixels[1] += (pixelsXY * conj(aYY2));
  pixels[2] = (pixelsYX * conj(aXX2));
  pixels[2] += (pixelsYY * conj(aXY2));
  pixels[3] = (pixelsYX * conj(aYX2));
  pixels[3] += (pixelsYY * conj(aYY2));
}

inline __device__ void apply_aterm(const float2 aXX1, const float2 aXY1,
                                   const float2 aYX1, const float2 aYY1,
                                   const float2 aXX2, const float2 aXY2,
                                   const float2 aYX2, const float2 aYY2,
                                   float2 &uvXX, float2 &uvXY, float2 &uvYX,
                                   float2 &uvYY) {
  float2 uv[NR_CORRELATIONS] = {uvXX, uvXY, uvYX, uvYY};

  apply_aterm(aXX1, aXY1, aYX1, aYY1, aXX2, aXY2, aYX2, aYY2, uv);

  uvXX = uv[0];
  uvXY = uv[1];
  uvYX = uv[2];
  uvYY = uv[3];
}

inline __device__ long index_subgrid(int subgrid_size, int s, int pol, int y,
                                     int x) {
  // subgrid: [nr_subgrids][NR_POLARIZATIONS][subgrid_size][subgrid_size]
  return s * NR_CORRELATIONS * subgrid_size * subgrid_size +
         pol * subgrid_size * subgrid_size + y * subgrid_size + x;
}

inline __device__ int index_visibility(int nr_channels, int time, int chan,
                                       int pol) {
  // visibilities: [nr_time][nr_channels][nr_polarizations]
  return time * nr_channels * NR_CORRELATIONS + chan * NR_CORRELATIONS + pol;
}

inline __device__ int index_aterm(int subgrid_size, int nr_stations,
                                  int aterm_index, int station, int y, int x) {
  // aterm: [nr_aterms][subgrid_size][subgrid_size][NR_CORRELATIONS]
  int aterm_nr = (aterm_index * nr_stations + station);
  return aterm_nr * subgrid_size * subgrid_size * NR_CORRELATIONS +
         y * subgrid_size * NR_CORRELATIONS + x * NR_CORRELATIONS;
}

inline __device__ void read_aterm(int subgrid_size, int nr_stations,
                                  int aterm_index, int station, int y, int x,
                                  const float2 *aterms_ptr, float2 *aXX,
                                  float2 *aXY, float2 *aYX, float2 *aYY) {
  int station_idx =
      index_aterm(subgrid_size, nr_stations, aterm_index, station, y, x);
  float4 *aterm_ptr = (float4 *)&aterms_ptr[station_idx];
  float4 atermA = aterm_ptr[0];
  float4 atermB = aterm_ptr[1];
  *aXX = make_float2(atermA.x, atermA.y);
  *aXY = make_float2(atermA.z, atermA.w);
  *aYX = make_float2(atermB.x, atermB.y);
  *aYY = make_float2(atermB.z, atermB.w);
}
