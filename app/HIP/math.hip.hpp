#pragma once

#include "common/parameters.hpp"
#include <hip/hip_runtime.h>

inline __device__ float2 conj(float2 a)
{
  return make_float2(a.x, -a.y);
}

inline __device__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
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
  pixels[0] = cmul(pixelsXX, aXX1);
  pixels[0] += cmul(pixelsYX, aXY1);
  pixels[1] = cmul(pixelsXY, aXX1);
  pixels[1] += cmul(pixelsYY, aXY1);
  pixels[2] = cmul(pixelsXX, aYX1);
  pixels[2] += cmul(pixelsYX, aYY1);
  pixels[3] = cmul(pixelsXY, aYX1);
  pixels[3] += cmul(pixelsYY, aYY1);

  pixelsXX = pixels[0];
  pixelsXY = pixels[1];
  pixelsYX = pixels[2];
  pixelsYY = pixels[3];

  // Apply aterm to subgrid: P = P * A2^H
  //    [ pixels[0], pixels[1];  =   [ pixelsXX, pixelsXY;  *  [ conj(aXX2),
  //    conj(aYX2);
  //      pixels[2], pixels[3] ]       pixelsYX, pixelsYY ]      conj(aXY2),
  //      conj(aYY2) ]
  pixels[0] = cmul(pixelsXX, conj(aXX2));
  pixels[0] += cmul(pixelsXY, conj(aXY2));
  pixels[1] = cmul(pixelsXX, conj(aYX2));
  pixels[1] += cmul(pixelsXY, conj(aYY2));
  pixels[2] = cmul(pixelsYX, conj(aXX2));
  pixels[2] += cmul(pixelsYY, conj(aXY2));
  pixels[3] = cmul(pixelsYX, conj(aYX2));
  pixels[3] += cmul(pixelsYY, conj(aYY2));
}


inline __device__ void apply_aterm_fp32(const float2 aXX1, const float2 aXY1,
                                   const float2 aYX1, const float2 aYY1,
                                   const float2 aXX2, const float2 aXY2,
                                   const float2 aYX2, const float2 aYY2,
                                   float2 pixels[NR_CORRELATIONS]) {
  float2 pixelsXX = pixels[0];
  float2 pixelsXY = pixels[1];
  float2 pixelsYX = pixels[2];
  float2 pixelsYY = pixels[3];

  pixels[0] = make_float2(0, 0);
  pixels[1] = make_float2(0, 0);
  pixels[2] = make_float2(0, 0);
  pixels[3] = make_float2(0, 0);
  
  float2 aXX1c1 = make_float2(aXX1.x, aXX1.x);
  float2 aXX1c2 = make_float2(-aXX1.y, aXX1.y);  
  float2 aXY1c1 = make_float2(aXY1.x, aXY1.x);
  float2 aXY1c2 = make_float2(-aXY1.y, aXY1.y);
  float2 aYX1c1 = make_float2(aYX1.x, aYX1.x);
  float2 aYX1c2 = make_float2(-aYX1.y, aYX1.y);  
  float2 aYY1c1 = make_float2(aYY1.x, aYY1.x);
  float2 aYY1c2 = make_float2(-aYY1.y, aYY1.y);
   
  float2 pixelsXX_i = make_float2(pixelsXX.y, pixelsXX.x); 
  float2 pixelsXY_i = make_float2(pixelsXY.y, pixelsXY.x); 
  float2 pixelsYX_i = make_float2(pixelsYX.y, pixelsYX.x);
  float2 pixelsYY_i = make_float2(pixelsYY.y, pixelsYY.x); 

  pixels[0] += pixelsXX * aXX1c1;
  pixels[0] += pixelsXX_i * aXX1c2;
  pixels[0] += pixelsYX * aXY1c1;
  pixels[0] += pixelsYX_i * aXY1c2;

  pixels[1] += pixelsXY * aXX1c1;
  pixels[1] += pixelsXY_i * aXX1c2;
  pixels[1] += pixelsYY * aXY1c1;
  pixels[1] += pixelsYY_i * aXY1c2;

  pixels[2] += pixelsXX * aYX1c1;
  pixels[2] += pixelsXX_i * aYX1c2;
  pixels[2] += pixelsYX * aYY1c1;
  pixels[2] += pixelsYX_i * aYY1c2;

  pixels[3] += pixelsXY * aYX1c1;
  pixels[3] += pixelsXY_i * aYX1c2;
  pixels[3] += pixelsYY * aYY1c1;
  pixels[3] += pixelsYY_i * aYY1c2;

  pixelsXX = pixels[0];
  pixelsXY = pixels[1];
  pixelsYX = pixels[2];
  pixelsYY = pixels[3];

  pixels[0] = make_float2(0, 0);
  pixels[1] = make_float2(0, 0);
  pixels[2] = make_float2(0, 0);
  pixels[3] = make_float2(0, 0);

  float2 aXX2c1 = make_float2(aXX2.x, aXX2.x);
  float2 aXX2c2 = make_float2(aXX2.y, -aXX2.y);  
  float2 aXY2c1 = make_float2(aXY2.x, aXY2.x);
  float2 aXY2c2 = make_float2(aXY2.y, -aXY2.y);
  float2 aYX2c1 = make_float2(aYX2.x, aYX2.x);
  float2 aYX2c2 = make_float2(aYX2.y, -aYX2.y);  
  float2 aYY2c1 = make_float2(aYY2.x, aYY2.x);
  float2 aYY2c2 = make_float2(aYY2.y, -aYY2.y);

  pixelsXX_i = make_float2(pixelsXX.y, pixelsXX.x); 
  pixelsXY_i = make_float2(pixelsXY.y, pixelsXY.x); 
  pixelsYX_i = make_float2(pixelsYX.y, pixelsYX.x);
  pixelsYY_i = make_float2(pixelsYY.y, pixelsYY.x); 

  pixels[0] += pixelsXX * aXX2c1;
  pixels[0] += pixelsXX_i * aXX2c2;
  pixels[0] += pixelsXY * aXY2c1;
  pixels[0] += pixelsXY_i * aXY2c2;

  pixels[1] += pixelsXX * aYX2c1;
  pixels[1] += pixelsXX_i * aYX2c2;
  pixels[1] += pixelsXY * aYY2c1;
  pixels[1] += pixelsXY_i * aYY2c2;

  pixels[2] += pixelsYX * aXX2c1;
  pixels[2] += pixelsYX_i * aXX2c2;
  pixels[2] += pixelsYY * aXY2c1;
  pixels[2] += pixelsYY_i * aXY2c2;

  pixels[3] += pixelsYX * aYX2c1;
  pixels[3] += pixelsYX_i * aYX2c2;
  pixels[3] += pixelsYY * aYY2c1;
  pixels[3] += pixelsYY_i * aYY2c2;
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

inline __device__ void apply_aterm_fp32(const float2 aXX1, const float2 aXY1,
                                   const float2 aYX1, const float2 aYY1,
                                   const float2 aXX2, const float2 aXY2,
                                   const float2 aYX2, const float2 aYY2,
                                   float2 &uvXX, float2 &uvXY, float2 &uvYX,
                                   float2 &uvYY) {
  float2 uv[NR_CORRELATIONS] = {uvXX, uvXY, uvYX, uvYY};

  apply_aterm_fp32(aXX1, aXY1, aYX1, aYY1, aXX2, aXY2, aYX2, aYY2, uv);

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
