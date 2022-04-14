#pragma once

#include <cmath>

#ifndef FUNCTION_ATTRIBUTES
#define FUNCTION_ATTRIBUTES
#endif

inline float FUNCTION_ATTRIBUTES compute_l(int x, int subgrid_size,
                                           float image_size) {
  return (x + 0.5 - (subgrid_size / 2)) * image_size / subgrid_size;
}

inline float FUNCTION_ATTRIBUTES compute_m(int y, int subgrid_size,
                                           float image_size) {
  return compute_l(y, subgrid_size, image_size);
}

inline float FUNCTION_ATTRIBUTES compute_n(float l, float m) {
  // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
  // accurately for small values of l and m
  const float tmp = (l * l) + (m * m);
  return tmp > 1.0 ? 1.0 : tmp / (1.0f + sqrtf(1.0f - tmp));
}

template <typename T>
inline FUNCTION_ATTRIBUTES void matmul(const T *a, const T *b, T *c) {
  c[0] = a[0] * b[0];
  c[1] = a[0] * b[1];
  c[2] = a[2] * b[0];
  c[3] = a[2] * b[1];
  c[0] += a[1] * b[2];
  c[1] += a[1] * b[3];
  c[2] += a[3] * b[2];
  c[3] += a[3] * b[3];
}

template <typename T>
inline FUNCTION_ATTRIBUTES void conjugate(const T *a, T *b) {
  float s[8] = {1, -1, 1, -1, 1, -1, 1, -1};
  float *a_ptr = (float *)a;
  float *b_ptr = (float *)b;

  for (unsigned i = 0; i < 8; i++) {
    b_ptr[i] = s[i] * a_ptr[i];
  }
}

template <typename T>
inline FUNCTION_ATTRIBUTES void transpose(const T *a, T *b) {
  b[0] = a[0];
  b[1] = a[2];
  b[2] = a[1];
  b[3] = a[3];
}

template <typename T>
inline FUNCTION_ATTRIBUTES void hermitian(const T *a, T *b) {
  T temp[4];
  conjugate(a, temp);
  transpose(temp, b);
}

template <typename T>
inline FUNCTION_ATTRIBUTES void apply_aterm_gridder(T *pixels, const T *aterm1,
                                                    const T *aterm2) {
  // Aterm 1 hermitian
  T aterm1_h[4];
  hermitian(aterm1, aterm1_h);

  // Apply aterm: P = A1^H * P
  T temp[4];
  matmul(aterm1_h, pixels, temp);

  // Apply aterm: P = P * A2
  matmul(temp, aterm2, pixels);
}

template <typename T>
inline FUNCTION_ATTRIBUTES void
apply_aterm_degridder(T *pixels, const T *aterm1, const T *aterm2) {
  // Apply aterm: P = A1 * P
  T temp[4];
  matmul(aterm1, pixels, temp);

  // Aterm 2 hermitian
  T aterm2_h[4];
  hermitian(aterm2, aterm2_h);

  // Apply aterm: P = P * A2^H
  matmul(temp, aterm2_h, pixels);
}
