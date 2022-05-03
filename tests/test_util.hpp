void fill_vector_rand(std::vector<float> &v) {
  srand(time(0));
  generate(v.begin(), v.end(), rand);
}

template <typename T> void print_vector(std::vector<T> v) {
  for (const auto e : v) {
    std::cout << e << std::endl;
  }
}

template <typename T> std::vector<T> copy_vector(std::vector<T> &v) {
  std::vector<T> vec(v);
  return vec;
}

bool check_value(double value, float threshold = 1e-2) {
  bool res = true;
  if (value > threshold) {
    res = false;
  }
  return res;
}

#define PRINT_ERRORS
#define N_ERRORS 1024
void check_error(const int n, const std::complex<float> *A,
                 const std::complex<float> *B) {
  double r_error = 0.0;
  double i_error = 0.0;
  int nnz = 0;

  float r_max = 1;
  float i_max = 1;
  for (int i = 0; i < n; i++) {
    float r_value = abs(A[i].real());
    float i_value = abs(A[i].imag());
    if (r_value > r_max) {
      r_max = r_value;
    }
    if (i_value > i_max) {
      i_max = i_value;
    }
  }

#if defined(PRINT_ERRORS)
  int nerrors = 0;
#endif

  for (int i = 0; i < n; i++) {
    float r_cmp = A[i].real();
    float i_cmp = A[i].imag();
    float r_ref = B[i].real();
    float i_ref = B[i].imag();
    double r_diff = r_ref - r_cmp;
    double i_diff = i_ref - i_cmp;
    if (1) { // abs(B[i]) > 0.0f) {
#if defined(PRINT_ERRORS)
      if (/*(abs(r_diff) > 0.0001f || abs(i_diff) > 0.0001f) &&*/ nerrors <
          N_ERRORS) {
        printf("%d: (%f, %f) - (%f, %f) = (%f, %f)\n", i, r_cmp, i_cmp, r_ref,
               i_ref, r_diff, i_diff);
        nerrors++;
      }
#endif
      nnz++;
      r_error += (r_diff * r_diff) / r_max;
      i_error += (i_diff * i_diff) / i_max;
    }
  }

#if defined(DEBUG)
  printf("r_error: %f\n", r_error);
  printf("i_error: %f\n", i_error);
  printf("nnz: %d\n", nnz);
#endif

  r_error /= std::max(1, nnz);
  i_error /= std::max(1, nnz);

  double mean_error = sqrt(r_error + i_error);

  bool equal = check_value(mean_error, 1e-5);

  if (equal) {
    std::cout << ">>> Result PASSED" << std::endl;
  } else {
    std::cout << ">>> Result FAILED" << std::endl;
  }
  std::cout << ">>> Error: " << mean_error << std::endl;
}

void compare_visibilities(
    idg::Array3D<idg::Visibility<std::complex<float>>> &cpu_visibilities,
    idg::Array3D<idg::Visibility<std::complex<float>>> &gpu_visibilities) {
  check_error(cpu_visibilities.size(),
              (std::complex<float> *)gpu_visibilities.data(),
              (std::complex<float> *)cpu_visibilities.data());
}

void compare_subgrids(idg::Array4D<std::complex<float>> &cpu_subgrids,
                      idg::Array4D<std::complex<float>> &gpu_subgrids) {
  check_error(cpu_subgrids.size(), gpu_subgrids.data(), cpu_subgrids.data());
}
