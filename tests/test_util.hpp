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

void compare_subgrids(idg::Array4D<std::complex<float>> &cpu_subgrids,
                      idg::Array4D<std::complex<float>> &gpu_subgrids) {
  bool equal = true;

  for (int i = 0; i < cpu_subgrids.size(); i++) {
    std::complex<float> diff = cpu_subgrids.data()[i] - gpu_subgrids.data()[i];
    if (abs(diff.real()) > 1e-2 || abs(diff.imag()) > 1e-2) {
      std::cout << "[" << i << "]: " << cpu_subgrids.data()[i]
                << " != " << gpu_subgrids.data()[i] << std::endl;
      std::cout << "diff: " << diff << std::endl;
      equal = false;
      break;
    }
  }

  if (equal) {
    std::cout << ">>> Result PASSED" << std::endl;
  }
}