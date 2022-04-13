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

bool check_value(std::complex<float> value, float threshold = 1e-2) {
  bool res = false;
  if (abs(value.real()) > threshold || abs(value.imag()) > threshold) {
    res = true;
  }
  return res;
}

void compare_visibilities(
    idg::Array3D<idg::Visibility<std::complex<float>>> &cpu_visibilities,
    idg::Array3D<idg::Visibility<std::complex<float>>> &gpu_visibilities) {
  bool equal = true;

  for (int i = 0; i < cpu_visibilities.size(); i++) {
    idg::Visibility<std::complex<float>> cpu_tmp = cpu_visibilities.data()[i];
    idg::Visibility<std::complex<float>> gpu_tmp = gpu_visibilities.data()[i];
    idg::Visibility<std::complex<float>> diff = cpu_tmp - gpu_tmp;

    if (check_value(diff.xx) || check_value(diff.xy) || check_value(diff.yx) ||
        check_value(diff.yy)) {
      std::cout << "[" << i << "]: " << cpu_tmp.xx << " != " << gpu_tmp.xx
                << std::endl
                << cpu_tmp.xy << " != " << gpu_tmp.xy << std::endl
                << cpu_tmp.yx << " != " << gpu_tmp.yx << std::endl
                << cpu_tmp.yy << " != " << gpu_tmp.yy << std::endl;
      std::cout << ">>> Result FAILED" << std::endl;
      equal = false;
      break;
    }
  }

  if (equal) {
    std::cout << ">>> Result PASSED" << std::endl;
  }
}

void compare_subgrids(idg::Array4D<std::complex<float>> &cpu_subgrids,
                      idg::Array4D<std::complex<float>> &gpu_subgrids) {
  bool equal = true;

  for (int i = 0; i < cpu_subgrids.size(); i++) {
    std::complex<float> diff = cpu_subgrids.data()[i] - gpu_subgrids.data()[i];
    if (check_value(diff)) {
      std::cout << "[" << i << "]: " << cpu_subgrids.data()[i]
                << " != " << gpu_subgrids.data()[i] << std::endl;
      std::cout << "diff: " << diff << std::endl;
      std::cout << ">>> Result FAILED" << std::endl;
      equal = false;
      break;
    }
  }

  if (equal) {
    std::cout << ">>> Result PASSED" << std::endl;
  }
}