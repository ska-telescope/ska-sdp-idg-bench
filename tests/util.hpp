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