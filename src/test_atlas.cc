// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifdef USE_ATLAS
extern "C" {
  #include <cblas.h>
}
#endif

#include <iostream>
#include <array>

template<size_t M, size_t N, size_t K>
void test_dgemm() {
  std::array<std::array<double, K>, M> A;
  std::array<std::array<double, K>, N> B;
  std::array<std::array<double, N>, M> C;

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K,
              0.01, reinterpret_cast<const double *>(&A), K,
              reinterpret_cast<const double *>(&B), K,
              0.01, reinterpret_cast<double *>(&C), N);
  std::cout << "ok" << std::endl;
}
              


int main(int argc, char *argv[]) {
  test_dgemm<5, 3, 10>();
  return 0;
}
