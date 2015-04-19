// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#include <iostream>
#include <random>
#include <typeinfo>
#include <chrono>

//#include <fenv.h>

#include "deep_learning/size.h"
#include "deep_learning/neural_networks.h"

template<typename T, size_t batch_size, size_t length>
using Arr = std::array<std::array<T, length>, batch_size>;

template<typename T, size_t batch_size, size_t length>
Arr<T, batch_size, length>* get_dummy_example() {
  using MyArr = Arr<T, batch_size, length>;
  MyArr* a = new MyArr;
  std::random_device rd{ };
  std::default_random_engine e{rd()};
  std::uniform_real_distribution<T> next(0, 1);
  for (size_t n = 0; n < batch_size; n++)
    for (size_t j = 0; j < length; j++)
      (*a)[n][j] = next(e);
  return a;
}

template<size_t batch_size, typename NN>
void test_performance() {
  using FC = typename NN::template ForwardComputation<batch_size, RMSE>;
  using GC = typename NN::template GradientComputation<batch_size, SoftMax>;
  using P1 = typename NN::Parameters;

  FC* fc = new FC;
  GC* gc = new GC;
  P1* p = new P1;
  P1* g = new P1;

  double fw_avg = 0.0;
  double bw_avg = 0.0;
  for (size_t t_no = 0; t_no < 10; t_no++) {
    typename FC::Inputs* x =
      get_dummy_example<double, batch_size, NN::InputSize::length>();
    typename FC::NetOutputs* t =
      get_dummy_example<double, batch_size, NN::OutputSize::length>();

    std::chrono::steady_clock::time_point a = std::chrono::steady_clock::now();
    fc->forward(*x, *p);
    double err = fc->error(*t);
    std::chrono::steady_clock::time_point b = std::chrono::steady_clock::now();
    double err2 = 0; //gc->computeGradient(*x, *p, *t, *g);
    std::chrono::steady_clock::time_point c = std::chrono::steady_clock::now();

    std::chrono::duration<double> fw_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(b - a);
    std::chrono::duration<double> bw_span =
      std::chrono::duration_cast<std::chrono::duration<double>>(c - b);
    std::cout << err << " in " << fw_span.count() << " \t and "
              << err2 << " in " << bw_span.count() << std::endl;
    fw_avg += fw_span.count();
    bw_avg += bw_span.count();
    delete x;
    delete t;
  }
  delete fc;
  delete gc;
  delete p;
  delete g;
  std::cout << "avg forward in " << fw_avg / 10.0 << " \t and "
            << "avg backprop in " << bw_avg / 10.0 << std::endl;
}

int main() {
  //feenableexcept(FE_INVALID | FE_OVERFLOW);
  using NN = FeedForwardNet<double,
                            Size<10>,
                            FullyConnected<1000, Logistic>,
                            FullyConnected<1000, ReLU>,
                            FullyConnected<2000, HyperbolicTangent>,
                            FullyConnected<100, Identity>>;
  test_performance<300, NN>();
  return 0;
}
