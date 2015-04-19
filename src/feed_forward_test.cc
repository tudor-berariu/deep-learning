// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#include <iostream>
#include <random>
#include <typeinfo>

#include <fenv.h>

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

int main() {
  feenableexcept(FE_INVALID | FE_OVERFLOW);
  using NN1 = FeedForwardNet<double,
                             Size<10>,
                             FullyConnected<1000, Logistic>,
                             FullyConnected<1000, ReLU>,
                             FullyConnected<200, HyperbolicTangent>,
                             FullyConnected<10, Identity>>;
  using P1 = NN1::Parameters;
  using FC = NN1::ForwardComputation<1000, RMSE>;
  using GC = NN1::GradientComputation<1000, SoftMax>;

  std::cout << typeid(NN1::DataType).name() << std::endl;
  std::cout << typeid(P1).name() << " " << sizeof(P1) << std::endl;
  std::cout << typeid(FC).name() << std::endl;

  FC* fc = new FC;
  GC* gc = new GC;
  P1* p = new P1;
  //std::cout << "Parameters: " << std::endl << (*p) << std::endl;
  P1* g = new P1;
  FC::Inputs* x = get_dummy_example<double, 1000, 10>();
  FC::NetOutputs* t = get_dummy_example<double, 1000, 10>();
  fc->forward(*x, *p);
  std::cout << fc->error(*t) << std::endl;
  std::cout << gc->computeGradient(*x, *p, *t, *g) << std::endl;
  delete fc;
  return 0;
}
