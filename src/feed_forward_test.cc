// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#include <iostream>
#include <random>
#include <typeinfo>

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
  using NN1 = FeedForwardNet<double,
                             Size<784>,
                             FullyConnected<20, Logistic>,
                             FullyConnected<30, Logistic>>;
  using P1 = NN1::Parameters;
  using FC = NN1::ForwardComputation<200, RMSE>;
  std::cout << typeid(NN1::DataType).name() << std::endl;
  std::cout << typeid(P1).name() << " " << sizeof(P1) << std::endl;
  std::cout << typeid(FC).name() << std::endl;
  FC* fc = new FC;
  P1* p = new P1;
  FC::Inputs* x = get_dummy_example<double, 200, 784>();
  FC::NetOutputs* t = get_dummy_example<double, 200, 30>();
  fc->forward(*x, *p);
  std::cout << fc->error(*t);
  delete fc;
  return 0;
}
