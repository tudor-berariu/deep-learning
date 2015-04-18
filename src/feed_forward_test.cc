// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#include <iostream>
#include <typeinfo>

#include "deep_learning/size.h"
#include "deep_learning/neural_networks.h"

int main() {
  using NN1 = FeedForwardNet<double,
                             Size<784>,
                             FullyConnected<20>,
                             FullyConnected<30>>;
  using P1 = NN1::Parameters;
  std::cout << typeid(NN1::DataType).name() << std::endl;
  std::cout << typeid(P1).name() << " " << sizeof(P1) << std::endl;
  return 0;
}
