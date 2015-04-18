// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#include <iostream>
#include <typeinfo>

#include "deep_learning/neural_networks/feed_forward_net.h"

int main(int argc, char *argv[]) {
  using NN1 = FeedForwardNet<double>;
  std::cout << typeid(NN1::DataType).name() << std::endl;
  return 0;
}
