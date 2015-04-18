// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef FEED_FORWARD_NET_H
#define FEED_FORWARD_NET_H

#include "deep_learning/neural_networks/parameters.h"

template<typename T, typename _InputSize, typename... LayersInfo>
struct FeedForwardNet {
  using DataType = T;
  using Parameters = _Parameters<T, _InputSize, LayersInfo...>;
};

#endif
