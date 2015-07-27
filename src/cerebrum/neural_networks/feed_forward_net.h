// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef FEED_FORWARD_NET_H
#define FEED_FORWARD_NET_H

#include "cerebrum/neural_networks/parameters.h"
#include "cerebrum/neural_networks/forward_computation.h"
#include "cerebrum/neural_networks/gradient_computation.h"

template<typename... info>
struct NetOutput;

template<typename T, typename _InputSize, typename... LayersInfo>
struct FeedForwardNet {
  using DataType = T;
  using Parameters = _Parameters<T, _InputSize, LayersInfo...>;

  using InputSize = _InputSize;
  using OutputSize = typename NetOutput<InputSize, LayersInfo...>::OutputSize;

  template <size_t batch_size, template<typename> class ErrorFunction>
  using ForwardComputation =
    _ForwardComputation<T, batch_size, ErrorFunction<T>,
                        ErrorFunction<T>::transforms_last_layer,
                        InputSize, LayersInfo...>;

  template <size_t batch_size, template<typename> class ErrorFunction>
  using GradientComputation =
    _GradientComputation<T, batch_size, ErrorFunction<T>,
                         ErrorFunction<T>::transforms_last_layer,
                         InputSize, LayersInfo...>;

};

/* Tudor:
 * Extract size info from last layer
 */

template <typename InputSize, typename LastLayer>
struct NetOutput<InputSize, LastLayer> {
  using OutputSize = typename LastLayer::template OutputSize<InputSize>;
};

template <typename InputSize, typename CrtLayer, typename... OtherLayers>
struct NetOutput<InputSize, CrtLayer, OtherLayers...> {
  using CrtSize = typename CrtLayer::template OutputSize<InputSize>;
  using OutputSize = typename NetOutput<CrtSize, OtherLayers...>::OutputSize;
};

#endif
