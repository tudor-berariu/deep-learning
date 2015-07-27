// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef FORWARD_COMPUTATION_H
#define FORWARD_COMPUTATION_H

#include <cstddef>
#include <array>
#include <type_traits>

#include "cerebrum/neural_networks/parameters.h"

template<typename T, size_t batch_size,
         typename ErrorFunction, bool computes,
         typename InputSize, typename... OtherLayers>
struct _ForwardComputation;

template<typename T, size_t batch_size, typename ErrorFunction,
         typename LastSize>
struct _ForwardComputation<T, batch_size, ErrorFunction, false, LastSize> {

  using NetOutputs = std::array<std::array<T, LastSize::length>, batch_size>;
  const NetOutputs* y;

  const NetOutputs&
  forward(const NetOutputs& outputs, bool parameters) {
    y = &outputs;
    return outputs;
  }

  T error(const NetOutputs& labels) {
    return ErrorFunction::template error<LastSize, batch_size>(*y, labels);
  }
};

template<typename T, size_t batch_size, typename ErrorFunction,
         typename LastSize>
struct _ForwardComputation<T, batch_size, ErrorFunction, true, LastSize> {

  using NetOutputs = std::array<std::array<T, LastSize::length>, batch_size>;
  NetOutputs y;

  const NetOutputs&
  forward(const NetOutputs& outputs, bool parameters) {
    ErrorFunction::f(outputs, y);
    return y;
  }

  T error(const NetOutputs& labels) {
    return ErrorFunction::template error<LastSize, batch_size>(y, labels);
  }
};

template<typename T, size_t batch_size, typename ErrorFunction, bool computes,
         typename InputSize, typename CrtLayer, typename... Others>
struct _ForwardComputation<T, batch_size, ErrorFunction, computes,
                           InputSize, CrtLayer, Others...> {

  using Inputs =
    typename CrtLayer::template Inputs<T, InputSize, batch_size>;
  using Hidden =
    typename CrtLayer::template Hidden<T, InputSize, batch_size>;
  using Outputs =
    typename CrtLayer::template Outputs<T, InputSize, batch_size>;

  using OutputSize = typename CrtLayer::template OutputSize<InputSize>;
  using NextComputation =
    _ForwardComputation<T, batch_size, ErrorFunction, computes,
                        OutputSize, Others...>;

  using NetOutputs = typename NextComputation::NetOutputs;
  using Parameters = _Parameters<T, InputSize, CrtLayer, Others...>;

  Hidden hidden;
  Outputs outputs;
  NextComputation next;

  const NetOutputs&
  forward(const Inputs& inputs, const Parameters& parameters) {
    CrtLayer::template
      forward<T, InputSize, batch_size, false>(inputs, parameters.values,
                                               hidden, outputs);
    return next.forward(outputs, parameters.next);
  }

  T error(const NetOutputs& labels) {
    return next.error(labels);
  }
};

#endif
