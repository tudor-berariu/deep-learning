// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef FORWARD_COMPUTATION_H
#define FORWARD_COMPUTATION_H

#include <cstddef>
#include <array>

#include "deep_learning/neural_networks/parameters.h"

template<typename T, size_t batch_size, typename ErrorFunction,
         typename InputSize, typename... OtherLayers>
struct _ForwardComputation;

template<typename T, size_t batch_size, typename ErrorFunction,
         typename OutputSize>
struct _ForwardComputation<T, batch_size, ErrorFunction, Size> {

  using NetOutputs = std::array<std::array<T, InputSize::length>, batch_size>;
  NetOutputs y;

  const NetOutputs&
  forward(const NetOutputs& outputs, bool parameters) {
    ErrorFunction::f(outputs, y);
    return y;
  }

  T error(const NetOutputs& labels) {
    return ErrorFunction::error(y, labels);
  }
};

template<typename T, size_t batch_size, typename ErrorFunction,
         typename InputSize, typename CrtLayer, typename... Others>
struct _ForwardComputation<T, batch_size, ErrorFunction,
                           InputSize, CrtLayer, Others...> {

  using Inputs =
    typename CrtLayer::template Inputs<T, InputSize, batch_size>;
  using Hidden =
    typename CrtLayer::template Hidden<T, InputSize, batch_size>;
  using Outputs =
    typename CrtLayer::template Outputs<T, InputSize, batch_size>;

  using OutputSize = typename CrtLayer::template OutputSize<InputSize>;
  using NextComputation =
    _ForwardComputation<T, batch_size, ErrorFunction, OutputSize, Others...>;

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
    return NextComputation::forward(outputs, parameters.next);
  }

  T error(const NetOutputs& labels) {
    return NextComputation::error(labels);
  }
};

#endif
