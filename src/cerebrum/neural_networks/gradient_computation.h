// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef GRADIENT_COMPUTATION_H
#define GRADIENT_COMPUTATION_H

#include <array>
#include "cerebrum/neural_networks/parameters.h"

template<typename T, size_t batch_size, typename ErrorFunction, bool computes,
         typename InputSize, typename... OtherLayers>
struct _GradientComputation;

template<typename T, size_t batch_size, typename ErrorFunction,
         typename InputSize>
struct _GradientComputation<T, batch_size, ErrorFunction, true, InputSize> {

  using NetOutputs = std::array<std::array<T, InputSize::length>, batch_size>;
  NetOutputs y;

  T computeGradient(const NetOutputs& outputs, bool parameters,
                    const NetOutputs& labels,
                    NetOutputs& prev_errors, bool gradient) {
    ErrorFunction::template f<InputSize, batch_size>(outputs, y);
    ErrorFunction::template
      dError<InputSize, batch_size>(y, labels, prev_errors);
    return ErrorFunction::template error<InputSize, batch_size>(y, labels);
  }
};

template<typename T, size_t batch_size, typename ErrorFunction,
         typename InputSize>
struct _GradientComputation<T, batch_size, ErrorFunction, false, InputSize> {

  using NetOutputs = std::array<std::array<T, InputSize::length>, batch_size>;

  T computeGradient(const NetOutputs& outputs, bool parameters,
                    const NetOutputs& labels,
                    NetOutputs& prev_errors, bool gradient) {
    ErrorFunction::dError(outputs, labels, prev_errors);
    return ErrorFunction::error(outputs, labels);
  }
};


template<typename T, size_t batch_size, typename ErrorFunction, bool computes,
         typename InputSize, typename CrtLayer, typename... Others>
struct _GradientComputation<T, batch_size, ErrorFunction, computes,
                            InputSize, CrtLayer, Others...> {

  using Inputs  = typename CrtLayer::template Inputs<T, InputSize, batch_size>;
  using Hidden  = typename CrtLayer::template Hidden<T, InputSize, batch_size>;
  using Outputs = typename CrtLayer::template Outputs<T, InputSize, batch_size>;

  using OutputSize = typename CrtLayer::template OutputSize<InputSize>;
  using NextComputation =
    _GradientComputation<T, batch_size, ErrorFunction, computes,
                         OutputSize, Others...>;

  using NetOutputs = typename NextComputation::NetOutputs;
  using Parameters = _Parameters<T, InputSize, CrtLayer, Others...>;

  Hidden hidden;
  Outputs outputs;
  Outputs errors;
  NextComputation next;

  T computeGradient(const Inputs& inputs, const Parameters& parameters,
                       const NetOutputs& labels, Inputs& prev_errors,
                       Parameters& gradient) {
    CrtLayer::template
      forward<T, InputSize, batch_size, true>(inputs, parameters.values,
                                              hidden, outputs);
    T err = next.computeGradient(outputs, parameters.next, labels,
                                 errors, gradient.next);
    CrtLayer::template
      backpropagate<T, InputSize, batch_size>(inputs, parameters.values,
                                              hidden, outputs, errors,
                                              gradient.values, prev_errors);
    return err;
  }

  T computeGradient(const Inputs& inputs, const Parameters& parameters,
                    const NetOutputs& labels, Parameters& gradient) {
    Inputs crt_errors;
    return computeGradient(inputs, parameters, labels, crt_errors, gradient);
  }

};

#endif
