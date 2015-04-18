// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include <cstddef>
#include <array>
#include <random>

#include "deep_learning/size.h"

template<size_t length>
struct FullyConnected {

  /* -------------------- OutputSize -------------------- */

 public:

  template<typename InputSize>
  using OutputSize = Size<length>;

  /* -------------------- Parameters -------------------- */

 public:

  template<typename InputSize>
  static constexpr size_t
  parameters_array_size() {
    return (InputSize::length + 1) * length;
  }

  template<typename InputSize>
  static constexpr size_t
  parameters_no() {
    return parameters_array_size<InputSize>();
  }

  template<typename T, typename InputSize>
  using Parameters = std::array<T, parameters_array_size<InputSize>()>;

 private:

  template<typename T, typename InputSize>
  using _Biases = std::array<T, OutputSize<InputSize>::length>;

  template<typename T, typename InputSize>
  using _Weights =
    std::array<std::array<T, InputSize::length>,
               OutputSize<InputSize>::length>;

 public:

  template<typename T, typename InputSize>
  inline static void
  init_parameters(Parameters<T, InputSize>& parameters, T value) {
    for (size_t i = 0; i < parameters_array_size<InputSize>(); i++)
      parameters[i] = value;
  }

  template<typename T, typename InputSize>
  inline static void
  init_parameters(Parameters<T, InputSize>& parameters, T min, T max) {
    std::random_device rd { };
    std::default_random_engine e {rd()};
    std::uniform_real_distribution<T> next_parameter(min, max);
    for (size_t i = 0; i < parameters_array_size<InputSize>(); i++)
      parameters[i] = next_parameter(e);
  }

  template<typename T, typename InputSize>
  inline static void
  init_parameters(Parameters<T, InputSize>& parameters) {
    std::random_device rd { };
    std::default_random_engine e {rd()};
    constexpr T mean = (T)1 / (T)length;
    std::normal_distribution<T> next_parameter(mean, 1.0);

    for (size_t i = 0; i < parameters_array_size<InputSize>(); i++)
      parameters[i] = next_parameter(e);
  }
};

#endif
