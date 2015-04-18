// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <cstddef>

template<typename T, typename InputSize, typename... Other>
struct _Parameters;

template<typename T, typename InputSize, typename LastLayer>
struct _Parameters<T, InputSize, LastLayer> {
  typename LastLayer::template Parameters<T, InputSize> values;
  bool next;
};

template<typename T, typename InputSize, typename CrtLayer, typename... Other>
struct _Parameters<T, InputSize, CrtLayer, Other...> {
  using NextParameters =
    _Parameters<T, typename CrtLayer::template OutputSize<InputSize>,
                Other...>;
  using CrtParameters = typename CrtLayer::template Parameters<T, InputSize>;
 
  CrtParameters values;
  NextParameters next;

  static constexpr size_t parameters_array_size =
    CrtLayer::template parameters_array_size<InputSize>();
  static constexpr size_t parameters_no =
    CrtLayer::template parameters_no<InputSize>();

  /* Tudor:
   * Constructors: default / default value / uniform from interval [min, max]
   */

  _Parameters() : next { } {
    CrtLayer::template init_parameters<T, InputSize>(values);
  }

  _Parameters(T value) : next {value} {
    CrtLayer::template init_parameters<T, InputSize>(values, value);
  }

  _Parameters(T min, T max) : next {min, max} {
    CrtLayer::template init_parameters<T, InputSize>(values, min, max);
  }
};

#endif
