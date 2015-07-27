// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <cstddef>
#include <iostream>
#include <random>

template<typename T, typename InputSize, typename... Other>
struct _Parameters;

template<typename T, typename InputSize, typename LastLayer>
struct _Parameters<T, InputSize, LastLayer> {
  typename LastLayer::template Parameters<T, InputSize> values;
  bool next;

  _Parameters() {
    LastLayer::template init_parameters<T, InputSize>(values);
  }

  _Parameters(T value) {
    for (size_t i = 0;
         i < LastLayer::template parameters_array_size<InputSize>();
         i++)
      values[i] = value;
  }

  _Parameters(T min, T max) {
    std::random_device rd { };
    std::default_random_engine e {rd()};
    std::uniform_real_distribution<T> next_parameter(min, max);
    for (size_t i = 0;
         i < LastLayer::template parameters_array_size<InputSize>();
         i++)
      values[i] = next_parameter(e);
  }

  friend std::ostream&
  operator<<(std::ostream& s, const _Parameters<T, InputSize, LastLayer>&){
    s << std::endl;
    return s;
  }

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

  _Parameters() {
    CrtLayer::template init_parameters<T, InputSize>(values);
  }

  _Parameters(T value) : next {value} {
    for (size_t i = 0;
         i < CrtLayer::template parameters_array_size<InputSize>();
         i++)
      values[i] = value;
  }

  _Parameters(T min, T max) : next {min, max} {
    std::random_device rd { };
    std::default_random_engine e {rd()};
    std::uniform_real_distribution<T> next_parameter(min, max);
    for (size_t i = 0;
         i < CrtLayer::template parameters_array_size<InputSize>();
         i++)
      values[i] = next_parameter(e);
  }

  friend std::ostream&
  operator<<(std::ostream& s,
             const _Parameters<T, InputSize, CrtLayer, Other...>& p) {
    for (size_t i = 0; i < p.parameters_no; i++) {
      s << p.values[i] << " ";
    }
    s << std::endl << "-----" << std::endl << p.next;
    return s;
  }
};

#endif
