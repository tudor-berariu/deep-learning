// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <cmath>
#include <array>

using namespace std;

template<typename T>
struct SoftMax {
  static constexpr bool transforms_last_layer = true;
  static constexpr bool supports_backpropagation = true;

  template<typename LayerSize>
  using Output = array<T, LayerSize::length>;

  template<typename LayerSize, size_t batch_size>
  using Outputs = array<Output<LayerSize>, batch_size>;

  template<typename LayerSize, size_t batch_size>
  inline static void
  f(const Outputs<LayerSize, batch_size>& a,
    Outputs<LayerSize, batch_size>& y) {
    for (size_t n = 0; n < batch_size; n++) {
      T sum = 0;
      Output<LayerSize>* const y_row =
        reinterpret_cast<Output<LayerSize>*>(y[n].data());
      for (size_t i = 0; i < LayerSize::length; i++) {
        (*y_row)[i] = exp(a[n][i]);
        sum += (*y_row)[i];
      }
      for (size_t i = 0; i < LayerSize::length; i++) {
        (*y_row)[i] = (*y_row)[i] / sum;
      }
    }
  }

  template<typename LayerSize, size_t batch_size>
  inline static T
  error(const Outputs<LayerSize, batch_size>& y,
        const Outputs<LayerSize, batch_size>& t) {
    T err = 0;
    for (size_t n = 0; n < batch_size; n++) {
      const Output<LayerSize>* const y_row =
        reinterpret_cast<const Output<LayerSize>*>(y[n].data());
      const Output<LayerSize>* const t_row =
        reinterpret_cast<const Output<LayerSize>*>(t[n].data());
      for (size_t i = 0; i < LayerSize::length; i++)
        err += (*t_row)[i] * log((*y_row)[i]);
    }
    return err;
  }

  template<typename LayerSize, size_t batch_size>
  inline static void
  dError(const Outputs<LayerSize, batch_size>& y,
         const Outputs<LayerSize, batch_size>& t,
         Outputs<LayerSize, batch_size>& e) {
    for (size_t n = 0; n < batch_size; n++) {
      const Output<LayerSize>* const y_row =
        reinterpret_cast<const Output<LayerSize>*>(y[n].data());
      const Output<LayerSize>* const t_row =
        reinterpret_cast<const Output<LayerSize>*>(t[n].data());
      Output<LayerSize>* const e_row =
        reinterpret_cast<Output<LayerSize>*>(e[n].data());
      for (size_t i = 0; i < LayerSize::length; i++)
        (*e_row)[i] = (*y_row)[i] - (*t_row)[i];
    }
  }
};

#endif
