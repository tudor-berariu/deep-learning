// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef RMSE_H
#define RMSE_H

#include <cmath>
#include <cstddef>
#include <array>

template<typename T>
struct RMSE {
  static constexpr bool transforms_last_layer = false;
  static constexpr bool supports_backpropagation = false;

  template<typename LayerSize>
  using Output = std::array<T, LayerSize::length>;

  template<typename LayerSize, size_t batch_size>
  using Outputs = std::array<Output<LayerSize>, batch_size>;

  template<typename LayerSize, size_t batch_size>
  inline static T
  error(const Outputs<LayerSize, batch_size>& y,
        const Outputs<LayerSize, batch_size>& t) {
    Output<LayerSize> avg_label;
    for (size_t i = 0; i < LayerSize::length; i++)
      avg_label[i] = 0;
    for (size_t n = 0; n < batch_size; n++)
      for (size_t i = 0; i < LayerSize::length; i++)
        avg_label[i] += t[n][i];
    for (size_t i = 0; i < LayerSize::length; i++)
      avg_label[i] /= (T)batch_size;

    T err = (T)0;
    T norm = (T)0;

    for (size_t n = 0; n < batch_size; n++) {
      for (size_t i = 0; i < LayerSize::length; i++)
        err += (t[n][i] - y[n][i]) * (t[n][i] - y[n][i]);
      for (size_t i = 0; i < LayerSize::length; i++)
        norm += (t[n][i] - avg_label[i]) * (t[n][i] - avg_label[i]);
    }

    return (err / norm);
  }

};

#endif
