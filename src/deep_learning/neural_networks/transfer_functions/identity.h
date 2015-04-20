// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef IDENTITY_H
#define IDENTITY_H

#include <cstring>
#include <cmath>
#include <array>

template<typename T>
struct Identity {
  inline static T f(T x) {
    return x;
  }

  inline static T df(T) {
    return (T)1;
  }

  /* All neurons on a layer */
  template<typename LayerSize>
  using Neurons = std::array<T, LayerSize::length>;

  template<typename LayerSize>
  inline static void
  f_layer(const Neurons<LayerSize>& Z, Neurons<LayerSize>& A) {
    std::memcpy(A.data(), Z.data(), sizeof(Z));
  }

  template<typename LayerSize>
  inline static void
  df_layer(const Neurons<LayerSize>& , Neurons<LayerSize>& ) {}

  /* All neurons in a batch */
  template<typename LayerSize, size_t batch_size>
  using Batch = std::array<Neurons<LayerSize>, batch_size>;

  template<typename LayerSize, size_t batch_size>
  inline static void
  f_batch(const Batch<LayerSize, batch_size>& Z,
          Batch<LayerSize, batch_size>& A) {
    std::memcpy(A.data(), Z.data(), sizeof(Z));
  }

  template<typename LayerSize, size_t batch_size>
  inline static void
  df_batch(const Batch<LayerSize, batch_size>&,
           Batch<LayerSize, batch_size>&) { }
};

#endif
