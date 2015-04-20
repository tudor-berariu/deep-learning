// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef HYPERBOLIC_TANGENT_H
#define HYPERBOLIC_TANGENT_H

#include <cmath>
#include <array>

template<typename T>
struct HyperbolicTangent;

template<typename T>
struct HyperbolicTangent {
  inline static T f(T x) {
    const T e2x = exp(-2 * x);
    return ((T)1 - e2x) / ((T)1 + e2x);
  }

  inline static T df(T y) {
    return (T)1 - y * y;
  }

  /* All neurons on a layer */
  template<typename LayerSize>
  using Neurons = std::array<T, LayerSize::length>;

  template<typename LayerSize>
  inline static void
  f_layer(const Neurons<LayerSize>& Z, Neurons<LayerSize>& A) {
    for (size_t j = 0; j < LayerSize::length; j++) {
      A[j] = f(Z[j]);
    }
  }

  template<typename LayerSize>
  inline static void
  df_layer(const Neurons<LayerSize>& A, Neurons<LayerSize>& Err){
    for (size_t j = 0; j < LayerSize::length; j++) {
      Err[j] *= ((T)1 - A[j] * A[j]);
    }
  }

  /* All neurons in a batch */
  template<typename LayerSize, size_t batch_size>
  using Batch = std::array<Neurons<LayerSize>, batch_size>;

  template<typename LayerSize, size_t batch_size>
  inline static void
  f_batch(const Batch<LayerSize, batch_size>& Z,
          Batch<LayerSize, batch_size>& A) {
    constexpr size_t full_length = LayerSize::length * batch_size;
    using LinearBatch = std::array<T, LayerSize::length * batch_size>;

    const LinearBatch& z = *reinterpret_cast<const LinearBatch*>(Z.data());
    LinearBatch& a = *reinterpret_cast<LinearBatch*>(A.data());

    for (size_t j = 0; j < full_length; j++)
      a[j] = f(z[j]);
  }

  template<typename LayerSize, size_t batch_size>
  inline static void
  df_batch(const Batch<LayerSize, batch_size>& A,
           Batch<LayerSize, batch_size>& Err) {
    constexpr size_t full_length = LayerSize::length * batch_size;
    using LinearBatch = std::array<T, LayerSize::length * batch_size>;

    const LinearBatch& a = *reinterpret_cast<const LinearBatch*>(A.data());
    LinearBatch& err = *reinterpret_cast<LinearBatch*>(Err.data());

    for (size_t j = 0; j < full_length; j++)
      err[j] *= ((T)1 - a[j] * a[j]);
  }
};

#endif
