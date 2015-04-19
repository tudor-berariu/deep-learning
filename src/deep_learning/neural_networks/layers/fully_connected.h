// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include <cstddef>
#include <array>
#include <random>

#include "deep_learning/size.h"

template<size_t length, template <typename> class TransferFunction>
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
  using Parameters = std::array<T, (InputSize::length + 1) * length>;

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


  /* -------------------- Inputs, Hidden, and Outputs -------------------- */
 public:

  template<typename T, typename InputSize>
  using Input = std::array<T, InputSize::length>;

  template<typename T, typename InputSize, size_t batch_size>
  using Inputs = std::array<Input<T, InputSize>, batch_size>;

  template<typename T, typename InputSize>
  using Output = std::array<T, OutputSize<InputSize>::length>;

  template<typename T, typename InputSize, size_t batch_size>
  using Outputs = std::array<Output<T, InputSize>, batch_size>;

  template<typename T, typename InputSize, size_t batch_size>
  using Hidden = Outputs<T, OutputSize<InputSize>, batch_size>;

  /* -------------------- Forward phase -------------------- */

  template<typename T, typename InputSize, size_t batch_size, bool train>
  struct _Forward;

  template<typename T, typename InputSize, size_t batch_size, bool train>
  inline static void
  forward(const Inputs<T, InputSize, batch_size>& inputs,
          const Parameters<T, InputSize>& parameters,
          Hidden<T, InputSize, batch_size>& hidden,
          Outputs<T, InputSize, batch_size>& outputs) {
    _Forward<T, InputSize, batch_size, train>::
      forward(inputs, parameters, hidden, outputs);
  }

  template<typename T, typename InputSize, size_t batch_size, bool train>
  struct _Forward {
    inline static void
    forward(const Inputs<T, InputSize, batch_size>& inputs,
            const Parameters<T, InputSize>& parameters,
            Hidden<T, InputSize, batch_size>& hidden,
            Outputs<T, InputSize, batch_size>& outputs) {
      const _Biases<T, InputSize>& biases =
        *reinterpret_cast<const _Biases<T, InputSize>*>(parameters.data());
      const _Weights<T, InputSize>& weights =
        *reinterpret_cast<const _Weights<T, InputSize>*>(&(parameters[length]));

      for (size_t n = 0; n < batch_size; n++) {
        for (size_t j = 0; j < length; j++) {
          hidden[n][j] = biases[j];
          for (size_t i = 0; i < InputSize::length; i++)
            hidden[n][j] += inputs[n][i] * weights[j][i];
          outputs[n][j] = TransferFunction<T>::f(hidden[n][j]);
        }
      }
   }
  };

  /* -------------------- Backpropagation phase -------------------- */

  template<typename T, typename InputSize, size_t batch_size>
  struct _Backpropagate;

  template<typename T, typename InputSize, size_t batch_size>
  static inline void
  backpropagate(const Inputs<T, InputSize, batch_size>& inputs,
                const Parameters<T, InputSize>& parameters,
                const Hidden<T, InputSize, batch_size>& hidden,
                const Outputs<T, InputSize, batch_size>& outputs,
                Outputs<T, InputSize, batch_size>& errors,
                Parameters<T, InputSize>& gradients,
                Inputs<T, InputSize, batch_size>& prev_errors) {
    _Backpropagate<T, InputSize, batch_size>::
      backpropagate(inputs, parameters, hidden, outputs, errors, gradients,
                    prev_errors);
  }

  template<typename T, typename InputSize, size_t batch_size>
  struct _Backpropagate {
    inline static void
    backpropagate(const Inputs<T, InputSize, batch_size>& inputs,
                  const Parameters<T, InputSize>& parameters,
                  const Hidden<T, InputSize, batch_size>& hidden,
                  const Outputs<T, InputSize, batch_size>& outputs,
                  Outputs<T, InputSize, batch_size>& errors,
                  Parameters<T, InputSize>& gradients,
                  Inputs<T, InputSize, batch_size>& prev_errors) {
      const _Weights<T, InputSize>& weights =
        *reinterpret_cast<const _Weights<T, InputSize>*>(&(parameters[length]));

      _Biases<T, InputSize>& g_biases =
        *reinterpret_cast<_Biases<T, InputSize>*>(gradients.data());
      _Weights<T, InputSize>& g_weights =
        *reinterpret_cast<_Weights<T, InputSize>*>(&(gradients[length]));

      for (size_t j = 0; j < length; j++)
        g_biases[j] = (T)0;
      for (size_t j = 0; j < length; j++)
        for (size_t i = 0; i < InputSize::length; i++)
          g_weights[j][i] = (T)0;

      for (size_t n = 0; n < batch_size; n++) {
        for (size_t j = 0; j < length; j++) {
          errors[n][j] *= TransferFunction<T>::df(hidden[n][j]);
          g_biases[j] += errors[n][j];
          for (size_t i = 0; i < InputSize::length; i++)
            g_weights[j][i] += inputs[n][i] * errors[n][j];
        }
        for (size_t i = 0; i < InputSize::length; i++) {
          prev_errors[n][i] = (T)0;
          for (size_t j = 0; j < length; j++)
            prev_errors[n][i] += errors[n][j] * weights[j][i];
        }
      }
    }
  };
};

#endif
