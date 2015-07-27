// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef DROPOUT_H
#define DROPOUT_H

#include <cstddef>
#include <cstring>
#include <array>
#include <random>

#include "cerebrum/include_cblas.h"
#include "cerebrum/size.h"

template<size_t active_no>
struct Dropout { 

  /* -------------------- OutputSize -------------------- */

 public:

  template<typename InputSize>
  using OutputSize = InputSize;

  /* -------------------- Parameters -------------------- */

 public:

  template<typename InputSize>
  static constexpr size_t
  parameters_array_size() {
    return 0ul;
  }

  template<typename InputSize>
  static constexpr size_t
  parameters_no() {
    return 0ul;
  }

  template<typename T, typename InputSize>
  using Parameters = std::array<T, 0>;

  template<typename T, typename InputSize>
  inline static void
  init_parameters(Parameters<T, InputSize>&) { }

  /* -------------------- Inputs, Hidden, and Outputs -------------------- */
 public:

  template<typename T, typename InputSize>
  using Input = std::array<T, InputSize::length>;

  template<typename T, typename InputSize, size_t batch_size>
  using Inputs = std::array<Input<T, InputSize>, batch_size>;

  template<typename T, typename InputSize>
  using Output = Input<T, InputSize>;

  template<typename T, typename InputSize, size_t batch_size>
  using Outputs = Inputs<T, InputSize, batch_size>;

  template<typename T, typename InputSize, size_t batch_size>
  using Hidden = Output<T, InputSize>;

 private:

  template<typename T, typename InputSize, size_t batch_size>
  using _LinearInputs = std::array<T, InputSize::length * batch_size>;

  template<typename T, typename InputSize, size_t batch_size>
  using _LinearOutputs = _LinearInputs<T, InputSize, batch_size>;

  /* -------------------- Forward phase -------------------- */

 private:

  template<typename T, typename InputSize, size_t batch_size, bool train>
  struct _Forward;

 public:

  template<typename T, typename InputSize, size_t batch_size, bool train>
  inline static void
  forward(const Inputs<T, InputSize, batch_size>& inputs,
          const Parameters<T, InputSize>& parameters,
          Hidden<T, InputSize, batch_size>& hidden,
          Outputs<T, InputSize, batch_size>& outputs) {
    _Forward<T, InputSize, batch_size, train>::
      forward(inputs, parameters, hidden, outputs);
  }

 private:

#ifdef USE_CBLAS
/*
 * Tudor
 * No cblas optimization yet as there is no element wise multiplication
 * routine. Using a diagonal matrix made things worse.
 */
#endif

  template<typename T, typename InputSize, size_t batch_size, bool train>
  struct _Forward {
    inline static void
    forward(const Inputs<T, InputSize, batch_size>& inputs,
            const Parameters<T, InputSize>&,
            Hidden<T, InputSize, batch_size>& hidden,
            Outputs<T, InputSize, batch_size>& outputs) {
      if (train) {
        constexpr double p = (double)active_no / (double)(InputSize::length);
        std::random_device rd { };
        std::default_random_engine e {rd()};
        std::bernoulli_distribution take(p);

        using InputRow = Input<T, InputSize>;
        using OutputRow = Output<T, InputSize>;

        for (size_t i = 0; i < InputSize::length; i++)
          hidden[i] = (T)take(e);
        for (size_t n = 0; n < batch_size; n++) {
          const InputRow& input_row =
            *reinterpret_cast<const InputRow*>(inputs[n].data());
          OutputRow& output_row =
            *reinterpret_cast<OutputRow*>(outputs[n].data());
          for (size_t i = 0; i < InputSize::length; i++)
            output_row[i] = hidden[i] * input_row[i];
        }
      } else {
        std::memcpy(outputs.data(), inputs.data(), sizeof(outputs));
      }
    }
  };

  /* -------------------- Backpropagation phase -------------------- */

 private:

  template<typename T, typename InputSize, size_t batch_size>
  struct _Backpropagate;

 public:

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

 private:

#ifdef USE_CBLAS
  /* Tudor:
   * Nothing good here yet
   */
#endif

  template<typename T, typename InputSize, size_t batch_size>
  struct _Backpropagate {
    inline static void
    backpropagate(const Inputs<T, InputSize, batch_size>&,
                  const Parameters<T, InputSize>&,
                  const Hidden<T, InputSize, batch_size>& hidden,
                  const Outputs<T, InputSize, batch_size>&,
                  Outputs<T, InputSize, batch_size>& errors,
                  Parameters<T, InputSize>&,
                  Inputs<T, InputSize, batch_size>& prev_errors) {
      using InputRow = Input<T, InputSize>;
      using OutputRow = Output<T, InputSize>;
      for (size_t n = 0; n < batch_size; n++) {
        const OutputRow& error_row =
          *reinterpret_cast<const OutputRow*>(errors[n].data());
        InputRow& prev_error_row =
          *reinterpret_cast<InputRow*>(prev_errors[n].data());
        for (size_t i = 0; i < InputSize::length; i++)
          prev_error_row[i] = hidden[i] * error_row[i];
      }
    }
  };
};

#endif
