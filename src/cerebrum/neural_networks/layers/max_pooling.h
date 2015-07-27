// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef MAX_POOLING_H
#define MAX_POOLING_H

#include <cstddef>
#include <limits>
#include <array>

#include "cerebrum/size.h"

template<size_t pool_height, size_t pool_width>
struct MaxPooling {

  /* -------------------- OutputSize -------------------- */

 public:

  template<typename InputSize>
  using OutputSize =
    Size<InputSize::maps_no,
         InputSize::height / pool_height,
         InputSize::width / pool_width>;

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

  template <typename T, typename InputSize, size_t batch_size>
  using Inputs = std::array<Input<T, InputSize>, batch_size>;

 private:

  template<typename T, typename InputSize>
  using _InputMap =
    std::array<std::array<T, InputSize::width>, InputSize::height>;

  template<typename T, typename InputSize>
  using _InputMaps =
    std::array<_InputMap<T, InputSize>, InputSize::maps_no>;

  template<typename T, typename InputSize, size_t batch_size>
  using _Inputs = std::array<_InputMaps<T, InputSize>, batch_size>;

 public:

  template<typename T, typename InputSize>
  using Output = std::array<T, OutputSize<InputSize>::length>;

  template <typename T, typename InputSize, size_t batch_size>
  using Outputs = std::array<Output<T, InputSize>, batch_size>;

 private:

  template<typename T, typename InputSize>
  using _OutputMap =
    std::array<std::array<T, OutputSize<InputSize>::width>,
               OutputSize<InputSize>::height>;

  template<typename T, typename InputSize>
  using _OutputMaps =
    std::array<_OutputMap<T, InputSize>, OutputSize<InputSize>::maps_no>;

  template<typename T, typename InputSize, size_t batch_size>
  using _Outputs = std::array<_OutputMaps<T, InputSize>, batch_size>;

  using HiddenValue = std::pair<size_t, size_t>;

 public:

  template <typename T, typename InputSize, size_t batch_size>
  using Hidden = Outputs<HiddenValue, InputSize, batch_size>;

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

  template<typename T, typename InputSize, size_t batch_size, bool train>
  struct _Forward {
    inline static void
    forward(const Inputs<T, InputSize, batch_size>& inputs,
            const Parameters<T, InputSize>&,
            Hidden<T, InputSize, batch_size>& hidden,
            Outputs<T, InputSize, batch_size>& outputs) {
      using __Inputs = _Inputs<T, InputSize, batch_size>;
      using __Outputs = _Outputs<T, InputSize, batch_size>;
      using __Hidden = _Outputs<HiddenValue, InputSize, batch_size>;

      using __InputMap = _InputMap<T, InputSize>;
      using __OutputMap = _OutputMap<T, InputSize>;
      using __HiddenMap = _OutputMap<HiddenValue, InputSize>;

      const __Inputs& _inputs = *reinterpret_cast<const __Inputs*>(&inputs);
      __Outputs& _outputs = *reinterpret_cast<__Outputs*>(&outputs);
      __Hidden& _hidden = *reinterpret_cast<__Hidden*>(&hidden);

      for (size_t n = 0; n < batch_size; n++) {
        for (size_t m = 0; m < OutputSize<InputSize>::maps_no; m++) {
          const __InputMap& input_map =
            *reinterpret_cast<const __InputMap*>(_inputs[n][m].data());
          __OutputMap& output_map =
            *reinterpret_cast<__OutputMap*>(_outputs[n][m].data());
          __HiddenMap& hidden_map =
            *reinterpret_cast<__HiddenMap*>(_hidden[n][m].data());

          for (int r = 0; r < OutputSize<InputSize>::height; r++) {
            for (int c = 0; c < OutputSize<InputSize>::width; c++) {
              T max = std::numeric_limits<T>::lowest();
              for (size_t i = r * pool_height; i < (r+1) * pool_height; i++) {
                for (size_t j = c * pool_width; j < (c+1) * pool_width; j++) {
                  if (input_map[i][j] > max) {
                    max = input_map[i][j];
                    hidden_map[r][c].first = i;
                    hidden_map[r][c].second = j;
                  }
                }
              }
              output_map[r][c] = max;
            }
          }
        }
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

  template<typename T, typename InputSize, size_t batch_size>
  struct _Backpropagate {
    static void backpropagate(const Inputs<T, InputSize, batch_size>&,
                              const Parameters<T, InputSize>&,
                              const Hidden<T, InputSize, batch_size>& hidden,
                              const Outputs<T, InputSize, batch_size>&,
                              Outputs<T, InputSize, batch_size>& errors,
                              Parameters<T, InputSize>&,
                              Inputs<T, InputSize, batch_size>& prev_errors) {
      using __Inputs = _Inputs<T, InputSize, batch_size>;
      using __Outputs = _Outputs<T, InputSize, batch_size>;
      using __Hidden = _Outputs<HiddenValue, InputSize, batch_size>;

      using __InputMap = _InputMap<T, InputSize>;
      using __OutputMap = _OutputMap<T, InputSize>;
      using __HiddenMap = _OutputMap<HiddenValue, InputSize>;

      __Inputs& _prev_errors = *reinterpret_cast<__Inputs*>(&prev_errors);
      const __Outputs& _errors = *reinterpret_cast<const __Outputs*>(&errors);
      const __Hidden& _hidden = *reinterpret_cast<const __Hidden*>(&hidden);

      for (size_t n = 0; n < batch_size; n++) {
        std::fill(prev_errors[n].begin(), prev_errors[n].end(), (T)0);

        for (size_t m = 0; m < OutputSize<InputSize>::maps_no; m++) {
          __InputMap& prev_errors_map =
            *reinterpret_cast<__InputMap*>(_prev_errors[n][m].data());
          const __OutputMap& errors_map =
            *reinterpret_cast<const __OutputMap*>(_errors[n][m].data());
          const __HiddenMap& hidden_map =
            *reinterpret_cast<const __HiddenMap*>(_hidden[n][m].data());

          for (size_t r = 0; r < OutputSize<InputSize>::height; r++) {
            for (size_t c = 0; c < OutputSize<InputSize>::width; c++) {
              prev_errors_map[hidden_map[r][c].first][hidden_map[r][c].second]=
                errors_map[r][c];
            }
          }
        }
      }
    }
  };
};

#endif
