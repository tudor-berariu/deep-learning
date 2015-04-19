// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#ifdef USE_ATLAS
extern "C" {
#include <cblas.h>
}
#endif

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
  using _WeightsRow = std::array<T, InputSize::length>;

  template<typename T, typename InputSize>
  using _Weights =
    std::array<_WeightsRow<T, InputSize>, OutputSize<InputSize>::length>;

 public:

  template<typename T, typename InputSize>
  inline static void
  init_parameters(Parameters<T, InputSize>& parameters) {
    std::random_device rd { };
    std::default_random_engine e {rd()};
    // constexpr T mean = (T)1 / (T)length;
    std::normal_distribution<T> next_parameter(0.0, 0.1);

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


#ifdef USE_ATLAS

  template<typename InputSize, size_t batch_size, bool train>
  struct _Forward<float, InputSize, batch_size, train> {
    static void forward(const Inputs<float, InputSize, batch_size>& inputs,
                        const Parameters<float, InputSize>& parameters,
                        Hidden<float, InputSize, batch_size>& hidden,
                        Outputs<float, InputSize, batch_size>& outputs) {
      for (size_t n = 0; n < batch_size; n++) {
        cblas_scopy(length,
                    reinterpret_cast<const float*>(parameters.data()), 1,
                    reinterpret_cast<float*>(hidden[n].data()), 1);
      }
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                  batch_size, length, InputSize::length,
                  1.0, reinterpret_cast<const float*>(inputs.data()),
                  InputSize::length,
                  reinterpret_cast<const float*>(&(parameters[length])),
                  InputSize::length,
                  1.0, reinterpret_cast<float*>(hidden.data()), length);
      for (size_t n = 0; n < batch_size; n++) {
        Output<float, InputSize>* const output_row =
          reinterpret_cast<Output<float, InputSize>*>(outputs[n].data());
        Output<float, InputSize>* const hidden_row =
          reinterpret_cast<Output<float, InputSize>*>(hidden[n].data());
        for (size_t j = 0; j < length; j++) {
          (*output_row)[j] = TransferFunction<float>::f((*hidden_row)[j]);
        }
      }
    }
  };

  template<typename InputSize, size_t batch_size, bool train>
  struct _Forward<double, InputSize, batch_size, train> {
    inline static void
    forward(const Inputs<double, InputSize, batch_size>& inputs,
            const Parameters<double, InputSize>& parameters,
            Hidden<double, InputSize, batch_size>& hidden,
            Outputs<double, InputSize, batch_size>& outputs){
      for (size_t n = 0; n < batch_size; n++) {
        cblas_dcopy(length,
                    reinterpret_cast<const double*>(parameters.data()), 1,
                    reinterpret_cast<double*>(hidden[n].data()), 1);
      }
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                  batch_size, length, InputSize::length,
                  1.0, reinterpret_cast<const double*>(inputs.data()),
                  InputSize::length,
                  reinterpret_cast<const double*>(&(parameters[length])),
                  InputSize::length,
                  1.0, reinterpret_cast<double*>(hidden.data()), length);
      for (size_t n = 0; n < batch_size; n++) {
        Output<double, InputSize>* const output_row =
          reinterpret_cast<Output<double, InputSize>*>(outputs[n].data());
        Output<double, InputSize>* const hidden_row =
          reinterpret_cast<Output<double, InputSize>*>(hidden[n].data());
        for (size_t j = 0; j < length; j++) {
          (*output_row)[j] = TransferFunction<double>::f((*hidden_row)[j]);
        }
      }
    }
  };
#endif

  template<typename T, typename InputSize, size_t batch_size, bool train>
  struct _Forward {
    inline static void
    forward(const Inputs<T, InputSize, batch_size>& inputs,
            const Parameters<T, InputSize>& parameters,
            Hidden<T, InputSize, batch_size>& hidden,
            Outputs<T, InputSize, batch_size>& outputs) {

      const _Biases<T, InputSize>* const biases =
        reinterpret_cast<const _Biases<T, InputSize>*>(parameters.data());
      const _Weights<T, InputSize>* const weights =
        reinterpret_cast<const _Weights<T, InputSize>*>(&(parameters[length]));

      for (size_t n = 0; n < batch_size; n++) {
        const Input<T, InputSize>* const input_row =
          reinterpret_cast<const Input<T, InputSize>*>(inputs[n].data());
        Output<T, InputSize>* const hidden_row =
          reinterpret_cast<Output<T, InputSize>*>(hidden[n].data());
        Output<T, InputSize>* const output_row =
          reinterpret_cast<Output<T, InputSize>*>(outputs[n].data());
        for (size_t j = 0; j < length; j++) {
          using _WR = _WeightsRow<T, InputSize>;
          const _WR* const weights_row =
            reinterpret_cast<const _WR*>((*weights)[j].data());

          (*hidden_row)[j] = (*biases)[j];
          for (size_t i = 0; i < InputSize::length; i++) {
            (*hidden_row)[j] += (*input_row)[i] * (*weights_row)[i];
          }
          (*output_row)[j] = TransferFunction<T>::f((*hidden_row)[j]);
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
                  const Outputs<T, InputSize, batch_size>&,
                  Outputs<T, InputSize, batch_size>& errors,
                  Parameters<T, InputSize>& gradients,
                  Inputs<T, InputSize, batch_size>& prev_errors) {
      /*
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
      */

      const _Weights<T, InputSize>& weights =
        *reinterpret_cast<const _Weights<T, InputSize>*>(&(parameters[length]));

      _Biases<T, InputSize>& g_biases =
        *reinterpret_cast<_Biases<T, InputSize>*>(gradients.data());
      _Weights<T, InputSize>& g_weights =
        *reinterpret_cast<_Weights<T, InputSize>*>(&(gradients[length]));

      for (T& gb : g_biases)
        gb = (T)0;
      for (_WeightsRow<T, InputSize>& gw_row : g_weights)
        for (T& gw : gw_row)
          gw = (T)0;

      for (size_t n = 0; n < batch_size; n++) {
        for (T& p_err : prev_errors[n])
          p_err = (T)0;

        Output<T, InputSize>* const errorsRow =
          reinterpret_cast<Output<T, InputSize>*>(errors[n].data());
        for (size_t j = 0; j < length; j++) {
          const _WeightsRow<T, InputSize>* const w_row =
            reinterpret_cast<const _WeightsRow<T, InputSize>*>(weights[j].data());
          _WeightsRow<T, InputSize>* const gw_row =
            reinterpret_cast<_WeightsRow<T, InputSize>*>(g_weights[j].data());
          const Input<T, InputSize>* const input_row =
            reinterpret_cast<const Input<T, InputSize>*>(inputs[n].data());
          Input<T, InputSize>* const prev_err_row =
            reinterpret_cast<Input<T, InputSize>*>(prev_errors[n].data());

          (*errorsRow)[j] *= TransferFunction<T>::df(hidden[n][j]);
          const T err_n_j = (*errorsRow)[j];
          g_biases[j] += err_n_j;
          for (size_t i = 0; i < InputSize::length; i++) {
            (*gw_row)[i] += (*input_row)[i] * err_n_j;
            (*prev_err_row)[i] += err_n_j * (*w_row)[i];
          }
        }
      }

    }
  };
};

#endif
