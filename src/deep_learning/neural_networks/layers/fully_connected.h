// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include <cstddef>
#include <array>
#include <random>

#include "deep_learning/include_cblas.h"
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

  template<typename T, typename InputSize, size_t batch_size>
  using _LinearInputs = std::array<T, InputSize::length * batch_size>;

  template<typename T, typename InputSize, size_t batch_size>
  using _LinearOutputs = std::array<T, length * batch_size>;

  template<typename T, typename InputSize>
  using _Biases = std::array<T, OutputSize<InputSize>::length>;

  template<typename T, typename InputSize>
  using _WeightsRow = std::array<T, InputSize::length>;

  template<typename T, typename InputSize>
  using _Weights =
    std::array<_WeightsRow<T, InputSize>, OutputSize<InputSize>::length>;

  template<typename T, typename InputSize>
  using _LinearWeights = std::array<T, InputSize::length * length>;

 public:

  template<typename T, typename InputSize>
  inline static void
  init_parameters(Parameters<T, InputSize>& parameters) {
    std::random_device rd { };
    std::default_random_engine e {rd()};
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


#ifdef USE_CBLAS

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

      TransferFunction<float>::template
        f_batch<OutputSize<InputSize>, batch_size>(hidden, outputs);
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

      TransferFunction<double>::template
        f_batch<OutputSize<InputSize>, batch_size>(hidden, outputs);
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

      using Biases = _Biases<T, InputSize>;
      using Weights = _Weights<T, InputSize>;
      using WeightsRow = _WeightsRow<T, InputSize>;
      using InputRow = Input<T, InputSize>;
      using OutputRow = Output<T, InputSize>;

      const Biases& biases =
        *reinterpret_cast<const Biases*>(parameters.data());
      const Weights& weights =
        *reinterpret_cast<const Weights*>(&(parameters[length]));

      for (size_t n = 0; n < batch_size; n++) {
        const InputRow& input_row =
          *reinterpret_cast<const InputRow*>(inputs[n].data());
        OutputRow& hidden_row =
          *reinterpret_cast<OutputRow*>(hidden[n].data());
        OutputRow& output_row =
          *reinterpret_cast<OutputRow*>(outputs[n].data());

        for (size_t j = 0; j < length; j++) {
          const WeightsRow& weights_row =
            *reinterpret_cast<const WeightsRow*>(weights[j].data());
          hidden_row[j] = biases[j];
          for (size_t i = 0; i < InputSize::length; i++) {
            hidden_row[j] += input_row[i] * weights_row[i];
          }
          output_row[j] = TransferFunction<T>::f(hidden_row[j]);
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

#ifdef USE_CBLAS

  template<typename InputSize, size_t batch_size>
  struct _Backpropagate<float, InputSize, batch_size> {
    inline static void
    backpropagate(const Inputs<float, InputSize, batch_size>& inputs,
                  const Parameters<float, InputSize>& parameters,
                  const Hidden<float, InputSize, batch_size>& hidden,
                  const Outputs<float, InputSize, batch_size>& outputs,
                  Outputs<float, InputSize, batch_size>& errors,
                  Parameters<float, InputSize>& gradients,
                  Inputs<float, InputSize, batch_size>& prev_errors) {
      TransferFunction<float>::template
        df_batch<OutputSize<InputSize>, batch_size>(outputs, errors);
      cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                  length, InputSize::length, batch_size,
                  1.0, reinterpret_cast<const float*>(errors.data()),
                  length,
                  reinterpret_cast<const float*>(inputs.data()),
                  InputSize::length,
                  0.0, reinterpret_cast<float*>(&(gradients[length])),
                  InputSize::length);
      // not sure if faster, tests need to be done
      using Biases = _Biases<float, InputSize>;
      Biases ones;
      std::fill(ones.begin(), ones.end(), 1.0);
      cblas_sgemv(CblasRowMajor, CblasTrans,
                  batch_size, length,
                  1.0, reinterpret_cast<const float*>(errors.data()),
                  length,
                  reinterpret_cast<const float*>(ones.data()), 1,
                  0.0, reinterpret_cast<float*>(gradients.data()), 1);

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  batch_size, InputSize::length, length,
                  1.0, reinterpret_cast<const float*>(errors.data()),
                  length,
                  reinterpret_cast<const float*>(&(parameters[length])),
                  InputSize::length,
                  0.0, reinterpret_cast<float*>(prev_errors.data()),
                  InputSize::length);
    }
  };

  template<typename InputSize, size_t batch_size>
  struct _Backpropagate<double, InputSize, batch_size> {
    inline static void
    backpropagate(const Inputs<double, InputSize, batch_size>& inputs,
                  const Parameters<double, InputSize>& parameters,
                  const Hidden<double, InputSize, batch_size>& hidden,
                  const Outputs<double, InputSize, batch_size>& outputs,
                  Outputs<double, InputSize, batch_size>& errors,
                  Parameters<double, InputSize>& gradients,
                  Inputs<double, InputSize, batch_size>& prev_errors) {
      TransferFunction<double>::template
        df_batch<OutputSize<InputSize>, batch_size>(outputs, errors);
      cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                  length, InputSize::length, batch_size,
                  1.0, reinterpret_cast<const double*>(errors.data()),
                  length,
                  reinterpret_cast<const double*>(inputs.data()),
                  InputSize::length,
                  0.0, reinterpret_cast<double*>(&(gradients[length])),
                  InputSize::length);
      // not sure if faster, tests need to be done
      using Biases = _Biases<double, InputSize>;
      Biases ones;
      std::fill(ones.begin(), ones.end(), 1.0);
      cblas_dgemv(CblasRowMajor, CblasTrans,
                  batch_size, length,
                  1.0, reinterpret_cast<const double*>(errors.data()),
                  length,
                  reinterpret_cast<const double*>(ones.data()), 1,
                  0.0, reinterpret_cast<double*>(gradients.data()), 1);

      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  batch_size, InputSize::length, length,
                  1.0, reinterpret_cast<const double*>(errors.data()),
                  length,
                  reinterpret_cast<const double*>(&(parameters[length])),
                  InputSize::length,
                  0.0, reinterpret_cast<double*>(prev_errors.data()),
                  InputSize::length);
    }
  };

#endif

  template<typename T, typename InputSize, size_t batch_size>
  struct _Backpropagate {
    inline static void
    backpropagate(const Inputs<T, InputSize, batch_size>& inputs,
                  const Parameters<T, InputSize>& parameters,
                  const Hidden<T, InputSize, batch_size>&,
                  const Outputs<T, InputSize, batch_size>& outputs,
                  Outputs<T, InputSize, batch_size>& errors,
                  Parameters<T, InputSize>& gradients,
                  Inputs<T, InputSize, batch_size>& prev_errors) {

      using Weights = _Weights<T, InputSize>;
      using LinearWeights = _LinearWeights<T, InputSize>;
      using WeightsRow = _WeightsRow<T, InputSize>;
      using Biases = _Biases<T, InputSize>;

      using InputRow = Input<T, InputSize>;
      using OutputRow = Output<T, InputSize>;

      const Weights& weights =
        *reinterpret_cast<const Weights*>(&(parameters[length]));

      Biases& g_biases = *reinterpret_cast<Biases*>(gradients.data());
      Weights& g_weights = *reinterpret_cast<Weights*>(&(gradients[length]));
      LinearWeights& g_linear_weights =
        *reinterpret_cast<LinearWeights*>(&g_weights);

      std::fill(g_biases.begin(), g_biases.end(), (T)0);
      std::fill(g_linear_weights.begin(), g_linear_weights.end(), (T)0);

      for (size_t n = 0; n < batch_size; n++) {
        std::fill(prev_errors[n].begin(), prev_errors[n].end(), (T)0);

        const OutputRow& output_row =
          *reinterpret_cast<const OutputRow*>(outputs[n].data());
        OutputRow& errors_row =
          *reinterpret_cast<OutputRow*>(errors[n].data());
        const InputRow& input_row =
          *reinterpret_cast<const InputRow*>(inputs[n].data());
        InputRow& prev_err_row =
          *reinterpret_cast<InputRow*>(prev_errors[n].data());

        for (size_t j = 0; j < length; j++) {
          const WeightsRow& weights_row =
            *reinterpret_cast<const WeightsRow*>(weights[j].data());
          WeightsRow& g_weights_row =
            *reinterpret_cast<WeightsRow*>(g_weights[j].data());

          errors_row[j] *= TransferFunction<T>::df(output_row[j]);

          const T err_n_j = errors_row[j];
          g_biases[j] += err_n_j;
          for (size_t i = 0; i < InputSize::length; i++) {
            g_weights_row[i] += input_row[i] * err_n_j;
            prev_err_row[i] += err_n_j * weights_row[i];
          }
        }
      }

    }
  };
};

#endif
