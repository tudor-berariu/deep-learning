// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <cstddef>
#include <array>
#include <random>

#include "deep_learning/include_cblas.h"
#include "deep_learning/size.h"
#include "deep_learning/meta/meta_matrix.h"

template<size_t maps_no, size_t conv_height, size_t conv_width, size_t stride,
         typename Mapping, template<typename> class TransferFunction>
struct _Convolution;

template<size_t maps_no, size_t conv_height, size_t conv_width, size_t stride,
         typename Mapping, template<typename> class TransferFunction>
struct Convolution;

template<size_t maps_no, size_t conv_length, size_t stride,
         typename Mapping, template<typename> class TransferFunction>
struct Convolution<maps_no, conv_length, conv_length, stride, Mapping,
                   TransferFunction>
    : public _Convolution<maps_no, conv_length, conv_length, stride, Mapping,
                          TransferFunction> { };

template<size_t maps_no, size_t conv_length, typename Mapping,
         template<typename> class TransferFunction>
struct Convolution<maps_no, conv_length, conv_length, 1ul, Mapping,
                   TransferFunction>
    : public _Convolution<maps_no, conv_length, conv_length, 1ul, Mapping,
                          TransferFunction> { };

template<size_t conv_length, typename Mapping,
         template<typename> class TransferFunction>
struct Convolution<0ul, conv_length, conv_length, 1ul, Mapping,
                   TransferFunction>
    : public _Convolution<0ul, conv_length, conv_length, 1ul, Mapping,
                          TransferFunction> { };

template<size_t maps_no, size_t conv_height, size_t conv_width, size_t stride,
         typename Mapping, template<typename> class TransferFunction>
struct Convolution
   : public _Convolution<maps_no, conv_height, conv_width, stride,
         Mapping, TransferFunction> { };

template<size_t maps_no, size_t conv_height, size_t conv_width, size_t stride,
         typename Mapping, template<typename> class TransferFunction>
struct _Convolution {

  /* -------------------- OutputSize -------------------- */
  template<typename InputSize>
  using OutputSize =
    Size<(maps_no > 0 ? maps_no : Mapping::rows_no),
         (InputSize::height - conv_height + 1ul) / stride,
         (InputSize::width - conv_width + 1ul) / stride>;

};

#endif
