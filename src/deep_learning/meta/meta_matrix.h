// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef META_MATRIX_H
#define META_MATRIX_H

#include <cstddef>

template<typename T, size_t height, size_t width, T... values>
struct MetaMatrix { };

template<typename T, size_t height, size_t width>
struct MetaMatrix<T, height, width> {
  static constexpr size_t left = 0ul;
};

template<typename T, size_t _height, size_t _width, T head, T... tail>
struct MetaMatrix<T, _height, _width, head, tail...> {
  using Type = T;
  using Rest = MetaMatrix<T, _height, _width, tail...>;
  static constexpr T first = head;
  static constexpr size_t height = _height;
  static constexpr size_t width = _width;
  static constexpr size_t left = 1ul + Rest::left;
};

#endif
