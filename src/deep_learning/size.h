// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef SIZE_H
#define SIZE_H

#include <cstddef>

/* Tudor:
 * Size carries static information about the size of a layer:
 *  - total number of units : length
 *  - number and size of maps
 */

template<size_t... info>
struct Size;

template<size_t length>
struct Size<length> : public Size<1ul, 1ul, length> { };

template<size_t height, size_t width>
struct Size<height, width> : public Size<1ul, height, width> { };

template<size_t _maps_no, size_t _height, size_t _width>
struct Size<_maps_no, _height, _width> {
  static constexpr size_t length  = _height * _width * _maps_no;
  static constexpr size_t heigth  = _height;
  static constexpr size_t width   = _width;
  static constexpr size_t maps_no = _maps_no;
};

#endif
