// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef RELU_H
#define RELU_H

template <typename T>
struct ReLU {
  inline static T f(T x) {
    return (x > 0) ? x : 0;
  }

  inline static T df(T y) {
    return (y > 0) ? 1 : 0;
  }
};

#endif
