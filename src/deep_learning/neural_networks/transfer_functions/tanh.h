// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef HYPERBOLIC_TANGENT_H
#define HYPERBOLIC_TANGENT_H

#include <cmath>

template<typename T>
struct HyperbolicTangent {
  inline static T f(T x) {
    const T e2x = exp(-2 * x);
    return ((T)1 - e2x) / ((T)1 + e2x);
  }

  inline static T df(T y) {
    return (T)1 - y * y;
  }
};

#endif
