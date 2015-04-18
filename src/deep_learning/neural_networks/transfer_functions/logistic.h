// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef LOGISTIC_H
#define LOGISTIC_H

#include <cmath>

template<typename T>
struct Logistic {
  inline static T f(T x) {
    return (T)1 / ((T)1 + exp(-x));
  }
  inline static T df(T y) {
    return y * ((T)1 - y);
  }
};

#endif
