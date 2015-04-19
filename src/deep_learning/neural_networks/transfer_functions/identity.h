// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef IDENTITY_H
#define IDENTITY_H

#include <cmath>

template<typename T>
struct Identity {
  inline static T f(T x) {
    return x;
  }

  inline static T df(T) {
    return (T)1;
  }
};

#endif
