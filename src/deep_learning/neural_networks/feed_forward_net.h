// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

template<typename... Info>
struct NetOutput;

template<typename T>
struct FeedForwardNet {
  using DataType = T;
};

#endif
