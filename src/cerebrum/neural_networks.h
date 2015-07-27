// Copyright (C) 2015 Tudor Berariu <tudor.berariu@gmail.com>

#ifndef NEURAL_NETWORKS
#define NEURAL_NETWORKS

#include "cerebrum/include_cblas.h"

#include "cerebrum/neural_networks/feed_forward_net.h"

#include "cerebrum/neural_networks/layers/fully_connected.h"
#include "cerebrum/neural_networks/layers/dropout.h"
#include "cerebrum/neural_networks/layers/max_pooling.h"

#include "cerebrum/neural_networks/transfer_functions/logistic.h"
#include "cerebrum/neural_networks/transfer_functions/tanh.h"
#include "cerebrum/neural_networks/transfer_functions/relu.h"
#include "cerebrum/neural_networks/transfer_functions/identity.h"

#include "cerebrum/neural_networks/error_functions/rmse.h"
#include "cerebrum/neural_networks/error_functions/sum_of_squares.h"
#include "cerebrum/neural_networks/error_functions/softmax.h"

#endif

