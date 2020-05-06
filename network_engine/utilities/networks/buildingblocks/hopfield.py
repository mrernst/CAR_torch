#!/usr/bin/python
#
# Project Titan
# _____________________________________________________________________________
#
#                                                                         _.oo.
# April 2020                                   _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# hopfieldnetwork.py                         oN88888UU[[[/;::-.        dP^
# Description Description                   dNMMNN888UU[[[/;:--.   .o@P^
# Description Description                  ,MMMMMMN888UU[[/;::-. o@^
#                                          NNMMMNN888UU[[[/~.o@P^
# Markus Ernst                             888888888UU[[[/o@^-..
#                                         oI8888UU[[[/o@P^:--..
#                                      .@^  YUU[[[/o@^;::---..
#                                    oMP     ^/o@P^;:::---..
#                                 .dMMM    .o@^ ^;::---...
#                                dMMMMMMM@^`       `^^^^
#                               YMMMUP^
#                                ^^
# _____________________________________________________________________________
#
#
# Copyright 2020 Markus Ernst
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# _____________________________________________________________________________


# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torch


# network class
# -----

class HopfieldNet(object):
    """
    A Hopfield network.
    """

    def __init__(self, num_units, scope='hopfield'):

        self._weights = torch.zeros(size=(num_units, num_units),
                                    dtype=torch.float32,
                                    requires_grad=False)

        self._thresholds = torch.zeros(size=(num_units,),
                                       dtype=torch.float32,
                                       requires_grad=False)
        self._hebb_counter = torch.zeros(size=(),
                                         dtype=torch.int32,
                                         requires_grad=False)

        self._second_moment = torch.zeros(size=self._weights.shape,
                                          dtype=self._weights.dtype,
                                          requires_grad=False)

    @property
    def weights(self):
        """
        Get the weight 2-D Tensor for the network.

        Rows correspond to inputs and columns correspond
        to outputs.
        """
        return self._weights

    @property
    def thresholds(self):
        """
        Get the threshold 1-D Tensor for the network.
        """
        return self._thresholds

    @property
    def hebb_counter(self):
        """
        Get the threshold 1-D Tensor for the network.
        """
        return self._hebb_counter

    @property
    def second_moment(self):
        """
        Get the threshold 1-D Tensor for the network.
        """
        return self._second_moment

    def step(self, states):
        """
        Apply the activation rules to the states.

        Args:
          states: a 1-D or 2-D Tensor of input states.
            2-D Tensors represent batches of states.
            States must have dtype torch.bool.

        Returns:
          The new state Tensor after one timestep.
        """
        assert states.dtype == torch.bool
        numerics = 2 * states.type(torch.float32) - 1

        if len(numerics.shape) == 1:
            numerics = torch.unsqueeze(numerics, dim=0)

        weighted_states = torch.matmul(numerics, self.weights)
        result = weighted_states >= self.thresholds
        if len(states.shape) == 1:
            return result[0]
        return result

    def hebbian_update(self, samples):
        """
        Create an Op that updates the weight matrix with the
        mini-batch via the Hebbian update rule.

        Args:
          samples: a mini-batch of samples. Should be a 2-D
            Tensor with a dtype of torch.bool.
          weights: the weight matrix to update. Should start
            out as all zeros.

        Returns:
          An Op that updates the weights such that the batch
            of samples is encouraged.

        Hebbian learning involves a running average over all
        of the training data. This is implemented via extra
        training-specific variables.
        """
        self._weights = self._second_moment_update(samples, self.weights)[0]
        pass

    def covariance_update(self, samples):
        """
        Create an Op that performs a statistically centered
        Hebbian update on the mini-batch.

        This is like hebbian_update(), except that the weights
        are trained on a zero-mean version of the samples, and
        the thresholds are tuned as well as the weights.
        """
        dtype = self.second_moment.dtype

        new_second, rate = self._second_moment_update(
            samples, self.second_moment, mask_diag=False)
        new_mean = -(torch.mean(samples.type(dtype=dtype) * 2 - 1, dim=0))
        new_thresh = self.thresholds + (rate * (new_mean - self.thresholds))

        outer = torch.matmul(torch.unsqueeze(new_thresh, dim=1),
                             torch.unsqueeze(new_thresh, dim=0))
        weights = new_second - outer
        self._thresholds = new_thresh
        self._second_moment = new_second
        self._weights = weights
        pass

    def extended_storkey_update(self, sample):
        """
        Create an Op that performs a step of the Extended
        Storkey Learning Rule.

        Args:
          sample: a 1-D sample Tensor of dtype torch.bool.
          weights: the weight matrix to update.

        Returns:
          An Op that updates the weights based on the sample.
        """
        scale = 1 / int(self.weights.shape[0])
        numerics = 2 * sample.type(dtype=self.weights.dtype) - 1
        row_sample = torch.unsqueeze(numerics, dim=0)
        row_h = torch.matmul(row_sample, self.weights)

        pos_term = (torch.matmul(torch.transpose(row_sample, 0, 1), row_sample) +
                    torch.matmul(torch.transpose(row_h, 0, 1), row_h))
        neg_term = (torch.matmul(torch.transpose(row_sample, 0, 1), row_h) +
                    torch.matmul(torch.transpose(row_h, 0, 1), row_sample))
        self._weights += scale * (pos_term - neg_term)
        pass

    def _second_moment_update(self, samples, weights, mask_diag=True):
        """
        Get an Op to do an uncentered second-moment update.

        Returns (update_op, updated_weights, running_avg_rate)
        """
        assert samples.dtype == torch.bool
        assert len(samples.shape) == 2

        dtype = weights.dtype

        old_count = self.hebb_counter.type(dtype=dtype)
        new_count = samples.shape[0]
        rate = new_count / (new_count + old_count)

        numerics = 2 * samples.type(dtype=dtype) - 1
        outer = torch.matmul(torch.transpose(
            numerics, 0, 1), numerics) / new_count
        if not mask_diag:
            rate_mask = rate
        else:
            diag_mask = 1 - torch.diag(torch.ones(weights.shape[0],))
            rate_mask = rate * diag_mask
        new_weights = weights + rate_mask * (outer - weights)
        self._hebb_counter += new_count

        # return seems odd here
        return new_weights, rate


# TODO implement Continuous Version and Local Weights
class ContinuousHopfieldNet(HopfieldNet):
    pass

class LocalHopfieldNet(HopfieldNet):
    pass

# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
