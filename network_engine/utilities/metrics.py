#!/usr/bin/python
#
# Project Titan
# _____________________________________________________________________________
#
#                                                                         _.oo.
# April 2020                                    _.u[[/;:,.         .odMMMMMM'
#                                             .o888UU[[[/;:-.  .o@P^    MMM^
# metrics.py                                 oN88888UU[[[/;::-.        dP^
# a collection of                           dNMMNN888UU[[[/;:--.   .o@P^
# metrics for                             ,MMMMMMN888UU[[/;::-. o@^
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
# Copyright 2021 Markus Ernst
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS 
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
# THE SOFTWARE.
#
# _____________________________________________________________________________


# ----------------
# import libraries
# ----------------

# standard libraries
# -----


import math
import random
import csv
import cProfile
import numpy as np
import hashlib

from fractions import Fraction

import torch

# calculate the gini coefficient from a numpy array
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

# calculate the gini coefficient from a torch array
def gini_torch(array):
    """Calculate the Gini coefficient of a torch array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if torch.amin(array) < 0:
        # Values cannot be negative:
        array -= torch.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = torch.sort(array)[0]
    # Index per array element:
    index = torch.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((torch.sum((2 * index - n  - 1) * array)) / (n * torch.sum(array)))



memoization = {}


class Similarity:
    """
    This class contains instances of similarity / distance metrics.
    These are used in centroid based clustering algorithms to identify similar
    patterns and put them into the same homogeneous sub sets
    :param minimum: the minimum distance between two patterns
    (so you don't divide by 0)
    """

    def __init__(self, minimum):
        self.e = minimum
        self.vector_operators = VectorOperations()

    def manhattan_distance(self, p_vec, q_vec):
        """
        This method implements the manhattan distance metric
        :param p_vec: vector one
        :param q_vec: vector two
        :return: the manhattan distance between vector one and two
        """
        return max(np.sum(np.fabs(p_vec - q_vec)), self.e)

    def square_euclidean_distance(self, p_vec, q_vec):
        """
        This method implements the squared euclidean distance metric
        :param p_vec: vector one
        :param q_vec: vector two
        :return: the squared euclidean distance between vector one and two
        """
        diff = p_vec - q_vec
        return max(np.sum(diff**2), self.e)

    def euclidean_distance(self, p_vec, q_vec):
        """
        This method implements the euclidean distance metric
        :param p_vec: vector one
        :param q_vec: vector two
        :return: the euclidean distance between vector one and two
        """
        return max(math.sqrt(self.square_euclidean_distance(p_vec, q_vec)),
                   self.e)

    def half_square_euclidean_distance(self, p_vec, q_vec):
        """
        This method implements the half squared euclidean distance metric
        :param p_vec: vector one
        :param q_vec: vector two
        :return: the half squared euclidean distance between vector one and two
        """
        return max(0.5 * self.square_euclidean_distance(p_vec, q_vec), self.e)

    def cosine_similarity(self, p_vec, q_vec):
        """
        This method implements the cosine similarity metric
        :param p_vec: vector one
        :param q_vec: vector two
        :return: the cosine similarity between vector one and two
        """
        pq = self.vector_operators.product(p_vec, q_vec)
        p_norm = self.vector_operators.norm(p_vec)
        q_norm = self.vector_operators.norm(q_vec)
        return max(pq / (p_norm * q_norm), self.e)

    def tanimoto_coefficient(self, p_vec, q_vec):
        """
        This method implements the cosine tanimoto coefficient metric
        :param p_vec: vector one
        :param q_vec: vector two
        :return: the tanimoto coefficient between vector one and two
        """
        pq = self.vector_operators.product(p_vec, q_vec)
        p_square = self.vector_operators.square(p_vec)
        q_square = self.vector_operators.square(q_vec)
        return max(pq / (p_square + q_square - pq), self.e)

    def fractional_distance(self, p_vec, q_vec, fraction=Fraction(1,2)):
        """
        This method implements the fractional distance metric. I have
        implemented memoization for this method to reduce
        the number of function calls required. The net effect is that the
        algorithm runs 400% faster. A similar approach
        can be used with any of the above distance metrics as well.
        :param p_vec: vector one
        :param q_vec: vector two
        :param fraction: the fractional distance value (power)
        :return: the fractional distance between vector one and two
        """

        memoize = False
        if memoize:
            key = self.get_key(p_vec, q_vec)
            x = memoization.get(key)
            if x is None:
                diff = p_vec - q_vec
                diff_fraction = np.abs(diff)**fraction
                return max(math.pow(np.sum(diff_fraction), 1/fraction), self.e)
            else:
                return x
        else:
            diff = p_vec - q_vec
            diff_fraction = np.abs(diff)**fraction
            return max(math.pow(np.sum(diff_fraction), 1/fraction), self.e)
    

        
    @staticmethod
    def get_key(p_vec, q_vec):
        """
        This method returns a unique hash value for two vectors. The hash value
        is equal to the concatenated string of the hash value for vector one
        and vector two. E.g. is hash(p_vec) = 1234 and hash(q_vec) = 5678 then
        get_key(p_vec, q_vec) = 12345678. Memoization improved the speed of
        this algorithm 400%.
        :param p_vec: vector one
        :param q_vec: vector two
        :return: a unique hash
        """
        # return str(hash(tuple(p_vec))) + str(hash(tuple(q_vec)))
        return str(hashlib.sha1(p_vec)) + str(hashlib.sha1(q_vec))


class VectorOperations():
    """
    This class contains useful implementations of methods which can be
    performed on vectors
    """

    @staticmethod
    def product(p_vec, q_vec):
        """
        This method returns the product of two lists / vectors
        :param p_vec: vector one
        :param q_vec: vector two
        :return: the product of p_vec and q_vec
        """
        return p_vec * q_vec

    @staticmethod
    def square(p_vec):
        """
        This method returns the square of a vector
        :param p_vec: the vector to be squared
        :return: the squared value of the vector
        """
        return p_vec**2

    @staticmethod
    def norm(p_vec):
        """
        This method returns the norm value of a vector
        :param p_vec: the vector to be normed
        :return: the norm value of the vector
        """
        return np.sqrt(p_vec)


# _____________________________________________________________________________


# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
