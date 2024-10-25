# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

from __future__ import annotations

import math
from typing import Iterable, Iterator, Dict, Tuple, Optional
from math import sqrt

from numpy.matlib import empty
from spacy.attrs import value

from .sieve import Sieve

class SparseDocumentVector:
    """
    A simple representation of a sparse document vector. The vector space has one dimension
    per vocabulary term, and our representation only lists the dimensions that have non-zero
    values.

    Being able to place text buffers, be they documents or queries, in a vector space and
    thinking of them as point clouds (or, equivalently, as vectors from the origin) enables us
    to numerically assess how similar they are according to some suitable metric. Cosine
    similarity (the inner product of the vectors normalized by their lengths) is a very
    common metric.
    """

    def __init__(self, values: Dict[str, float]):
        # An alternative, effective representation would be as a
        # [(term identifier, weight)] list kept sorted by integer
        # term identifiers. Computing dot products would then be done
        # pretty much in the same way we do posting list AND-scans.
        self._values = {}
        for key, val in values.items():  # prune zero-valued entries on construction
            if not math.isclose(val, 0.0):
                self._values[key] = val

        # We cache the length. It might get used over and over, e.g., for cosine
        # computations. A value of None triggers lazy computation.
        self._length : Optional[float] = None

    def __iter__(self):
        return iter(self._values.items())

    def __getitem__(self, term: str) -> float:
        return self._values.get(term, 0.0)

    def __setitem__(self, term: str, weight: float) -> None:
        self._values[term] = weight
        self._length = None

    def __contains__(self, term: str) -> bool:
        return term in self._values

    def __len__(self) -> int:
        """
        Enables use of the built-in len/1 function to count the number of non-zero
        dimensions in the vector. It is not for computing the vector's norm.
        """
        return len(self._values)

    def _calc_length(self):
        """
        Recalculates L^2 norm of vector.
        """
        square_norm = 0
        for val in self._values.values():
            square_norm += val ** 2

        self._length = math.sqrt(square_norm)

    def get_length(self) -> float:
        """
        Returns the length (L^2 norm, also called the Euclidian norm) of the vector.
        """
        if self._length is None:  # only calculates length if necessary
            self._calc_length()

        return self._length

    def normalize(self) -> None:
        """
        Divides all weights by the length of the vector, thus rescaling it to
        have unit length.
        """
        for key, val in self._values.items():
            self._values[key] = val/self._length

        self._calc_length()  # recalculate L^2 norm

    def top(self, count: int) -> Iterable[Tuple[str, float]]:
        """
        Returns the top weighted terms, i.e., the "most important" terms and their weights.
        """
        assert count >= 0
        if count == 0:
            return (nothing for nothing in [])

        # sift elements through sieve to get top
        sieve = Sieve(count)

        for key, val in self._values.items():
            sieve.sift(val, key)

        for val, key in sieve.winners():
            yield key, val

    def truncate(self, count: int) -> None:
        """
        Truncates the vector so that it contains no more than the given number of terms,
        by removing the lowest-weighted terms.
        """
        new_dict = {}  # ads the top terms to new dict
        for key, val in self.top(count):
            if not math.isclose(val, 0.0):
                new_dict[key] = val

        self._values = new_dict  # keeps only top elements
        self._calc_length() # recalculate L^2 norm

    def scale(self, factor: float) -> None:
        """
        Multiplies every vector component by the given factor.
        """
        if math.isclose(factor, 0.0):
            self._values.clear()  # empty vector if scaled to zero
            return

        for key, val in self._values.items():
            self._values[key] = val*factor

        self._calc_length() # recalculate L^2 norm

    def dot(self, other: SparseDocumentVector) -> float:
        """
        Returns the dot product (inner product, scalar product) between this vector
        and the other vector.
        """
        dot = 0
        for self_key, self_value in self._values.items():
            dot += self_value * other[self_key]

        return dot

    def cosine(self, other: SparseDocumentVector) -> float:
        """
        Returns the cosine of the angle between this vector and the other vector.
        See also https://en.wikipedia.org/wiki/Cosine_similarity.
        """
        norm = self.get_length() * other.get_length()

        if math.isclose(norm, 0.0):
            return 0  # returns zero if denominator is zero

        return self.dot(other)/norm

    @staticmethod
    def centroid(vectors: Iterator[SparseDocumentVector]) -> SparseDocumentVector:
        """
        Computes the centroid of all the vectors, i.e., the average vector.
        """
        num_vecs = 0
        c_o_m_vec = {}
        for vector in vectors:  # iterate over all vectors and all non-zero elements in vectors
            num_vecs += 1
            for key, val in vector._values.items():
                c_o_m_vec.setdefault(key, 0)  # add key with value zero, if not already in vec
                c_o_m_vec[key] += val  # add corresponding value from other vec

        if num_vecs <= 0:  # return empty vector if no vectors were given
            return SparseDocumentVector({})

        centroid_vec = SparseDocumentVector(c_o_m_vec)
        centroid_vec.scale(1/num_vecs)  # rescale by inverse number of vectors
        return centroid_vec