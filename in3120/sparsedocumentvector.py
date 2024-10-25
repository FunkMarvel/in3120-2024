# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

from __future__ import annotations

import math
from typing import Iterable, Iterator, Dict, Tuple, Optional
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

# example run of assignments.py d-1:
r"""
(venv) PS E:\Documents\in3120-2024\tests> python .\assignments.py d-1
test_document_id_mismatch (test_betterranker.TestBetterRanker.test_document_id_mismatch) ... ok
test_inverse_document_frequency (test_betterranker.TestBetterRanker.test_inverse_document_frequency) ... ok
test_static_quality_score (test_betterranker.TestBetterRanker.test_static_quality_score) ... ok
test_term_frequency (test_betterranker.TestBetterRanker.test_term_frequency) ... ok
test_shingled_mesh_corpus (test_shinglegenerator.TestShingleGenerator.test_shingled_mesh_corpus) ... ok
test_spans (test_shinglegenerator.TestShingleGenerator.test_spans) ... ok
test_strings (test_shinglegenerator.TestShingleGenerator.test_strings) ... ok
test_tokens (test_shinglegenerator.TestShingleGenerator.test_tokens) ... ok
test_uses_yield (test_shinglegenerator.TestShingleGenerator.test_uses_yield) ... ok
test_spans (test_wordshinglegenerator.TestWordShingleGenerator.test_spans) ... ok
test_spans_cover_surface_forms_but_strings_are_normalized (test_wordshinglegenerator.TestWordShingleGenerator.test_spans_cover_surface_forms_but_strings_are_normalized) ... ok
test_strings (test_wordshinglegenerator.TestWordShingleGenerator.test_strings) ... ok
test_tokens (test_wordshinglegenerator.TestWordShingleGenerator.test_tokens) ... ok
test_uses_yield (test_wordshinglegenerator.TestWordShingleGenerator.test_uses_yield) ... ok
test_centroid (test_sparsedocumentvector.TestSparseDocumentVector.test_centroid) ... ok
test_cosine (test_sparsedocumentvector.TestSparseDocumentVector.test_cosine) ... ok
test_dot_product (test_sparsedocumentvector.TestSparseDocumentVector.test_dot_product) ... ok
test_dunderscore_contains (test_sparsedocumentvector.TestSparseDocumentVector.test_dunderscore_contains) ... ok
test_dunderscore_getitem (test_sparsedocumentvector.TestSparseDocumentVector.test_dunderscore_getitem) ... ok
test_dunderscore_len (test_sparsedocumentvector.TestSparseDocumentVector.test_dunderscore_len) ... ok
test_dunderscore_setitem (test_sparsedocumentvector.TestSparseDocumentVector.test_dunderscore_setitem) ... ok
test_length (test_sparsedocumentvector.TestSparseDocumentVector.test_length) ... ok
test_normalize_empty (test_sparsedocumentvector.TestSparseDocumentVector.test_normalize_empty) ... ok
test_normalize_nonempty (test_sparsedocumentvector.TestSparseDocumentVector.test_normalize_nonempty) ... ok
test_only_non_zero_elements_are_kept (test_sparsedocumentvector.TestSparseDocumentVector.test_only_non_zero_elements_are_kept) ... ok
test_scale (test_sparsedocumentvector.TestSparseDocumentVector.test_scale) ... ok
test_scale_zero (test_sparsedocumentvector.TestSparseDocumentVector.test_scale_zero) ... ok
test_top (test_sparsedocumentvector.TestSparseDocumentVector.test_top) ... ok
test_truncate (test_sparsedocumentvector.TestSparseDocumentVector.test_truncate) ... ok

----------------------------------------------------------------------
Ran 29 tests in 0.811s

OK
"""

# example run of repl.py d-1:
r"""
(venv) PS E:\Documents\in3120-2024\tests> python .\repl.py d-1       
Indexing MeSH corpus...
Enter a query and find matching documents.
Lookup options are {'debug': False, 'hit_count': 5, 'match_threshold': 0.5}.
Normalizer is SimpleNormalizer.
Tokenizer is ShingleGenerator.
Ranker is SimpleRanker.
Ctrl-C to exit.
query>OrGaNiK KeMmIsTrY
[{'document': {'document_id': 16981, 'fields': {'body': 'organic chemistry processes', 'meta': '27'}},
  'score': 8.0},
 {'document': {'document_id': 16980, 'fields': {'body': 'organic chemistry phenomena', 'meta': '27'}},
  'score': 8.0},
 {'document': {'document_id': 4411, 'fields': {'body': 'chemistry, organic', 'meta': '18'}},
  'score': 8.0},
 {'document': {'document_id': 4410, 'fields': {'body': 'chemistry, inorganic', 'meta': '20'}},
  'score': 8.0},
 {'document': {'document_id': 4408, 'fields': {'body': 'chemistry, bioinorganic', 'meta': '23'}},
  'score': 8.0}]
Evaluation took 0.0023331000011239666 seconds.
query>
"""