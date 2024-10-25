# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

from typing import Iterator, Tuple, Optional
from collections import deque
from itertools import islice
from .tokenizer import Tokenizer
from .normalizer import Normalizer, DummyNormalizer


class ShingleGenerator(Tokenizer):
    """
    Does character-level shingling: Tokenizes a buffer into overlapping shingles
    having a specified width, as measured by character count. For example, the
    3-shingles for "mouse" are {"mou", "ous", "use"}.

    If the buffer is shorter than the shingle width then this produces a single
    shorter-than-usual shingle.

    The current implementation is simplistic and not whitespace- or punctuation-aware,
    and doesn't treat the beginning or end of the buffer in any special way.

    Character-level shingles are also called "k-grams", and can be used for various
    kinds of tolerant string matching, or near-duplicate detection. For details, see
    https://nlp.stanford.edu/IR-book/html/htmledition/k-gram-indexes-for-wildcard-queries-1.html,
    https://nlp.stanford.edu/IR-book/html/htmledition/k-gram-indexes-for-spelling-correction-1.html,
    or https://nlp.stanford.edu/IR-book/html/htmledition/near-duplicates-and-shingling-1.html.

    Note that 1-shingles reduces to a simple unigram tokenizer.
    """

    def __init__(self, width: int):
        assert width > 0
        self.__width = width

    def spans(self, buffer: str) -> Iterator[Tuple[int, int]]:
        return (span for _, span in self.tokens(buffer))

    def strings(self, buffer: str) -> Iterator[str]:
        if len(buffer) <= 0:  # return empty if empty buffer
            return (empty for empty in [])
        else:
            return (string for string, _ in self.tokens(buffer))

    def tokens(self, buffer: str) -> Iterator[Tuple[str, Tuple[int, int]]]:
        buffer_length = len(buffer)
        if buffer_length == 0: # return empty if empty buffer
            return (empty for empty in [])

        elif buffer_length < self.__width:  # return entire buffer if too short for more than one shingle
            yield buffer, (0, buffer_length)

        # yield each of the l-k+1 possible shingles
        for i in range(buffer_length - self.__width + 1):
            yield buffer[i:i+self.__width], (i, i+self.__width)


class WordShingleGenerator(Tokenizer):
    """
    Does token-level shingling: Tokenizes a buffer into overlapping shingles
    having a specified width, as measured by token count. For example, the
    2-shingles for "foo bar baz gog" are {"foo bar", "bar baz", "baz gog"}.

    If the buffer is shorter than the shingle width then this produces a single
    shorter-than-usual shingle.

    We delegate to another tokenizer exactly how to split the buffer into individual
    tokens. Optionally, individual tokens can be normalized.

    For 2-shingles used with a traditional tokenizer, the tokens are also called "biwords".
    For details, see https://nlp.stanford.edu/IR-book/html/htmledition/biword-indexes-1.html.

    Note that 1-shingles is just a roundabout way of invoking the embedded tokenizer, but
    with the added convenience of doing normalization as part of the tokenization process.
    """

    def __init__(self, width: int, tokenizer: Tokenizer, normalizer: Optional[Normalizer]):
        assert width > 0
        self.__width = width
        self.__tokenizer = tokenizer
        self.__normalizer = normalizer or DummyNormalizer()

    def spans(self, buffer: str) -> Iterator[Tuple[int, int]]:
        return (span for _, span in self.tokens(buffer))

    def strings(self, buffer: str) -> Iterator[str]:
        return (string for string, _ in self.tokens(buffer))

    def tokens(self, buffer: str) -> Iterator[Tuple[str, Tuple[int, int]]]:
        tokens = self.__tokenizer.tokens(buffer)
        header = ((self.__normalizer.normalize(string), _) for string, _ in islice(tokens, self.__width))
        window = deque(header, self.__width)
        while True:
            if len(window) > 0:
                oldest_span = window[0][1]
                newest_span = window[-1][1]
                yield (self.join(window), (oldest_span[0], newest_span[1]))
            string, span = next(tokens, (None, None))
            if string is None:
                break
            window.append((self.__normalizer.normalize(string), span))

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