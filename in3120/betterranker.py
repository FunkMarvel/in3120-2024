# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import math
from .ranker import Ranker
from .corpus import Corpus
from .posting import Posting
from .invertedindex import InvertedIndex


class BetterRanker(Ranker):
    """
    A ranker that does traditional TF-IDF ranking, possibly combining it with
    a static document score (if present).

    The static document score is assumed accessible in a document field named
    "static_quality_score". If the field is missing or doesn't have a value, a
    default value of 0.0 is assumed for the static document score.

    See Section 7.1.4 in https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf.
    """

    # These values could be made configurable. Hardcode them for now.
    _dynamic_score_weight = 1.0
    _static_score_weight = 1.0
    _static_score_field_name = "static_quality_score"
    _static_score_default_value = 0.0

    def __init__(self, corpus: Corpus, inverted_index: InvertedIndex):
        self._score = 0.0
        self._document_id = None
        self._corpus = corpus
        self._inverted_index = inverted_index

    def reset(self, document_id: int) -> None:
        self._score = 0
        self._document_id = document_id

    def update(self, term: str, multiplicity: int, posting: Posting) -> None:
        assert self._document_id == posting.document_id

        # get number of documents with the current term
        num_docs_with_term = len(list(self._inverted_index.get_postings_iterator(term)))

        # calculate relative term frequency and inverse document frequency
        tf = 1 + math.log(posting.term_frequency)
        idf = math.log(self._corpus.size() / num_docs_with_term)

        self._score += tf*idf

    def evaluate(self) -> float:
        # static score only retrieved once per document
        static_score = self._corpus.get_document(self._document_id).get_field(
            self._static_score_field_name, self._static_score_default_value)

        # calculates score as weighted sum of tf-idf score and static score
        return self._dynamic_score_weight*self._score + self._static_score_weight*static_score

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