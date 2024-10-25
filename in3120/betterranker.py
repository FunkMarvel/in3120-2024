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
