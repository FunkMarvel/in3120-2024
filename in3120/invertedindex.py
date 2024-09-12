# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=unnecessary-pass
# pylint: disable=unused-argument

from abc import ABC, abstractmethod
from typing import Iterable, Iterator, List, Tuple, Dict
from .dictionary import InMemoryDictionary
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .corpus import Corpus
from .posting import Posting
from .postinglist import InMemoryPostingList, PostingList


class InvertedIndex(ABC):
    """
    Abstract base class for a simple inverted index.
    """

    def __getitem__(self, term: str) -> Iterator[Posting]:
        return self.get_postings_iterator(term)

    def __contains__(self, term: str) -> bool:
        return self.get_document_frequency(term) > 0

    @abstractmethod
    def get_terms(self, buffer: str) -> Iterator[str]:
        """
        Processes the given text buffer and returns an iterator that yields normalized
        terms as they are indexed. Useful when both query strings and documents need to
        be identically processed. The terms produced from the given text buffer may or
        may not have been encountered index construction.
        """
        pass

    @abstractmethod
    def get_indexed_terms(self) -> Iterator[str]:
        """
        Returns an iterator over all unique terms that have been encountered during the
        construction of the inverted index, i.e., our term vocabulary or all terms for which
        there exists a posting list. The vocabulary is not listed in any particular order.
        This method might be useful if a client needs to build up some additional data
        structure over the term vocabulary, that is external to the inverted index.
        """
        pass

    @abstractmethod
    def get_postings_iterator(self, term: str) -> Iterator[Posting]:
        """
        Returns an iterator that can be used to iterate over the term's associated
        posting list. For out-of-vocabulary terms we associate empty posting lists.
        """
        pass

    @abstractmethod
    def get_document_frequency(self, term: str) -> int:
        """
        Returns the number of documents in the indexed corpus that contain the given term.
        """
        pass

    def get_collection_frequency(self, term: str) -> int:
        """
        Returns the number of times the given term occurs in the indexed corpus, across
        all documents.
        """
        new_term = next(self.get_terms(term), None)
        if new_term is None:
            return 0

        return sum(p.term_frequency for p in self.get_postings_iterator(term))


class InMemoryInvertedIndex(InvertedIndex):
    """
    A simple in-memory implementation of an inverted index, suitable for small corpora.

    In a serious application we'd have configuration to allow for field-specific NLP,
    scale beyond current memory constraints, have a positional index, and so on.

    If index compression is enabled, only the posting lists are compressed. Dictionary
    compression is currently not supported.
    """

    def __init__(self, corpus: Corpus, fields: Iterable[str], normalizer: Normalizer, tokenizer: Tokenizer, compressed: bool = False):
        self._corpus = corpus
        self._normalizer = normalizer
        self._tokenizer = tokenizer
        self._posting_lists: List[PostingList] = []
        self._dictionary = InMemoryDictionary()
        self._build_index(fields, compressed)

    def __repr__(self):
        return str({term: self._posting_lists[term_id] for term, term_id in self._dictionary})

    def _build_index(self, fields: Iterable[str], compressed: bool) -> None:
        """
        Kicks off the indexing process. Basically implements a flavor of SPIMI indexing as described in
        https://nlp.stanford.edu/IR-book/html/htmledition/single-pass-in-memory-indexing-1.html but with
        the vastly simplifying assumption that everything fits in memory so we just have a single block
        and thus no need to merge per-block results.

        Note that we currently don't keep track of which field each term occurs in. If we were to allow
        fielded searches (e.g., "find documents that contain 'foo' in the 'title' field") then we would
        have to keep track of that, either as a synthetic term in the dictionary (e.g., 'foo.title') or
        as extra data in the posting. See https://nlp.stanford.edu/IR-book/html/htmledition/parametric-and-zone-indexes-1.html
        for further details.

        Also note that we are building a non-positional index, for simplicity. With a positional index
        we could offer clients the ability to do, e.g., phrase searches and proximity-based filtering and
        ranking. See https://nlp.stanford.edu/IR-book/html/htmledition/positional-indexes-1.html for
        further details.
        """
        for document in self._corpus: # wasn't sure if I should parse the corpus into a buffer of terms first,
            for field in fields:      # or if handling it document by document, like this, is ok.
                terms = self.get_terms(document.get_field(field, None))
                if terms is None:
                    continue

                for term in terms:
                    term_id = self._dictionary.add_if_absent(term)
                    if len(self._posting_lists) <= term_id:
                        self._posting_lists.append(InMemoryPostingList())

                    posting = next((posting for posting in self._posting_lists[term_id] if posting.document_id == document.document_id), None)
                    if posting is None:
                        posting = Posting(document.document_id, 0)
                        self._posting_lists[term_id].append_posting(posting)

                    posting.term_frequency += 1  # manipulates element in posting-list, due to being a shallow copy

    def _add_to_dictionary(self, term: str) -> int:
        """
        Adds the given term to the dictionary, if it's not already present. If it's already present,
        the dictionary stays unchanged. Returns the term identifier assigned to the term.
        """
        return self._dictionary.add_if_absent(term)

    def _append_to_posting_list(self, term_id: int, document_id: int, term_frequency: int, compressed: bool) -> None:
        """
        Appends a new posting to the right posting list. The posting lists
        must be kept sorted so that we can efficiently traverse and
        merge them when querying the inverted index.
        """
        assert term_id >= 0
        assert term_frequency >= 0
        if len(self._posting_lists) <= term_id:
            self._posting_lists.append(InMemoryPostingList())
        self._posting_lists[term_id].append_posting(Posting(document_id, term_frequency))

    def _finalize_index(self):
        """
        Invoked at the very end after all documents have been processed. Provides
        implementations that need it with the chance to tie up any loose ends,
        if needed.
        """
        for postings in self._posting_lists:
            postings.finalize_postings()

    def get_terms(self, buffer: str) -> Iterator[str]:
        # In a serious large-scale application there could be field-specific tokenizers.
        # We choose to keep it simple here.
        tokens = self._tokenizer.strings(self._normalizer.canonicalize(buffer))
        return (self._normalizer.normalize(t) for t in tokens)

    def get_indexed_terms(self) -> Iterator[str]:
        # Assume that everything fits in memory. This would not be the case in a serious
        # large-scale application, even with compression.
        return (s for s, _ in self._dictionary)

    def get_postings_iterator(self, term: str) -> Iterator[Posting]:
        term_id = self._dictionary.get_term_id(term)
        if term_id is not None:
            return self._posting_lists[term_id].get_iterator()
        else:
            return iter([])

    def get_document_frequency(self, term: str) -> int:
        normalized_term = next(self.get_terms(term), None)
        if normalized_term is None:
            return 0

        term_id = self._dictionary.get_term_id(normalized_term)
        if term_id is not None:
            return len(self._posting_lists[term_id])
        else:
            return 0


class DummyInMemoryInvertedIndex(InMemoryInvertedIndex):
    """
    Creates a fake or dummy inverted index with no posting lists. Useful if the only effect we're
    after is the ability to infer a term's document frequency in the corpus, and we want to allow
    this to happen whether we have a real inverted index available or not: If we have a real inverted
    index at hand then use that, otherwise we can create and use this dummy version.
    """

    def __init__(self, corpus: Corpus, fields: Iterable[str], normalizer: Normalizer, tokenizer: Tokenizer):
        self._document_frequencies: Dict[int, int] = {}  # Maps a term identifier to its document frequency.
        super().__init__(corpus, fields, normalizer, tokenizer, False)

    def __repr__(self):
        return str({term: self._document_frequencies[term_id] for term, term_id in self._dictionary})

    def _append_to_posting_list(self, term_id: int, document_id: int, term_frequency: int, compressed: bool) -> None:
        # Actually, don't append to the posting list. Introduce a side-effect instead.
        self._document_frequencies[term_id] = self._document_frequencies.get(term_id, 0) + 1

    def _finalize_index(self):
        # No posting lists!
        pass

    def get_postings_iterator(self, term: str) -> Iterator[Posting]:
        # No posting lists!
        return iter([])

    def get_document_frequency(self, term: str) -> int:
        return self._document_frequencies.get(self._dictionary.get_term_id(term), 0)


class AccessLoggedInvertedIndex(InvertedIndex):
    """
    Wraps another inverted index, and keeps an in-memory log of which postings
    that have been accessed. Facilitates testing.
    """

    class AccessLoggedIterator(Iterator[Posting]):
        """
        Wraps another iterator, and updates an in-memory log of which postings
        that have been accessed. Facilitates testing.
        """

        def __init__(self, term: str, accesses: List[Tuple[str, int]], wrapped: Iterator[Posting]):
            self._term = term
            self._accesses = accesses
            self._wrapped = wrapped

        def __next__(self):
            posting = next(self._wrapped)
            self._accesses.append((self._term, posting.document_id))
            return posting

    def __init__(self, wrapped: InvertedIndex):
        self._wrapped = wrapped
        self._accesses = []

    def get_terms(self, buffer: str) -> Iterator[str]:
        return self._wrapped.get_terms(buffer)

    def get_indexed_terms(self) -> Iterator[str]:
        return self._wrapped.get_indexed_terms()

    def get_postings_iterator(self, term: str) -> Iterator[Posting]:
        return __class__.AccessLoggedIterator(term, self._accesses, self._wrapped.get_postings_iterator(term))

    def get_document_frequency(self, term: str) -> int:
        return self._wrapped.get_document_frequency(term)

    def get_history(self) -> List[Tuple[str, int]]:
        """
        Returns the list of postings that clients have accessed so far.
        """
        return self._accesses


# don't know if this is needed, but
# example run of unittests:
r"""
(venv) PS E:\Documents\in3120-2024\tests> python.exe .\assignments.py a
test_access_postings (test_inmemoryinvertedindexwithoutcompression.TestInMemoryInvertedIndexWithoutCompression.test_access_postings) ... ok
test_access_vocabulary (test_inmemoryinvertedindexwithoutcompression.TestInMemoryInvertedIndexWithoutCompression.test_access_vocabulary) ... ok
test_mesh_corpus (test_inmemoryinvertedindexwithoutcompression.TestInMemoryInvertedIndexWithoutCompression.test_mesh_corpus) ... ok
test_multiple_fields (test_inmemoryinvertedindexwithoutcompression.TestInMemoryInvertedIndexWithoutCompression.test_multiple_fields) ... ok
test_empty_lists (test_postingsmerger.TestPostingsMerger.test_empty_lists) ... ok
test_ends_with_same_so_tail_is_empty (test_postingsmerger.TestPostingsMerger.test_ends_with_same_so_tail_is_empty) ... ok
test_order_dependence (test_postingsmerger.TestPostingsMerger.test_order_dependence) ... ok
test_order_independence (test_postingsmerger.TestPostingsMerger.test_order_independence) ... ok
test_uncompressed_mesh_corpus (test_postingsmerger.TestPostingsMerger.test_uncompressed_mesh_corpus) ... ok
test_uses_yield (test_postingsmerger.TestPostingsMerger.test_uses_yield) ... ok
test_malformed_queries (test_booleansearchengine.TestBooleanSearchEngine.test_malformed_queries) ... ok
test_optimization (test_booleansearchengine.TestBooleanSearchEngine.test_optimization) ... ok
test_valid_expressions (test_booleansearchengine.TestBooleanSearchEngine.test_valid_expressions) ... ok

----------------------------------------------------------------------
Ran 13 tests in 0.739s

OK
(venv) PS E:\Documents\in3120-2024\tests> 
"""

# example run of repl.py a-1:
r"""
(venv) PS E:\Documents\in3120-2024\tests> python.exe .\repl.py a-1
Building inverted index from Cranfield corpus...
Enter one or more index terms and inspect their posting lists.
Ctrl-C to exit.
terms>stop break quantum
{'break': [{'document_id': 176, 'term_frequency': 1},
           {'document_id': 372, 'term_frequency': 1},
           {'document_id': 520, 'term_frequency': 1},
           {'document_id': 1247, 'term_frequency': 1}],
 'quantum': [{'document_id': 777, 'term_frequency': 1}],
 'stop': []}
Evaluation took 5.5999997130129486e-05 seconds.
terms>
"""