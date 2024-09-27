# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import sys
from bisect import bisect_left, bisect
from itertools import takewhile
from typing import Any, Dict, Iterator, Iterable, Tuple, List
from collections import Counter

from pygments.lexer import default
from spacy.lang.fi.tokenizer_exceptions import suffix

from .corpus import Corpus
from .normalizer import Normalizer
from .tokenizer import Tokenizer


class SuffixArray:
    """
    A simple suffix array implementation. Allows us to conduct efficient substring searches.
    The prefix of a suffix is an infix!
    In a serious application we'd make use of least common prefixes (LCPs), pay more attention
    to memory usage, and add more lookup/evaluation features.
    """

    def __init__(self, corpus: Corpus, fields: Iterable[str], normalizer: Normalizer, tokenizer: Tokenizer):
        self.__corpus = corpus
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer
        self.__haystack: List[Tuple[int, str]] = []  # The (<document identifier>, <searchable content>) pairs.
        self.__suffixes: List[Tuple[int, int]] = []  # The sorted (<haystack index>, <start offset>) pairs.
        self.__build_suffix_array(fields)  # Construct the haystack and the suffix array itself.

    def __get_suffix_string(self, suffix_tuple):
        haystack_idx = suffix_tuple[0]
        offset = suffix_tuple[1]
        return self.__haystack[haystack_idx][1][offset:]

    def __build_suffix_array(self, fields: Iterable[str]) -> None:
        """
        Builds a simple suffix array from the set of named fields in the document collection.
        The suffix array allows us to search across all named fields in one go.
        """
        for document in self.__corpus:
            buffer = ""
            for field in fields:
                text = document.get_field(field, None)
                if text is None:
                    continue

                buffer += " " + text
            self.__haystack.append((document.document_id, self.__normalize(buffer)))

        for haystack_idx, (_, text) in enumerate(self.__haystack):
            for token_start_idx, _ in self.__tokenizer.spans(text):
               self.__suffixes.append((haystack_idx, token_start_idx))

        self.__suffixes.sort(key=self.__get_suffix_string)

        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def __normalize(self, buffer: str) -> str:
        """
        Produces a normalized version of the given string. Both queries and documents need to be
        identically processed for lookups to succeed.
        """
        tokens = self.__tokenizer.tokens(self.__normalizer.canonicalize(buffer))
        terms = ((self.__normalizer.normalize(t), (start_idx, stop_idx)) for t, (start_idx, stop_idx) in tokens)
        new_text = self.__tokenizer.join(terms)
        return new_text
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def __binary_search(self, needle: str) -> int:
        """
        Does a binary search for a given normalized query (the needle) in the suffix array (the haystack).
        Returns the position in the suffix array where the normalized query is either found, or, if not found,
        should have been inserted.

        Kind of silly to roll our own binary search instead of using the bisect module, but seems needed
        prior to Python 3.10 due to how we represent the suffixes via (index, offset) tuples. Version 3.10
        added support for specifying a key.
        """
        return bisect_left(self.__suffixes, needle, key=self.__get_suffix_string)
        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def evaluate(self, query: str, options: dict) -> Iterator[Dict[str, Any]]:
        """
        Evaluates the given query, doing a "phrase prefix search".  E.g., for a supplied query phrase like
        "to the be", we return documents that contain phrases like "to the bearnaise", "to the best",
        "to the behemoth", and so on. I.e., we require that the query phrase starts on a token boundary in the
        document, but it doesn't necessarily have to end on one.

        The matching documents are ranked according to how many times the query substring occurs in the document,
        and only the "best" matches are yielded back to the client. Ties are resolved arbitrarily.

        The client can supply a dictionary of options that controls this query evaluation process: The maximum
        number of documents to return to the client is controlled via the "hit_count" (int) option.

        The results yielded back to the client are dictionaries having the keys "score" (int) and
        "document" (Document).
        """
        normalized_query = self.__normalize(query)

        first_idx = self.__binary_search(normalized_query)
        results: Dict[int, int] = {}  # keeping track of scores for each document.

        if len(normalized_query) < 1 or first_idx < 0 or first_idx >= len(self.__suffixes):
            return

        for i in range(first_idx, len(self.__suffixes)):
            haystack_idx = self.__suffixes[i][0]
            offset = self.__suffixes[i][1]
            document_id = self.__haystack[haystack_idx][0]

            if len(self.__haystack[haystack_idx][1][offset:]) < len(normalized_query):
                continue

            if normalized_query == self.__haystack[haystack_idx][1][offset:][:len(normalized_query)]:
                score = results.setdefault(document_id, 0)  # retrieve existing score, or set zero if first hit
                results[document_id] = score + 1
            else:
                break

        # I know sorting all matches is inefficient, but I ran out of time for refactoring it to use sieve
        matches = list(results.items())
        matches.sort(key=lambda x: x[1], reverse=True)

        hit_count = options.setdefault("hit_count", None)
        if hit_count is None:
            return

        for i, (document_id, score) in enumerate(matches):
            if i+1 > hit_count:
                return

            yield {"document": self.__corpus.get_document(document_id), "score": score}

        #raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

# example run: (the fail is from stringfinder.py)
r"""
(venv) PS E:\Documents\in3120-2024\tests> python.exe .\assignments.py b-1
test_canonicalized_corpus (test_suffixarray.TestSuffixArray.test_canonicalized_corpus) ... ok
test_cran_corpus (test_suffixarray.TestSuffixArray.test_cran_corpus) ... ok
test_memory_usage (test_suffixarray.TestSuffixArray.test_memory_usage) ... ok
test_multiple_fields (test_suffixarray.TestSuffixArray.test_multiple_fields) ... ok
test_uses_yield (test_suffixarray.TestSuffixArray.test_uses_yield) ... ok
test_add_is_idempotent (test_trie.TestTrie.test_add_is_idempotent) ... ok
test_add_is_idempotent_unless_meta_data_differs (test_trie.TestTrie.test_add_is_idempotent_unless_meta_data_differs) ... ok
test_child (test_trie.TestTrie.test_child) ... ok
test_consume_and_final (test_trie.TestTrie.test_consume_and_final) ... ok
test_containment (test_trie.TestTrie.test_containment) ... ok
test_dump_strings (test_trie.TestTrie.test_dump_strings) ... ok
test_transitions (test_trie.TestTrie.test_transitions) ... ok
test_with_meta_data (test_trie.TestTrie.test_with_meta_data) ... ok
test_mesh_terms_in_cran_corpus (test_stringfinder.TestStringFinder.test_mesh_terms_in_cran_corpus) ... ok
test_relative_insensitivity_to_dictionary_size (test_stringfinder.TestStringFinder.test_relative_insensitivity_to_dictionary_size) ... 1.4522292816520965
FAIL
test_scan_matches_and_spans (test_stringfinder.TestStringFinder.test_scan_matches_and_spans) ... ok
test_scan_matches_and_surface_forms_only (test_stringfinder.TestStringFinder.test_scan_matches_and_surface_forms_only) ... ok
test_uses_yield (test_stringfinder.TestStringFinder.test_uses_yield) ... ok
test_with_phonetic_normalizer_and_meta (test_stringfinder.TestStringFinder.test_with_phonetic_normalizer_and_meta) ... ok
test_with_unigram_tokenizer_for_finding_arbitrary_substrings (test_stringfinder.TestStringFinder.test_with_unigram_tokenizer_for_finding_arbitrary_substrings) ... ok

======================================================================
FAIL: test_relative_insensitivity_to_dictionary_size (test_stringfinder.TestStringFinder.test_relative_insensitivity_to_dictionary_size)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "E:\Documents\in3120-2024\tests\test_stringfinder.py", line 96, in test_relative_insensitivity_to_dictionary_size
    self.assertLessEqual(ratio - slack, 1.0)
AssertionError: 1.1022292816520967 not less than or equal to 1.0

----------------------------------------------------------------------
Ran 20 tests in 1.840s

FAILED (failures=1)
(venv) PS E:\Documents\in3120-2024\tests> 
"""