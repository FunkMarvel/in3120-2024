# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods

from typing import Iterator, Dict, Any, List, Tuple
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .trie import Trie


class StringFinder:
    """
    Given a trie encoding a dictionary of strings, efficiently finds the subset of strings in the dictionary
    that are also present in a given text buffer. I.e., in a sense computes the "intersection" or "overlap"
    between the dictionary and the text buffer.

    Uses a trie-walk algorithm similar to the Aho-Corasick algorithm with some simplifications and some minor
    NLP extensions. The running time of this algorithm is virtually independent of the size of the dictionary,
    and linear in the length of the buffer we are searching in.

    The tokenizer we use when scanning the input buffer is assumed to be the same as the one that was used
    when adding strings to the trie.
    """

    def __init__(self, trie: Trie, normalizer: Normalizer, tokenizer: Tokenizer):
        self.__trie = trie
        self.__normalizer = normalizer  # The same as was used for trie building.
        self.__tokenizer = tokenizer  # The same as was used for trie building.

    def scan(self, buffer: str) -> Iterator[Dict[str, Any]]:
        """
        Scans the given buffer and finds all dictionary entries in the trie that are also present in the
        buffer. We only consider matches that begin and end on token boundaries.

        The matches, if any, are yielded back to the client as dictionaries having the keys "match" (str),
        "surface" (str), "meta" (Optional[Any]), and "span" (Tuple[int, int]). Note that "match" refers to
        the matching dictionary entry, "surface" refers to the content of the input buffer that triggered the
        match (the surface form), and "span" refers to the exact location in the input buffer where the surface
        form is found. Depending on the normalizer that is used, "match" and "surface" may or may not differ.

        A space-normalized version of the surface form is emitted as "surface", for convenience. Clients
        that require an exact surface form that is not space-normalized can easily reconstruct the desired
        string using the emitted "span" value.

        In a serious application we'd add more lookup/evaluation features, e.g., support for prefix matching,
        support for leftmost-longest matching (instead of reporting all matches), and more.
        """

        tokens = self.__tokenizer.tokens(self.__normalizer.canonicalize(buffer))
        terms = ((self.__normalizer.normalize(t), (s_idx, e_idx)) for (t, (s_idx, e_idx)) in tokens)

        # tried using the states array suggested by @aleksaoh in the mattermost chat
        live_states: List[Tuple[Trie, int, str]] = []

        for term, (start_idx, stop_idx) in terms:
            for i, (state_node, state_start, state_term) in enumerate(live_states):
                live_states[i] = (state_node, -1, "")

                # joining terms with tokenizer to capture correct delimiter.
                state_stop = len(state_term) + state_start
                combined_term = self.__tokenizer.join(iter([
                    (state_term, (state_start, state_stop)), (term, (start_idx, stop_idx))
                ]))

                # consuming from end of previous term in sequence to capture any whitespace
                # that is in the Trie, but not in the normalized terms.
                new_node = state_node.consume(combined_term[state_stop-state_start:])
                if new_node is None:
                    continue

                if new_node.is_final():
                    tokenized_buffer = self._tokenize_buffer(buffer, state_start, stop_idx)
                    yield {
                        "match": combined_term,
                        "surface": tokenized_buffer,
                        "span": (state_start, stop_idx),
                        "meta": new_node.get_meta()
                    }
                live_states[i] = (new_node, state_start, combined_term)

            # removing dead states from state list.
            live_states = [t for t in live_states if t[1] >= 0]

            # checking if current term can iterate from root.
            node = self.__trie.consume(term)
            if node is None:
                continue

            if node.is_final():
                tokenized_buffer = self._tokenize_buffer(buffer, start_idx, stop_idx)
                yield {
                    "match": term,
                    "surface": tokenized_buffer,
                    "span": (start_idx, stop_idx),
                    "meta": node.get_meta()
                }
            live_states.append((node, start_idx, term))

    def _tokenize_buffer(self, buffer, start_idx, stop_idx):
        tokens = self.__tokenizer.tokens(buffer[start_idx:stop_idx])
        tokenized_buffer = self.__tokenizer.join(tokens)
        return tokenized_buffer

# BUG: I could not for the life of me get it to pass the relative_insensitivity_to_dictionary_size test.
# I assume the ratio means my implementation runs roughly 10-15% slower than the max allowed slack,
# but I can't find how to speed it up.
# I used the guide described here: https://github.com/FunkMarvel/in3120-2024/blob/725f88d55d6cb9c4e6b15303428197bee34066c0/seminars/gruppe1/uke04/uke04.pdf
# as reference, but I suspect my handling of live states is suboptimal. Since I have run out of time,
# I hand in my attempt as is.

# example run:
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
test_relative_insensitivity_to_dictionary_size (test_stringfinder.TestStringFinder.test_relative_insensitivity_to_dictionary_size) ... FAIL
test_scan_matches_and_spans (test_stringfinder.TestStringFinder.test_scan_matches_and_spans) ... ok
test_scan_matches_and_surface_forms_only (test_stringfinder.TestStringFinder.test_scan_matches_and_surface_forms_only) ... ok
test_uses_yield (test_stringfinder.TestStringFinder.test_uses_yield) ... ok
test_with_phonetic_normalizer_and_meta (test_stringfinder.TestStringFinder.test_with_phonetic_normalizer_and_meta) ... ok
test_with_unigram_tokenizer_for_finding_arbitrary_substrings (test_stringfinder.TestStringFinder.test_with_unigram_tokenizer_for_finding_arbitrary_substrings) ... ok

======================================================================
FAIL: test_relative_insensitivity_to_dictionary_size (test_stringfinder.TestStringFinder.test_relative_insensitivity_to_dictionary_size)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "E:\Documents\in3120-2024\tests\test_stringfinder.py", line 95, in test_relative_insensitivity_to_dictionary_size
    self.assertLessEqual(ratio - slack, 1.0)
AssertionError: 1.1029411934646065 not less than or equal to 1.0

----------------------------------------------------------------------
Ran 20 tests in 1.822s

FAILED (failures=1)
(venv) PS E:\Documents\in3120-2024\tests> 
"""