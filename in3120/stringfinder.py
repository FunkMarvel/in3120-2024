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
        terms = [(self.__normalizer.normalize(t), (s_idx, e_idx)) for (t, (s_idx, e_idx)) in tokens]

        live_states: List[Tuple[Trie, int, str]] = []

        for term, (start_idx, stop_idx) in terms:
            for state in list(live_states):
                state_node = state[0]
                state_start = state[1]
                state_term = state[2]
                state_stop = len(state_term) + state_start
                combined_term = self.__tokenizer.join(iter([
                    (state_term, (state_start, state_stop)), (term, (start_idx, stop_idx))
                ]))

                new_node = state_node.consume(combined_term[state_stop-state_start:])
                if new_node is not None:
                    if new_node.is_final():
                        tokens = self.__tokenizer.tokens(buffer[state_start:stop_idx])
                        tokenized_buffer = self.__tokenizer.join(tokens)
                        yield {
                            "match": combined_term,
                            "surface": tokenized_buffer,
                            "span": (state_start, stop_idx),
                            "meta": new_node.get_meta()
                        }
                    live_states.remove(state)
                    live_states.append((new_node, state_start, combined_term))
                else:
                    live_states.remove(state)

            node = self.__trie.consume(term)
            if node is not None:
                if node.is_final():
                    tokens = self.__tokenizer.tokens(buffer[start_idx:stop_idx])
                    tokenized_buffer = self.__tokenizer.join(tokens)
                    yield {
                        "match": term,
                        "surface": tokenized_buffer,
                        "span": (start_idx, stop_idx),
                        "meta": node.get_meta()
                    }
                live_states.append((node, start_idx, term))
        # raise NotImplementedError("You need to implement this as part of the obligatory assignment.")
