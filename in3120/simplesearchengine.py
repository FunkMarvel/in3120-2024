# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-locals
import collections
from cgitb import small
from getpass import fallback_getpass
from heapq import heappush, heappop
from typing import Iterator, Dict, Any
from .sieve import Sieve
from .ranker import Ranker
from .corpus import Corpus
from .invertedindex import InvertedIndex


class SimpleSearchEngine:
    """
    Realizes a simple query evaluator that efficiently performs N-of-M matching over an inverted index.
    I.e., if the query contains M unique query terms, each document in the result set should contain at
    least N of these m terms. For example, 2-of-3 matching over the query 'orange apple banana' would be
    logically equivalent to the following predicate:

       (orange AND apple) OR (orange AND banana) OR (apple AND banana)
       
    Note that N-of-M matching can be viewed as a type of "soft AND" evaluation, where the degree of match
    can be smoothly controlled to mimic either an OR evaluation (1-of-M), or an AND evaluation (M-of-M),
    or something in between.

    The evaluator uses the client-supplied ratio T = N/M as a parameter as specified by the client on a
    per-query basis. For example, for the query 'john paul george ringo' we have M = 4 and a specified
    threshold of T = 0.7 would imply that at least 3 of the 4 query terms have to be present in a matching
    document.
    """

    def __init__(self, corpus: Corpus, inverted_index: InvertedIndex):
        self.__corpus = corpus
        self.__inverted_index = inverted_index

    def evaluate(self, query: str, options: Dict[str, Any], ranker: Ranker) -> Iterator[Dict[str, Any]]:
        """
        Evaluates the given query, doing N-out-of-M ranked retrieval. I.e., for a supplied query having M
        unique terms, a document is considered to be a match if it contains at least N <= M of those terms.

        The matching documents, if any, are ranked by the supplied ranker, and only the "best" matches are yielded
        back to the client as dictionaries having the keys "score" (float) and "document" (Document).

        The client can supply a dictionary of options that controls the query evaluation process: The value of
        N is inferred from the query via the "match_threshold" (float) option, and the maximum number of documents
        to return to the client is controlled via the "hit_count" (int) option.
        """
        term_counter = collections.Counter(self.__inverted_index.get_terms(query))
        m = len(term_counter)
        t = options.setdefault("match_threshold", 0)
        n = min(1, max(m, int(t * m)))
        ranking_sieve = Sieve(options.setdefault("hit_count", 1))

        posting_lists = {}
        smallest_id = -1
        for term in term_counter.keys():
            term_iter = self.__inverted_index.get_postings_iterator(term)
            first_posting = next(term_iter, None)

            if first_posting is not None:
                smallest_id = first_posting.document_id
                posting_lists[term] = (first_posting, term_iter)

        postings_left = True

        while postings_left:
            postings_left = False
            smallest_id = int(1e64)
            for (term, (posting, _)) in posting_lists.items():
                if posting is not None and posting.document_id < smallest_id:
                    smallest_id = posting.document_id

            if smallest_id < 0:
                break

            ranker.reset(smallest_id)
            hit_count = 0
            for (term, (posting, iterator)) in posting_lists.items():
                if posting is not None and posting.document_id == smallest_id:
                    hit_count +=1
                    ranker.update(term, term_counter[term], posting)
                    posting_lists[term] = (next(iterator, None), iterator)
                    postings_left = True

            if hit_count >= n:
                score = ranker.evaluate()
                ranking_sieve.sift(score, smallest_id)

        winners = ranking_sieve.winners()
        for (score, doc_id) in winners:
            yield {"score": score, "document": self.__corpus.get_document(doc_id)}
        # raise NotImplementedError("You need to implement this as part of the obligatory assignment.")
