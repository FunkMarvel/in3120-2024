# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-locals
import collections
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

        # find terms and multiplicity:
        term_multiplicity = collections.Counter(self.__inverted_index.get_terms(query))
        terms = list(term_multiplicity.keys())
        terms.sort()  # ensure terms are accessed in lexicographical order for each document

        # get parameters for N-out-of-M retrieval:
        m = len(term_multiplicity)
        t = options.setdefault("match_threshold", 0.25)  # default threshold of 25% just in case
        n = max(1, min(m, int(t * m)))

        ranking_sieve = Sieve(options.setdefault("hit_count", 1))  # default hit count of 1 just in case

        posting_list_heap = []
        for term in terms:
            # get posting lists for terms that occur in corpus:
            posting_iter = self.__inverted_index.get_postings_iterator(term)
            first_posting = next(posting_iter, None)

            if first_posting is not None:
                # decided to use min-heap for document-at-a-time traversal
                heappush(posting_list_heap, (first_posting.document_id, (term, first_posting, posting_iter)))

        while len(posting_list_heap) > 0:
            # peek the smallest document_id and associated posting
            smallest_id = posting_list_heap[0][0]
            posting = posting_list_heap[0][1][1]
            if posting is None:
                continue

            ranker.reset(smallest_id)
            term_occurrences = 0

            while posting.document_id == smallest_id:
                posting_tuple = heappop(posting_list_heap)  # pop next posting for current doc
                term = posting_tuple[1][0]
                posting = posting_tuple[1][1]
                posting_iter = posting_tuple[1][2]

                # rank and count term occurrences in doc
                ranker.update(term, term_multiplicity[term], posting)
                term_occurrences += 1

                posting = next(posting_iter, None)
                if posting is not None:  # if posting list not exhausted, push next posting back on heap
                    heappush(posting_list_heap, (posting.document_id, (term, posting, posting_iter)))

                if len(posting_list_heap) <= 0:  # stop if no more postings in any list
                    break

                posting = posting_list_heap[0][1][1]  # peek next smallest id and posting on heap

            if term_occurrences >= n:  # sift all documents with at least n occurrences of terms
                score = ranker.evaluate()
                ranking_sieve.sift(score, smallest_id)

        winners = ranking_sieve.winners()
        for (score, doc_id) in winners:
            yield {"score": score, "document": self.__corpus.get_document(doc_id)}

# example run of assignments.py c-1:
r"""
(venv) PS E:\Documents\in3120-2024\tests> python.exe .\assignments.py c-1
test_canonicalized_corpus (test_simplesearchengine.TestSimpleSearchEngine.test_canonicalized_corpus) ... ok
test_document_at_a_time_traversal_mesh_corpus (test_simplesearchengine.TestSimpleSearchEngine.test_document_at_a_time_traversal_mesh_corpus) ... ok
test_mesh_corpus (test_simplesearchengine.TestSimpleSearchEngine.test_mesh_corpus) ... ok
test_synthetic_corpus (test_simplesearchengine.TestSimpleSearchEngine.test_synthetic_corpus) ... ok
test_uses_yield (test_simplesearchengine.TestSimpleSearchEngine.test_uses_yield) ... ok

----------------------------------------------------------------------
Ran 5 tests in 0.528s

OK
"""

# example run of repl.py c-1:
r"""
Indexing English news corpus...
Enter a query and find matching documents.
Lookup options are {'debug': False, 'hit_count': 5, 'match_threshold': 0.5}.
Tokenizer is SimpleTokenizer.
Ranker is SimpleRanker.
Ctrl-C to exit.
query>pollUtion waTer
[{'document': {'document_id': 9699, 'fields': {'body': 'While there are not many people in the water during the winter months, there are plenty playing by the shore with their jeans rolled up just enough so that they can feel the cool water lap up against their feet.'}},
  'score': 2.0},
 {'document': {'document_id': 7398, 'fields': {'body': 'The elevated salt levels in the water threatened some of the wildlife in the area that depend on a supply of fresh water.'}},
  'score': 2.0},
 {'document': {'document_id': 5854, 'fields': {'body': 'Polluted air, on the other hand, contains water-soluble particles, leading to clouds with more, yet smaller, water droplets.'}},
  'score': 2.0},
 {'document': {'document_id': 4515, 'fields': {'body': 'Kate Kralman, who shot the video of the MAX going through the water, was helping a friend load equipment nearby when she saw one light rail train go through the water.'}},
  'score': 2.0},
 {'document': {'document_id': 354, 'fields': {'body': "A lot of people are disputing that climate change is a reality because they don't see everybody going under water."}},
  'score': 1.0}]
Evaluation took 0.0002347999998164596 seconds.
"""